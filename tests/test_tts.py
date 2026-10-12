"""Tests for :mod:`app.pipeline.tts`.

OmniVoice is never imported, no weights are downloaded and no inference runs: the
cloning engine is replaced by the module's own adapter interface, and the one place
where the real adapter is exercised - the loader being asked for an engine - is
replaced by an in-process fake. Audio is synthesized with ``numpy`` and written with
``soundfile``, so every test here runs in milliseconds without a GPU, FFmpeg or
network access.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import tts
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import (
    InvalidDialogueError,
    InvalidEngineError,
    InvalidInputError,
    InvalidVoiceProfileError,
    MissingInputError,
    MissingOutputError,
    MissingVoiceProfileError,
    PerformanceControls,
    SynthesisError,
    VoiceCloningEngine,
    reset_engine_cache,
    resolve_tts_directory,
    synthesize_dialogue,
)
from app.pipeline.voice_profiles import (
    REFERENCE_FILENAME,
    VoiceProfile,
    portable_path,
    speaker_directory_name,
)

STEM_RATE = 48_000
CLIP_RATE = 24_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``.

    The engine is pinned to the cloning path: this file exercises that path, so it has
    to ask for it rather than inherit whatever the project default happens to be.
    """

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
        "voice_profile_dir": root / "voices",
        "tts_engine": "omnivoice",
        "omnivoice_model": "k2-fsa/OmniVoice",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, rate, format="WAV", subtype="PCM_16")
    return path


def _stem(root: Path, *, seconds: float = 16.0, rate: int = STEM_RATE) -> Path:
    """Write a speech-stem stand-in whose every sample is identifiable."""

    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    data = 0.5 * np.sin(2.0 * np.pi * 3.0 * times) + 0.2 * np.sin(2.0 * np.pi * 37.0 * times)
    return _write_wav(root / "movie_speech.wav", data.astype(np.float32), rate)


def _profile(
    root: Path,
    speaker_id: str = "SPEAKER_00",
    *,
    seconds: float = 5.0,
    write: bool = True,
) -> VoiceProfile:
    """Return a valid profile, with its reference audio written by default."""

    reference = root / "voices" / speaker_directory_name(speaker_id) / REFERENCE_FILENAME
    if write:
        _write_wav(reference, np.full(int(seconds * CLIP_RATE), 0.4, dtype=np.float32), CLIP_RATE)
    return VoiceProfile(
        speaker_id=speaker_id,
        reference_audio=reference,
        reference_start=10.0,
        reference_end=10.0 + seconds,
    )


def _line(
    *,
    speaker_id: str = "SPEAKER_00",
    start: float = 2.0,
    end: float = 6.0,
    amharic: str = "ሰላም ዓለም።",
    emotion: str = "neutral",
    intensity: float = 0.5,
    delivery: str = "calm and conversational",
    pause_before: float = 0.0,
    pause_after: float = 0.0,
) -> AdaptedDialogue:
    """Return one adapted dialogue line."""

    return AdaptedDialogue(
        speaker_id=speaker_id,
        start=start,
        end=end,
        source_text="hello world",
        amharic=amharic,
        emotion=emotion,
        intensity=intensity,
        delivery=delivery,
        pause_before=pause_before,
        pause_after=pause_after,
    )


# ---------------------------------------------------------------------------
# Fake engine
# ---------------------------------------------------------------------------


class FakeOmnivoice(VoiceCloningEngine):
    """In-process stand-in for the OmniVoice cloning adapter."""

    name = "fake-omnivoice"

    def __init__(
        self,
        *,
        model: str = "fake/omnivoice",
        seconds: float = 0.75,
        rate: int = CLIP_RATE,
        error: Exception | None = None,
        error_at: int | None = None,
        write: bool = True,
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.model = model
        self.seconds = seconds
        self.rate = rate
        self.error = error
        self.error_at = error_at
        self.write = write
        self.events = events if events is not None else []
        self.calls: list[dict[str, object]] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self.rate

    def synthesize(
        self,
        *,
        text: str,
        voice_reference: Path,
        destination: Path,
        speaking_rate: float | None = None,
        reference_text: str | None = None,
    ) -> Path:
        self.calls.append(
            {
                "text": text,
                "voice_reference": Path(voice_reference),
                "destination": Path(destination),
                "speaking_rate": speaking_rate,
                "reference_text": reference_text,
            }
        )
        self.events.append(("omnivoice", Path(destination).name))
        if self.error is not None and (self.error_at is None or self.error_at == len(self.calls) - 1):
            raise self.error
        self._loaded = True
        if not self.write:
            return Path(destination)
        times = np.arange(int(self.seconds * self.rate), dtype=np.float64) / self.rate
        data = 0.3 * np.sin(2.0 * np.pi * 220.0 * times)
        return _write_wav(Path(destination), data.astype(np.float32), self.rate)


@dataclass
class Rig:
    """A synthetic stem, one voice profile and a fake cloning engine."""

    root: Path
    stem: Path
    profile: VoiceProfile
    engine: FakeOmnivoice
    events: list[tuple[str, str]]

    def settings(self, **overrides: object) -> Settings:
        return _settings(self.root, **overrides)

    def run(self, *lines: AdaptedDialogue, **kwargs: object) -> list[tts.TtsClip]:
        settings = kwargs.pop("settings", self.settings())
        return synthesize_dialogue(
            list(lines) or [_line()],
            self.stem,
            {self.profile.speaker_id: self.profile},
            settings=settings,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.fixture(autouse=True)
def _clean_engine_cache() -> None:
    """Keep the process-wide engine cache from leaking between tests."""

    reset_engine_cache()
    yield
    reset_engine_cache()


@pytest.fixture(scope="module")
def shared_stem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One speech stem for the whole module: the stem is only ever read."""

    return _stem(tmp_path_factory.mktemp("stem"))


@pytest.fixture
def rig(tmp_path: Path, shared_stem: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    events: list[tuple[str, str]] = []
    rig = Rig(
        root=tmp_path,
        stem=shared_stem,
        profile=_profile(tmp_path),
        engine=FakeOmnivoice(events=events),
        events=events,
    )
    # The loader is what a run actually asks for an engine, so replacing it is what
    # keeps the default path from reaching OmniVoice - and what lets a test swap the
    # engine out mid-test by assigning to ``rig.engine``.
    monkeypatch.setitem(tts.VOICE_CLONING_LOADERS, "omnivoice", lambda *, settings: rig.engine)
    return rig


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


def test_the_cloning_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        VoiceCloningEngine()  # type: ignore[abstract]


def test_the_fake_engine_is_an_adapter_of_that_interface(rig: Rig) -> None:
    assert isinstance(rig.engine, VoiceCloningEngine)
    assert rig.engine.is_loaded is False


def test_models_are_not_loaded_at_import_time() -> None:
    """Importing the module must not touch the runtime either."""

    probe = (
        "import sys, app.pipeline.tts as tts;"
        "mods = {'omnivoice', 'omnivoice_tts', 'hydra', 'omegaconf'};"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in mods);"
        "assert not loaded, loaded;"
        "assert tts._OMNIVOICE_ENGINES == {};"
        "assert tts._MMS_ENGINES == {};"
        "assert set(tts.VOICE_CLONING_LOADERS) == {'omnivoice'};"
        "print('lazy')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout


def test_constructing_the_real_engine_loads_nothing() -> None:
    engine = tts.OmniVoiceEngine(
        model="k2-fsa/OmniVoice",
        device="cpu",
        steps=16,
        guidance_scale=2.0,
        torch_dtype="float16",
    )

    assert engine.is_loaded is False
    assert "omnivoice" not in sys.modules
    assert engine.model == "k2-fsa/OmniVoice"
    assert engine.device == "cpu"
    assert engine.sample_rate == tts.OMNIVOICE_SAMPLE_RATE


def test_engine_names_are_reported_in_the_clip(rig: Rig) -> None:
    (clip,) = rig.run()

    assert clip.performance_engine == "fake-omnivoice"
    assert clip.style_engine == "none"
    assert rig.engine.is_loaded is True


# ---------------------------------------------------------------------------
# Engine selection and caching
# ---------------------------------------------------------------------------


def test_the_engine_is_resolved_once_for_the_whole_run(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _loader(*, settings: Settings) -> VoiceCloningEngine:
        calls.append("omnivoice")
        return rig.engine

    monkeypatch.setitem(tts.VOICE_CLONING_LOADERS, "omnivoice", _loader)

    clips = rig.run(
        _line(start=2.0, end=4.0), _line(start=4.0, end=6.0), _line(start=6.0, end=8.0)
    )

    assert len(clips) == 3
    assert calls == ["omnivoice"]
    assert len(rig.engine.calls) == 3


def test_an_injected_engine_is_used_without_loading_the_default_one(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*, settings: Settings) -> VoiceCloningEngine:
        raise AssertionError("the default engine must not be resolved")

    monkeypatch.setitem(tts.VOICE_CLONING_LOADERS, "omnivoice", _boom)

    (clip,) = tts.synthesize_dialogue_detailed(
        [_line()],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
        tts_engine=rig.engine,
    ).clips

    assert clip.performance_engine == "fake-omnivoice"


def test_a_wrong_engine_adapter_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidEngineError, match="VoiceCloningEngine"):
        tts.synthesize_dialogue_detailed(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
            tts_engine=object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_dialogue_returns_no_clips_and_touches_nothing(rig: Rig) -> None:
    clips = synthesize_dialogue(
        [],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert clips == []
    assert rig.engine.calls == []
    assert not (rig.root / "tts").exists()


def test_non_dialogue_input_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidDialogueError, match="dialogue\\[0\\]"):
        rig.run("not a line")  # type: ignore[arg-type]


def test_unordered_dialogue_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidDialogueError, match="chronological"):
        rig.run(_line(start=20.0, end=22.0), _line(start=10.0, end=12.0))


def test_missing_voice_profile_is_rejected(rig: Rig) -> None:
    with pytest.raises(MissingVoiceProfileError, match="SPEAKER_09"):
        synthesize_dialogue(
            [_line(speaker_id="SPEAKER_09")],
            rig.stem,
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
        )


def test_a_profile_stored_under_the_wrong_key_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="stored under the key"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {"SPEAKER_99": rig.profile},
            settings=rig.settings(),
        )


def test_a_non_profile_value_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="expected a VoiceProfile"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: object()},  # type: ignore[dict-item]
            settings=rig.settings(),
        )


def test_a_non_mapping_profile_collection_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="must be a mapping"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            [rig.profile],  # type: ignore[arg-type]
            settings=rig.settings(),
        )


def test_a_missing_reference_audio_is_reported_before_any_model_runs(
    tmp_path: Path, rig: Rig
) -> None:
    profile = _profile(tmp_path, speaker_id="SPEAKER_07", write=False)

    with pytest.raises(MissingInputError, match="voice reference of speaker"):
        synthesize_dialogue(
            [_line(speaker_id="SPEAKER_07")],
            rig.stem,
            {profile.speaker_id: profile},
            settings=rig.settings(),
        )

    assert rig.engine.calls == []


def test_a_reference_that_is_not_a_file_is_reported(rig: Rig) -> None:
    reference = rig.root / "voices" / speaker_directory_name("SPEAKER_00") / REFERENCE_FILENAME
    reference.unlink()
    reference.mkdir()

    with pytest.raises(InvalidInputError, match="not a regular file"):
        rig.run()

    assert rig.engine.calls == []


# ---------------------------------------------------------------------------
# Performance metadata
# ---------------------------------------------------------------------------


def test_the_seed_is_a_pure_function_of_the_line() -> None:
    line = _line(amharic="ተመሳሳይ መስመር")
    seed = int(tts._line_digest(line, model="m")[:8], 16)

    first = PerformanceControls.from_dialogue(line, seed=seed)
    second = PerformanceControls.from_dialogue(line, seed=seed)

    assert first.seed == second.seed
    assert first.seed != int(tts._line_digest(_line(amharic="ሌላ መስመር"), model="m")[:8], 16)


def test_the_clip_keeps_the_whole_line_and_its_performance(rig: Rig) -> None:
    line = _line(emotion="afraid", intensity=0.8, delivery="nervous and hesitant", pause_after=0.25)

    (clip,) = rig.run(line)

    assert clip.dialogue is line
    assert clip.speaker_id == line.speaker_id
    assert clip.start == line.start
    assert clip.end == line.end
    assert clip.amharic == line.amharic
    assert clip.original_duration == 4.0
    assert clip.performance.emotion == "afraid"
    assert clip.performance.intensity == 0.8
    assert clip.performance.delivery == "nervous and hesitant"
    assert clip.performance.pause_after == 0.25
    assert clip.dialogue.pause_after == 0.25


# ---------------------------------------------------------------------------
# Pauses and duration metadata
# ---------------------------------------------------------------------------


def test_pauses_are_rendered_as_silence(rig: Rig) -> None:
    line = _line(pause_before=0.3, pause_after=0.5)

    (clip,) = rig.run(line)

    samples, rate = sf.read(str(clip.audio_path), dtype="float32")
    lead = int(round(0.3 * rate))
    trail = int(round(0.5 * rate))

    assert clip.rendered_pause_before == pytest.approx(0.3)
    assert clip.rendered_pause_after == pytest.approx(0.5)
    assert np.allclose(samples[:lead], 0.0)
    assert np.allclose(samples[-trail:], 0.0)
    assert clip.speech_duration == pytest.approx((samples.shape[0] - lead - trail) / rate)
    assert clip.duration == pytest.approx(samples.shape[0] / rate, abs=1.0 / rate)


def test_pauses_are_capped_and_the_declared_value_is_kept(rig: Rig) -> None:
    line = _line(pause_before=5.0, pause_after=0.0)

    (clip,) = rig.run(line, settings=rig.settings(tts_max_pause_seconds=0.4))

    assert clip.rendered_pause_before == pytest.approx(0.4)
    assert clip.performance.pause_before == 5.0
    assert clip.dialogue.pause_before == 5.0


def test_duration_metadata_describes_the_delivered_audio(rig: Rig) -> None:
    (clip,) = rig.run(_line(pause_after=0.2))

    samples, rate = sf.read(str(clip.audio_path), dtype="float32", always_2d=True)

    assert clip.sample_rate == rate == CLIP_RATE
    assert samples.shape[0] / rate == pytest.approx(clip.duration, abs=1.0 / rate)
    assert clip.take_path.is_file()
    # No supported engine cuts a performance prompt from the stem any more, so the clip
    # records its absence rather than a path that was never written.
    assert clip.performance_reference_path is None
    assert clip.voice_reference_path == rig.profile.resolve_reference_audio()


def test_no_duration_is_forced_on_the_generated_audio(rig: Rig) -> None:
    """The clip is as long as the engine made it, not as long as the line."""

    line = _line(start=0.0, end=16.0)

    (clip,) = rig.run(line)

    # The fake engine emits its own fixed length, and nothing here stretches that to
    # the 16 s the line occupied.
    assert clip.original_duration == 16.0
    assert clip.speech_duration == pytest.approx(rig.engine.seconds, abs=0.01)
    assert clip.duration != pytest.approx(clip.original_duration)


def test_metadata_is_json_safe_and_complete(rig: Rig) -> None:
    line = _line(emotion="sad", intensity=0.2, delivery="soft", pause_before=0.1)

    (clip,) = rig.run(line)
    payload = clip.to_dict()

    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["speaker_id"] == "SPEAKER_00"
    assert decoded["amharic"] == clip.dialogue.amharic
    assert decoded["performance"]["emotion"] == "sad"
    assert decoded["performance"]["pause_before"] == 0.1
    assert payload["speech_duration"] == clip.speech_duration
    assert payload["duration"] == clip.duration
    assert payload["performance_engine"] == "fake-omnivoice"
    assert payload["style_engine"] == "none"
    assert payload["performance_reference_path"] is None
    assert payload["audio_path"] == portable_path(clip.audio_path)
    assert "/" in payload["audio_path"] and "\\" not in payload["audio_path"]


# ---------------------------------------------------------------------------
# Deterministic paths and reuse
# ---------------------------------------------------------------------------


def test_artifacts_live_under_the_work_directory(rig: Rig) -> None:
    (clip,) = rig.run()

    base = resolve_tts_directory(settings=rig.settings())
    assert base == rig.root / tts.TTS_DIRECTORY_NAME
    for path in (clip.audio_path, clip.take_path):
        assert path.parent.parent == base
    assert clip.audio_path.parent.name == "clips"
    assert clip.take_path.parent.name == "takes"
    assert clip.voice_reference_path == rig.profile.resolve_reference_audio()


def test_artifact_names_are_deterministic_and_order_independent(rig: Rig) -> None:
    lines = [_line(start=2.0, end=4.0), _line(start=4.0, end=6.0)]

    together = rig.run(*lines)
    separately = [rig.run(line)[0] for line in reversed(lines)]

    assert len(together) == 2
    assert together[0].audio_path == separately[1].audio_path
    assert together[1].audio_path == separately[0].audio_path
    assert together[0].audio_path.name.startswith("SPEAKER_00_")
    assert together[0].audio_path.name == together[0].take_path.name


def test_a_different_line_gets_a_different_path(rig: Rig) -> None:
    (first,) = rig.run(_line(amharic="አንድ"))
    (second,) = rig.run(_line(amharic="ሁለት"))

    assert first.audio_path != second.audio_path


def test_a_different_engine_gets_a_different_path(rig: Rig) -> None:
    """Artifacts are keyed by the engine *and* the checkpoint that shapes them.

    A cloning engine names itself, so two engines that speak different weights must
    not share one line's audio.
    """

    line = _line()
    (default,) = rig.run(line)

    rig.engine = FakeOmnivoice(model="someone/else")
    (other,) = rig.run(line)

    assert default.audio_path != other.audio_path


def test_existing_artifacts_are_reused_instead_of_regenerated(rig: Rig) -> None:
    line = _line(pause_after=0.2)
    (first,) = rig.run(line)

    # A second run whose engine would fail if it were called again.
    rig.engine = FakeOmnivoice(error=SynthesisError("must not be called"))
    (second,) = rig.run(line)

    assert rig.engine.calls == []
    assert second.audio_path == first.audio_path
    assert second.to_dict()["duration"] == first.to_dict()["duration"]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_a_failing_engine_call_is_reported_with_the_line(rig: Rig) -> None:
    rig.engine = FakeOmnivoice(error=SynthesisError("no speaker tokens"), error_at=1)

    with pytest.raises(SynthesisError, match="line 1 of speaker 'SPEAKER_00'"):
        rig.run(_line(start=2.0, end=4.0), _line(start=4.0, end=6.0))


def test_the_failing_line_is_identified_by_its_window(rig: Rig) -> None:
    rig.engine = FakeOmnivoice(error=SynthesisError("reference too short"))

    with pytest.raises(SynthesisError, match=r"\(2\.000s-6\.000s\)"):
        rig.run()


# ---------------------------------------------------------------------------
# Lines that cannot be dubbed
# ---------------------------------------------------------------------------


def _short_line() -> AdaptedDialogue:
    """The 0.22s window in which the actor says "Walt.".

    It used to be dropped as undubbable, which left a hole in the dub exactly where the
    audience expects a name to be called. A brief English line is normal rather than a
    defect; the Amharic that replaces it needs more time, and the timing stage finds
    that time by moving the lines after it.
    """

    return _line(start=5.836, end=6.056, amharic="ዋልት።")


def test_a_short_window_is_voiced_rather_than_dropped(rig: Rig) -> None:
    result = tts.synthesize_dialogue_detailed(
        [_short_line()],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert result.skipped == ()
    assert [clip.index for clip in result.clips] == [0]
    assert result.clips[0].dialogue.amharic == "ዋልት።"
    (note,) = result.notes
    assert "shorter than the" in note
    assert "line(s) 0" in note


def test_the_short_window_is_reported_even_though_the_line_is_spoken(rig: Rig) -> None:
    """A run still has to be able to see that a line had almost no room."""

    result = tts.synthesize_dialogue_detailed(
        [_short_line()],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    payload = result.as_dict()
    assert payload["notes"]
    assert payload["skipped"] == 0


def test_the_engine_is_not_called_for_a_line_with_nothing_to_pronounce(rig: Rig) -> None:
    """The guard has to run *before* the engine, or it costs a wasted synthesis."""

    tts.synthesize_dialogue_detailed(
        [_line(amharic="።፣?!")],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert rig.engine.calls == []


def test_a_short_line_does_not_stop_the_lines_around_it(rig: Rig) -> None:
    result = tts.synthesize_dialogue_detailed(
        [_line(start=2.0, end=4.0), _short_line(), _line(start=8.0, end=10.0)],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert [clip.index for clip in result.clips] == [0, 1, 2]
    assert result.skipped == ()
    assert result.attempted == 3


def test_a_line_with_nothing_to_pronounce_is_skipped(rig: Rig) -> None:
    """Punctuation and whitespace cannot become speech."""

    result = tts.synthesize_dialogue_detailed(
        [_line(amharic="።፣?!")],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    (skipped,) = result.skipped
    assert "nothing to pronounce" in skipped.reason
    assert result.clips == ()


def test_a_latin_only_line_is_attempted_rather_than_dropped(rig: Rig) -> None:
    """An English-derived word is a word, and dropping the line would be worse.

    A borrowed word is meant to survive - an Amharic speaker says "ኮምፒውተር", not a
    dictionary equivalent - so a line that arrives in Roman script is voiced and
    reported, not silently removed. Getting it into Fidel is the adaptation stage's
    job, not this one's.
    """

    result = tts.synthesize_dialogue_detailed(
        [_line(amharic="Walt")],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert result.skipped == ()
    assert len(result.clips) == 1


def test_a_mixed_script_line_is_attempted(rig: Rig) -> None:
    """Fidel plus a borrowed word is the normal case, not an error."""

    result = tts.synthesize_dialogue_detailed(
        [_line(amharic="ዋልት ኮምፒውተር ገዛ")],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert result.skipped == ()
    assert len(result.clips) == 1


def test_the_reporting_threshold_is_configurable(rig: Rig) -> None:
    """A pod can silence the note for a film whose turns really are that short."""

    result = tts.synthesize_dialogue_detailed(
        [_short_line()],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(tts_min_line_seconds=0.05),
    )

    assert result.skipped == ()
    assert result.notes == ()
    assert len(result.clips) == 1


def test_synthesize_dialogue_still_returns_the_clips(rig: Rig) -> None:
    """The delegating function keeps its contract for existing callers."""

    clips = tts.synthesize_dialogue(
        [_line(start=2.0, end=4.0), _short_line()],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert isinstance(clips, list)
    assert [clip.index for clip in clips] == [0, 1]


def test_an_engine_failure_is_skipped_when_the_caller_asks(rig: Rig) -> None:
    """Off by default, because a failure should be visible - see the tests above."""

    rig.engine = FakeOmnivoice(error=SynthesisError("no speaker tokens"), error_at=1)

    result = tts.synthesize_dialogue_detailed(
        [_line(start=2.0, end=4.0), _line(start=4.0, end=6.0)],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(tts_continue_on_failure=True),
    )

    assert [clip.index for clip in result.clips] == [0]
    (skipped,) = result.skipped
    assert skipped.index == 1
    assert skipped.reason.startswith("synthesis failed")
    assert "no speaker tokens" in skipped.reason
    assert result.failed == result.skipped


def test_a_skipped_line_is_json_safe(rig: Rig) -> None:
    result = tts.synthesize_dialogue_detailed(
        [_line(start=5.836, end=6.056, amharic="።፣?!")],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    payload = result.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["clips"] == 0
    assert payload["skipped"] == 1
    assert payload["skipped_lines"][0]["duration"] == 0.22


def test_an_engine_that_writes_nothing_is_reported(rig: Rig) -> None:
    rig.engine = FakeOmnivoice(write=False)

    with pytest.raises(MissingOutputError, match="fake-omnivoice reported success"):
        rig.run()


def test_unexpected_engine_errors_propagate_unchanged(rig: Rig) -> None:
    rig.engine = FakeOmnivoice(error=RuntimeError("cuda out of memory"))

    with pytest.raises(RuntimeError, match="cuda out of memory"):
        rig.run()
