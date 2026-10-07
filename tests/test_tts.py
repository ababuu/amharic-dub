"""Tests for :mod:`app.pipeline.tts`.

Chatterbox and Seed-VC are never imported, no weights are downloaded and no
inference runs: both engines are replaced by the module's own adapter
interfaces, and the one place where the real adapters are exercised - the
Chatterbox loader and the Seed-VC wrapper - is replaced by an in-process fake.
Audio is synthesized with ``numpy`` and written with ``soundfile``, so every test
here runs in milliseconds without a GPU, FFmpeg or network access.
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
import torch

from app.config import Settings
from app.pipeline import tts
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import (
    BASE_CFG_WEIGHT,
    BASE_EXAGGERATION,
    BASE_TEMPERATURE,
    CFG_WEIGHT_BOUNDS,
    CHATTERBOX_SAMPLE_RATE,
    EXAGGERATION_BOUNDS,
    TEMPERATURE_BOUNDS,
    ChatterboxAmharicEngine,
    ChatterboxPerformanceEngine,
    ConfigurationError,
    ConversionError,
    EngineLoadError,
    InvalidAudioError,
    InvalidDialogueError,
    InvalidEngineError,
    InvalidInputError,
    InvalidVoiceProfileError,
    MissingInputError,
    MissingOutputError,
    MissingVoiceProfileError,
    PerformanceControls,
    SeedVcV2Engine,
    SynthesisError,
    VoiceConversionEngine,
    extract_performance_reference,
    load_chatterbox_engine,
    load_seed_vc_engine,
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
SEED_VC_RATE = 22_050
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``."""

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
        "voice_profile_dir": root / "voices",
        "seed_vc_repo_path": root / "seed-vc",
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
        _write_wav(reference, np.full(int(seconds * 24_000), 0.4, dtype=np.float32), 24_000)
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
# Fake engines
# ---------------------------------------------------------------------------


class FakeChatterbox(ChatterboxPerformanceEngine):
    """In-process stand-in for the Chatterbox Amharic adapter."""

    name = "fake-chatterbox"

    def __init__(
        self,
        *,
        seconds: float = 0.75,
        rate: int = 24_000,
        error: Exception | None = None,
        error_at: int | None = None,
        write: bool = True,
        events: list[tuple[str, str]] | None = None,
    ) -> None:
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
        performance_reference: Path,
        controls: PerformanceControls,
        destination: Path,
    ) -> Path:
        self.calls.append(
            {
                "text": text,
                "performance_reference": Path(performance_reference),
                "controls": controls,
                "destination": Path(destination),
            }
        )
        self.events.append(("chatterbox", Path(destination).name))
        if self.error is not None and (self.error_at is None or self.error_at == len(self.calls) - 1):
            raise self.error
        self._loaded = True
        if not self.write:
            return Path(destination)
        times = np.arange(int(self.seconds * self.rate), dtype=np.float64) / self.rate
        data = 0.3 * np.sin(2.0 * np.pi * 220.0 * times)
        return _write_wav(Path(destination), data.astype(np.float32), self.rate)


class FakeSeedVc(VoiceConversionEngine):
    """In-process stand-in for the Seed-VC V2 converter."""

    name = "fake-seedvc"

    def __init__(
        self,
        *,
        rate: int = SEED_VC_RATE,
        error: Exception | None = None,
        write: bool = True,
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.rate = rate
        self.error = error
        self.write = write
        self.events = events if events is not None else []
        self.calls: list[dict[str, Path]] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def convert(
        self,
        *,
        source_audio: Path,
        identity_reference: Path,
        destination: Path,
    ) -> Path:
        self.calls.append(
            {
                "source_audio": Path(source_audio),
                "identity_reference": Path(identity_reference),
                "destination": Path(destination),
            }
        )
        self.events.append(("seed-vc", Path(destination).name))
        if self.error is not None:
            raise self.error
        self._loaded = True
        if not self.write:
            return Path(destination)

        # Half the source duration, so a duration that accidentally matched the
        # original line would be impossible.
        samples, rate = sf.read(str(source_audio), dtype="float32", always_2d=True)
        times = np.arange(max(1, samples.shape[0] // 2), dtype=np.float64) / self.rate
        data = 0.4 * np.sin(2.0 * np.pi * 180.0 * times)
        return _write_wav(Path(destination), data.astype(np.float32), self.rate)


@dataclass
class Rig:
    """A synthetic stem, one voice profile and a pair of fake engines."""

    root: Path
    stem: Path
    profile: VoiceProfile
    chatterbox: FakeChatterbox
    seedvc: FakeSeedVc
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
            performance_engine=self.chatterbox,
            style_engine=self.seedvc,
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
def rig(tmp_path: Path, shared_stem: Path) -> Rig:
    events: list[tuple[str, str]] = []
    return Rig(
        root=tmp_path,
        stem=shared_stem,
        profile=_profile(tmp_path),
        chatterbox=FakeChatterbox(events=events),
        seedvc=FakeSeedVc(events=events),
        events=events,
    )


# ---------------------------------------------------------------------------
# Adapter interfaces
# ---------------------------------------------------------------------------


def test_engine_interfaces_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ChatterboxPerformanceEngine()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        VoiceConversionEngine()  # type: ignore[abstract]


def test_fake_engines_are_adapters_of_their_interfaces(rig: Rig) -> None:
    assert isinstance(rig.chatterbox, ChatterboxPerformanceEngine)
    assert isinstance(rig.seedvc, VoiceConversionEngine)
    assert rig.chatterbox.is_loaded is False
    assert rig.seedvc.is_loaded is False


def test_engine_names_are_reported_in_the_clip(rig: Rig) -> None:
    (clip,) = rig.run()

    assert clip.performance_engine == "fake-chatterbox"
    assert clip.style_engine == "fake-seedvc"
    assert rig.chatterbox.is_loaded is True
    assert rig.seedvc.is_loaded is True


def test_models_are_not_loaded_at_import_time() -> None:
    """Importing the module must not touch either runtime."""

    probe = (
        "import sys, app.pipeline.tts as tts;"
        "mods = {'chatterbox', 'hydra', 'omegaconf', 'peft', 'triton'};"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in mods);"
        "assert not loaded, loaded;"
        "assert tts._CHATTERBOX_ENGINES == {};"
        "assert tts._SEED_VC_ENGINES == {};"
        "assert tts._LOADER_MODULES == {};"
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


def test_constructing_the_real_adapters_loads_nothing(tmp_path: Path) -> None:
    chatterbox = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)
    seedvc = SeedVcV2Engine(repo_path=tmp_path / "seed-vc", device="cpu")

    assert chatterbox.is_loaded is False
    assert seedvc.is_loaded is False
    assert "chatterbox" not in sys.modules
    assert "hydra" not in sys.modules
    assert chatterbox.model == "gabar-tech/chatterbox-amharic"
    assert chatterbox.device == "cpu"
    assert chatterbox.sample_rate == CHATTERBOX_SAMPLE_RATE
    assert seedvc.convert_style is True


# ---------------------------------------------------------------------------
# Engine selection and caching
# ---------------------------------------------------------------------------


def test_engines_are_resolved_once_for_the_whole_run(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _chatterbox(*, settings: Settings) -> ChatterboxPerformanceEngine:
        calls.append("chatterbox")
        return rig.chatterbox

    def _seedvc(*, settings: Settings) -> VoiceConversionEngine:
        calls.append("seed-vc")
        return rig.seedvc

    monkeypatch.setattr(tts, "load_chatterbox_engine", _chatterbox)
    monkeypatch.setattr(tts, "load_seed_vc_engine", _seedvc)

    clips = synthesize_dialogue(
        [_line(start=2.0, end=4.0), _line(start=4.0, end=6.0), _line(start=6.0, end=8.0)],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
    )

    assert len(clips) == 3
    assert calls == ["chatterbox", "seed-vc"]
    assert len(rig.chatterbox.calls) == 3
    assert len(rig.seedvc.calls) == 3


def test_injected_engines_are_used_without_loading_the_default_ones(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*, settings: Settings) -> ChatterboxPerformanceEngine:
        raise AssertionError("the default engine must not be resolved")

    monkeypatch.setattr(tts, "load_chatterbox_engine", _boom)
    monkeypatch.setattr(tts, "load_seed_vc_engine", _boom)

    (clip,) = rig.run()

    assert clip.performance_engine == "fake-chatterbox"


def test_a_wrong_performance_engine_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidEngineError, match="ChatterboxPerformanceEngine"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
            performance_engine=rig.seedvc,  # type: ignore[arg-type]
            style_engine=rig.seedvc,
        )


def test_a_wrong_style_engine_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidEngineError, match="VoiceConversionEngine"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=object(),  # type: ignore[arg-type]
        )


def test_default_engines_are_cached_per_device_and_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path, device="cuda")

    first = load_chatterbox_engine(settings=settings)
    assert load_chatterbox_engine(settings=settings) is first
    assert isinstance(first, ChatterboxAmharicEngine)
    assert first.device == "cuda"
    assert first.model == settings.tts_model

    other = load_chatterbox_engine(settings=_settings(tmp_path, device="cpu"))
    assert other is not first
    assert other.device == "cpu"

    style = load_seed_vc_engine(settings=settings)
    assert load_seed_vc_engine(settings=settings) is style
    assert style.repo_path == settings.seed_vc_repo_path
    assert style.name == tts.SEED_VC_ENGINE_NAME


def test_resetting_the_cache_releases_the_engines(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    first = load_chatterbox_engine(settings=settings)
    reset_engine_cache()
    second = load_chatterbox_engine(settings=settings)

    assert first is not second
    assert tts._LOADER_MODULES == {}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_dialogue_returns_no_clips_and_touches_nothing(rig: Rig) -> None:
    clips = synthesize_dialogue(
        [],
        rig.stem,
        {rig.profile.speaker_id: rig.profile},
        settings=rig.settings(),
        performance_engine=rig.chatterbox,
        style_engine=rig.seedvc,
    )

    assert clips == []
    assert rig.chatterbox.calls == []
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
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


def test_a_profile_stored_under_the_wrong_key_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="stored under the key"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {"SPEAKER_99": rig.profile},
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


def test_a_non_profile_value_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="expected a VoiceProfile"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: object()},  # type: ignore[dict-item]
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


def test_a_non_mapping_profile_collection_is_rejected(rig: Rig) -> None:
    with pytest.raises(InvalidVoiceProfileError, match="must be a mapping"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            [rig.profile],  # type: ignore[arg-type]
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
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
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )

    assert rig.chatterbox.calls == []
    assert rig.seedvc.calls == []


def test_a_missing_speech_stem_is_reported(rig: Rig) -> None:
    with pytest.raises(MissingInputError, match="speech stem"):
        synthesize_dialogue(
            [_line()],
            rig.root / "nope.wav",
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


def test_an_unreadable_speech_stem_is_reported(rig: Rig) -> None:
    broken = rig.root / "broken.wav"
    broken.write_text("not audio", encoding="utf-8")

    with pytest.raises(InvalidAudioError, match="speech stem"):
        synthesize_dialogue(
            [_line()],
            broken,
            {rig.profile.speaker_id: rig.profile},
            settings=rig.settings(),
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


def test_contradictory_reference_durations_are_rejected(rig: Rig) -> None:
    settings = rig.settings(
        tts_performance_reference_min_duration=12.0,
        tts_performance_reference_max_duration=6.0,
    )

    with pytest.raises(ConfigurationError, match="MAX_DURATION"):
        synthesize_dialogue(
            [_line()],
            rig.stem,
            {rig.profile.speaker_id: rig.profile},
            settings=settings,
            performance_engine=rig.chatterbox,
            style_engine=rig.seedvc,
        )


# ---------------------------------------------------------------------------
# Chatterbox to Seed-VC ordering and reference routing
# ---------------------------------------------------------------------------


def test_chatterbox_runs_before_seed_vc_for_every_line(rig: Rig) -> None:
    clips = rig.run(_line(start=2.0, end=4.0), _line(start=4.0, end=6.0))

    assert len(clips) == 2
    assert [kind for kind, _ in rig.events] == [
        "chatterbox",
        "seed-vc",
        "chatterbox",
        "seed-vc",
    ]


def test_seed_vc_converts_the_chatterbox_take_of_the_same_line(rig: Rig) -> None:
    (clip,) = rig.run()

    (chatterbox_call,) = rig.chatterbox.calls
    (seedvc_call,) = rig.seedvc.calls

    assert seedvc_call["source_audio"] == chatterbox_call["destination"]
    assert clip.take_path == chatterbox_call["destination"]
    assert clip.audio_path != clip.take_path


def test_chatterbox_gets_the_stem_reference_and_seed_vc_the_profile(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted: list[Path] = []
    real = tts.extract_performance_reference

    def _record(*args: object, **kwargs: object) -> Path:
        result = real(*args, **kwargs)  # type: ignore[arg-type]
        extracted.append(Path(result))
        return result

    monkeypatch.setattr(tts, "extract_performance_reference", _record)

    (clip,) = rig.run()

    (chatterbox_call,) = rig.chatterbox.calls
    (seedvc_call,) = rig.seedvc.calls

    assert extracted == [chatterbox_call["performance_reference"]]
    assert chatterbox_call["performance_reference"] == clip.performance_reference_path
    # The identity reference only ever reaches Seed-VC.
    assert seedvc_call["identity_reference"] == rig.profile.resolve_reference_audio()
    assert chatterbox_call["performance_reference"] != rig.profile.resolve_reference_audio()
    assert clip.voice_reference_path == rig.profile.resolve_reference_audio()


def test_the_performance_reference_is_the_original_stem_segment(rig: Rig) -> None:
    line = _line(start=2.0, end=6.0)

    (clip,) = rig.run(line)

    extracted, rate = sf.read(str(clip.performance_reference_path), dtype="float32", always_2d=True)
    stem, stem_rate = sf.read(str(rig.stem), dtype="float32", always_2d=True)

    assert rate == stem_rate == STEM_RATE
    total = stem.shape[0] / STEM_RATE
    begin, finish = tts._performance_window(
        line.start, line.end, total=total, minimum=6.0, maximum=12.0
    )
    expected = stem[int(begin * STEM_RATE) : int(finish * STEM_RATE)]
    assert extracted.shape == expected.shape
    assert extracted.shape[0] / STEM_RATE == pytest.approx(6.0, abs=1.0 / STEM_RATE)
    assert np.allclose(extracted, expected, atol=1.0 / 32768.0)


def test_the_amharic_text_and_controls_reach_chatterbox(rig: Rig) -> None:
    line = _line(amharic="ጤና ይስጥልኝ።", emotion="furious", intensity=0.9, delivery="shouting")

    (clip,) = rig.run(line)

    (call,) = rig.chatterbox.calls
    assert call["text"] == "ጤና ይስጥልኝ።"
    assert call["controls"] is clip.performance
    assert call["controls"].intensity == 0.9
    assert call["controls"].emotion == "furious"
    assert call["controls"].delivery == "shouting"


# ---------------------------------------------------------------------------
# Performance metadata
# ---------------------------------------------------------------------------


def test_the_mapping_starts_from_the_adapter_defaults() -> None:
    controls = PerformanceControls.from_dialogue(
        _line(emotion="neutral", intensity=0.5, delivery="unmapped direction"),
        seed=7,
    )

    assert controls.exaggeration == pytest.approx(BASE_EXAGGERATION - 0.04)
    assert controls.cfg_weight == pytest.approx(BASE_CFG_WEIGHT)
    assert controls.temperature == pytest.approx(BASE_TEMPERATURE)
    assert controls.seed == 7


def test_emotion_and_delivery_are_preserved_verbatim() -> None:
    line = _line(
        emotion="quietly heartbroken",
        intensity=0.35,
        delivery="soft and vulnerable",
        pause_before=0.4,
        pause_after=0.9,
    )

    controls = PerformanceControls.from_dialogue(line, seed=1)

    assert controls.emotion == "quietly heartbroken"
    assert controls.delivery == "soft and vulnerable"
    assert controls.intensity == 0.35
    assert controls.pause_before == 0.4
    assert controls.pause_after == 0.9
    assert controls.to_dict()["pause_before"] == 0.4


def test_the_mapping_stays_inside_its_bounds() -> None:
    loudest = PerformanceControls.from_dialogue(
        _line(emotion="furious", intensity=1.0, delivery="shouting"), seed=1
    )
    quietest = PerformanceControls.from_dialogue(
        _line(emotion="sad", intensity=0.0, delivery="whispering"), seed=1
    )

    for controls in (loudest, quietest):
        assert EXAGGERATION_BOUNDS[0] <= controls.exaggeration <= EXAGGERATION_BOUNDS[1]
        assert CFG_WEIGHT_BOUNDS[0] <= controls.cfg_weight <= CFG_WEIGHT_BOUNDS[1]
        assert TEMPERATURE_BOUNDS[0] <= controls.temperature <= TEMPERATURE_BOUNDS[1]

    assert loudest.exaggeration > quietest.exaggeration
    assert loudest.temperature > quietest.temperature


def test_clamping_is_recorded_rather_than_hidden() -> None:
    controls = PerformanceControls.from_dialogue(
        _line(emotion="furious", intensity=1.0, delivery="screaming"), seed=1
    )

    assert controls.exaggeration == EXAGGERATION_BOUNDS[1]
    assert any("clamped down" in note for note in controls.notes)


def test_an_unknown_direction_says_so_and_changes_nothing() -> None:
    controls = PerformanceControls.from_dialogue(
        _line(emotion="wistful", intensity=0.5, delivery="musing to himself"), seed=1
    )

    assert controls.matched_emotion_cue is None
    assert controls.matched_delivery_cue is None
    assert any("matches no cue" in note for note in controls.notes)
    assert controls.exaggeration == pytest.approx(BASE_EXAGGERATION)
    assert controls.cfg_weight == pytest.approx(BASE_CFG_WEIGHT)
    assert controls.temperature == pytest.approx(BASE_TEMPERATURE)


def test_matched_cues_are_reported() -> None:
    controls = PerformanceControls.from_dialogue(
        _line(emotion="angry", intensity=0.5, delivery="whispered threat"), seed=1
    )

    assert controls.matched_emotion_cue == "angry"
    assert controls.matched_delivery_cue == "whisper"
    assert controls.cfg_weight > BASE_CFG_WEIGHT


def test_singing_is_reported_as_unsupported_instead_of_guessed() -> None:
    controls = PerformanceControls.from_dialogue(
        _line(emotion="happy", intensity=0.5, delivery="singing softly"), seed=1
    )

    assert any("pitch conditioning" in note for note in controls.notes)
    assert controls.cfg_weight == pytest.approx(BASE_CFG_WEIGHT)


def test_a_bad_mapping_input_is_rejected() -> None:
    with pytest.raises(InvalidDialogueError):
        PerformanceControls.from_dialogue("not a line", seed=1)  # type: ignore[arg-type]

    with pytest.raises(InvalidDialogueError, match="intensity"):
        PerformanceControls(
            emotion="neutral",
            intensity=1.5,
            delivery="calm",
            pause_before=0.0,
            pause_after=0.0,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.6,
            seed=1,
        )


def test_the_seed_is_a_pure_function_of_the_line(rig: Rig) -> None:
    line = _line(amharic="ተመሳሳይ መስመር")

    first = PerformanceControls.from_dialogue(line, seed=int(tts._line_digest(line, model="m")[:8], 16))
    second = PerformanceControls.from_dialogue(line, seed=int(tts._line_digest(line, model="m")[:8], 16))

    assert first.seed == second.seed
    assert first.seed != int(tts._line_digest(_line(amharic="ሌላ መስመር"), model="m")[:8], 16)


def test_the_clip_keeps_the_whole_line_and_the_mapped_performance(rig: Rig) -> None:
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
    assert clip.performance.matched_emotion_cue == "afraid"
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

    assert clip.sample_rate == rate == SEED_VC_RATE
    assert samples.shape[0] / rate == pytest.approx(clip.duration, abs=1.0 / rate)
    assert clip.take_path.is_file()
    assert clip.performance_reference_path.is_file()


def test_no_duration_is_forced_on_the_generated_audio(rig: Rig) -> None:
    """The clip is as long as the engines made it, not as long as the line."""

    line = _line(start=0.0, end=16.0)

    (clip,) = rig.run(line)

    # The fake converter emits half of the take's frames at its own rate, and
    # nothing here stretches that to the 16 s the line occupied.
    expected = 0.5 * rig.chatterbox.seconds * rig.chatterbox.rate / SEED_VC_RATE
    assert clip.original_duration == 16.0
    assert clip.speech_duration == pytest.approx(expected, abs=0.01)
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
    assert payload["performance_engine"] == "fake-chatterbox"
    assert payload["style_engine"] == "fake-seedvc"
    assert payload["performance_reference_path"].endswith(".wav")
    assert payload["audio_path"] == portable_path(clip.audio_path)
    assert "/" in payload["audio_path"] and "\\" not in payload["audio_path"]


# ---------------------------------------------------------------------------
# Deterministic paths and reuse
# ---------------------------------------------------------------------------


def test_artifacts_live_under_the_work_directory(rig: Rig) -> None:
    (clip,) = rig.run()

    base = resolve_tts_directory(settings=rig.settings())
    assert base == rig.root / tts.TTS_DIRECTORY_NAME
    for path in (
        clip.audio_path,
        clip.take_path,
        clip.performance_reference_path,
    ):
        assert path.parent.parent == base
    assert clip.audio_path.parent.name == "clips"
    assert clip.take_path.parent.name == "chatterbox"
    assert clip.performance_reference_path.parent.name == "performance"
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


def test_a_different_model_gets_a_different_path(rig: Rig) -> None:
    (default,) = rig.run(_line(), settings=rig.settings())
    (other,) = rig.run(_line(), settings=rig.settings(tts_model="someone/else"))

    assert default.audio_path != other.audio_path


def test_existing_artifacts_are_reused_instead_of_regenerated(rig: Rig) -> None:
    line = _line(pause_after=0.2)
    (first,) = rig.run(line)

    # A second run whose engines would fail if they were called again.
    cached = Rig(
        root=rig.root,
        stem=rig.stem,
        profile=rig.profile,
        chatterbox=FakeChatterbox(error=SynthesisError("must not be called")),
        seedvc=FakeSeedVc(error=ConversionError("must not be called")),
        events=[],
    )
    (second,) = cached.run(line)

    assert cached.chatterbox.calls == []
    assert cached.seedvc.calls == []
    assert second.audio_path == first.audio_path
    assert second.to_dict()["duration"] == first.to_dict()["duration"]


def test_a_cached_performance_reference_is_not_re_extracted(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.run()

    def _boom(*args: object, **kwargs: object) -> Path:
        raise AssertionError("the reference must be reused")

    monkeypatch.setattr(tts, "extract_performance_reference", _boom)

    (clip,) = rig.run()

    assert clip.performance_reference_path.is_file()


# ---------------------------------------------------------------------------
# The performance-reference extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (10.0, 20.0, (10.0, 20.0)),  # inside the preferred window
        (10.0, 11.0, (7.5, 13.5)),  # short line, grown around its centre
        (5.0, 25.0, (9.0, 21.0)),  # long line, trimmed around its centre
        (0.0, 1.0, (0.0, 6.0)),  # near the start, shifted inside
        (29.0, 30.0, (24.0, 30.0)),  # near the end, shifted inside
    ],
)
def test_the_prompt_window_is_centred_and_inside_the_stem(
    tmp_path: Path, start: float, end: float, expected: tuple[float, float]
) -> None:
    stem = _stem(tmp_path, seconds=30.0)

    window = tts._performance_window(start, end, total=30.0, minimum=6.0, maximum=12.0)

    assert window == pytest.approx(expected)


def test_a_stem_shorter_than_the_minimum_is_used_whole(tmp_path: Path) -> None:
    stem = _stem(tmp_path, seconds=4.0)

    out = extract_performance_reference(
        stem, start=1.0, end=2.0, destination=tmp_path / "ref.wav", minimum_duration=6.0
    )
    samples, rate = sf.read(str(out), dtype="float32")

    assert rate == STEM_RATE
    assert samples.shape[0] / rate == pytest.approx(4.0, abs=1.0 / rate)


def test_the_extracted_reference_is_sample_exact(tmp_path: Path) -> None:
    stem = _stem(tmp_path, seconds=30.0)

    out = extract_performance_reference(
        stem, start=10.0, end=20.0, destination=tmp_path / "ref.wav"
    )
    info = sf.info(str(out))
    extracted, _ = sf.read(str(out), dtype="float32", always_2d=True)
    original, _ = sf.read(str(stem), dtype="float32", always_2d=True)

    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert np.allclose(extracted, original[10 * STEM_RATE : 20 * STEM_RATE], atol=1.0 / 32768.0)


def test_extraction_validates_its_inputs(tmp_path: Path) -> None:
    stem = _stem(tmp_path, seconds=10.0)

    with pytest.raises(MissingInputError):
        extract_performance_reference(stem.parent / "nope.wav", start=1.0, end=2.0, destination=tmp_path / "a.wav")

    with pytest.raises(InvalidInputError, match="greater than start"):
        extract_performance_reference(stem, start=2.0, end=2.0, destination=tmp_path / "b.wav")

    with pytest.raises(InvalidInputError, match="stem is only"):
        extract_performance_reference(stem, start=11.0, end=12.0, destination=tmp_path / "c.wav")

    with pytest.raises(ConfigurationError, match="MIN_DURATION"):
        extract_performance_reference(
            stem,
            start=1.0,
            end=2.0,
            destination=tmp_path / "d.wav",
            minimum_duration=12.0,
            maximum_duration=6.0,
        )


def test_an_unreadable_stem_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_text("not audio", encoding="utf-8")

    with pytest.raises(InvalidAudioError, match="speech stem"):
        extract_performance_reference(broken, start=0.0, end=1.0, destination=tmp_path / "x.wav")


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_a_failing_chatterbox_call_is_reported_with_the_line(rig: Rig) -> None:
    rig.chatterbox = FakeChatterbox(error=SynthesisError("no speaker tokens"), error_at=1)

    with pytest.raises(SynthesisError, match="line 1 of speaker 'SPEAKER_00'"):
        rig.run(_line(start=2.0, end=4.0), _line(start=4.0, end=6.0))


def test_a_failing_conversion_is_reported_with_the_line(rig: Rig) -> None:
    rig.seedvc = FakeSeedVc(error=ConversionError("reference too short"))

    with pytest.raises(ConversionError, match="line 0 of speaker 'SPEAKER_00'"):
        rig.run()


def test_the_failing_line_is_identified_by_its_window(rig: Rig) -> None:
    rig.seedvc = FakeSeedVc(error=ConversionError("reference too short"))

    with pytest.raises(ConversionError, match=r"\(2\.000s-6\.000s\)"):
        rig.run()


def test_an_engine_that_writes_nothing_is_reported(rig: Rig) -> None:
    rig.chatterbox = FakeChatterbox(write=False)

    with pytest.raises(MissingOutputError, match="fake-chatterbox reported success"):
        rig.run()


def test_a_converter_that_writes_nothing_is_reported(rig: Rig) -> None:
    rig.seedvc = FakeSeedVc(write=False)

    with pytest.raises(MissingOutputError, match="fake-seedvc reported success"):
        rig.run()


def test_unexpected_engine_errors_propagate_unchanged(rig: Rig) -> None:
    rig.chatterbox = FakeChatterbox(error=RuntimeError("cuda out of memory"))

    with pytest.raises(RuntimeError, match="cuda out of memory"):
        rig.run()


# ---------------------------------------------------------------------------
# The real Chatterbox adapter, with the loader replaced
# ---------------------------------------------------------------------------

_FAKE_LOADER = '''
"""Stand-in for the adapter repository's amharic_tts.py."""

# The real loader publishes these, and the engine reads the base pin from here
# instead of keeping a second copy of it.
REPO_ID = "gabar-tech/chatterbox-amharic"
BASE_REPO = "ResembleAI/chatterbox"
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_FILE = "t3_mtl23ls_v3.safetensors"
S3GEN_FILE = "s3gen.pt"
VE_FILE = "ve.pt"
CONDS_FILE = "conds.pt"
BASE_FILES = (T3_FILE, S3GEN_FILE, VE_FILE, CONDS_FILE)

CALLS = []
GENERATE_CALLS = []


class _Tensor:
    """Mimics the ``[1, N]`` torch tensor the real adapter returns."""

    def __init__(self, samples):
        self._samples = samples

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._samples


class _FakeAmharicTTS:
    sr = 24000

    def generate(self, text, **kwargs):
        import numpy

        GENERATE_CALLS.append(dict(text=text, **kwargs))
        if FAIL:
            raise RuntimeError("chatterbox exploded")
        frames = int(0.5 * self.sr)
        ramp = numpy.linspace(-0.5, 0.5, frames, dtype="float32")
        return _Tensor(ramp.reshape(1, -1))


def load_amharic_tts(device="cuda", adapter_dir=None, base_dir=None, merge_adapter=True):
    CALLS.append(
        {
            "device": device,
            "adapter_dir": adapter_dir,
            "base_dir": base_dir,
            "merge_adapter": merge_adapter,
        }
    )
    return _FakeAmharicTTS()


FAIL = False
'''


class FakeSnapshots:
    """Records ``snapshot_download`` calls instead of reaching the network."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        repo_id: str,
        *,
        cache_dir: Path,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
    ) -> Path:
        self.calls.append(
            {
                "repo_id": repo_id,
                "cache_dir": Path(cache_dir),
                "revision": revision,
                "allow_patterns": allow_patterns,
            }
        )
        target = self.root / repo_id.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        return target


@dataclass(frozen=True)
class LoaderStub:
    """The fake Amharic loader plus the snapshot recorder handed to an engine."""

    path: Path
    snapshots: FakeSnapshots


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LoaderStub:
    """Replace the Hub download and the weight snapshots with in-process fakes."""

    path = tmp_path / "amharic_tts.py"
    path.write_text(_FAKE_LOADER, encoding="utf-8")
    monkeypatch.setattr(tts, "_download_amharic_loader", lambda model, *, cache_dir: path)

    snapshots = FakeSnapshots(tmp_path)
    monkeypatch.setattr(tts, "_cached_snapshot", snapshots)
    return LoaderStub(path=path, snapshots=snapshots)


def test_the_loader_is_downloaded_and_executed_once(tmp_path: Path, loader: LoaderStub) -> None:
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)

    first = engine._load()
    second = engine._load()

    assert first is second
    assert engine.is_loaded is True
    (call,) = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"].CALLS
    assert call["device"] == "cpu"
    assert call["merge_adapter"] is True


def test_the_engine_passes_its_device_and_controls_to_the_adapter(
    tmp_path: Path, loader: LoaderStub
) -> None:
    engine = ChatterboxAmharicEngine(device="cuda:1", cache_dir=tmp_path)
    reference = _write_wav(tmp_path / "prompt.wav", np.zeros(2400, dtype=np.float32), 24_000)
    controls = PerformanceControls.from_dialogue(
        _line(emotion="angry", intensity=0.9, delivery="shouting"), seed=42
    )

    out = engine.synthesize(
        text="ሰላም",
        performance_reference=reference,
        controls=controls,
        destination=tmp_path / "take.wav",
    )

    module = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"]
    assert module.CALLS[0]["device"] == "cuda:1"
    (call,) = module.GENERATE_CALLS
    assert call["text"] == "ሰላም"
    assert call["audio_prompt_path"] == str(reference)
    assert call["temperature"] == controls.temperature
    assert call["cfg_weight"] == controls.cfg_weight
    assert call["exaggeration"] == controls.exaggeration

    info = sf.info(str(out))
    assert info.samplerate == CHATTERBOX_SAMPLE_RATE
    assert info.channels == 1
    assert info.frames == int(0.5 * CHATTERBOX_SAMPLE_RATE)


def test_the_engine_wraps_adapter_failures(tmp_path: Path, loader: LoaderStub) -> None:
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)
    engine._load()
    tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"].FAIL = True
    reference = _write_wav(tmp_path / "prompt.wav", np.zeros(2400, dtype=np.float32), 24_000)

    with pytest.raises(SynthesisError, match="chatterbox exploded"):
        engine.synthesize(
            text="ሰላም",
            performance_reference=reference,
            controls=PerformanceControls.from_dialogue(_line(), seed=1),
            destination=tmp_path / "take.wav",
        )


def test_a_download_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(model: str, *, cache_dir: Path | None) -> Path:
        raise EngineLoadError(f"could not download amharic_tts.py from {model!r}")

    monkeypatch.setattr(tts, "_download_amharic_loader", _boom)
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)

    with pytest.raises(EngineLoadError, match="amharic_tts.py"):
        engine._load()


def test_a_loader_without_the_entry_point_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "amharic_tts.py"
    path.write_text("SOMETHING = 1\n", encoding="utf-8")
    monkeypatch.setattr(tts, "_download_amharic_loader", lambda model, *, cache_dir: path)

    with pytest.raises(EngineLoadError, match="load_amharic_tts"):
        ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)._load()


def test_a_missing_performance_reference_is_rejected_before_loading(
    tmp_path: Path, loader: LoaderStub
) -> None:
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)

    with pytest.raises(MissingInputError, match="performance reference"):
        engine.synthesize(
            text="ሰላም",
            performance_reference=tmp_path / "nope.wav",
            controls=PerformanceControls.from_dialogue(_line(), seed=1),
            destination=tmp_path / "take.wav",
        )
    assert engine.is_loaded is False


def test_empty_text_is_rejected(tmp_path: Path, loader: LoaderStub) -> None:
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=tmp_path)

    with pytest.raises(InvalidDialogueError, match="must be non-empty"):
        engine.synthesize(
            text="   ",
            performance_reference=tmp_path / "nope.wav",
            controls=PerformanceControls.from_dialogue(_line(), seed=1),
            destination=tmp_path / "take.wav",
        )


def test_the_adapter_and_its_pinned_base_are_cached_where_the_project_asks(
    tmp_path: Path, loader: LoaderStub
) -> None:
    """Both Chatterbox downloads belong in MODEL_CACHE_DIR, not HF's default cache."""

    cache = tmp_path / "models_cache"
    engine = ChatterboxAmharicEngine(device="cpu", cache_dir=cache)

    engine._load()

    adapter_call, base_call = loader.snapshots.calls
    assert adapter_call["repo_id"] == "gabar-tech/chatterbox-amharic"
    assert adapter_call["cache_dir"] == cache
    assert base_call["repo_id"] == "ResembleAI/chatterbox"
    assert base_call["cache_dir"] == cache
    assert base_call["revision"] == "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
    assert base_call["allow_patterns"] == [
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "ve.pt",
        "conds.pt",
    ]

    module = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"]
    assert module.CALLS[0]["adapter_dir"] == tmp_path / "gabar-tech--chatterbox-amharic"
    assert module.CALLS[0]["base_dir"] == tmp_path / "ResembleAI--chatterbox"


def test_the_base_pin_is_read_from_the_adapter_not_duplicated(
    tmp_path: Path, loader: LoaderStub
) -> None:
    cache = tmp_path / "models_cache"
    ChatterboxAmharicEngine(device="cpu", cache_dir=cache)._load()

    module = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"]
    module.BASE_REVISION = "0" * 40
    module.BASE_FILES = ("only-this.pt",)
    loader.snapshots.calls.clear()

    ChatterboxAmharicEngine(device="cpu", cache_dir=cache)._load()

    assert loader.snapshots.calls[1]["revision"] == "0" * 40
    assert loader.snapshots.calls[1]["allow_patterns"] == ["only-this.pt"]


def test_explicit_directories_skip_the_snapshots(tmp_path: Path, loader: LoaderStub) -> None:
    adapter = tmp_path / "local-adapter"
    base = tmp_path / "local-base"

    ChatterboxAmharicEngine(
        device="cpu", cache_dir=tmp_path, adapter_dir=adapter, base_dir=base
    )._load()

    assert loader.snapshots.calls == []
    module = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"]
    assert module.CALLS[0]["adapter_dir"] == adapter
    assert module.CALLS[0]["base_dir"] == base


def test_without_a_cache_directory_the_loader_keeps_its_own_download(
    tmp_path: Path, loader: LoaderStub
) -> None:
    ChatterboxAmharicEngine(device="cpu", cache_dir=None)._load()

    assert loader.snapshots.calls == []
    module = tts._LOADER_MODULES["gabar-tech/chatterbox-amharic"]
    assert module.CALLS[0]["adapter_dir"] is None
    assert module.CALLS[0]["base_dir"] is None


# ---------------------------------------------------------------------------
# The real Seed-VC V2 adapter, with the wrapper replaced
# ---------------------------------------------------------------------------


class FakeVcWrapper:
    """Records what the engine asks of Seed-VC V2."""

    def __init__(self, *, chunks: list[object] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.checkpoints: list[tuple[object, object]] = []
        self.caches: list[dict[str, object]] = []
        self.conversions: list[dict[str, object]] = []
        self.device: str | None = None
        self.evaluated = False

    def load_checkpoints(self, ar_checkpoint_path: object, cfm_checkpoint_path: object) -> None:
        self.checkpoints.append((ar_checkpoint_path, cfm_checkpoint_path))

    def to(self, device: str) -> "FakeVcWrapper":
        self.device = device
        return self

    def eval(self) -> "FakeVcWrapper":
        self.evaluated = True
        return self

    def setup_ar_caches(self, **kwargs: object) -> None:
        self.caches.append(kwargs)

    def convert_voice_with_streaming(self, **kwargs: object):
        self.conversions.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.chunks is None:
            return
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def seed_vc_repo(tmp_path: Path) -> Path:
    """A Seed-VC checkout skeleton: the config is the only file that is read."""

    config = tmp_path / "seed-vc" / "configs" / "v2" / "vc_wrapper.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("_target_: modules.vc_wrapper.Wrapper\n", encoding="utf-8")
    return tmp_path / "seed-vc"


@pytest.fixture
def vc(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SeedVcV2Engine, FakeVcWrapper, list[str]]:
    # The real generator yields ``(mp3_bytes, full_audio)`` pairs, where
    # ``full_audio`` is None until the final chunk and ``(sample_rate, samples)``
    # there - see modules/v2/vc_wrapper.py and inference_v2.py.
    wrapper = FakeVcWrapper(
        chunks=[
            (b"mp3-chunk", None),
            (b"mp3-chunk", (SEED_VC_RATE, np.full(2205, 0.25, dtype=np.float32))),
        ]
    )
    built: list[str] = []

    def _build(repo: Path, config: Path) -> FakeVcWrapper:
        built.append(str(config))
        return wrapper

    monkeypatch.setattr(tts, "_seed_vc_runtime", _build)
    monkeypatch.setattr(tts, "_seed_vc_dtype", lambda device: "float16")

    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cuda", diffusion_steps=25)
    return engine, wrapper, built


def test_seed_vc_runs_with_style_conversion_on(vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path) -> None:
    engine, wrapper, _ = vc
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    out = engine.convert(
        source_audio=source, identity_reference=reference, destination=tmp_path / "voice.wav"
    )

    (call,) = wrapper.conversions
    assert call["convert_style"] is True
    assert call["anonymization_only"] is False
    assert call["source_audio_path"] == str(source)
    assert call["target_audio_path"] == str(reference)
    assert call["stream_output"] is True
    assert call["dtype"] == "float16"
    assert call["diffusion_steps"] == 25
    assert call["intelligebility_cfg_rate"] == 0.7

    info = sf.info(str(out))
    assert info.samplerate == 22_050
    assert info.channels == 1
    assert info.frames == 2205
    assert engine.is_loaded is True


def test_seed_vc_is_handed_a_real_torch_device(
    vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path
) -> None:
    """Seed-VC dereferences ``device.type``, so a settings string is not enough."""

    engine, wrapper, _ = vc
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    engine.convert(
        source_audio=source, identity_reference=reference, destination=tmp_path / "voice.wav"
    )

    (call,) = wrapper.conversions
    assert isinstance(call["device"], torch.device)
    assert call["device"].type == "cuda"
    assert isinstance(wrapper.caches[0]["device"], torch.device)
    assert wrapper.caches[0]["device"].type == "cuda"


def test_the_converted_audio_comes_from_the_final_streaming_chunk(
    vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path
) -> None:
    """Intermediate chunks carry no audio; only the last one carries the result."""

    engine, wrapper, _ = vc
    marker = np.linspace(-0.5, 0.5, 2205, dtype=np.float32)
    wrapper.chunks = [
        (b"mp3-chunk", None),
        (b"mp3-chunk", None),
        (b"mp3-chunk", (SEED_VC_RATE, marker)),
    ]
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    out = engine.convert(
        source_audio=source, identity_reference=reference, destination=tmp_path / "voice.wav"
    )

    written, rate = sf.read(str(out), dtype="float32")
    assert rate == SEED_VC_RATE
    assert written.shape[0] == marker.shape[0]
    assert np.allclose(written, marker, atol=1.0 / 32768.0)


def test_a_stream_that_never_delivers_audio_is_reported(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts, "_seed_vc_runtime", lambda repo, config: FakeVcWrapper(chunks=[(b"mp3", None)])
    )
    monkeypatch.setattr(tts, "_seed_vc_dtype", lambda device: "float32")
    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cpu")
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    with pytest.raises(ConversionError, match="produced no audio"):
        engine.convert(
            source_audio=source, identity_reference=reference, destination=tmp_path / "out.wav"
        )


def test_a_malformed_streamed_item_is_reported(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts, "_seed_vc_runtime", lambda repo, config: FakeVcWrapper(chunks=[b"only-mp3-bytes"])
    )
    monkeypatch.setattr(tts, "_seed_vc_dtype", lambda device: "float32")
    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cpu")
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    with pytest.raises(ConversionError, match="yielded an unexpected result"):
        engine.convert(
            source_audio=source, identity_reference=reference, destination=tmp_path / "out.wav"
        )


def test_the_seed_vc_wrapper_is_built_once(vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path) -> None:
    engine, wrapper, built = vc
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    for name in ("one.wav", "two.wav"):
        engine.convert(
            source_audio=source,
            identity_reference=reference,
            destination=tmp_path / name,
        )

    assert len(built) == 1
    assert len(wrapper.conversions) == 2
    assert wrapper.checkpoints == [(None, None)]
    assert wrapper.caches[0]["device"] == torch.device("cuda")
    assert wrapper.caches[0]["max_batch_size"] == 1
    assert wrapper.evaluated is True


def test_a_missing_seed_vc_checkout_is_reported(tmp_path: Path) -> None:
    engine = SeedVcV2Engine(repo_path=tmp_path / "missing", device="cpu")

    with pytest.raises(EngineLoadError, match="SEED_VC_REPO_PATH"):
        engine._load()


def test_a_checkout_without_the_v2_config_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "seed-vc").mkdir()
    engine = SeedVcV2Engine(repo_path=tmp_path / "seed-vc", device="cpu")

    with pytest.raises(EngineLoadError, match="vc_wrapper.yaml"):
        engine._load()


def test_a_failing_wrapper_load_is_reported(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(repo: Path, config: Path) -> object:
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(tts, "_seed_vc_runtime", _boom)
    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cuda")

    with pytest.raises(EngineLoadError, match="no CUDA device"):
        engine._load()


def test_seed_vc_failures_are_wrapped(vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path) -> None:
    engine, wrapper, _ = vc
    wrapper.error = RuntimeError("tensor mismatch")
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    with pytest.raises(ConversionError, match="tensor mismatch"):
        engine.convert(
            source_audio=source, identity_reference=reference, destination=tmp_path / "out.wav"
        )


def test_an_empty_seed_vc_result_is_reported(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tts, "_seed_vc_runtime", lambda repo, config: FakeVcWrapper(chunks=None))
    monkeypatch.setattr(tts, "_seed_vc_dtype", lambda device: "float32")
    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cpu")
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    with pytest.raises(ConversionError, match="produced no audio"):
        engine.convert(
            source_audio=source, identity_reference=reference, destination=tmp_path / "out.wav"
        )


def test_an_unexpected_seed_vc_result_is_reported(
    tmp_path: Path, seed_vc_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts,
        "_seed_vc_runtime",
        lambda repo, config: FakeVcWrapper(chunks=[(b"mp3", (SEED_VC_RATE,))]),
    )
    monkeypatch.setattr(tts, "_seed_vc_dtype", lambda device: "float32")
    engine = SeedVcV2Engine(repo_path=seed_vc_repo, device="cpu")
    source = _write_wav(tmp_path / "take.wav", np.full(4800, 0.2, dtype=np.float32), 24_000)
    reference = _write_wav(tmp_path / "reference.wav", np.full(2400, 0.3, dtype=np.float32), 24_000)

    with pytest.raises(ConversionError, match="unexpected result"):
        engine.convert(
            source_audio=source, identity_reference=reference, destination=tmp_path / "out.wav"
        )


def test_seed_vc_requires_its_inputs_to_exist(vc: tuple[SeedVcV2Engine, FakeVcWrapper, list[str]], tmp_path: Path) -> None:
    engine, _, _ = vc

    with pytest.raises(MissingInputError, match="Seed-VC source audio"):
        engine.convert(
            source_audio=tmp_path / "nope.wav",
            identity_reference=tmp_path / "also-nope.wav",
            destination=tmp_path / "out.wav",
        )


def test_bad_seed_vc_options_are_rejected(seed_vc_repo: Path) -> None:
    with pytest.raises(ConfigurationError, match="SEED_VC_DIFFUSION_STEPS"):
        SeedVcV2Engine(repo_path=seed_vc_repo, diffusion_steps=0)

    with pytest.raises(ConfigurationError, match="top_p"):
        SeedVcV2Engine(repo_path=seed_vc_repo, top_p=1.5)

    with pytest.raises(ConfigurationError, match="device"):
        SeedVcV2Engine(repo_path=seed_vc_repo, device="  ")
