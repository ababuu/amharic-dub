"""Tests for the MMS-TTS Amharic engine and the single-voice synthesis path.

The VITS model, its tokenizer and uroman are all replaced by fakes, so nothing is
downloaded, torch is never needed for real, and the adapter's own behaviour -
romanising Fidel before synthesis, seeding the duration predictor, honouring the
config, writing a playable clip - is what is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import tts
from app.pipeline.tts import (
    MMS_ENGINE_NAME,
    ConfigurationError,
    EngineLoadError,
    InvalidEngineError,
    InvalidInputError,
    MmsAmharicEngine,
    SynthesisError,
    TextToSpeechEngine,
    synthesize_dialogue_detailed,
)
from app.pipeline.translation import AdaptedDialogue


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeWaveform:
    def __init__(self, samples) -> None:
        self._samples = np.asarray(samples, dtype=np.float32)
        self.shape = self._samples.shape

    def squeeze(self):
        return self

    def detach(self):
        return self

    def to(self, device):
        return self

    def numpy(self):
        return self._samples


class FakeNetwork:
    """Stand-in for ``transformers.VitsModel``."""

    instances: list["FakeNetwork"] = []
    fail: Exception | None = None
    samples: np.ndarray = np.full(4800, 0.3, dtype=np.float32)
    speaking_rate_seen: float | None = None

    def __init__(self, name, cache_dir=None) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.config = SimpleNamespace(sampling_rate=16_000, speaking_rate=1.0)
        self.device = "cpu"
        self.moved_to = None
        self.calls: list[dict] = []
        FakeNetwork.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, cache_dir=None):
        if cls.fail is not None:
            raise cls.fail
        return cls(name, cache_dir=cache_dir)

    def to(self, device):
        self.moved_to = device
        self.device = str(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([SimpleNamespace(device=self.device)])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        FakeNetwork.speaking_rate_seen = self.config.speaking_rate
        return SimpleNamespace(waveform=FakeWaveform(FakeNetwork.samples))


class FakeTokenizer:
    """Stand-in for ``transformers.AutoTokenizer``."""

    instances: list["FakeTokenizer"] = []

    def __init__(self, name, cache_dir=None) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.texts: list[str] = []
        FakeTokenizer.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, cache_dir=None):
        return cls(name, cache_dir=cache_dir)

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        return {"input_ids": _Moved()}


class _Moved:
    """Anything with ``.to()``, which is all the engine needs of an encoding."""

    def to(self, device):
        return self


class FakeTorch:
    device_calls: list[str] = []
    seeded: list[int] = []

    @staticmethod
    def device(name):
        FakeTorch.device_calls.append(name)
        return f"torch.device({name})"

    @staticmethod
    def manual_seed(value):
        FakeTorch.seeded.append(value)

    @staticmethod
    def no_grad():
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()


class FakeUroman:
    """Stand-in for ``uroman.Uroman``."""

    instances: list["FakeUroman"] = []
    fail: Exception | None = None

    def __init__(self) -> None:
        self.seen: list[str] = []
        FakeUroman.instances.append(self)

    def romanize_string(self, text):
        if FakeUroman.fail is not None:
            raise FakeUroman.fail
        self.seen.append(text)
        return "romanised:" + text


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNetwork.instances = []
    FakeNetwork.fail = None
    FakeNetwork.samples = np.full(4800, 0.3, dtype=np.float32)
    FakeNetwork.speaking_rate_seen = None
    FakeTokenizer.instances = []
    FakeTorch.device_calls = []
    FakeTorch.seeded = []
    FakeUroman.instances = []
    FakeUroman.fail = None

    monkeypatch.setattr(tts, "_require_vits", lambda: (FakeTorch, FakeTokenizer, FakeNetwork))
    monkeypatch.setattr(tts, "_require_uroman", lambda: FakeUroman)


def _speech(root: Path, *, seconds: float = 8.0, rate: int = 24_000) -> Path:
    """Write a stand-in speech stem. A single-voice engine does not read it, but the
    stage still validates what it is handed, so a real file keeps that path covered."""

    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    data = (0.4 * np.sin(2.0 * np.pi * 140.0 * times)).astype(np.float32)
    path = root / "speech.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, format="WAV", subtype="PCM_16")
    return path


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
        "voice_profile_dir": root / "voices",
        "tts_engine": "mms",
        "tts_model": "facebook/mms-tts-amh",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _line(
    *,
    start: float = 0.0,
    end: float = 2.0,
    amharic: str = "ሰላም ዓለም።",
    speaker_id: str = "SPEAKER_00",
) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id=speaker_id,
        start=start,
        end=end,
        source_text="hello world",
        amharic=amharic,
        emotion="neutral",
        intensity=0.5,
        delivery="neutral",
        pause_before=0.0,
        pause_after=0.0,
    )


# ---------------------------------------------------------------------------
# laziness and configuration
# ---------------------------------------------------------------------------


def test_constructing_the_engine_loads_nothing(monkeypatch) -> None:
    _patch(monkeypatch)

    engine = MmsAmharicEngine(device="cpu")

    assert engine.is_loaded is False
    assert engine.model == "facebook/mms-tts-amh"
    assert FakeNetwork.instances == []
    assert FakeTokenizer.instances == []
    assert FakeUroman.instances == []


def test_bad_options_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="repository id"):
        MmsAmharicEngine(model="  ")
    with pytest.raises(ConfigurationError, match="device"):
        MmsAmharicEngine(device="   ")
    with pytest.raises(ConfigurationError, match="MMS_SEED"):
        MmsAmharicEngine(seed=-1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="MMS_SPEAKING_RATE"):
        MmsAmharicEngine(speaking_rate=0.0)


def test_it_is_a_text_to_speech_engine() -> None:
    assert issubclass(MmsAmharicEngine, TextToSpeechEngine)
    assert MmsAmharicEngine.name == MMS_ENGINE_NAME


# ---------------------------------------------------------------------------
# romanisation: the model takes Latin, the pipeline speaks Fidel
# ---------------------------------------------------------------------------


def test_fidel_text_is_romanised_before_synthesis(monkeypatch) -> None:
    """The checkpoint takes Latin input, so Fidel has to be converted first."""

    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    assert engine.romanize("ሰላም") == "romanised:ሰላም"
    assert FakeUroman.instances[0].seen == ["ሰላም"]


def test_the_romaniser_is_built_once(monkeypatch) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    for _ in range(4):
        engine.romanize("ሰላም")

    assert len(FakeUroman.instances) == 1


def test_an_injected_romaniser_is_used(monkeypatch) -> None:
    _patch(monkeypatch)
    mine = FakeUroman()
    engine = MmsAmharicEngine(device="cpu", romanizer=mine)

    engine.romanize("ሰላም")

    assert mine.seen == ["ሰላም"]
    assert FakeUroman.instances == [mine]


def test_empty_text_is_rejected(monkeypatch) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    with pytest.raises(InvalidInputError, match="nothing to synthesize"):
        engine.romanize("   ")


def test_a_romaniser_failure_is_reported(monkeypatch) -> None:
    _patch(monkeypatch)
    FakeUroman.fail = RuntimeError("no data dir")
    engine = MmsAmharicEngine(device="cpu")

    with pytest.raises(SynthesisError, match="could not romanise"):
        engine.romanize("ሰላም")


def test_a_missing_romaniser_package_is_reported(monkeypatch) -> None:
    _patch(monkeypatch)

    def _missing():
        raise EngineLoadError("MMS-TTS needs the 'uroman' package")

    monkeypatch.setattr(tts, "_require_uroman", _missing)
    engine = MmsAmharicEngine(device="cpu")

    with pytest.raises(EngineLoadError, match="uroman"):
        engine.romanize("ሰላም")


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------


def test_a_line_is_synthesized_to_a_playable_file(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu", cache_dir=tmp_path)
    destination = tmp_path / "out.wav"

    written = engine.synthesize(text="ሰላም ዓለም", destination=destination)

    assert written == destination
    assert destination.is_file()
    info = sf.info(str(destination))
    assert info.samplerate == 16_000
    assert info.channels == 1
    # The romanised text is what reached the model, not the Fidel original.
    assert FakeTokenizer.instances[0].texts == ["romanised:ሰላም ዓለም"]


def test_the_engine_reports_the_rate_the_model_produced(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    FakeNetwork.samples = np.full(8000, 0.2, dtype=np.float32)
    engine = MmsAmharicEngine(device="cpu")

    engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")

    assert engine.sample_rate == 16_000


def test_generation_is_seeded(monkeypatch, tmp_path: Path) -> None:
    """Unseeded, the same line would change length between runs."""

    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu", seed=7)

    engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")

    assert FakeTorch.seeded == [7]


def test_the_speaking_rate_reaches_the_model(monkeypatch, tmp_path: Path) -> None:
    """Asking for a rate is how a line is made to fit without being stretched."""

    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu", speaking_rate=1.2)

    engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")

    assert FakeNetwork.speaking_rate_seen == 1.2


def test_the_model_is_loaded_once_for_many_lines(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    for index in range(4):
        engine.synthesize(text="ሰላም", destination=tmp_path / f"{index}.wav")

    assert len(FakeNetwork.instances) == 1
    assert len(FakeTokenizer.instances) == 1


def test_the_device_and_cache_are_honoured(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cuda:2", cache_dir=tmp_path)

    engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")

    assert FakeNetwork.instances[0].moved_to == "torch.device(cuda:2)"
    assert FakeNetwork.instances[0].cache_dir == tmp_path
    assert FakeTokenizer.instances[0].cache_dir == tmp_path


def test_an_empty_waveform_is_reported(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    FakeNetwork.samples = np.zeros(0, dtype=np.float32)
    engine = MmsAmharicEngine(device="cpu")

    with pytest.raises(SynthesisError, match="empty waveform"):
        engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")


def test_a_model_load_failure_names_the_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    FakeNetwork.fail = OSError("no such repo")
    engine = MmsAmharicEngine(model="bad/model", device="cpu")

    with pytest.raises(EngineLoadError, match="bad/model"):
        engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")


def test_a_generation_failure_is_wrapped(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(FakeNetwork, "__call__", _boom)
    engine = MmsAmharicEngine(device="cpu")

    with pytest.raises(SynthesisError, match="failed to synthesize"):
        engine.synthesize(text="ሰላም", destination=tmp_path / "out.wav")


# ---------------------------------------------------------------------------
# the single-voice stage path
# ---------------------------------------------------------------------------


def test_the_stage_runs_a_single_voice_engine_without_profiles(
    monkeypatch, tmp_path: Path
) -> None:
    """No profiles, no performance prompt, no conversion - and it still works."""

    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")
    settings = _settings(tmp_path)

    result = synthesize_dialogue_detailed(
        [_line()],
        _speech(tmp_path),
        None,
        settings=settings,
        tts_engine=engine,
    )

    assert len(result.clips) == 1
    assert result.skipped == ()
    (clip,) = result.clips
    assert clip.performance_engine == MMS_ENGINE_NAME
    assert clip.style_engine == "none"
    # A single-voice engine follows no prompt and converts no identity, so there is
    # no reference to point at - and the clip says so rather than inventing one.
    assert clip.performance_reference_path is None
    assert clip.voice_reference_path is None
    assert clip.audio_path.is_file()


def test_the_clip_is_usable_by_later_stages(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    result = synthesize_dialogue_detailed(
        [_line()], _speech(tmp_path), None, settings=_settings(tmp_path), tts_engine=engine
    )

    (clip,) = result.clips
    info = sf.info(str(clip.audio_path))
    assert info.channels == 1
    assert clip.speech_duration > 0
    assert clip.duration == pytest.approx(clip.speech_duration, abs=1e-6)


def test_pauses_are_rendered_around_the_take(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")
    line = _line()
    object.__setattr__(line, "pause_before", 0.25)
    object.__setattr__(line, "pause_after", 0.5)

    result = synthesize_dialogue_detailed(
        [line], _speech(tmp_path), None, settings=_settings(tmp_path), tts_engine=engine
    )

    (clip,) = result.clips
    assert clip.rendered_pause_before == pytest.approx(0.25)
    assert clip.rendered_pause_after == pytest.approx(0.5)


def test_unvoiceable_lines_are_still_skipped(monkeypatch, tmp_path: Path) -> None:
    """The guard applies to every engine, not just the prompt-and-convert pair."""

    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    result = synthesize_dialogue_detailed(
        [_line(start=5.836, end=6.056), _line(start=8.0, end=10.0)],
        _speech(tmp_path),
        None,
        settings=_settings(tmp_path),
        tts_engine=engine,
    )

    assert [clip.index for clip in result.clips] == [1]
    assert [line.index for line in result.skipped] == [0]


def test_an_engine_of_the_wrong_kind_is_rejected(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)

    with pytest.raises(InvalidEngineError, match="TextToSpeechEngine"):
        synthesize_dialogue_detailed(
            [_line()],
            _speech(tmp_path),
            None,
            settings=_settings(tmp_path),
            tts_engine=object(),  # type: ignore[arg-type]
        )


def test_the_engine_is_resolved_from_settings_when_not_injected(
    monkeypatch, tmp_path: Path
) -> None:
    """TTS_ENGINE=mms is what makes the default path single-voice."""

    _patch(monkeypatch)
    settings = _settings(tmp_path)

    result = synthesize_dialogue_detailed(
        [_line()], _speech(tmp_path), None, settings=settings
    )

    assert len(result.clips) == 1
    assert result.clips[0].performance_engine == MMS_ENGINE_NAME


def test_the_single_voice_engine_is_cached_per_settings(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    settings = _settings(tmp_path)

    first = tts.load_tts_engine(settings=settings)

    assert tts.load_tts_engine(settings=settings) is first


def test_an_unknown_single_voice_engine_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="TTS_ENGINE"):
        tts.load_tts_engine(settings=_settings(tmp_path, tts_engine="chatterbox"))


def test_the_result_is_json_safe(monkeypatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    engine = MmsAmharicEngine(device="cpu")

    result = synthesize_dialogue_detailed(
        [_line()], _speech(tmp_path), None, settings=_settings(tmp_path), tts_engine=engine
    )

    payload = result.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    # A single-voice clip records no references, and the manifest says so.
    assert payload["clips"] == 1
    assert result.clips[0].to_dict()["voice_reference_path"] is None


# ---------------------------------------------------------------------------
# peak limiting
# ---------------------------------------------------------------------------


def test_a_signal_inside_the_ceiling_is_left_alone() -> None:
    samples = (0.5 * np.ones(64)).astype(np.float32)

    limited, gain = tts._peak_limit(samples)

    assert gain == 0.0
    assert np.array_equal(limited, samples)


def test_a_hot_signal_is_attenuated_rather_than_truncated() -> None:
    """Truncation is irreversible distortion - the defect this exists to prevent."""

    hot = (1.8 * np.ones(64)).astype(np.float32)

    limited, gain = tts._peak_limit(hot)

    assert float(np.max(np.abs(limited))) <= tts.CLIP_CEILING + 1e-6
    assert gain < 0.0
    # The waveform keeps its shape: this is a scale, not a clamp.
    assert np.allclose(limited / limited.max(), hot / hot.max())


def test_an_all_zero_signal_is_untouched() -> None:
    limited, gain = tts._peak_limit(np.zeros(16, dtype=np.float32))

    assert gain == 0.0
    assert float(np.max(np.abs(limited))) == 0.0


def test_a_written_clip_is_never_truncated(monkeypatch, tmp_path: Path) -> None:
    """The defect measured on a real run: 33 of 38 clips were pinned at 1.000."""

    _patch(monkeypatch)
    FakeNetwork.samples = (1.6 * np.sin(np.linspace(0, 40, 4800))).astype(np.float32)
    engine = MmsAmharicEngine(device="cpu")

    result = synthesize_dialogue_detailed(
        [_line()], _speech(tmp_path), None, settings=_settings(tmp_path), tts_engine=engine
    )

    (clip,) = result.clips
    data, _ = sf.read(str(clip.audio_path), dtype="float32")
    assert float(np.max(np.abs(data))) < 1.0
    # Truncation would pile samples up at exactly full scale; a scale does not.
    assert not np.any(np.abs(data) >= 0.9999)
