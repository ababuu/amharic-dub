"""Tests for the OmniVoice engine and the voice-cloning synthesis path.

The model is replaced by a fake, so nothing is downloaded and torch is never needed
for real. What is exercised is this adapter's own behaviour: naming Amharic the way
OmniVoice spells it, asking for a requested rate, pinning the sampling temperatures so
a run is reproducible, building one cloning prompt per character rather than per line,
and writing a playable clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import tts
from app.pipeline.tts import (
    ConfigurationError,
    EngineLoadError,
    InvalidInputError,
    OmniVoiceEngine,
    SynthesisError,
    VoiceCloningEngine,
    synthesize_dialogue_detailed,
)
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.voice_profiles import VoiceProfile
from app.pipeline.dialogue_context import PacingPlan

SPEAKER = "SPEAKER_00"
OTHER_SPEAKER = "SPEAKER_01"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeConfig:
    """Stand-in for ``omnivoice.OmniVoiceGenerationConfig``."""

    def __init__(self, **fields: object) -> None:
        self.fields = dict(fields)
        for key, value in fields.items():
            setattr(self, key, value)


class FakePrompt:
    """Stand-in for a ``VoiceClonePrompt``."""

    def __init__(self, ref_audio: str, ref_text: str | None) -> None:
        self.ref_audio = ref_audio
        self.ref_text = ref_text


class FakeOmniVoice:
    """Stand-in for ``omnivoice.OmniVoice``."""

    instances: list["FakeOmniVoice"] = []
    fail_load: Exception | None = None
    fail_generate: Exception | None = None
    output: list | None = None

    def __init__(self, name: str, **kwargs: object) -> None:
        self.name = name
        self.kwargs = dict(kwargs)
        self.prompts: list[FakePrompt] = []
        self.calls: list[dict] = []
        FakeOmniVoice.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        if cls.fail_load is not None:
            raise cls.fail_load
        return cls(name, **kwargs)

    def create_voice_clone_prompt(
        self, ref_audio=None, ref_text=None, preprocess_prompt=True
    ) -> FakePrompt:
        prompt = FakePrompt(ref_audio, ref_text)
        self.prompts.append(prompt)
        return prompt

    def generate(self, **kwargs):
        if FakeOmniVoice.fail_generate is not None:
            raise FakeOmniVoice.fail_generate
        self.calls.append(kwargs)
        if FakeOmniVoice.output is not None:
            return FakeOmniVoice.output
        return [np.full(24_000, 0.3, dtype=np.float32)]


class FakeTorch:
    float16 = "torch.float16"
    float32 = "torch.float32"


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOmniVoice.instances = []
    FakeOmniVoice.fail_load = None
    FakeOmniVoice.fail_generate = None
    FakeOmniVoice.output = None
    monkeypatch.setattr(
        tts, "_require_omnivoice", lambda: (FakeTorch, FakeOmniVoice, FakeConfig)
    )
    tts.reset_engine_cache()


class _NoRateEngine(OmniVoiceEngine):
    """An engine that meets the cloning contract but cannot control duration."""

    name = "no-rate-engine"

    @property
    def supports_speaking_rate(self) -> bool:
        return False


def _engine(**overrides: object) -> OmniVoiceEngine:
    values: dict[str, object] = {
        "model": "k2-fsa/OmniVoice",
        "device": "cpu",
        "steps": 16,
        "guidance_scale": 2.0,
        "torch_dtype": "float16",
    }
    values.update(overrides)
    return OmniVoiceEngine(**values)  # type: ignore[arg-type]


def _reference(root: Path, name: str = "reference.wav", *, seconds: float = 5.0) -> Path:
    """Write a stand-in character reference recording."""

    rate = 24_000
    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    data = (0.4 * np.sin(2.0 * np.pi * 150.0 * times)).astype(np.float32)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, format="WAV", subtype="PCM_16")
    return path


def _profile(
    root: Path, *, speaker_id: str = SPEAKER, reference_text: str | None = "hello world"
) -> VoiceProfile:
    reference = _reference(root, f"{speaker_id}.wav")
    return VoiceProfile(
        speaker_id=speaker_id,
        reference_audio=reference,
        reference_start=0.0,
        reference_end=5.0,
        reference_text=reference_text,
    )


def _line(
    *,
    start: float = 0.0,
    end: float = 2.0,
    amharic: str = "ሰላም ዓለም።",
    speaker_id: str = SPEAKER,
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


def _settings(root: Path, **overrides: object) -> Settings:
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


# ---------------------------------------------------------------------------
# the contract the engine claims to meet
# ---------------------------------------------------------------------------


def test_the_engine_meets_the_voice_cloning_contract() -> None:
    assert issubclass(OmniVoiceEngine, VoiceCloningEngine)
    assert _engine().name == "omnivoice"
    assert _engine().sample_rate == 24_000


def test_the_engine_says_it_can_honour_a_rate() -> None:
    """The caller has to know, or it will stretch audio that could have been asked for."""

    assert _engine().supports_speaking_rate is True


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def test_amharic_is_named_the_way_omnivoice_spells_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Its language map is keyed by name, so ``amh`` is not a code it accepts."""

    _patch(monkeypatch)
    engine = _engine()

    engine.synthesize(
        text="ሰላም", voice_reference=_reference(tmp_path), destination=tmp_path / "o.wav"
    )

    call = FakeOmniVoice.instances[0].calls[0]
    assert call["language"] == tts.OMNIVOICE_AMHARIC == "am"
    assert call["language"] != "amh"


def test_sampling_is_pinned_so_a_run_can_be_reproduced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model samples by default, which would make two runs incomparable."""

    _patch(monkeypatch)

    _engine().synthesize(
        text="ሰላም", voice_reference=_reference(tmp_path), destination=tmp_path / "o.wav"
    )

    config = FakeOmniVoice.instances[0].calls[0]["generation_config"]
    assert config.position_temperature == 0.0
    assert config.class_temperature == 0.0
    assert config.num_step == 16
    assert config.guidance_scale == 2.0


def test_a_requested_rate_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Asking for a duration beats stretching audio to reach one."""

    _patch(monkeypatch)

    _engine().synthesize(
        text="ሰላም",
        voice_reference=_reference(tmp_path),
        destination=tmp_path / "o.wav",
        speaking_rate=1.25,
    )

    assert FakeOmniVoice.instances[0].calls[0]["speed"] == 1.25


def test_no_rate_is_requested_when_none_is_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model's own pacing is left alone rather than being pinned to a guess."""

    _patch(monkeypatch)

    _engine().synthesize(
        text="ሰላም", voice_reference=_reference(tmp_path), destination=tmp_path / "o.wav"
    )

    assert "speed" not in FakeOmniVoice.instances[0].calls[0]


def test_synthesis_writes_a_playable_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    destination = tmp_path / "take.wav"

    engine = _engine()
    engine.synthesize(
        text="ሰላም", voice_reference=_reference(tmp_path), destination=destination
    )

    assert destination.is_file()
    info = sf.info(str(destination))
    assert info.samplerate == 24_000
    assert info.channels == 1
    assert info.frames == 24_000
    assert engine.is_loaded is True


def test_the_model_is_loaded_once_for_many_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    reference = _reference(tmp_path)
    engine = _engine()

    for index in range(4):
        engine.synthesize(
            text="ሰላም",
            voice_reference=reference,
            destination=tmp_path / f"{index}.wav",
        )

    assert len(FakeOmniVoice.instances) == 1


# ---------------------------------------------------------------------------
# per-character identity
# ---------------------------------------------------------------------------


def test_one_prompt_is_built_per_character(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prompt is a character's identity, so a film pays for it once, not per line."""

    _patch(monkeypatch)
    reference = _reference(tmp_path)
    engine = _engine()

    for index in range(3):
        engine.synthesize(
            text=f"ሰላም {index}",
            voice_reference=reference,
            destination=tmp_path / f"{index}.wav",
            reference_text="hello world",
        )

    assert len(FakeOmniVoice.instances[0].prompts) == 1


def test_two_characters_get_two_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    engine = _engine()

    engine.synthesize(
        text="ሰላም",
        voice_reference=_reference(tmp_path, "a.wav"),
        destination=tmp_path / "a.wav",
        reference_text="one",
    )
    engine.synthesize(
        text="ሰላም",
        voice_reference=_reference(tmp_path, "b.wav"),
        destination=tmp_path / "b.wav",
        reference_text="two",
    )

    prompts = FakeOmniVoice.instances[0].prompts
    assert len(prompts) == 2
    assert prompts[0].ref_audio != prompts[1].ref_audio


def test_the_reference_transcript_is_passed_when_it_is_known(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Supplying it is what stops a run loading an ASR model it does not need."""

    _patch(monkeypatch)

    _engine().synthesize(
        text="ሰላም",
        voice_reference=_reference(tmp_path),
        destination=tmp_path / "o.wav",
        reference_text="  hello world  ",
    )

    assert FakeOmniVoice.instances[0].prompts[0].ref_text == "hello world"


def test_the_transcript_is_left_to_the_model_when_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty transcript means "not known", never "the empty string"."""

    _patch(monkeypatch)

    _engine().synthesize(
        text="ሰላም",
        voice_reference=_reference(tmp_path),
        destination=tmp_path / "o.wav",
        reference_text="   ",
    )

    assert FakeOmniVoice.instances[0].prompts[0].ref_text is None


# ---------------------------------------------------------------------------
# failures are named, not swallowed
# ---------------------------------------------------------------------------


def test_a_reference_that_cannot_be_read_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)

    with pytest.raises(Exception, match="voice reference"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=tmp_path / "missing.wav",
            destination=tmp_path / "o.wav",
        )


def test_a_load_failure_is_reported_as_an_engine_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    FakeOmniVoice.fail_load = RuntimeError("no weights")

    with pytest.raises(EngineLoadError, match="could not load the OmniVoice model"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


def test_a_generation_failure_is_reported_as_a_synthesis_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    FakeOmniVoice.fail_generate = RuntimeError("boom")

    with pytest.raises(SynthesisError, match="failed to synthesize"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


def test_an_empty_result_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    FakeOmniVoice.output = []

    with pytest.raises(SynthesisError, match="returned no audio"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


def test_an_empty_waveform_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    FakeOmniVoice.output = [np.zeros(0, dtype=np.float32)]

    with pytest.raises(SynthesisError, match="empty waveform"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


def test_text_with_nothing_to_say_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)

    with pytest.raises(InvalidInputError, match="nothing to synthesize"):
        _engine().synthesize(
            text="   ",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_a_rate_that_is_not_a_usable_number_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rate: float
) -> None:
    _patch(monkeypatch)

    with pytest.raises(ConfigurationError, match="positive number"):
        _engine().synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
            speaking_rate=rate,
        )


def test_unusable_construction_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="non-empty id"):
        _engine(model="  ")
    with pytest.raises(ConfigurationError, match="non-empty string"):
        _engine(device="  ")
    with pytest.raises(ConfigurationError, match="OMNIVOICE_STEPS"):
        _engine(steps=0)
    with pytest.raises(ConfigurationError, match="OMNIVOICE_GUIDANCE_SCALE"):
        _engine(guidance_scale=0.0)


def test_a_dtype_the_runtime_does_not_have_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A typo in the dtype must not silently pick some other precision."""

    _patch(monkeypatch)

    with pytest.raises(ConfigurationError, match="not a torch dtype"):
        _engine(torch_dtype="float128").synthesize(
            text="ሰላም",
            voice_reference=_reference(tmp_path),
            destination=tmp_path / "o.wav",
        )


# ---------------------------------------------------------------------------
# the synthesis stage's dispatch
# ---------------------------------------------------------------------------


def test_only_the_engines_that_read_a_reference_need_profiles() -> None:
    assert tts.engine_needs_profiles("omnivoice") is True
    assert tts.engine_needs_profiles("chatterbox") is True
    assert tts.engine_needs_profiles("mms") is False


def test_the_artifact_name_distinguishes_engines(tmp_path: Path) -> None:
    """Two engines must not collide on one line's audio."""

    settings = _settings(tmp_path)

    names = {
        tts.artifact_model_name("chatterbox", settings=settings),
        tts.artifact_model_name("omnivoice", settings=settings),
        tts.artifact_model_name("mms", settings=settings),
    }

    assert len(names) == 3
    assert tts.artifact_model_name("omnivoice", settings=settings) == (
        f"omnivoice:{settings.omnivoice_model}"
    )
    assert tts.artifact_model_name("mms", settings=settings) == (
        f"mms:{settings.tts_model}"
    )


def test_the_cloning_path_needs_no_conversion_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clip records the character's own voice and no converted take."""

    _patch(monkeypatch)
    settings = _settings(tmp_path)
    profiles = {SPEAKER: _profile(tmp_path)}

    result = synthesize_dialogue_detailed(
        [_line()],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        profiles,
        output_dir=tmp_path / "tts",
        settings=settings,
    )

    (clip,) = result.clips
    assert clip.performance_engine == "omnivoice"
    assert clip.style_engine == "none"
    assert clip.performance_reference_path is None
    assert clip.voice_reference_path == profiles[SPEAKER].resolve_reference_audio()
    assert clip.audio_path.is_file()
    assert clip.take_path.is_file()


def test_the_cloning_path_gives_each_speaker_their_own_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    settings = _settings(tmp_path)
    profiles = {
        SPEAKER: _profile(tmp_path, speaker_id=SPEAKER, reference_text="one"),
        OTHER_SPEAKER: _profile(tmp_path, speaker_id=OTHER_SPEAKER, reference_text="two"),
    }

    result = synthesize_dialogue_detailed(
        [_line(), _line(start=2.0, end=4.0, amharic="ሁለት።", speaker_id=OTHER_SPEAKER)],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        profiles,
        output_dir=tmp_path / "tts",
        settings=settings,
    )

    assert len(result.clips) == 2
    # One prompt per character, built from each character's own reference.
    assert len(FakeOmniVoice.instances[0].prompts) == 2
    references = {prompt.ref_audio for prompt in FakeOmniVoice.instances[0].prompts}
    assert references == {
        str(profiles[SPEAKER].resolve_reference_audio()),
        str(profiles[OTHER_SPEAKER].resolve_reference_audio()),
    }


def test_the_chatterbox_prompt_bounds_are_not_imposed_on_a_cloning_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Those bounds describe a prompt this path never cuts, so they cannot apply."""

    _patch(monkeypatch)
    settings = _settings(
        tmp_path,
        tts_performance_reference_min_duration=12.0,
        tts_performance_reference_max_duration=6.0,
    )

    result = synthesize_dialogue_detailed(
        [_line()],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        {SPEAKER: _profile(tmp_path)},
        output_dir=tmp_path / "tts",
        settings=settings,
    )

    assert result.clips


def test_an_engine_that_meets_no_contract_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)

    with pytest.raises(tts.InvalidEngineError, match="VoiceCloningEngine"):
        synthesize_dialogue_detailed(
            [_line()],
            _reference(tmp_path, "stem.wav", seconds=8.0),
            {SPEAKER: _profile(tmp_path)},
            output_dir=tmp_path / "tts",
            settings=_settings(tmp_path),
            tts_engine=object(),  # type: ignore[arg-type]
        )


def test_a_missing_reference_fails_before_any_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A character with no usable reference is a configuration error, not a hole."""

    _patch(monkeypatch)
    profile = _profile(tmp_path)
    profile.reference_audio.unlink()

    with pytest.raises(Exception, match="voice reference"):
        synthesize_dialogue_detailed(
            [_line()],
            _reference(tmp_path, "stem.wav", seconds=8.0),
            {SPEAKER: profile},
            output_dir=tmp_path / "tts",
            settings=_settings(tmp_path),
        )


def test_resetting_the_cache_drops_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    first = tts.load_omnivoice_engine(settings=_settings(tmp_path))
    assert tts.load_omnivoice_engine(settings=_settings(tmp_path)) is first

    tts.reset_engine_cache()

    assert tts.load_omnivoice_engine(settings=_settings(tmp_path)) is not first


# ---------------------------------------------------------------------------
# asking the model for a length instead of stretching afterwards
# ---------------------------------------------------------------------------


def test_a_long_line_is_corrected_within_the_local_band() -> None:
    """A single long line moves by the local band; the film rate carries any real bias.

    Asking for the full 4.0 this line "wants" would be a request the timing stage's own
    limits would not allow, and it would make this line's delivery unlike every other.
    """

    plan = PacingPlan(film_rate=1.0, minimum=0.8, maximum=1.25)
    # 16 syllables at 4/second is 4 seconds of speech in a 2-second window.
    line = _line(start=0.0, end=2.0, amharic="ሰ" * 16)

    assert tts.requested_speaking_rate(line, plan=plan) == 1.1


def test_the_rate_never_leaves_the_band_the_timing_stage_uses() -> None:
    """One policy governs both this request and the stretch applied afterwards."""

    plan = PacingPlan(film_rate=1.25, minimum=0.8, maximum=1.25)
    line = _line(start=0.0, end=1.0, amharic="ሰ" * 40)

    asked = tts.requested_speaking_rate(line, plan=plan)

    assert asked is not None
    assert 0.8 <= asked <= 1.25


def test_a_slow_line_is_left_to_the_timing_stage() -> None:
    """Fitting one down into a longer window needs no request to the model.

    Drawing a line out by asking for a slower rate buys nothing over letting the timing
    stage place it, and it costs a generation.
    """

    plan = PacingPlan(film_rate=1.0, minimum=0.8, maximum=1.25)
    # 4 syllables at 4/second is 1 second of speech in a 4-second window.
    line = _line(start=0.0, end=4.0, amharic="ሰ" * 4)

    assert tts.requested_speaking_rate(line, plan=plan) is None


def test_a_line_that_needs_nothing_is_not_speed_up_by_the_film_rate() -> None:
    """A comfortable line must not inherit a film-wide speed-up it does not need."""

    plan = PacingPlan(film_rate=1.25, minimum=0.8, maximum=1.25)
    # 8 syllables at 4/second is 2 seconds, exactly the window.
    line = _line(start=0.0, end=2.0, amharic="ሰ" * 8)

    assert tts.requested_speaking_rate(line, plan=plan) is None


def test_a_line_that_already_fits_asks_for_nothing() -> None:
    """A needless request would spend a different generation on an inaudible change."""

    plan = PacingPlan(film_rate=1.0, minimum=0.8, maximum=1.25)
    # 8 syllables at 4/second is 2 seconds, exactly the window.
    line = _line(start=0.0, end=2.0, amharic="ሰ" * 8)

    assert tts.requested_speaking_rate(line, plan=plan) is None


def test_an_outlier_line_only_moves_slightly_from_the_film_rate() -> None:
    """The film rate does the work; one line is not allowed to wander off on its own."""

    plan = PacingPlan(film_rate=1.1, minimum=0.8, maximum=1.25)
    # This line on its own would want 4.0; the local band caps how far it may go.
    line = _line(start=0.0, end=2.0, amharic="ሰ" * 32)

    asked = tts.requested_speaking_rate(line, plan=plan)

    assert asked is not None
    assert asked < 4.0
    assert asked <= 1.25


def test_a_line_with_nothing_to_say_asks_for_nothing() -> None:
    plan = PacingPlan(film_rate=1.0, minimum=0.8, maximum=1.25)
    line = _line(start=0.0, end=2.0, amharic="።")

    assert tts.requested_speaking_rate(line, plan=plan) is None


def test_the_rate_is_asked_for_through_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stage asks only when the engine says it can honour a rate."""

    _patch(monkeypatch)
    settings = _settings(tmp_path)
    profile = _profile(tmp_path)
    # 40 syllables in two seconds is far past what the clamp allows.
    line = _line(start=0.0, end=2.0, amharic="ሰ" * 40)

    synthesize_dialogue_detailed(
        [line],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        {SPEAKER: profile},
        output_dir=tmp_path / "tts",
        settings=settings,
    )

    assert FakeOmniVoice.instances[0].calls[0]["speed"] == settings.timing_max_tempo


def test_the_rate_request_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A comparison run has to be able to attribute a difference to this."""

    _patch(monkeypatch)
    settings = _settings(tmp_path, tts_request_rate=False)
    profile = _profile(tmp_path)

    synthesize_dialogue_detailed(
        [_line(start=0.0, end=2.0, amharic="ሰ" * 40)],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        {SPEAKER: profile},
        output_dir=tmp_path / "tts",
        settings=settings,
    )

    assert "speed" not in FakeOmniVoice.instances[0].calls[0]


def test_an_engine_without_rate_control_is_never_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Asking an engine that cannot honour a rate would fail or be silently ignored.

    The contract reports this rather than the stage assuming it, which is what keeps an
    engine with no duration control usable instead of crashing it.
    """

    _patch(monkeypatch)
    limited = _NoRateEngine(model="k2-fsa/OmniVoice", device="cpu")

    synthesize_dialogue_detailed(
        [_line(start=0.0, end=2.0, amharic="ሰ" * 40)],
        _reference(tmp_path, "stem.wav", seconds=8.0),
        {SPEAKER: _profile(tmp_path)},
        output_dir=tmp_path / "tts",
        settings=_settings(tmp_path),
        tts_engine=limited,
    )

    assert "speed" not in FakeOmniVoice.instances[0].calls[0]
