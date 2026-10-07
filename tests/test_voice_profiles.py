"""Tests for :mod:`app.pipeline.voice_profiles`.

OmniVoice is never imported, no model weights are downloaded, and no network
call is made: the only model-specific hook - the voice-clone encoder - is
injected as a plain callable. Dialogue stems are synthesised with ``numpy`` and
written with ``soundfile`` so the selection logic runs against real audio, and
FFmpeg is replaced by an in-process fake everywhere except the one end-to-end
extraction test.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import diarization, transcription, voice_profiles
from app.pipeline.diarization import SpeakerSegment
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.voice_profiles import (
    CLONE_PROMPT_FILENAME,
    PROFILE_FILENAME,
    REFERENCE_FILENAME,
    AudioAnalysisError,
    ClonePromptError,
    ConfigurationError,
    InvalidInputError,
    InvalidProfileError,
    InvalidSegmentError,
    MissingInputError,
    NoUsableReferenceError,
    ProfilePersistenceError,
    ReferenceExtractionError,
    ReferencePreprocessingError,
    VoiceProfile,
    build_voice_profiles,
    load_voice_profiles,
    portable_path,
    resolve_project_path,
    resolve_voice_profile_dir,
    save_voice_profiles,
)

STEM_RATE = 48_000
REFERENCE_RATE = voice_profiles.REFERENCE_SAMPLE_RATE

#: Shape of the synthetic speech used here: a burst of about 150 ms followed by a
#: gap, i.e. the syllabic rhythm of real dialogue rather than a continuous tone.
#: The module's VAD (correctly) treats a steady tone as non-speech, so the
#: positive tests must hand it speech-like audio.
SPEECH_SYLLABLE_HZ = 4.0
SPEECH_DUTY = 0.6
SPEECH_FLOOR_RATIO = 0.02


@dataclass(frozen=True)
class Signal:
    """A synthetic audio region: what to render, and how loud."""

    kind: str
    level: float


#: Speech-like audio: about -29 dBFS at ``LOUD``, about -57 dBFS at ``QUIET``
#: (audible but clearly worse).
LOUD = Signal("speech", 0.1)
QUIET = Signal("speech", 0.004)
#: A loud *continuous* bed - music, a tone, hum or bleed. Deliberately louder
#: than ``LOUD`` so a test can prove that level alone must not win.
STEADY = Signal("music", 0.3)
SILENCE = Signal("silence", 0.0)
#: Speech-like audio squashed against full scale, i.e. distorted.
CLIPPED = Signal("clipped", 1.6)


def _render(signal: Signal, seconds: float) -> np.ndarray:
    """Return ``seconds`` of the requested synthetic signal."""

    total = int(round(seconds * STEM_RATE))
    time = np.arange(total) / STEM_RATE
    voice = np.sin(2.0 * np.pi * 220.0 * time)

    if signal.kind == "silence":
        return np.zeros(total, dtype=np.float32)
    if signal.kind == "music":
        return (signal.level * voice).astype(np.float32)

    phase = (time * SPEECH_SYLLABLE_HZ) % 1.0
    burst = np.where(phase < SPEECH_DUTY, np.sin(np.pi * phase / SPEECH_DUTY) ** 2, 0.0)
    speech = signal.level * (burst * voice + SPEECH_FLOOR_RATIO * voice)
    if signal.kind == "clipped":
        speech = np.clip(speech, -1.0, 1.0)
    return speech.astype(np.float32)


def _write_stem(path: Path, regions: list[tuple[float, float, Signal]]) -> Path:
    """Write a mono stem; ``regions`` are ``(start, end, signal)`` triangles."""

    total = max(end for _, end, _ in regions)
    audio = np.zeros(int(round(total * STEM_RATE)), dtype=np.float32)
    for start, end, signal in regions:
        first = int(round(start * STEM_RATE))
        last = int(round(end * STEM_RATE))
        audio[first:last] = _render(signal, (last - first) / STEM_RATE)
    sf.write(str(path), audio, STEM_RATE, subtype="PCM_16")
    return path


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``."""

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
        "voice_profile_dir": root / "voices",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _profile(root: Path, **overrides: object) -> VoiceProfile:
    """Return a valid profile inside ``root``, with optional field overrides."""

    values: dict[str, object] = {
        "speaker_id": "SPEAKER_00",
        "reference_audio": root / "voices" / "SPEAKER_00" / REFERENCE_FILENAME,
        "reference_start": 10.0,
        "reference_end": 20.0,
    }
    values.update(overrides)
    return VoiceProfile(**values)  # type: ignore[arg-type]


def _candidate(
    *,
    start: float = 10.0,
    end: float = 20.0,
    speech_ratio: float = 0.6,
    dynamic_range_db: float = 30.0,
    rms_dbfs: float = -29.0,
    overlap_seconds: float = 0.0,
    clipped_fraction: float = 0.0,
    text: str | None = None,
    variety: float | None = None,
) -> voice_profiles._ReferenceCandidate:
    """Return a candidate window with hand-made measurements."""

    return voice_profiles._ReferenceCandidate(
        speaker_id="SPEAKER_00",
        start=start,
        end=end,
        stats=voice_profiles._WindowStats(
            speech_ratio=speech_ratio,
            dynamic_range_db=dynamic_range_db,
            rms_dbfs=rms_dbfs,
            peak_dbfs=-20.0,
            clipped_fraction=clipped_fraction,
        ),
        overlap_seconds=overlap_seconds,
        text=text,
        variety=variety,
    )


class FakeFfmpeg:
    """In-process stand-in for the FFmpeg extraction subprocess.

    It records the argv it was handed and writes a real WAV so the post-extraction
    verification has something to inspect.
    """

    def __init__(
        self,
        *,
        rate: int = REFERENCE_RATE,
        channels: int = 1,
        duration_scale: float = 1.0,
    ) -> None:
        self.rate = rate
        self.channels = channels
        self.duration_scale = duration_scale
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> None:
        self.calls.append(list(arguments))
        destination = Path(arguments[-1])
        duration = float(arguments[arguments.index("-t") + 1]) * self.duration_scale
        frames = max(1, int(round(duration * self.rate)))
        data = np.full((frames, self.channels), 0.25, dtype=np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), data, self.rate, subtype="PCM_16")

    @property
    def last(self) -> list[str]:
        """Return the argv of the most recent call."""

        return self.calls[-1]


class FakeEncoder:
    """Stand-in for the OmniVoice clone-prompt encoder."""

    def __init__(
        self,
        *,
        payload: bytes = b"fake-prompt",
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, reference_audio: Path, destination: Path) -> Path:
        self.calls.append((Path(reference_audio), Path(destination)))
        if self.error is not None:
            raise self.error
        Path(destination).write_bytes(self.payload)
        return Path(destination)


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFfmpeg:
    """Replace the FFmpeg subprocess seam with an in-process fake."""

    runner = FakeFfmpeg()
    monkeypatch.setattr(voice_profiles, "_run_ffmpeg", runner)
    return runner


# ---------------------------------------------------------------------------
# VoiceProfile validation
# ---------------------------------------------------------------------------


def test_profile_accepts_a_valid_reference(tmp_path: Path) -> None:
    profile = _profile(tmp_path, reference_text="hello", quality_score=0.75)

    assert profile.speaker_id == "SPEAKER_00"
    assert profile.reference_start == 10.0
    assert profile.reference_end == 20.0
    assert profile.reference_duration == 10.0
    assert profile.reference_text == "hello"
    assert profile.quality_score == 0.75
    assert profile.clone_prompt_path is None


def test_profile_is_frozen_and_slotted(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    assert not hasattr(profile, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.speaker_id = "SPEAKER_01"  # type: ignore[misc]


def test_profile_duration_is_derived_not_stored() -> None:
    fields = set(VoiceProfile.__dataclass_fields__)

    assert "reference_duration" not in fields
    assert "duration" not in fields


def test_profile_keeps_the_speaker_id_exactly_as_given(tmp_path: Path) -> None:
    profile = _profile(tmp_path, speaker_id="SPEAKER_07")

    assert profile.speaker_id == "SPEAKER_07"


def test_profile_rejects_an_empty_speaker_id(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="speaker_id must be a non-empty string"):
        _profile(tmp_path, speaker_id="   ")

    with pytest.raises(InvalidProfileError, match="speaker_id"):
        _profile(tmp_path, speaker_id=7)


def test_profile_rejects_a_non_finite_timestamp(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="reference_start must be finite"):
        _profile(tmp_path, reference_start=float("nan"))

    with pytest.raises(InvalidProfileError, match="reference_end must be finite"):
        _profile(tmp_path, reference_end=float("inf"))


def test_profile_rejects_a_boolean_timestamp(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="must be a number"):
        _profile(tmp_path, reference_start=True)


def test_profile_rejects_a_negative_start(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="reference_start must be >= 0"):
        _profile(tmp_path, reference_start=-1.0, reference_end=10.0)


def test_profile_rejects_an_end_that_is_not_after_the_start(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="greater than reference_start"):
        _profile(tmp_path, reference_start=10.0, reference_end=10.0)


def test_profile_rejects_a_duration_outside_the_cloning_range(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="voice cloning"):
        _profile(tmp_path, reference_start=10.0, reference_end=10.5)

    with pytest.raises(InvalidProfileError, match="voice cloning"):
        _profile(tmp_path, reference_start=0.0, reference_end=60.0)


def test_profile_rejects_an_empty_reference_audio_path(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="reference_audio"):
        _profile(tmp_path, reference_audio="")

    with pytest.raises(InvalidProfileError, match="reference_audio"):
        _profile(tmp_path, reference_audio=None)


def test_profile_rejects_a_directory_as_reference_audio(tmp_path: Path) -> None:
    directory = tmp_path / "reference.wav"
    directory.mkdir()

    with pytest.raises(InvalidProfileError, match="must be a file path"):
        _profile(tmp_path, reference_audio=directory)


def test_profile_accepts_reference_audio_as_a_string(tmp_path: Path) -> None:
    profile = _profile(tmp_path, reference_audio="voices/SPEAKER_00/reference.wav")

    assert profile.reference_audio == Path("voices/SPEAKER_00/reference.wav")


def test_profile_normalises_reference_text(tmp_path: Path) -> None:
    assert _profile(tmp_path, reference_text="  hello  ").reference_text == "hello"
    assert _profile(tmp_path, reference_text="   ").reference_text is None
    assert _profile(tmp_path, reference_text=None).reference_text is None


def test_profile_rejects_non_string_reference_text(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="reference_text must be a string"):
        _profile(tmp_path, reference_text=12)


def test_profile_rejects_a_directory_as_the_clone_prompt(tmp_path: Path) -> None:
    directory = tmp_path / "voice_clone.pt"
    directory.mkdir()

    with pytest.raises(InvalidProfileError, match="clone_prompt_path"):
        _profile(tmp_path, clone_prompt_path=directory)


def test_profile_accepts_a_clone_prompt_path(tmp_path: Path) -> None:
    prompt = tmp_path / "voices" / "SPEAKER_00" / CLONE_PROMPT_FILENAME

    profile = _profile(tmp_path, clone_prompt_path=str(prompt))

    assert profile.clone_prompt_path == prompt
    assert profile.resolve_clone_prompt() == prompt


def test_profile_rejects_an_out_of_range_quality_score(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="quality_score must be between 0 and 1"):
        _profile(tmp_path, quality_score=1.4)

    with pytest.raises(InvalidProfileError, match="quality_score"):
        _profile(tmp_path, quality_score=float("nan"))


def test_profile_strips_the_selection_reason(tmp_path: Path) -> None:
    assert _profile(tmp_path, selection_reason="  best  ").selection_reason == "best"
    assert _profile(tmp_path).selection_reason == ""


def test_profile_has_no_performance_metadata() -> None:
    fields = set(VoiceProfile.__dataclass_fields__)

    assert fields == {
        "speaker_id",
        "reference_audio",
        "reference_start",
        "reference_end",
        "reference_text",
        "clone_prompt_path",
        "quality_score",
        "selection_reason",
    }
    for forbidden in ("emotion", "intensity", "delivery", "pause_before", "pause_after"):
        assert forbidden not in fields


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_profile_serializes_project_paths_relative() -> None:
    project_reference = (
        voice_profiles.PROJECT_ROOT / "data" / "working" / "voices" / "SPEAKER_00" / REFERENCE_FILENAME
    )

    payload = _profile(Path("/tmp"), reference_audio=project_reference).to_dict()

    assert payload["reference_audio"] == "data/working/voices/SPEAKER_00/reference.wav"


def test_profile_serializes_outside_paths_as_absolute_posix(tmp_path: Path) -> None:
    payload = _profile(tmp_path).to_dict()

    assert "\\" not in str(payload["reference_audio"])
    assert str(payload["reference_audio"]).endswith("/voices/SPEAKER_00/reference.wav")


def test_profile_json_round_trip(tmp_path: Path) -> None:
    original = _profile(
        tmp_path,
        reference_text="hello",
        clone_prompt_path=tmp_path / "voices" / "SPEAKER_00" / CLONE_PROMPT_FILENAME,
        quality_score=0.5,
        selection_reason="best",
    )

    restored = VoiceProfile.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original
    assert restored.reference_duration == original.reference_duration


def test_profile_from_dict_rejects_a_non_mapping() -> None:
    with pytest.raises(InvalidProfileError, match="must be a JSON object"):
        VoiceProfile.from_dict(["SPEAKER_00"])


def test_profile_from_dict_rejects_missing_fields(tmp_path: Path) -> None:
    payload = _profile(tmp_path).to_dict()
    del payload["reference_start"]

    with pytest.raises(InvalidProfileError, match="missing reference_start"):
        VoiceProfile.from_dict(payload)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    profiles = {"SPEAKER_00": _profile(tmp_path)}

    written = save_voice_profiles(profiles, directory=tmp_path / "out")

    assert written["SPEAKER_00"] == tmp_path / "out" / "SPEAKER_00" / PROFILE_FILENAME
    assert written["SPEAKER_00"].is_file()
    assert load_voice_profiles(directory=tmp_path / "out") == profiles


def test_save_voice_profiles_rejects_a_mismatched_key(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="stored under the key"):
        save_voice_profiles({"SPEAKER_09": _profile(tmp_path)}, directory=tmp_path / "out")


def test_save_voice_profiles_uses_the_settings_directory(tmp_path: Path) -> None:
    written = save_voice_profiles({"SPEAKER_00": _profile(tmp_path)}, settings=_settings(tmp_path))

    assert written["SPEAKER_00"].parent.parent == tmp_path / "voices"


def test_load_voice_profiles_requires_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="directory not found"):
        load_voice_profiles(directory=tmp_path / "missing")


def test_load_voice_profiles_reports_malformed_json(tmp_path: Path) -> None:
    manifest = tmp_path / "voices" / "SPEAKER_00" / PROFILE_FILENAME
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{not json", encoding="utf-8")

    with pytest.raises(ProfilePersistenceError, match="could not read"):
        load_voice_profiles(directory=tmp_path / "voices")


def test_load_voice_profiles_ignores_directories_without_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "voices" / "SPEAKER_00").mkdir(parents=True)
    (tmp_path / "voices" / "loose-file.txt").write_text("x", encoding="utf-8")

    assert load_voice_profiles(directory=tmp_path / "voices") == {}


def test_portable_and_resolved_paths_are_inverse() -> None:
    absolute = voice_profiles.PROJECT_ROOT / "data" / "working" / "voices"

    assert resolve_project_path(portable_path(absolute)) == absolute
    assert resolve_project_path(Path("outside")) == voice_profiles.PROJECT_ROOT / "outside"


def test_resolve_voice_profile_dir_prefers_the_explicit_directory(tmp_path: Path) -> None:
    assert resolve_voice_profile_dir(directory=tmp_path / "x") == tmp_path / "x"
    assert resolve_voice_profile_dir(settings=_settings(tmp_path)) == tmp_path / "voices"


# ---------------------------------------------------------------------------
# build_voice_profiles: input validation
# ---------------------------------------------------------------------------


def test_empty_segments_return_an_empty_mapping_without_touching_audio(tmp_path: Path) -> None:
    profiles = build_voice_profiles([], tmp_path / "does-not-exist.wav", settings=_settings(tmp_path))

    assert profiles == {}


def test_missing_source_audio_fails(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError, match="source audio file not found"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 1.0, 5.0)],
            tmp_path / "missing.wav",
            settings=_settings(tmp_path),
        )


def test_source_audio_must_be_a_file(tmp_path: Path) -> None:
    directory = tmp_path / "movie"
    directory.mkdir()

    with pytest.raises(InvalidInputError, match="not a file"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 1.0, 5.0)],
            directory,
            settings=_settings(tmp_path),
        )


def test_invalid_speaker_segments_are_rejected(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(1.0, 11.0, LOUD)])

    with pytest.raises(InvalidSegmentError, match="SpeakerSegment objects"):
        build_voice_profiles([(1.0, 11.0)], stem, settings=_settings(tmp_path))

    with pytest.raises(InvalidSegmentError, match="iterable"):
        build_voice_profiles(None, stem, settings=_settings(tmp_path))  # type: ignore[arg-type]


def test_performance_objects_are_not_accepted_as_segments(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(1.0, 11.0, LOUD)])
    dialogue = AdaptedDialogue(
        speaker_id="SPEAKER_00",
        start=1.0,
        end=11.0,
        source_text="hello",
        amharic="\u1230\u120b\u121d",
        emotion="neutral",
        intensity=0.5,
        delivery="calm",
        pause_before=0.0,
        pause_after=0.0,
    )

    with pytest.raises(InvalidSegmentError, match="AdaptedDialogue"):
        build_voice_profiles([dialogue], stem, settings=_settings(tmp_path))  # type: ignore[list-item]


def test_invalid_transcript_objects_are_rejected(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(1.0, 11.0, LOUD)])
    segments = [SpeakerSegment("SPEAKER_00", 1.0, 11.0)]

    with pytest.raises(InvalidInputError, match="TranscriptSegment objects"):
        build_voice_profiles(
            segments, stem, transcript=[("SPEAKER_00", 1.0, 2.0, "hi")], settings=_settings(tmp_path)
        )


def test_settings_bounds_are_validated(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(1.0, 11.0, LOUD)])
    segments = [SpeakerSegment("SPEAKER_00", 1.0, 11.0)]

    with pytest.raises(ConfigurationError, match="must be at least"):
        build_voice_profiles(
            segments, stem, settings=_settings(tmp_path, voice_reference_min_duration=0.5)
        )

    with pytest.raises(ConfigurationError, match="VOICE_REFERENCE_MIN_DURATION"):
        build_voice_profiles(
            segments,
            stem,
            settings=_settings(
                tmp_path, voice_reference_min_duration=6.0, voice_reference_target_duration=4.0
            ),
        )

    with pytest.raises(ConfigurationError, match="VOICE_REFERENCE_MAX_DURATION"):
        build_voice_profiles(
            segments, stem, settings=_settings(tmp_path, voice_reference_max_duration=60.0)
        )


def test_default_settings_are_used_when_none_is_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(1.0, 11.0, LOUD)])
    monkeypatch.setattr(voice_profiles, "get_settings", lambda: _settings(tmp_path))

    profiles = build_voice_profiles([SpeakerSegment("SPEAKER_00", 1.0, 11.0)], stem)

    assert (
        profiles["SPEAKER_00"].resolve_reference_audio()
        == tmp_path / "voices" / "SPEAKER_00" / REFERENCE_FILENAME
    )


def test_unreadable_audio_is_reported(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = tmp_path / "speech.wav"
    stem.write_text("not audio", encoding="utf-8")

    with pytest.raises(AudioAnalysisError, match="could not open the dialogue stem"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 1.0, 5.0)], stem, settings=_settings(tmp_path)
        )


# ---------------------------------------------------------------------------
# build_voice_profiles: reference selection
# ---------------------------------------------------------------------------


def test_best_candidate_prefers_the_louder_clean_window(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(
        tmp_path / "speech.wav",
        [(10.0, 20.0, QUIET), (30.0, 40.0, LOUD)],
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert profile.reference_end == 40.0


def test_longer_clean_speech_is_preferred(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 14.0, LOUD), (30.0, 39.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 14.0),
        SpeakerSegment("SPEAKER_00", 30.0, 39.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_duration == 9.0
    assert profile.reference_start == 30.0


def test_overlapping_speech_is_penalized(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 18.0, LOUD), (40.0, 48.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 18.0),
        SpeakerSegment("SPEAKER_00", 40.0, 48.0),
        SpeakerSegment("SPEAKER_01", 40.0, 48.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 10.0
    assert "0.0s overlapped" in profile.selection_reason


def test_overlapping_speech_is_reported_in_the_reason(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (15.0, 25.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_01", 15.0, 25.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert "5.0s overlapped" in profile.selection_reason
    assert profile.quality_score < 1.0


def test_a_clipped_candidate_loses_to_a_clean_one(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, CLIPPED), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert "clipped" in profile.selection_reason


def test_too_short_candidates_are_rejected(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 12.0, LOUD)])

    with pytest.raises(NoUsableReferenceError, match="at least 3.00s is required"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 12.0)], stem, settings=_settings(tmp_path)
        )


def test_a_short_but_clean_reference_is_used_when_it_is_all_there_is(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 15.0, LOUD)])

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 15.0)], stem, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_duration == 5.0


def test_silence_is_not_a_usable_reference(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(
        tmp_path / "speech.wav",
        [(10.0, 20.0, SILENCE), (30.0, 40.0, LOUD)],
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_01", 30.0, 40.0),
    ]

    with pytest.raises(NoUsableReferenceError, match="no speech was detected"):
        build_voice_profiles(segments, stem, settings=_settings(tmp_path))


def test_contiguous_turns_are_joined_into_one_reference(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 18.1, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 14.0),
        SpeakerSegment("SPEAKER_00", 14.1, 18.1),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 10.0
    assert abs(profile.reference_duration - 8.1) < 1e-6


def test_distant_turns_are_not_concatenated(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    """Two short turns either side of another speaker are never stitched together."""

    stem = _write_stem(
        tmp_path / "speech.wav",
        [(10.0, 14.0, LOUD), (14.0, 20.0, LOUD), (20.0, 24.0, LOUD)],
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 14.0),
        SpeakerSegment("SPEAKER_01", 14.0, 20.0),
        SpeakerSegment("SPEAKER_00", 20.0, 24.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_duration == 4.0
    assert profile.reference_start == 10.0


def test_a_long_monologue_is_scanned_with_a_sliding_window(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 30.0, QUIET), (30.0, 50.0, LOUD)])

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 50.0)], stem, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert profile.reference_duration == 10.0


def test_ties_are_broken_deterministically(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 18.0, LOUD), (30.0, 38.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 18.0),
        SpeakerSegment("SPEAKER_00", 30.0, 38.0),
    ]

    first = build_voice_profiles(
        segments, stem, settings=_settings(tmp_path, voice_profile_dir=tmp_path / "first")
    )["SPEAKER_00"]
    second = build_voice_profiles(
        segments, stem, settings=_settings(tmp_path, voice_profile_dir=tmp_path / "second")
    )["SPEAKER_00"]

    assert first.reference_start == 10.0
    # Everything except the profile directory is identical between the two runs.
    assert (
        first.reference_start,
        first.reference_end,
        first.quality_score,
        first.selection_reason,
    ) == (
        second.reference_start,
        second.reference_end,
        second.quality_score,
        second.selection_reason,
    )


def test_every_speaker_gets_a_reference_from_their_own_speech(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_01", 30.0, 40.0),
    ]

    profiles = build_voice_profiles(segments, stem, settings=_settings(tmp_path))

    assert set(profiles) == {"SPEAKER_00", "SPEAKER_01"}
    assert profiles["SPEAKER_00"].reference_start == 10.0
    assert profiles["SPEAKER_01"].reference_start == 30.0
    assert profiles["SPEAKER_00"].reference_audio != profiles["SPEAKER_01"].reference_audio
    assert profiles["SPEAKER_00"].speaker_id == "SPEAKER_00"


def test_a_speaker_without_usable_speech_never_borrows_another_reference(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, SILENCE)])
    segments = [
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
        SpeakerSegment("SPEAKER_01", 10.0, 20.0),
    ]

    with pytest.raises(NoUsableReferenceError, match="SPEAKER_00"):
        build_voice_profiles(segments, stem, settings=_settings(tmp_path))

    assert not (tmp_path / "voices" / "SPEAKER_00").exists()


def test_missing_reference_error_explains_the_requirement(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 12.0, LOUD)])

    try:
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_03", 10.0, 12.0)], stem, settings=_settings(tmp_path)
        )
    except NoUsableReferenceError as error:
        message = str(error)
    else:  # pragma: no cover - the call above must fail
        raise AssertionError("expected NoUsableReferenceError")

    assert "SPEAKER_03" in message
    assert "2.00s" in message
    assert "VOICE_REFERENCE_MIN_DURATION" in message


def test_the_same_input_produces_the_same_profiles(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, QUIET)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]

    first = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]
    second = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# Reference text
# ---------------------------------------------------------------------------


def test_reference_text_uses_only_lines_inside_the_window(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    transcript = [
        TranscriptSegment("SPEAKER_00", 8.0, 9.0, "before"),
        TranscriptSegment("SPEAKER_00", 10.5, 12.0, "hello"),
        TranscriptSegment("SPEAKER_00", 12.0, 13.0, "there"),
        TranscriptSegment("SPEAKER_00", 19.0, 21.0, "outside"),
    ]

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        transcript=transcript,
        settings=_settings(tmp_path),
    )["SPEAKER_00"]

    assert profile.reference_text == "hello there"


def test_reference_text_ignores_other_speakers(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    transcript = [
        TranscriptSegment("SPEAKER_01", 11.0, 12.0, "not mine"),
        TranscriptSegment("SPEAKER_00", 13.0, 14.0, "mine"),
    ]

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        transcript=transcript,
        settings=_settings(tmp_path),
    )["SPEAKER_00"]

    assert profile.reference_text == "mine"


def test_reference_text_is_none_without_a_transcript(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_text is None


def test_reference_text_is_none_when_no_line_is_usable(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    transcript = [TranscriptSegment("SPEAKER_00", 5.0, 6.0, "elsewhere")]

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        transcript=transcript,
        settings=_settings(tmp_path),
    )["SPEAKER_00"]

    assert profile.reference_text is None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_ffmpeg_arguments_request_mono_24k_pcm_wav(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )

    arguments = ffmpeg.last
    assert arguments[0] == "-nostdin"
    assert arguments[arguments.index("-ss") + 1] == "10.000000"
    assert arguments[arguments.index("-t") + 1] == "10.000000"
    assert arguments[arguments.index("-i") + 1] == str(stem)
    assert arguments[arguments.index("-ac") + 1] == "1"
    assert arguments[arguments.index("-ar") + 1] == str(REFERENCE_RATE)
    assert arguments[arguments.index("-c:a") + 1] == "pcm_s16le"
    assert arguments[arguments.index("-f") + 1] == "wav"
    assert "-vn" in arguments
    assert "-y" in arguments
    assert arguments[-1].endswith(REFERENCE_FILENAME)
    assert len(ffmpeg.calls) == 1


def test_hot_references_are_attenuated_but_never_amplified(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, CLIPPED)])

    build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )

    arguments = ffmpeg.last
    assert "-af" in arguments
    assert arguments[arguments.index("-af") + 1].startswith("volume=-")
    assert arguments[arguments.index("-af") + 1].endswith("dB")


def test_a_normal_reference_is_not_processed(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )

    assert "-af" not in ffmpeg.last


def test_reference_files_are_written_per_speaker(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    segments = [SpeakerSegment("SPEAKER_00", 10.0, 20.0)]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_audio == (
        tmp_path / "voices" / "SPEAKER_00" / REFERENCE_FILENAME
    )
    assert profile.resolve_reference_audio().is_file()


def test_missing_ffmpeg_is_reported_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_profiles, "shutil", types.SimpleNamespace(which=lambda name: None))

    with pytest.raises(ReferenceExtractionError, match="ffmpeg was not found on PATH"):
        voice_profiles._run_ffmpeg(["-i", "in.wav", "out.wav"])


def test_ffmpeg_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        returncode = 1
        stderr = "boom: no such file\n"

    monkeypatch.setattr(
        voice_profiles, "shutil", types.SimpleNamespace(which=lambda name: "/usr/bin/ffmpeg")
    )
    monkeypatch.setattr(
        voice_profiles, "subprocess", types.SimpleNamespace(run=lambda *a, **k: _Result())
    )

    with pytest.raises(ReferenceExtractionError, match="exit code 1"):
        voice_profiles._run_ffmpeg(["-i", "in.wav", "out.wav"])


def test_extraction_failure_reaches_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    def _boom(arguments: list[str]) -> None:
        raise ReferenceExtractionError("ffmpeg exploded")

    monkeypatch.setattr(voice_profiles, "_run_ffmpeg", _boom)

    with pytest.raises(ReferenceExtractionError, match="ffmpeg exploded"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
        )


def test_wrong_sample_rate_output_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    monkeypatch.setattr(voice_profiles, "_run_ffmpeg", FakeFfmpeg(rate=STEM_RATE))

    with pytest.raises(ReferencePreprocessingError, match="24000 Hz mono is required"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
        )


def test_multi_channel_output_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    monkeypatch.setattr(voice_profiles, "_run_ffmpeg", FakeFfmpeg(channels=2))

    with pytest.raises(ReferencePreprocessingError, match="channel"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
        )


def test_truncated_extraction_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    monkeypatch.setattr(voice_profiles, "_run_ffmpeg", FakeFfmpeg(duration_scale=0.25))

    with pytest.raises(ReferencePreprocessingError, match="only"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
        )


def test_reference_is_extracted_with_real_ffmpeg(tmp_path: Path) -> None:
    """End-to-end extraction, using the FFmpeg the project already requires."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 25.0, LOUD)])

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 12.0, 22.0)], stem, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    info = sf.info(str(profile.reference_audio))
    assert info.samplerate == REFERENCE_RATE
    assert info.channels == 1
    assert abs(info.duration - 10.0) < 0.02

    samples, rate = sf.read(str(profile.reference_audio), dtype="float32")
    assert rate == REFERENCE_RATE
    assert float(np.max(np.abs(samples))) > 0.05


# ---------------------------------------------------------------------------
# Clone prompts
# ---------------------------------------------------------------------------


def test_clone_prompt_is_created_by_the_injected_encoder(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    encoder = FakeEncoder()

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        settings=_settings(tmp_path),
        clone_encoder=encoder,
    )["SPEAKER_00"]

    expected = tmp_path / "voices" / "SPEAKER_00" / CLONE_PROMPT_FILENAME
    assert profile.clone_prompt_path == expected
    assert expected.is_file()
    assert encoder.calls == [(profile.reference_audio, expected)]


def test_an_existing_clone_prompt_is_reused_without_re_encoding(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    encoder = FakeEncoder()
    segments = [SpeakerSegment("SPEAKER_00", 10.0, 20.0)]

    first = build_voice_profiles(
        segments, stem, settings=_settings(tmp_path), clone_encoder=encoder
    )["SPEAKER_00"]
    second = build_voice_profiles(
        segments, stem, settings=_settings(tmp_path), clone_encoder=encoder
    )["SPEAKER_00"]

    assert first.clone_prompt_path == second.clone_prompt_path
    assert len(encoder.calls) == 1


def test_clone_prompt_path_is_none_without_an_encoder(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.clone_prompt_path is None
    assert profile.resolve_clone_prompt() is None
    assert not (tmp_path / "voices" / "SPEAKER_00" / CLONE_PROMPT_FILENAME).exists()


def test_encoder_failure_is_wrapped_and_identifies_the_speaker(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    encoder = FakeEncoder(error=RuntimeError("cuda out of memory"))

    with pytest.raises(ClonePromptError, match="SPEAKER_00"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
            stem,
            settings=_settings(tmp_path),
            clone_encoder=encoder,
        )


def test_an_encoder_that_writes_nothing_fails(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])

    class _Silent:
        def __call__(self, reference_audio: Path, destination: Path) -> Path:
            return destination

    with pytest.raises(ClonePromptError, match="did not write a prompt"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
            stem,
            settings=_settings(tmp_path),
            clone_encoder=_Silent(),
        )


def test_an_empty_clone_prompt_file_is_not_reused(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    empty = tmp_path / "voices" / "SPEAKER_00" / CLONE_PROMPT_FILENAME
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    encoder = FakeEncoder()

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        settings=_settings(tmp_path),
        clone_encoder=encoder,
    )["SPEAKER_00"]

    assert len(encoder.calls) == 1
    assert profile.clone_prompt_path == empty


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_pipeline_types_are_reused_not_duplicated() -> None:
    assert voice_profiles.SpeakerSegment is diarization.SpeakerSegment
    assert voice_profiles.TranscriptSegment is transcription.TranscriptSegment


def test_no_voice_model_is_required() -> None:
    assert not hasattr(voice_profiles, "OmniVoice")
    assert not [name for name in sys.modules if name.startswith("omnivoice")]


# ---------------------------------------------------------------------------
# Reference score
# ---------------------------------------------------------------------------


def test_the_score_stays_within_zero_and_one() -> None:
    perfect = _candidate(variety=1.0)
    terrible = _candidate(
        speech_ratio=0.0,
        dynamic_range_db=0.0,
        rms_dbfs=-90.0,
        overlap_seconds=10.0,
        clipped_fraction=0.5,
        variety=0.0,
    )

    assert perfect.quality_score(10.0) == 1.0
    # Only the duration term still applies to the terrible window.
    assert terrible.quality_score(10.0) <= voice_profiles.SCORE_DURATION_WEIGHT
    assert terrible.quality_score(10.0) < perfect.quality_score(10.0)
    # Renormalised when the transcript cannot supply a variety signal.
    assert _candidate().quality_score(10.0) == 1.0


def test_dynamics_contribute_to_the_score() -> None:
    flat = _candidate(dynamic_range_db=0.0)
    lively = _candidate(dynamic_range_db=30.0)

    assert lively.quality_score(10.0) > flat.quality_score(10.0)


def test_speech_coverage_contributes_to_the_score() -> None:
    sparse = _candidate(speech_ratio=0.1)
    covered = _candidate(speech_ratio=voice_profiles.SPEECH_COVERAGE_TARGET)

    assert covered.quality_score(10.0) > sparse.quality_score(10.0)


def test_variety_contributes_to_the_score_only_when_it_is_available() -> None:
    repetitive = _candidate(variety=0.0)
    rich = _candidate(variety=1.0)
    unavailable = _candidate(variety=None)

    assert rich.quality_score(10.0) > repetitive.quality_score(10.0)
    assert unavailable.quality_score(10.0) > repetitive.quality_score(10.0)


def test_a_candidate_is_unusable_without_activation_or_duration() -> None:
    assert not _candidate(speech_ratio=voice_profiles.MIN_SPEECH_RATIO / 2).is_usable(3.0)
    assert not _candidate(start=10.0, end=11.0).is_usable(3.0)
    assert _candidate().is_usable(3.0)


def test_the_selection_reason_reports_the_measurements() -> None:
    reason = _candidate(variety=0.75).selection_reason()

    assert "10.0s of continuous speech" in reason
    assert "60% active speech" in reason
    assert "30 dB dynamic range" in reason
    assert "0.0s overlapped" in reason
    assert "-29.0 dBFS" in reason
    assert "0.0% clipped" in reason
    assert "phonetic variety 0.75" in reason


def test_public_api_is_exported() -> None:
    for name in (
        "VoiceProfile",
        "build_voice_profiles",
        "save_voice_profiles",
        "load_voice_profiles",
        "speaker_directory_name",
        "VoiceProfileError",
        "NoUsableReferenceError",
        "ClonePromptEncoder",
    ):
        assert name in voice_profiles.__all__
        assert hasattr(voice_profiles, name)


# ---------------------------------------------------------------------------
# Speaker id filesystem safety
# ---------------------------------------------------------------------------


def test_ordinary_speaker_ids_are_used_as_directory_names() -> None:
    assert voice_profiles.speaker_directory_name("SPEAKER_00") == "SPEAKER_00"
    assert voice_profiles.speaker_directory_name("SPEAKER_17") == "SPEAKER_17"
    assert voice_profiles.speaker_directory_name("speaker-7_b") == "speaker-7_b"


def test_speaker_directory_names_never_contain_a_separator() -> None:
    for hostile in (
        "../evil",
        "../../evil",
        "a/b",
        "a\\b",
        "C:/tmp/evil",
        "/etc/passwd",
        "..",
    ):
        name = voice_profiles.speaker_directory_name(hostile)

        assert "/" not in name and "\\" not in name, hostile
        assert not name.startswith("."), hostile
        assert name not in {".", ".."}, hostile
        assert len(Path(name).parts) == 1, hostile


def test_speaker_directory_names_remain_distinct_after_sanitising() -> None:
    names = {
        voice_profiles.speaker_directory_name(hostile)
        for hostile in ("a/b", "a\\b", "a:b", "a b", "a|b")
    }

    assert len(names) == 5


def test_speaker_directory_names_are_deterministic() -> None:
    first = voice_profiles.speaker_directory_name("../evil")

    assert first == voice_profiles.speaker_directory_name("../evil")


def test_dot_only_ids_still_produce_a_safe_directory_name() -> None:
    for hostile in (".", "..", "..."):
        name = voice_profiles.speaker_directory_name(hostile)

        assert name
        assert not name.startswith(".")
        assert len(Path(name).parts) == 1


def test_reserved_windows_device_names_are_not_used() -> None:
    name = voice_profiles.speaker_directory_name("NUL")

    assert name != "NUL"
    assert name.startswith("NUL")


def test_empty_speaker_ids_are_rejected_by_the_sanitiser() -> None:
    for value in ("", "   ", None, 7):
        with pytest.raises(InvalidInputError, match="speaker_id must be a non-empty string"):
            voice_profiles.speaker_directory_name(value)


def test_a_hostile_speaker_id_cannot_escape_the_profile_directory(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD)])
    hostile = "../../escape"

    profiles = build_voice_profiles(
        [SpeakerSegment(hostile, 10.0, 20.0)], stem, settings=_settings(tmp_path)
    )

    profile = profiles[hostile]
    assert profile.speaker_id == hostile  # the identity itself is untouched
    assert profile.reference_audio.parent.parent == tmp_path / "voices"
    assert profile.resolve_reference_audio().is_file()
    assert {item.name for item in tmp_path.iterdir()} == {"speech.wav", "voices"}


def test_save_voice_profiles_sanitises_the_directory(tmp_path: Path) -> None:
    hostile = "../escape"

    written = save_voice_profiles(
        {hostile: _profile(tmp_path, speaker_id=hostile)}, directory=tmp_path / "out"
    )

    assert written[hostile].parent.parent == tmp_path / "out"
    assert written[hostile].name == PROFILE_FILENAME
    assert load_voice_profiles(directory=tmp_path / "out")[hostile].speaker_id == hostile


def test_the_speaker_directory_guard_rejects_an_escaping_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        voice_profiles, "speaker_directory_name", lambda speaker_id: "../escape"
    )

    with pytest.raises(InvalidInputError, match="unsafe speaker directory name"):
        voice_profiles._speaker_directory(tmp_path, "SPEAKER_00")


# ---------------------------------------------------------------------------
# Speech presence
# ---------------------------------------------------------------------------


def test_speech_like_audio_is_measured_as_active() -> None:
    stats = voice_profiles._measure_window(_render(LOUD, 10.0), STEM_RATE)

    assert stats.speech_ratio > 0.4
    assert stats.dynamic_range_db > voice_profiles.MIN_DYNAMIC_RANGE_DB


def test_steady_and_silent_audio_are_not_measured_as_speech() -> None:
    steady = voice_profiles._measure_window(_render(STEADY, 10.0), STEM_RATE)
    silence = voice_profiles._measure_window(_render(SILENCE, 10.0), STEM_RATE)

    assert steady.speech_ratio < voice_profiles.MIN_SPEECH_RATIO
    assert silence.speech_ratio == 0.0
    assert silence.dynamic_range_db == 0.0


def test_speech_over_a_music_bed_is_still_detected() -> None:
    bed = _render(Signal("music", 0.02), 10.0)
    mixed = _render(LOUD, 10.0) + bed

    stats = voice_profiles._measure_window(mixed, STEM_RATE)

    assert stats.speech_ratio > 0.25


def test_a_loud_music_window_never_outranks_quieter_dialogue(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """The steady bed here is louder than the speech, and must still lose."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, STEADY), (30.0, 40.0, QUIET)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert "active speech" in profile.selection_reason


def test_a_speaker_with_only_steady_audio_gets_no_profile(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, STEADY)])

    with pytest.raises(NoUsableReferenceError, match="no speech was detected"):
        build_voice_profiles(
            [SpeakerSegment("SPEAKER_00", 10.0, 20.0)], stem, settings=_settings(tmp_path)
        )


# ---------------------------------------------------------------------------
# Phonetic variety
# ---------------------------------------------------------------------------

RICH_TEXT = "I never told you the truth about the harbour, the money or the woman"
REPETITIVE_TEXT = "no no no no no no no no"


def test_variety_is_unavailable_without_text() -> None:
    assert voice_profiles._phonetic_variety(None) is None
    assert voice_profiles._phonetic_variety("   ") is None
    assert voice_profiles._phonetic_variety("123 456") is None


def test_variety_rewards_a_richer_vocabulary() -> None:
    repetitive = voice_profiles._phonetic_variety(REPETITIVE_TEXT)
    rich = voice_profiles._phonetic_variety(RICH_TEXT)

    assert repetitive is not None and rich is not None
    assert repetitive < rich
    assert 0.0 <= repetitive <= 1.0
    assert rich <= 1.0


def test_variety_penalises_a_window_that_says_too_little() -> None:
    variety = voice_profiles._phonetic_variety("yes")

    assert variety is not None
    assert variety < 0.25


def test_variety_is_recognised_in_other_scripts() -> None:
    variety = voice_profiles._phonetic_variety("\u1210\u120b\u121d \u1230\u120b\u121d")

    assert variety is not None
    assert 0.0 < variety <= 1.0


def test_a_richer_window_wins_when_both_are_transcribed(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """Both windows are acoustically identical, so only variety can break the tie."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    transcript = [
        TranscriptSegment("SPEAKER_00", 10.5, 14.0, REPETITIVE_TEXT),
        TranscriptSegment("SPEAKER_00", 30.5, 34.0, RICH_TEXT),
    ]

    profile = build_voice_profiles(
        segments, stem, transcript=transcript, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert "phonetic variety" in profile.selection_reason


def test_identical_windows_tie_without_a_transcript(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]

    profile = build_voice_profiles(segments, stem, settings=_settings(tmp_path))["SPEAKER_00"]

    assert profile.reference_start == 10.0
    assert "phonetic variety" not in profile.selection_reason


def test_variety_is_dropped_for_a_window_no_line_covers(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """The documented cost of never inventing a variety value.

    A window the transcript does not cover is scored without the term, so here it
    beats an identical window whose text is repetitive. The term is deliberately
    modest so that it cannot outweigh a real quality difference.
    """

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, LOUD), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    transcript = [TranscriptSegment("SPEAKER_00", 10.5, 14.0, REPETITIVE_TEXT)]

    profile = build_voice_profiles(
        segments, stem, transcript=transcript, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert profile.reference_text is None
    assert "phonetic variety" not in profile.selection_reason


def test_variety_cannot_outweigh_a_clear_audio_difference(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, QUIET), (30.0, 40.0, LOUD)])
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    transcript = [
        TranscriptSegment("SPEAKER_00", 10.5, 14.0, RICH_TEXT),
        TranscriptSegment("SPEAKER_00", 30.5, 34.0, REPETITIVE_TEXT),
    ]

    profile = build_voice_profiles(
        segments, stem, transcript=transcript, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    # Much better recorded speech wins even though its transcript is repetitive.
    assert profile.reference_start == 30.0
