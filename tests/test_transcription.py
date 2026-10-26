"""Tests for :mod:`app.pipeline.transcription`.

``faster_whisper`` is never imported or executed here: the model class and the
audio decoder are replaced by in-memory fakes. The tests therefore need no
Whisper weights, no CUDA device, and no real audio file - the input file is an
empty placeholder on disk because the decoder is fake too.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

from app.config import (
    DEFAULT_TRANSCRIPTION_COMPUTE_TYPE,
    DEFAULT_TRANSCRIPTION_MODEL,
    Settings,
)
from app.pipeline import transcription
from app.pipeline.diarization import SpeakerSegment
from app.pipeline.transcription import (
    AudioDecodeError,
    InvalidInputError,
    InvalidSegmentError,
    InvalidSpeakerSegmentsError,
    MissingInputError,
    ModelInitializationError,
    TranscriptSegment,
    TranscriptionError,
    TranscriptionInferenceError,
    UnsupportedOutputError,
    transcribe,
)


class FakeSegment:
    """Stand-in for ``faster_whisper.transcribe.Segment``.

    Only the attributes this module is allowed to read are provided, so the tests
    fail loudly if the implementation starts depending on more of Whisper's
    segment object than it should.
    """

    def __init__(
        self,
        start: object,
        end: object,
        text: object,
        *,
        words: object = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class FakeWord:
    """Stand-in for ``faster_whisper.transcribe.Word``."""

    def __init__(self, start: object, end: object, word: object) -> None:
        self.start = start
        self.end = end
        self.word = word


class FakeWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel`` recording every interaction."""

    # Recorded state, reset by :func:`_patch`.
    loads: list[dict[str, object]] = []
    instances: list["FakeWhisperModel"] = []
    calls: list[dict[str, object]] = []
    detect_calls: list[int] = []

    # Scripted behaviour, set by individual tests.
    load_error: Exception | None = None
    detect_error: Exception | None = None
    transcribe_error: Exception | None = None
    lazy_error: Exception | None = None
    detected: object = ("en", 0.98, [("en", 0.98)])
    segments: list[FakeSegment] = []
    responses: list[list[FakeSegment]] = []

    def __init__(
        self,
        model: str,
        device: str | None = None,
        device_index: int | None = None,
        compute_type: str | None = None,
        download_root: str | None = None,
    ) -> None:
        self.model = model
        FakeWhisperModel.loads.append(
            {
                "model": model,
                "device": device,
                "device_index": device_index,
                "compute_type": compute_type,
                "download_root": download_root,
            }
        )
        if FakeWhisperModel.load_error is not None:
            raise FakeWhisperModel.load_error
        FakeWhisperModel.instances.append(self)

    @staticmethod
    def _lazy_segments():
        """Yield segments lazily, exactly like faster-whisper does."""

        if FakeWhisperModel.lazy_error is not None:
            raise FakeWhisperModel.lazy_error
        for segment in FakeWhisperModel.segments:
            yield segment

    def detect_language(self, audio=None, **kwargs):
        FakeWhisperModel.detect_calls.append(0 if audio is None else int(len(audio)))
        if FakeWhisperModel.detect_error is not None:
            raise FakeWhisperModel.detect_error
        return FakeWhisperModel.detected

    def transcribe(self, audio, **kwargs):
        index = len(FakeWhisperModel.calls)
        FakeWhisperModel.calls.append({"samples": int(len(audio)), "kwargs": dict(kwargs)})
        if FakeWhisperModel.transcribe_error is not None:
            raise FakeWhisperModel.transcribe_error
        if index < len(FakeWhisperModel.responses):
            return iter(list(FakeWhisperModel.responses[index])), object()
        return FakeWhisperModel._lazy_segments(), object()


_FAKE_DECODE = types.SimpleNamespace(duration=60.0, error=None, calls=[])


def _fake_decode_audio(path, sampling_rate=transcription.SAMPLE_RATE):
    """Stand-in for ``faster_whisper.decode_audio``."""

    _FAKE_DECODE.calls.append((str(path), sampling_rate))
    if _FAKE_DECODE.error is not None:
        raise _FAKE_DECODE.error
    return np.zeros(int(_FAKE_DECODE.duration * sampling_rate), dtype=np.float32)


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fakes and reset all recorded state."""

    FakeWhisperModel.loads = []
    FakeWhisperModel.instances = []
    FakeWhisperModel.calls = []
    FakeWhisperModel.detect_calls = []
    FakeWhisperModel.load_error = None
    FakeWhisperModel.detect_error = None
    FakeWhisperModel.transcribe_error = None
    FakeWhisperModel.lazy_error = None
    FakeWhisperModel.detected = ("en", 0.98, [("en", 0.98)])
    FakeWhisperModel.segments = []
    FakeWhisperModel.responses = []

    _FAKE_DECODE.duration = 60.0
    _FAKE_DECODE.error = None
    _FAKE_DECODE.calls = []

    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcription, "decode_audio", _fake_decode_audio)


def _settings(
    tmp_path: Path,
    *,
    device: str = "cuda",
    language: str | None = None,
    model: str = DEFAULT_TRANSCRIPTION_MODEL,
    compute_type: str = DEFAULT_TRANSCRIPTION_COMPUTE_TYPE,
) -> Settings:
    return Settings(
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        model_cache_dir=tmp_path / "models",
        device=device,
        transcription_model=model,
        transcription_compute_type=compute_type,
        transcription_language=language,
    )


def _audio(tmp_path: Path, name: str = "speech.wav") -> Path:
    """Create the placeholder input file (the decoder is mocked, never reads it)."""

    path = tmp_path / name
    path.write_bytes(b"")
    return path


def _region(speaker_id: str, start: float, end: float) -> SpeakerSegment:
    return SpeakerSegment(speaker_id=speaker_id, start=start, end=end)


def _rows(segments) -> list[tuple[str, float, float, str]]:
    """Return rounded ``(speaker, start, end, text)`` rows for comparison."""

    return [
        (segment.speaker_id, round(segment.start, 6), round(segment.end, 6), segment.text)
        for segment in segments
    ]


# ---------------------------------------------------------------------------
# TranscriptSegment validation
# ---------------------------------------------------------------------------


def test_transcript_segment_keeps_its_values():
    segment = TranscriptSegment(
        speaker_id="SPEAKER_00", start=12.43, end=16.82, text="I'm here"
    )

    assert segment.speaker_id == "SPEAKER_00"
    assert segment.start == 12.43
    assert segment.end == 16.82
    assert segment.text == "I'm here"
    assert segment.duration == 16.82 - 12.43


def test_transcript_segment_normalises_integers_to_float():
    segment = TranscriptSegment(speaker_id="SPEAKER_00", start=1, end=3, text="hi")

    assert isinstance(segment.start, float)
    assert isinstance(segment.end, float)
    assert segment.duration == 2.0


def test_transcript_segment_strips_its_text():
    segment = TranscriptSegment(
        speaker_id="SPEAKER_00", start=0.0, end=1.0, text="  hello  "
    )

    assert segment.text == "hello"


def test_transcript_segment_is_immutable():
    segment = TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="hi")

    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.text = "changed"


def test_empty_speaker_id_is_rejected():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        TranscriptSegment(speaker_id="", start=0.0, end=1.0, text="hi")


def test_whitespace_only_speaker_id_is_rejected():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        TranscriptSegment(speaker_id="   ", start=0.0, end=1.0, text="hi")


def test_non_string_speaker_id_is_rejected():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        TranscriptSegment(speaker_id=None, start=0.0, end=1.0, text="hi")


def test_zero_length_segment_is_rejected():
    with pytest.raises(InvalidSegmentError, match="greater than start"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=4.0, end=4.0, text="hi")


def test_end_before_start_is_rejected():
    with pytest.raises(InvalidSegmentError, match="greater than start"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=4.0, end=2.0, text="hi")


def test_negative_start_is_rejected():
    with pytest.raises(InvalidSegmentError, match="start must be"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=-0.5, end=1.0, text="hi")


def test_non_finite_timestamp_is_rejected():
    with pytest.raises(InvalidSegmentError, match="finite"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=float("inf"), text="hi")


def test_non_numeric_timestamp_is_rejected():
    with pytest.raises(InvalidSegmentError, match="number of seconds"):
        TranscriptSegment(speaker_id="SPEAKER_00", start="0", end=1.0, text="hi")


def test_empty_text_is_rejected():
    with pytest.raises(InvalidSegmentError, match="text"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="")


def test_whitespace_only_text_is_rejected():
    with pytest.raises(InvalidSegmentError, match="text"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="   \n\t ")


def test_non_string_text_is_rejected():
    with pytest.raises(InvalidSegmentError, match="text"):
        TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text=None)


def test_segment_and_input_errors_are_also_value_errors():
    assert issubclass(InvalidSegmentError, ValueError)
    assert issubclass(InvalidSegmentError, TranscriptionError)
    assert issubclass(InvalidSpeakerSegmentsError, ValueError)
    assert issubclass(InvalidSpeakerSegmentsError, TranscriptionError)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_missing_audio_file_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingInputError, match="not found"):
        transcribe(
            tmp_path / "does-not-exist.wav",
            [_region("SPEAKER_00", 0.0, 1.0)],
            settings=_settings(tmp_path),
        )

    assert FakeWhisperModel.loads == []
    assert _FAKE_DECODE.calls == []


def test_directory_input_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(InvalidInputError, match="not a file"):
        transcribe(
            directory, [_region("SPEAKER_00", 0.0, 1.0)], settings=_settings(tmp_path)
        )

    assert FakeWhisperModel.loads == []


def test_non_segment_input_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerSegmentsError, match="expected a SpeakerSegment"):
        transcribe(
            _audio(tmp_path),
            [("SPEAKER_00", 0.0, 1.0)],
            settings=_settings(tmp_path),
        )

    assert FakeWhisperModel.loads == []


def test_non_iterable_speaker_segments_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerSegmentsError, match="iterable"):
        transcribe(_audio(tmp_path), 42, settings=_settings(tmp_path))

    assert FakeWhisperModel.loads == []


def test_no_diarized_segments_returns_empty_without_loading_the_model(monkeypatch, tmp_path):
    _patch(monkeypatch)

    assert transcribe(_audio(tmp_path), [], settings=_settings(tmp_path)) == []
    assert FakeWhisperModel.loads == []
    assert _FAKE_DECODE.calls == []


# ---------------------------------------------------------------------------
# model and device configuration
# ---------------------------------------------------------------------------


def test_model_is_loaded_with_the_configured_settings(monkeypatch, tmp_path):
    _patch(monkeypatch)
    settings = _settings(tmp_path, model="medium", compute_type="int8_float16")

    transcribe(_audio(tmp_path), [_region("SPEAKER_00", 0.0, 2.0)], settings=settings)

    assert FakeWhisperModel.loads == [
        {
            "model": "medium",
            "device": "cuda",
            "device_index": 0,
            "compute_type": "int8_float16",
            "download_root": str(settings.model_cache_dir),
        }
    ]


def test_gpu_device_is_used_and_there_is_no_cpu_fallback(monkeypatch, tmp_path):
    _patch(monkeypatch)

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 2.0)],
        settings=_settings(tmp_path, device="cuda"),
    )

    # Exactly one load, on the configured device: nothing retried on cpu.
    assert [load["device"] for load in FakeWhisperModel.loads] == ["cuda"]


def test_device_index_is_forwarded_for_a_specific_gpu(monkeypatch, tmp_path):
    _patch(monkeypatch)

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 2.0)],
        settings=_settings(tmp_path, device="cuda:1"),
    )

    assert FakeWhisperModel.loads[0]["device"] == "cuda"
    assert FakeWhisperModel.loads[0]["device_index"] == 1


def test_malformed_device_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(ModelInitializationError, match="DEVICE"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path, device="cuda:gpu"),
        )


def test_audio_is_decoded_once_at_the_whisper_sample_rate(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.responses = [[], [], []]

    transcribe(
        _audio(tmp_path),
        [_region(f"SPEAKER_0{index}", index * 5.0, index * 5.0 + 2.0) for index in range(3)],
        settings=_settings(tmp_path),
    )

    assert len(_FAKE_DECODE.calls) == 1
    assert _FAKE_DECODE.calls[0][1] == transcription.SAMPLE_RATE


# ---------------------------------------------------------------------------
# region-by-region transcription and absolute timestamps
# ---------------------------------------------------------------------------


def test_each_region_is_sent_to_whisper_on_its_own(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.responses = [[], []]

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 1.0, 3.0), _region("SPEAKER_01", 4.0, 6.0)],
        settings=_settings(tmp_path),
    )

    # Two calls on 2 s of audio each, not one call on the whole file.
    assert [call["samples"] for call in FakeWhisperModel.calls] == [
        2 * transcription.SAMPLE_RATE,
        2 * transcription.SAMPLE_RATE,
    ]


def test_whisper_local_timestamps_are_moved_onto_the_movie_timeline(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # Whisper sees only the 12.0s-17.0s region, so it reports local times.
    FakeWhisperModel.segments = [FakeSegment(0.43, 4.82, "I'm here")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 12.0, 17.0)],
        settings=_settings(tmp_path),
    )

    assert _rows(segments) == [("SPEAKER_00", 12.43, 16.82, "I'm here")]


def test_timestamps_are_clamped_to_the_diarized_region(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # Whisper overshoots the region on both sides.
    FakeWhisperModel.segments = [FakeSegment(-0.5, 9.0, "noisy")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 12.0)],
        settings=_settings(tmp_path),
    )

    assert _rows(segments) == [("SPEAKER_00", 10.0, 12.0, "noisy")]


def test_region_samples_cover_the_whole_diarized_turn(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = []

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 12.0, 17.0)],
        settings=_settings(tmp_path),
    )

    assert FakeWhisperModel.calls[0]["samples"] == 5 * transcription.SAMPLE_RATE


def test_degenerate_whisper_segment_is_dropped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # Local timestamps before the region collapse to a zero-length turn.
    FakeWhisperModel.segments = [FakeSegment(-9.0, -8.0, "nothing")]

    assert (
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )
        == []
    )


def test_region_outside_the_decoded_audio_is_skipped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    _FAKE_DECODE.duration = 10.0

    assert (
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 30.0, 32.0)],
            settings=_settings(tmp_path),
        )
        == []
    )
    assert FakeWhisperModel.calls == []


def test_a_diarization_artefact_too_short_to_speak_is_skipped(monkeypatch, tmp_path):
    """A 0.02s turn is not speech, and faster-whisper cannot word-align it.

    This is the exact shape that aborted a real run: diarization emitted a
    0.02-second turn, Whisper emitted a token for the sliver but none survived
    filtering, and its alignment step then indexed an empty timestamp array.
    """

    _patch(monkeypatch)
    _FAKE_DECODE.duration = 10.0
    FakeWhisperModel.segments = [FakeSegment(0.0, 0.02, "Thank you")]

    assert (
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 8.84, 8.86)],
            settings=_settings(tmp_path),
        )
        == []
    )
    # Skipped before Whisper is asked at all, so it cannot fail on the sliver.
    assert FakeWhisperModel.calls == []


def test_a_short_but_speakable_turn_is_still_transcribed(monkeypatch, tmp_path):
    """The floor must not swallow a real short utterance such as "No!"."""

    _patch(monkeypatch)
    _FAKE_DECODE.duration = 10.0
    FakeWhisperModel.segments = [FakeSegment(0.0, 0.2, "No!")]

    rows = _rows(
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 4.0, 4.2)],
            settings=_settings(tmp_path),
        )
    )

    assert rows == [("SPEAKER_00", 4.0, 4.2, "No!")]
    assert len(FakeWhisperModel.calls) == 1


def test_a_real_turn_survives_beside_a_diarization_artefact(monkeypatch, tmp_path):
    """One spurious turn must not cost the lines around it."""

    _patch(monkeypatch)
    _FAKE_DECODE.duration = 20.0
    FakeWhisperModel.segments = [FakeSegment(0.0, 1.0, "I love you.")]

    rows = _rows(
        transcribe(
            _audio(tmp_path),
            [
                _region("SPEAKER_00", 8.84, 8.86),  # artefact
                _region("SPEAKER_00", 9.0, 10.0),  # real line
            ],
            settings=_settings(tmp_path),
        )
    )

    assert rows == [("SPEAKER_00", 9.0, 10.0, "I love you.")]
    assert len(FakeWhisperModel.calls) == 1


def test_word_timestamps_are_requested_and_the_task_is_transcribe(monkeypatch, tmp_path):
    _patch(monkeypatch)

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 2.0)],
        settings=_settings(tmp_path),
    )

    kwargs = FakeWhisperModel.calls[0]["kwargs"]
    assert kwargs["word_timestamps"] is True
    assert kwargs["task"] == "transcribe"  # this stage never translates


# ---------------------------------------------------------------------------
# speakers, ordering, and text handling
# ---------------------------------------------------------------------------


def test_multiple_speakers_keep_their_own_lines(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.10, 1.00, "First line")],
        [FakeSegment(0.05, 0.90, "Second line")],
        [FakeSegment(0.20, 1.50, "Third line")],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [
            _region("SPEAKER_00", 10.0, 12.0),
            _region("SPEAKER_01", 20.0, 22.0),
            _region("SPEAKER_02", 30.0, 32.0),
        ],
        settings=_settings(tmp_path),
    )

    assert _rows(segments) == [
        ("SPEAKER_00", 10.10, 11.00, "First line"),
        ("SPEAKER_01", 20.05, 20.90, "Second line"),
        ("SPEAKER_02", 30.20, 31.50, "Third line"),
    ]


def test_speaker_id_is_preserved_exactly(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, 1.0, "line")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("speaker-A.1", 0.0, 2.0)],
        settings=_settings(tmp_path),
    )

    # No renaming, case folding, or renumbering.
    assert [segment.speaker_id for segment in segments] == ["speaker-A.1"]


def test_output_is_chronological_even_when_the_input_is_not(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 1.0, "at five")],
        [FakeSegment(0.0, 1.0, "at twenty")],
        [FakeSegment(0.0, 1.0, "at fifty")],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [
            _region("SPEAKER_02", 50.0, 52.0),
            _region("SPEAKER_00", 5.0, 7.0),
            _region("SPEAKER_01", 20.0, 22.0),
        ],
        settings=_settings(tmp_path),
    )

    assert [segment.text for segment in segments] == ["at five", "at twenty", "at fifty"]
    assert [segment.start for segment in segments] == sorted(
        segment.start for segment in segments
    )
    assert [segment.speaker_id for segment in segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    ]


def test_whisper_segments_inside_one_region_are_sorted(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [
        FakeSegment(4.0, 5.0, "second"),
        FakeSegment(1.0, 2.0, "first"),
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 10.0)],
        settings=_settings(tmp_path),
    )

    assert [segment.text for segment in segments] == ["first", "second"]


def test_empty_whisper_text_is_dropped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [
        FakeSegment(0.0, 1.0, "   "),
        FakeSegment(1.0, 2.0, "real words"),
        FakeSegment(2.0, 3.0, ""),
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 10.0)],
        settings=_settings(tmp_path),
    )

    assert [segment.text for segment in segments] == ["real words"]


def test_whisper_text_is_stripped_in_the_output(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, 1.0, "  padded line  ")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 2.0)],
        settings=_settings(tmp_path),
    )

    assert [segment.text for segment in segments] == ["padded line"]


# ---------------------------------------------------------------------------
# simultaneous speech: recovering the speaker the exclusive view dropped
# ---------------------------------------------------------------------------


def _crosstalk(speakers, start, end):
    from app.pipeline.diarization import CrosstalkRegion

    return CrosstalkRegion(speaker_ids=speakers, start=start, end=end)


def test_a_speaker_the_exclusive_view_dropped_is_recovered(monkeypatch, tmp_path):
    """Two people talking at once must both reach the transcript.

    The exclusive diarization gives every instant to one speaker, so the second
    person's turn - and everything they said - disappears. The overlap-aware turns
    say who else was there and when, which is enough to re-listen to that stretch and
    give each word to the speaker who was saying it.
    """

    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        # The exclusive turn: one clean decode, as before.
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        # The re-listened crosstalk: words with their own timings.
        [
            FakeSegment(
                0.0,
                0.7,
                "thank you",
                words=[
                    FakeWord(0.05, 0.3, " thank"),
                    FakeWord(0.35, 0.65, " you"),
                ],
            )
        ],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
    )

    assert _rows(segments) == [
        ("SPEAKER_00", 10.0, 14.0, "I do not say that lightly."),
        ("SPEAKER_01", 12.25, 12.85, "thank you"),
    ]


def test_recovery_is_skipped_without_the_overlap_aware_turns(monkeypatch, tmp_path):
    """No overlap-aware annotation means no evidence, so nothing is invented."""

    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, 4.0, "I do not say that lightly.")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
    )

    assert _rows(segments) == [("SPEAKER_00", 10.0, 14.0, "I do not say that lightly.")]
    # Exactly one decode: the crosstalk was not re-listened to.
    assert len(FakeWhisperModel.calls) == 1


def test_a_recovered_line_that_restates_the_other_speaker_is_dropped(
    monkeypatch, tmp_path
):
    """One mixture transcribed twice reports the dominant voice twice.

    Two characters saying one line is a worse defect than one of them saying
    nothing, so text that restates what the region already has is discarded.
    """

    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [
            FakeSegment(
                0.0,
                2.0,
                "I do not say that lightly",
                words=[
                    FakeWord(0.05, 0.5, " I"),
                    FakeWord(0.6, 1.0, " do"),
                    FakeWord(1.1, 1.5, " not"),
                    FakeWord(1.6, 1.9, " say"),
                ],
            )
        ],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 10.0, 13.0)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 12.0),
            _region("SPEAKER_01", 12.0, 13.0),
        ],
    )

    speakers = {segment.speaker_id for segment in segments}
    assert speakers == {"SPEAKER_00"}


def test_the_kept_speakers_text_is_never_replaced_by_a_mixture(monkeypatch, tmp_path):
    """The one speaker the exclusive view kept already had a clean decode."""

    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [
            FakeSegment(
                0.0,
                0.6,
                "we are done",
                words=[FakeWord(0.05, 0.55, " we are done")],
            )
        ],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
    )

    kept = [segment for segment in segments if segment.speaker_id == "SPEAKER_00"]
    assert [segment.text for segment in kept] == ["I do not say that lightly."]


def test_a_crosstalk_too_short_to_hold_a_word_is_not_re_listened_to(
    monkeypatch, tmp_path
):
    """0.2s of a mixture is a syllable; any text from it would be invented."""

    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, 4.0, "I do not say that lightly.")]

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.4)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.4),
        ],
    )

    assert len(FakeWhisperModel.calls) == 1


# ---------------------------------------------------------------------------
# language handling
# ---------------------------------------------------------------------------


def test_language_is_detected_once_and_reused_for_every_region(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.detected = ("am", 0.91, [("am", 0.91)])
    FakeWhisperModel.responses = [[], []]

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 1.0, 3.0), _region("SPEAKER_01", 4.0, 6.0)],
        settings=_settings(tmp_path),
    )

    assert len(FakeWhisperModel.detect_calls) == 1
    assert [call["kwargs"]["language"] for call in FakeWhisperModel.calls] == ["am", "am"]


def test_configured_language_skips_detection(monkeypatch, tmp_path):
    _patch(monkeypatch)

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 1.0, 3.0)],
        settings=_settings(tmp_path, language="fr"),
    )

    assert FakeWhisperModel.detect_calls == []
    assert FakeWhisperModel.calls[0]["kwargs"]["language"] == "fr"


def test_undetectable_language_lets_whisper_decide_per_region(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.detected = ("", 0.0, [])

    transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 1.0, 3.0)],
        settings=_settings(tmp_path),
    )

    assert FakeWhisperModel.calls[0]["kwargs"]["language"] is None


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_missing_faster_whisper_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.setattr(transcription, "WhisperModel", None)
    monkeypatch.setattr(transcription, "decode_audio", None)

    with pytest.raises(ModelInitializationError, match="not installed"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )


def test_audio_decode_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    _FAKE_DECODE.error = RuntimeError("no decoder for this container")

    with pytest.raises(AudioDecodeError, match="could not decode") as excinfo:
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_model_load_failure_is_wrapped_with_its_cause(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.load_error = RuntimeError("CUDA is not available")

    with pytest.raises(ModelInitializationError, match="could not load") as excinfo:
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_transcription_failure_is_wrapped_with_its_cause(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.transcribe_error = RuntimeError("boom")

    with pytest.raises(
        TranscriptionInferenceError, match="failed for speaker SPEAKER_00"
    ) as excinfo:
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_failure_raised_while_consuming_the_lazy_iterator_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.lazy_error = RuntimeError("CUDA out of memory")

    with pytest.raises(TranscriptionInferenceError, match="out of memory"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )


def test_language_detection_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.detect_error = RuntimeError("detector exploded")

    with pytest.raises(TranscriptionInferenceError, match="detect the source language"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )


def test_non_numeric_whisper_timestamp_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(None, 1.0, "text")]

    with pytest.raises(UnsupportedOutputError, match="non-numeric start"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )


def test_non_finite_whisper_timestamp_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, float("nan"), "text")]

    with pytest.raises(UnsupportedOutputError, match="non-finite end"):
        transcribe(
            _audio(tmp_path),
            [_region("SPEAKER_00", 0.0, 2.0)],
            settings=_settings(tmp_path),
        )


# ---------------------------------------------------------------------------
# sentences: one thought, however many pauses the transcriber split it at
# ---------------------------------------------------------------------------


def _segment(start: float, end: float, text: str, speaker: str = "SPEAKER_00") -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker, start=start, end=end, text=text)


def test_a_sentence_split_at_a_pause_is_one_sentence() -> None:
    """The transcriber segments on pauses; the sentence, not the segment, is the unit."""

    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 2.0, "It is the world"),
            _segment(2.5, 5.0, "that has been pulled over your eyes."),
        ]
    )

    assert len(sentences) == 1
    sentence = sentences[0]
    assert sentence.text == "It is the world that has been pulled over your eyes."
    assert (sentence.start, sentence.end) == (0.0, 5.0)
    assert len(sentence.parts) == 2


def test_a_lowercase_continuation_joins_even_after_a_full_stop() -> None:
    """A transcriber writes a full stop it does not believe, and lowercases the rest."""

    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 2.0, "You can see it when you look out your window."),
            _segment(2.4, 4.0, "or when you turned on your television."),
        ]
    )

    assert len(sentences) == 1
    assert (
        sentences[0].text
        == "You can see it when you look out your window, "
        "or when you turned on your television."
    )


def test_two_sentences_a_beat_apart_become_one_line() -> None:
    """A pause this short is a beat inside one delivery, not a full stop.

    Whisper punctuates every pause-delimited fragment as a sentence, and speaking
    each one separately is what makes a dub sound cut together: two clips, two
    intonation contours, a seam in the middle of one thought. Below
    ``SENTENCE_BREAK_PAUSE_SECONDS`` the fragments stay one line.
    """

    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 1.0, "The matrix is everywhere."),
            _segment(1.4, 2.4, "It is all around us."),
        ]
    )

    assert [sentence.text for sentence in sentences] == [
        "The matrix is everywhere. It is all around us.",
    ]


def test_two_sentences_a_real_pause_apart_stay_two() -> None:
    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 1.0, "The matrix is everywhere."),
            _segment(2.6, 3.6, "It is all around us."),
        ]
    )

    assert [sentence.text for sentence in sentences] == [
        "The matrix is everywhere.",
        "It is all around us.",
    ]


def test_a_different_speaker_always_starts_a_sentence() -> None:
    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 1.0, "Are you ready"),
            _segment(1.1, 2.0, "yes I am.", speaker="SPEAKER_01"),
        ]
    )

    assert [sentence.speaker_id for sentence in sentences] == ["SPEAKER_00", "SPEAKER_01"]


def test_a_long_pause_ends_a_sentence_whatever_the_words_say() -> None:
    """Two seconds of silence is a beat in the scene, not a comma."""

    sentences = transcription.assemble_sentences(
        [
            _segment(0.0, 1.0, "I never said that"),
            _segment(4.0, 5.0, "and I never would."),
        ]
    )

    assert len(sentences) == 2


def test_a_sentence_never_grows_past_its_caps() -> None:
    """An unpunctuated monologue must not become one unbounded unit."""

    transcript = [
        _segment(float(i), float(i) + 2.0, "and then the thing happened")
        for i in range(0, 40, 2)
    ]

    sentences = transcription.assemble_sentences(transcript)

    assert len(sentences) > 1
    assert all(
        sentence.duration <= transcription.MAX_SENTENCE_SECONDS
        for sentence in sentences
    )
    assert all(
        len(sentence.parts) <= transcription.MAX_SENTENCE_PARTS
        for sentence in sentences
    )


def test_no_segments_means_no_sentences() -> None:
    assert transcription.assemble_sentences([]) == []


def test_a_sentence_rejects_parts_from_another_speaker() -> None:
    with pytest.raises(InvalidSegmentError, match="belongs to"):
        transcription.DialogueSentence(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=2.0,
            text="hello",
            parts=(
                _segment(0.0, 1.0, "hello"),
                _segment(1.0, 2.0, "there", speaker="SPEAKER_01"),
            ),
        )


def test_a_crosstalk_the_exclusive_view_left_empty_is_not_guessed_at(
    monkeypatch, tmp_path
):
    """With no authoritative text, a recovery has nothing to check itself against."""

    _patch(monkeypatch)
    FakeWhisperModel.segments = [FakeSegment(0.0, 1.0, "elsewhere")]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 0.0, 1.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 20.0, 21.0)],
        overlap_turns=[
            _region("SPEAKER_00", 20.0, 20.5),
            _region("SPEAKER_01", 20.5, 21.0),
        ],
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]
    assert len(FakeWhisperModel.calls) == 1


def test_the_same_interjection_is_recovered_in_two_different_crosstalks(
    monkeypatch, tmp_path
):
    """One region's recovered line must not veto an identical one in another region."""

    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 1.0, "first line")],
        [FakeSegment(0.0, 1.0, "second line")],
        [FakeSegment(0.0, 0.6, "no", words=[FakeWord(0.05, 0.45, " no")])],
        [FakeSegment(0.0, 0.6, "no", words=[FakeWord(0.05, 0.45, " no")])],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [
            _region("SPEAKER_00", 10.0, 11.0),
            _region("SPEAKER_00", 11.0, 12.0),
        ],
        settings=_settings(tmp_path),
        crosstalk=[
            _crosstalk(("SPEAKER_00", "SPEAKER_01"), 10.0, 10.6),
            _crosstalk(("SPEAKER_00", "SPEAKER_01"), 11.0, 11.6),
        ],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 12.0),
            _region("SPEAKER_01", 10.0, 10.6),
            _region("SPEAKER_01", 11.0, 11.6),
        ],
    )

    recovered = [segment for segment in segments if segment.speaker_id == "SPEAKER_01"]
    assert len(recovered) == 2


def test_a_recovered_line_whisper_was_not_confident_about_is_not_invented(
    monkeypatch, tmp_path
):
    """A mixture is what Whisper fills in from its language model.

    On a real clip it produced "RUN!" from a quarter-second of two people talking at
    once. A recovered line has to look like speech by the decoder's own diagnostics
    before it is allowed into somebody's mouth.
    """

    import types

    _patch(monkeypatch)
    doubtful = FakeSegment(
        0.0, 0.6, "RUN!", words=[FakeWord(0.05, 0.55, " RUN!")]
    )
    doubtful.avg_logprob = -1.9
    doubtful.no_speech_prob = 0.82
    doubtful.compression_ratio = 0.9
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [doubtful],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]


def test_a_confident_recovered_line_is_still_kept(monkeypatch, tmp_path):
    """The filter must not throw away real simultaneous speech."""

    _patch(monkeypatch)
    confident = FakeSegment(
        0.0, 0.7, "thank you", words=[FakeWord(0.05, 0.65, " thank you")]
    )
    confident.avg_logprob = -0.3
    confident.no_speech_prob = 0.05
    confident.compression_ratio = 1.1
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [confident],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]


# ---------------------------------------------------------------------------
# Who is actually audible in the recycled crosstalk audio
# ---------------------------------------------------------------------------


def _tone(seconds: float, hz: float, rate: int = transcription.SAMPLE_RATE):
    """Return a voiced, harmonic signal at ``hz`` - something the tracker can measure."""

    time = np.arange(int(round(seconds * rate))) / rate
    signal = np.sin(2 * np.pi * hz * time)
    signal += 0.5 * np.sin(2 * np.pi * 2 * hz * time)
    return (0.3 * signal).astype(np.float32)


def test_a_mixture_that_sounds_like_the_other_speaker_is_not_recovered(
    monkeypatch, tmp_path
):
    """The defect this gate exists for.

    A crosstalk is decoded, and Whisper returns whichever voice dominates it. Handing
    that text to the *other* character is heard as one character saying the other's
    line in the wrong voice. On the real clip three such lines were produced - "Oh,
    yeah.", "The police." and "RUN!" - and every one measured as the character who was
    already talking.
    """

    _patch(monkeypatch)
    # The decoded audio measures 220 Hz: the kept speaker's register, not the 110 Hz
    # speaker the line would be credited to.
    monkeypatch.setattr(
        transcription,
        "decode_audio",
        lambda path, sampling_rate=transcription.SAMPLE_RATE: _tone(120.0, 220.0),
    )
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [FakeSegment(0.0, 0.4, "Oh, yeah.", words=[FakeWord(0.05, 0.35, " Oh, yeah.")])],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
        # SPEAKER_00 is the 200 Hz voice; the audio here is 110 Hz, i.e. the other one.
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 110.0},
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]


def test_a_mixture_that_sounds_like_the_missing_speaker_is_recovered(
    monkeypatch, tmp_path
):
    """The gate must not disable recovery - only the parts it cannot attribute."""

    _patch(monkeypatch)
    monkeypatch.setattr(
        transcription,
        "decode_audio",
        lambda path, sampling_rate=transcription.SAMPLE_RATE: _tone(120.0, 110.0),
    )
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [
            FakeSegment(
                0.0, 0.6, "thank you", words=[FakeWord(0.05, 0.55, " thank you")]
            )
        ],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 110.0},
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_an_unmeasurable_recovered_stretch_is_refused(monkeypatch, tmp_path):
    """No pitched speech means no evidence, and a guess is not evidence."""

    _patch(monkeypatch)
    FakeWhisperModel.responses = [
        [FakeSegment(0.0, 4.0, "I do not say that lightly.")],
        [FakeSegment(0.0, 0.6, "noise", words=[FakeWord(0.05, 0.55, " noise")])],
    ]

    segments = transcribe(
        _audio(tmp_path),
        [_region("SPEAKER_00", 10.0, 14.0)],
        settings=_settings(tmp_path),
        crosstalk=[_crosstalk(("SPEAKER_00", "SPEAKER_01"), 12.2, 12.9)],
        overlap_turns=[
            _region("SPEAKER_00", 10.0, 14.0),
            _region("SPEAKER_01", 12.2, 12.9),
        ],
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 110.0},
    )

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]
