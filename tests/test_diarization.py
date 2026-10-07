"""Tests for :mod:`app.pipeline.diarization`.

``pyannote.audio`` is never imported or executed here: the pyannote pipeline and
the ``torch.device`` factory it is moved with are both replaced by in-memory
fakes. The tests therefore need no Hugging Face token, no CUDA device, no model
weights, and no audio backend. The input "audio" is an empty placeholder file
because pyannote itself is mocked and never opens it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.config import DEFAULT_DIARIZATION_MODEL, Settings
from app.pipeline import diarization
from app.pipeline.diarization import (
    DiarizationError,
    DiarizationInferenceError,
    InvalidInputError,
    InvalidSegmentError,
    InvalidSpeakerCountError,
    MissingHuggingFaceTokenError,
    MissingInputError,
    ModelInitializationError,
    SpeakerSegment,
    UnsupportedOutputError,
    diarize,
)


class FakeTurn:
    """Stand-in for a ``pyannote.core.Segment``."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    """Stand-in for ``pyannote.core.Annotation``."""

    def __init__(self, turns: list[tuple[float, float, object]]) -> None:
        self.turns = list(turns)
        self.yield_label: bool | None = None

    def itertracks(self, *, yield_label: bool = False):
        self.yield_label = yield_label
        for start, end, speaker_id in self.turns:
            yield FakeTurn(start, end), "track", speaker_id


class FakeDiarizeOutput:
    """Stand-in for pyannote's ``DiarizeOutput``.

    Attributes are only created when a value is passed, so a test can model a
    pipeline that exposes just the exclusive annotation, or neither.
    """

    def __init__(
        self,
        speaker_diarization: object | None = None,
        exclusive_speaker_diarization: object | None = None,
    ) -> None:
        if speaker_diarization is not None:
            self.speaker_diarization = speaker_diarization
        if exclusive_speaker_diarization is not None:
            self.exclusive_speaker_diarization = exclusive_speaker_diarization


class FakePipeline:
    """Stand-in for ``pyannote.audio.Pipeline`` that records every interaction."""

    # Recorded state, reset by :func:`_patch`.
    requests: list[dict[str, object]] = []
    devices: list[object] = []
    calls: list[tuple[str, dict[str, object]]] = []
    instances: list["FakePipeline"] = []

    # Scripted behaviour, set by individual tests.
    load_error: Exception | None = None
    move_error: Exception | None = None
    infer_error: Exception | None = None
    returns_none: bool = False
    result: object = None

    def __init__(self, model: str, *, token: str | None = None, cache_dir=None) -> None:
        self.model = model
        self.token = token
        self.cache_dir = cache_dir
        self.device: object = None
        FakePipeline.instances.append(self)

    @classmethod
    def from_pretrained(cls, model: str, *, token=None, cache_dir=None):
        cls.requests.append({"model": model, "token": token, "cache_dir": cache_dir})
        if cls.load_error is not None:
            raise cls.load_error
        if cls.returns_none:
            return None
        return cls(model, token=token, cache_dir=cache_dir)

    def to(self, device):
        if FakePipeline.move_error is not None:
            raise FakePipeline.move_error
        self.device = device
        FakePipeline.devices.append(device)
        return self

    def __call__(self, audio_path: str, **kwargs):
        FakePipeline.calls.append((audio_path, kwargs))
        if FakePipeline.infer_error is not None:
            raise FakePipeline.infer_error
        return FakePipeline.result


class FakeTorch:
    """Minimal stand-in for the only part of ``torch`` this module uses."""

    requested: list[str] = []

    @staticmethod
    def device(name: str) -> str:
        FakeTorch.requested.append(name)
        return f"torch.device({name})"


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fakes and reset all recorded state."""

    FakePipeline.requests = []
    FakePipeline.devices = []
    FakePipeline.calls = []
    FakePipeline.instances = []
    FakePipeline.load_error = None
    FakePipeline.move_error = None
    FakePipeline.infer_error = None
    FakePipeline.returns_none = False
    FakePipeline.result = None
    FakeTorch.requested = []

    monkeypatch.setattr(diarization, "Pipeline", FakePipeline)
    monkeypatch.setattr(diarization, "torch", FakeTorch)


def _settings(
    tmp_path: Path,
    *,
    device: str = "cuda",
    token: str | None = "hf_test_token",
    model: str = DEFAULT_DIARIZATION_MODEL,
) -> Settings:
    return Settings(
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        model_cache_dir=tmp_path / "models",
        huggingface_token=token,
        device=device,
        diarization_model=model,
    )


def _audio(tmp_path: Path, name: str = "speech.wav") -> Path:
    """Create the placeholder input file (pyannote is mocked, so it is never read)."""

    path = tmp_path / name
    path.write_bytes(b"")
    return path


def _pipeline_output(turns) -> FakeDiarizeOutput:
    """Build the ``DiarizeOutput`` pyannote returns for ``turns``.

    Only the exclusive annotation carries the result this module consumes.
    """

    return FakeDiarizeOutput(exclusive_speaker_diarization=FakeAnnotation(turns))


# ---------------------------------------------------------------------------
# segment conversion, ordering, and speakers
# ---------------------------------------------------------------------------


def test_exclusive_turns_are_converted_to_speaker_segments(monkeypatch, tmp_path):
    _patch(monkeypatch)
    annotation = FakeAnnotation([(12.43, 16.82, "SPEAKER_00")])
    FakePipeline.result = FakeDiarizeOutput(exclusive_speaker_diarization=annotation)

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert segments == [SpeakerSegment(speaker_id="SPEAKER_00", start=12.43, end=16.82)]
    assert annotation.yield_label is True  # labels must be requested, not silently dropped


def test_segments_are_sorted_chronologically(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output(
        [
            (17.10, 20.54, "SPEAKER_01"),
            (12.43, 16.82, "SPEAKER_00"),
            (30.00, 31.00, "SPEAKER_00"),
        ]
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert [(segment.speaker_id, segment.start) for segment in segments] == [
        ("SPEAKER_00", 12.43),
        ("SPEAKER_01", 17.10),
        ("SPEAKER_00", 30.00),
    ]


def test_multiple_speakers_are_preserved_with_their_timestamps(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output(
        [
            (0.50, 2.25, "SPEAKER_00"),
            (2.25, 4.00, "SPEAKER_02"),
            (4.00, 6.75, "SPEAKER_01"),
        ]
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert [segment.speaker_id for segment in segments] == [
        "SPEAKER_00",
        "SPEAKER_02",
        "SPEAKER_01",
    ]
    assert [segment.duration for segment in segments] == [1.75, 1.75, 2.75]


def test_no_speech_yields_an_empty_list(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([])

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == []


# ---------------------------------------------------------------------------
# pyannote output: the exclusive annotation is authoritative
# ---------------------------------------------------------------------------


def test_exclusive_annotation_wins_when_both_are_present(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation([(1.0, 2.0, "SPEAKER_00")]),
        exclusive_speaker_diarization=FakeAnnotation([(9.0, 10.0, "SPEAKER_09")]),
    )

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == [
        SpeakerSegment("SPEAKER_09", 9.0, 10.0)
    ]


def test_overlapping_speaker_diarization_is_not_used(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # The non-exclusive field contains two overlapping tracks. The exclusive field
    # resolves them to one speaker, and that is what must come back.
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation(
            [(1.0, 3.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
        ),
        exclusive_speaker_diarization=FakeAnnotation([(1.0, 4.0, "SPEAKER_00")]),
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert segments == [SpeakerSegment("SPEAKER_00", 1.0, 4.0)]
    assert len(segments) == 1  # the same audio is not attributed twice


def test_exclusive_annotation_is_read_from_a_pipeline_that_only_has_it(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(
        exclusive_speaker_diarization=FakeAnnotation([(5.0, 6.0, "SPEAKER_01")])
    )

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == [
        SpeakerSegment("SPEAKER_01", 5.0, 6.0)
    ]


def test_regular_speaker_diarization_alone_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # Only the non-exclusive field is present: not good enough, and no fallback.
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation([(1.0, 2.0, "SPEAKER_00")])
    )

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_output_without_the_exclusive_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput()

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_malformed_exclusive_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(exclusive_speaker_diarization=object())

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_bare_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # A ``legacy=True`` pipeline returns the bare, non-exclusive annotation.
    FakePipeline.result = FakeAnnotation([(1.0, 2.0, "SPEAKER_00")])

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_unexpected_pyannote_output_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = object()

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


# ---------------------------------------------------------------------------
# invalid segments
# ---------------------------------------------------------------------------


def test_empty_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, "")])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_whitespace_only_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, "   ")])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_non_string_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, None)])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_end_before_start_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(5.0, 2.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="greater than start"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_zero_length_turn_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(2.0, 2.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="greater than start"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_negative_start_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(-1.0, 3.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="start must be"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_non_finite_timestamp_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, float("inf"), "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="finite"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_speaker_segment_normalises_integers_to_float():
    segment = SpeakerSegment(speaker_id="SPEAKER_00", start=1, end=3)

    assert isinstance(segment.start, float)
    assert isinstance(segment.end, float)
    assert segment.duration == 2.0


def test_speaker_segment_is_immutable():
    segment = SpeakerSegment("SPEAKER_00", 0.0, 1.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.speaker_id = "SPEAKER_01"


def test_speaker_segment_rejects_non_string_speaker_id():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        SpeakerSegment(speaker_id=None, start=0.0, end=1.0)


def test_speaker_segment_rejects_non_numeric_start():
    with pytest.raises(InvalidSegmentError, match="number of seconds"):
        SpeakerSegment(speaker_id="SPEAKER_00", start="0", end=1.0)


def test_segment_and_count_errors_are_also_value_errors():
    assert issubclass(InvalidSegmentError, ValueError)
    assert issubclass(InvalidSpeakerCountError, ValueError)
    assert issubclass(InvalidSegmentError, DiarizationError)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_missing_input_file_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingInputError, match="not found"):
        diarize(tmp_path / "does-not-exist.wav", settings=_settings(tmp_path))

    assert FakePipeline.requests == []  # the model is never touched


def test_directory_input_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(InvalidInputError, match="not a file"):
        diarize(directory, settings=_settings(tmp_path))

    assert FakePipeline.requests == []


# ---------------------------------------------------------------------------
# configuration forwarding
# ---------------------------------------------------------------------------


def test_pipeline_is_loaded_with_the_configured_model_token_and_cache(monkeypatch, tmp_path):
    _patch(monkeypatch)
    settings = _settings(tmp_path, model="pyannote/some-other-model")
    FakePipeline.result = _pipeline_output([])

    diarize(_audio(tmp_path), settings=settings)

    assert FakePipeline.requests == [
        {
            "model": "pyannote/some-other-model",
            "token": "hf_test_token",
            "cache_dir": settings.model_cache_dir,
        }
    ]


def test_pipeline_is_moved_to_the_configured_device(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([])

    diarize(_audio(tmp_path), settings=_settings(tmp_path, device="cuda:1"))

    # Exactly one device is requested, and it is the configured one: there is no
    # silent fall back to cpu.
    assert FakeTorch.requested == ["cuda:1"]
    assert FakePipeline.devices == ["torch.device(cuda:1)"]


def test_audio_path_is_passed_to_the_pipeline(monkeypatch, tmp_path):
    _patch(monkeypatch)
    source = _audio(tmp_path)
    FakePipeline.result = _pipeline_output([])

    diarize(source, settings=_settings(tmp_path))

    assert FakePipeline.calls == [(str(source), {})]


def test_speaker_count_hints_are_forwarded(monkeypatch, tmp_path):
    _patch(monkeypatch)
    source = _audio(tmp_path)
    FakePipeline.result = _pipeline_output([])

    diarize(source, settings=_settings(tmp_path), min_speakers=2, max_speakers=6)

    assert FakePipeline.calls == [(str(source), {"min_speakers": 2, "max_speakers": 6})]


def test_unusable_speaker_count_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerCountError, match="num_speakers"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path), num_speakers=0)

    assert FakePipeline.requests == []


def test_min_speakers_above_max_speakers_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerCountError, match="must not exceed"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path), min_speakers=5, max_speakers=2)

    assert FakePipeline.requests == []


# ---------------------------------------------------------------------------
# authentication and model failures
# ---------------------------------------------------------------------------


def test_missing_huggingface_token_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingHuggingFaceTokenError, match="HUGGINGFACE_TOKEN"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path, token=None))

    assert FakePipeline.requests == []


def test_blank_huggingface_token_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingHuggingFaceTokenError, match="HUGGINGFACE_TOKEN"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path, token="   "))

    assert FakePipeline.requests == []


def test_missing_pyannote_runtime_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.setattr(diarization, "Pipeline", None)

    with pytest.raises(ModelInitializationError, match="not installed"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_pipeline_load_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.load_error = RuntimeError("gated repo, access denied")

    with pytest.raises(ModelInitializationError, match="could not load") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_unavailable_checkpoint_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.returns_none = True

    with pytest.raises(ModelInitializationError, match="could not load"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_device_move_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.move_error = RuntimeError("CUDA is not available")

    with pytest.raises(ModelInitializationError, match="could not move") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_inference_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.infer_error = RuntimeError("boom")

    with pytest.raises(DiarizationInferenceError, match="diarization failed") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
