"""Speaker diarization ("who spoke when") with **pyannote Community-1**.

This module is the only place in the pipeline that talks to ``pyannote.audio``.
It consumes the dialogue stem written by :mod:`app.pipeline.separation` and
returns the speaker turns every later stage needs::

    speech.wav -> diarize() -> [SpeakerSegment(speaker_id, start, end), ...]

The output is deliberately limited to *who spoke when*, with exactly one speaker
attributed per region: pyannote's **exclusive** diarization is the authoritative
source. Transcript text, emotions, character names, and voice embeddings belong
to :mod:`app.pipeline.transcription`, :mod:`app.pipeline.voice_profiles`, and
:mod:`app.pipeline.tts`; they must not be added here.

Contract with the ``pyannote.audio`` package (checked against 4.0.7)
-------------------------------------------------------------------
* ``Pipeline.from_pretrained(model, token=..., cache_dir=...)`` loads the
  pipeline. There is **no** device argument; the call returns ``None`` when the
  checkpoint config cannot be fetched (unknown id, or a token without access).
* ``pipeline.to(torch.device(...))`` selects the compute device.
* ``pipeline(audio_path, num_speakers=..., min_speakers=..., max_speakers=...)``
  runs diarization and returns a ``DiarizeOutput``. Its
  ``exclusive_speaker_diarization`` annotation is the authoritative result: it
  carries no overlapping turns, so every instant of speech is attributed to
  exactly one speaker. The overlapping ``speaker_diarization`` variant is
  deliberately ignored, and a bare ``Annotation`` (what ``legacy=True`` pipelines
  return) is rejected rather than accepted as an overlap-free result.
* ``Annotation.itertracks(yield_label=True)`` yields ``(turn, track, label)``.

Deliberate non-goals
--------------------
* **No CUDA -> CPU fallback.** The device comes from the project settings, and an
  explicit ``cuda`` request that is unavailable must fail loudly.
* **No fallback to the non-exclusive diarization.** If the exclusive annotation
  is missing or malformed, this module raises :class:`UnsupportedOutputError`
  instead of returning potentially overlapping speaker turns.
* **No model downloads at import time** (and none in the test suite).
* **No pyannote objects in the public API.** Callers only ever see
  :class:`SpeakerSegment`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

try:  # Heavy runtime dependencies. They are guarded so that this module stays
    # importable (and testable) where the CUDA/PyTorch stack is not installed.
    import torch
    from pyannote.audio import Pipeline
except ImportError:  # pragma: no cover - only without the AI runtime installed
    torch = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]


#: ``DiarizeOutput`` field holding the authoritative annotation. pyannote
#: Community-1 assigns every instant of speech to exactly one speaker there, so
#: every region maps to a single speaker; the overlapping ``speaker_diarization``
#: variant is deliberately not used.
_EXCLUSIVE_ANNOTATION_ATTRIBUTE = "exclusive_speaker_diarization"


class DiarizationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(DiarizationError):
    """The requested input audio file does not exist."""


class InvalidInputError(DiarizationError):
    """The input path exists but is not a regular file."""


class MissingHuggingFaceTokenError(DiarizationError):
    """No Hugging Face token is configured for the gated diarization model."""


class ModelInitializationError(DiarizationError):
    """The pyannote pipeline could not be loaded or moved to the device."""


class DiarizationInferenceError(DiarizationError):
    """pyannote failed while running diarization."""


class UnsupportedOutputError(DiarizationError):
    """pyannote returned an object this module cannot convert."""


class InvalidSegmentError(DiarizationError, ValueError):
    """A speaker segment has an empty speaker id or invalid timestamps."""


class InvalidSpeakerCountError(DiarizationError, ValueError):
    """A requested speaker-count hint is not a usable positive integer."""


def _seconds(name: str, value: object) -> float:
    """Return ``value`` as a finite ``float`` number of seconds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSegmentError(
            f"{name} must be a number of seconds, got {type(value).__name__}"
        )

    seconds = float(value)
    if not math.isfinite(seconds):
        raise InvalidSegmentError(f"{name} must be finite, got {value!r}")
    return seconds


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One diarized speaker turn, in seconds.

    Instances are immutable and always valid: an empty speaker id, a negative
    ``start``, or an ``end`` that does not come strictly after ``start`` is
    rejected at construction time, so later stages never have to defend
    themselves against a malformed turn.
    """

    speaker_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidSegmentError("speaker_id must be a non-empty string")

        start = _seconds("start", self.start)
        end = _seconds("end", self.end)

        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        # Normalise ints to float so the declared type is guaranteed.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        """Length of the turn in seconds."""

        return self.end - self.start


def _validate_audio_path(path: str | Path) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path` pointing at an existing file."""

    source = Path(path)
    if not source.exists():
        raise MissingInputError(f"input audio file not found: {source}")
    if not source.is_file():
        raise InvalidInputError(f"input audio path is not a file: {source}")
    return source


def _require_huggingface_token(settings: Settings) -> str:
    """Return the configured Hugging Face token, or explain what is missing."""

    token = (settings.huggingface_token or "").strip()
    if not token:
        raise MissingHuggingFaceTokenError(
            "HUGGINGFACE_TOKEN is not set; the pyannote Community-1 diarization "
            "pipeline is a gated model and requires a Hugging Face token that has "
            "accepted its conditions (see .env.example)"
        )
    return token


def _validate_speaker_counts(
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """Reject unusable speaker-count hints before touching the model."""

    hints = (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    )
    for name, value in hints:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidSpeakerCountError(
                f"{name} must be a positive integer, got {value!r}"
            )

    if min_speakers is not None and max_speakers is not None:
        if min_speakers > max_speakers:
            raise InvalidSpeakerCountError(
                f"min_speakers ({min_speakers}) must not exceed "
                f"max_speakers ({max_speakers})"
            )


def _load_pipeline(settings: Settings, token: str) -> Any:
    """Load the pyannote pipeline into memory on the configured device."""

    if Pipeline is None or torch is None:  # pragma: no cover - missing AI runtime
        raise ModelInitializationError(
            "pyannote.audio is not installed; install the runtime dependencies "
            "(torch, torchaudio, pyannote.audio) before running diarization"
        )

    try:
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model,
            token=token,
            cache_dir=settings.model_cache_dir,
        )
    except Exception as exc:
        raise ModelInitializationError(
            f"could not load the pyannote pipeline "
            f"{settings.diarization_model!r}: {type(exc).__name__}: {exc}"
        ) from exc

    # ``from_pretrained`` returns None instead of raising when the checkpoint
    # config cannot be fetched at all.
    if pipeline is None:
        raise ModelInitializationError(
            f"pyannote could not load {settings.diarization_model!r}: the pipeline "
            "is unknown, or HUGGINGFACE_TOKEN has not been granted access to it"
        )

    try:
        pipeline.to(torch.device(settings.device))
    except Exception as exc:
        raise ModelInitializationError(
            f"could not move the pyannote pipeline to device "
            f"{settings.device!r}: {type(exc).__name__}: {exc}"
        ) from exc

    return pipeline


def _apply_pipeline(
    pipeline: Any,
    audio_path: Path,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Any:
    """Run diarization, wrapping any pyannote failure with context."""

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        return pipeline(str(audio_path), **kwargs)
    except Exception as exc:
        raise DiarizationInferenceError(
            f"pyannote diarization failed for {audio_path}: {type(exc).__name__}: {exc}"
        ) from exc


def _extract_annotation(result: Any) -> Any:
    """Return the exclusive diarization ``Annotation`` carried by ``result``.

    pyannote 4.x returns a ``DiarizeOutput``; its ``exclusive_speaker_diarization``
    field contains no overlapping turns, so every region is attributed to exactly
    one speaker. There is no fallback to the overlapping ``speaker_diarization``
    field: a result without a usable exclusive annotation is an error to report,
    not something to paper over.
    """

    annotation = getattr(result, _EXCLUSIVE_ANNOTATION_ATTRIBUTE, None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation

    raise UnsupportedOutputError(
        f"pyannote returned {type(result).__name__} without a usable "
        f"{_EXCLUSIVE_ANNOTATION_ATTRIBUTE!r}; pyannote Community-1 exclusive "
        "diarization is required so that every region is attributed to exactly "
        "one speaker, and this module does not fall back to the non-exclusive "
        "output"
    )


def _to_segments(result: Any) -> list[SpeakerSegment]:
    """Convert a pyannote result into chronologically sorted segments."""

    annotation = _extract_annotation(result)

    segments: list[SpeakerSegment] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker_id = label.strip() if isinstance(label, str) else ""
        segments.append(
            SpeakerSegment(
                speaker_id=speaker_id,
                start=turn.start,
                end=turn.end,
            )
        )

    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return segments


def diarize(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Diarize ``audio_path`` into chronologically sorted speaker turns.

    Parameters
    ----------
    audio_path:
        Dialogue audio (typically the ``*_speech.wav`` stem written by
        :func:`app.pipeline.separation.separate_stems`).
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        The model id, device, model cache, and Hugging Face token all come from
        here - nothing is hard-coded in this module.
    num_speakers, min_speakers, max_speakers:
        Optional hints for pyannote's clustering. Leave unset to let pyannote
        estimate the speaker count. ``num_speakers`` wins over the bounds.

    Returns
    -------
    list[SpeakerSegment]
        One immutable segment per speaker turn, sorted by ``start``. An empty
        list means pyannote found no speech.

    Raises
    ------
    MissingInputError, InvalidInputError
        The input is not an existing regular file.
    MissingHuggingFaceTokenError
        ``HUGGINGFACE_TOKEN`` is not configured.
    ModelInitializationError
        The pyannote pipeline could not be loaded or moved to the device.
    DiarizationInferenceError
        pyannote failed while running diarization.
    UnsupportedOutputError
        pyannote returned an object this module cannot convert.
    """

    settings = settings if settings is not None else get_settings()

    source = _validate_audio_path(audio_path)
    token = _require_huggingface_token(settings)
    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)

    pipeline = _load_pipeline(settings, token)
    result = _apply_pipeline(
        pipeline,
        source,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    return _to_segments(result)


__all__ = [
    "DiarizationError",
    "DiarizationInferenceError",
    "InvalidInputError",
    "InvalidSegmentError",
    "InvalidSpeakerCountError",
    "MissingHuggingFaceTokenError",
    "MissingInputError",
    "ModelInitializationError",
    "SpeakerSegment",
    "UnsupportedOutputError",
    "diarize",
]
