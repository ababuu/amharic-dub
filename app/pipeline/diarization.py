"""Speaker diarization ("who spoke when") with **pyannote Community-1**.

This module is the only place in the pipeline that talks to ``pyannote.audio``.
It consumes the dialogue stem written by :mod:`app.pipeline.separation` and
returns the speaker turns every later stage needs::

    speech.wav -> diarize() -> [SpeakerSegment(speaker_id, start, end), ...]

Two views of the same pyannote result are used for different purposes:

* the **exclusive** annotation provides the starting attribution - it carries no
  overlapping turns, so every instant of speech belongs to exactly one speaker and
  a transcribed line can never be attributed to two people. Clean, sufficiently
  long turns are then checked against Community-1's speaker embeddings, and only a
  strong disagreement can change the label.
* the **overlapping** annotation is used twice, and never to relabel a turn. It
  *flags* crosstalk, because the exclusive view necessarily hides one of the two
  speakers when they talk at once. And its turns are returned as
  ``overlap_turns``, so the transcription stage can find the words of the speaker
  the exclusive view dropped: that is the difference between a two-character
  argument and a monologue. Overlapping turns are excluded from verification.

Transcript text, emotions, character names, and voice-cloning embeddings belong
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
  ``exclusive_speaker_diarization`` annotation provides non-overlapping turns.
  Speaker embeddings are aligned to the labels in ``speaker_diarization`` and are
  used to verify clean, long turns. A bare ``Annotation`` (what ``legacy=True``
  pipelines return) is rejected rather than accepted as an overlap-free result.
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

import logging
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


#: ``DiarizeOutput`` field holding the annotation that is authoritative for
#: *attribution*. pyannote Community-1 assigns every instant of speech to exactly
#: one speaker there, so every region maps to a single speaker and a line is never
#: attributed to two people at once.
_EXCLUSIVE_ANNOTATION_ATTRIBUTE = "exclusive_speaker_diarization"

#: ``DiarizeOutput`` field holding the overlap-aware annotation. It is read for two
#: things the exclusive view cannot provide: *where* two speakers talk at once
#: (reported as crosstalk), and *which* speaker the exclusive view had to drop
#: (returned as ``overlap_turns`` so their words can be recovered). It never
#: relabels a turn.
#: Unlike the exclusive field this one is optional: a pyannote result without it
#: simply means "no crosstalk could be reported", not an error.
_OVERLAPPING_ANNOTATION_ATTRIBUTE = "speaker_diarization"

# A turn this short is not reliable material for speaker verification. In
# particular, pyannote's own embedding backends require at least 0.5 seconds.
_MIN_VERIFICATION_TURN_SECONDS = 1.5
_MIN_REASSIGNMENT_SIMILARITY = 0.55
_MIN_REASSIGNMENT_MARGIN = 0.15
_LOGGER = logging.getLogger(__name__)

#: Two turns of the same speaker separated by at most this long are one utterance
#: in the overlap-aware view. pyannote splits a turn wherever another speaker's
#: boundary crosses it, so a 6-second sentence can arrive as five pieces; merging
#: them back is what makes the recovered stretch long enough to transcribe.
_OVERLAP_TURN_MERGE_SECONDS = 0.25


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


@dataclass(frozen=True, slots=True)
class CrosstalkRegion:
    """An interval in which two or more speakers are talking at the same time.

    Reported for information only: it never changes which speaker a line is
    attributed to. It exists because the exclusive diarization this pipeline
    otherwise relies on has to pick one speaker per instant, so simultaneous
    speech would otherwise be invisible - and simultaneous speech is where a dub
    most often goes wrong.
    """

    speaker_ids: tuple[str, ...]
    start: float
    end: float

    def __post_init__(self) -> None:
        speakers = tuple(str(speaker).strip() for speaker in self.speaker_ids)
        if len(speakers) < 2 or any(not speaker for speaker in speakers):
            raise InvalidSegmentError(
                "a crosstalk region needs at least two non-empty speaker ids, got "
                f"{self.speaker_ids!r}"
            )

        start = _seconds("start", self.start)
        end = _seconds("end", self.end)
        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        # Sorted and de-duplicated so two regions covering the same speakers and
        # instant compare equal however pyannote happened to order its turns.
        object.__setattr__(self, "speaker_ids", tuple(sorted(set(speakers))))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        """Length of the overlap in seconds."""

        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    """Everything diarization found: attributed turns plus reported crosstalk.

    ``turns`` is the exclusive, non-overlapping attribution with high-confidence
    speaker-embedding corrections applied. ``crosstalk`` reports where speakers
    overlapped. ``overlap_turns`` is the overlap-aware view of the same audio:
    every speaker's own turn, *allowing* two of them to cover the same instant.

    Why both views are returned
    ---------------------------
    The exclusive view is authoritative for *text*: one instant, one speaker, so a
    transcribed word can only be credited to one character. But it necessarily
    shortens - sometimes removes entirely - the second speaker's turn, and the
    words spoken in that lost stretch are heard by nobody downstream. On a film
    where two characters argue, that is most of a scene collapsing into a
    monologue. ``overlap_turns`` carries the information needed to recover them:
    which other speaker was talking, and exactly when. It is never used to
    *relabel* a turn; only to notice that a speaker the exclusive view dropped was
    there, so the transcription stage can listen for them.

    ``corrections`` records every changed turn and the scores that justified it.
    """

    turns: tuple[SpeakerSegment, ...]
    crosstalk: tuple[CrosstalkRegion, ...] = ()
    corrections: tuple["SpeakerAttributionCorrection", ...] = ()
    #: Speaker turns from the overlap-aware annotation, in chronological order.
    #: Empty when the pyannote result carries no such annotation; everything
    #: downstream must keep working in that case.
    overlap_turns: tuple[SpeakerSegment, ...] = ()
    #: Turns whose label :func:`_verify_speaker_attributions` actually checked against
    #: Community-1's speaker embeddings and left as diarization reported them. Not
    #: "probably right": these were compared with the model-space prototype of every
    #: speaker and no other one won by a decisive margin.
    #:
    #: This matters because it ranks two kinds of evidence. An embedding compares the
    #: voice itself; a pitch median measures one acoustic dimension of it, and a
    #: trained actor's expressive range covers most of an octave either way - a hushed
    #: line and a shouted one from the same person can sit an octave apart. So a later
    #: pitch-based repair must not overturn a confirmation made here, and the turn is
    #: published for exactly that reason.
    verified: tuple[SpeakerSegment, ...] = ()


    @property
    def speaker_ids(self) -> tuple[str, ...]:
        """Every speaker id that appears in ``turns``, sorted."""

        return tuple(sorted({segment.speaker_id for segment in self.turns}))

    @property
    def crosstalk_seconds(self) -> float:
        """Total time covered by reported crosstalk."""

        return sum(region.duration for region in self.crosstalk)


@dataclass(frozen=True, slots=True)
class SpeakerAttributionCorrection:
    """A diarized turn reassigned after a high-confidence embedding comparison."""

    start: float
    end: float
    original_speaker_id: str
    corrected_speaker_id: str
    corrected_similarity: float
    similarity_margin: float


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


def _runtime_summary() -> str:
    """Describe the compute stack, so a device or compiler failure is diagnosable.

    Every lookup is defensive: this runs on an error path (and against the fakes
    in the test suite), where raising would hide the failure it is meant to explain.
    """

    if torch is None:  # pragma: no cover - only without the AI runtime
        return "torch is not installed"

    parts: list[str] = []
    try:
        parts.append(f"torch {torch.__version__}")
    except Exception:  # pragma: no cover - a stubbed runtime
        parts.append("torch (version unknown)")
    try:
        cuda_build = torch.version.cuda
        parts.append(f"built for CUDA {cuda_build or 'none (CPU build)'}")
    except Exception:  # pragma: no cover - a stubbed runtime
        pass
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            parts.append(f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    except Exception:  # pragma: no cover - diagnostics must never fail the stage
        pass
    try:
        import pyannote.audio as pyannote_audio

        parts.append(f"pyannote.audio {pyannote_audio.__version__}")
    except Exception:  # pragma: no cover - version metadata may be absent
        pass
    return ", ".join(parts)


#: NVRTC's own wording when it rejects the architecture Inductor asked for.
_ARCH_MISMATCH_MARKERS = ("nvrtc", "gpu-architecture")


def _is_compiler_arch_failure(exc: BaseException) -> bool:
    """``True`` when a CUDA kernel could not be compiled for this GPU's arch."""

    message = str(exc).lower()
    return any(marker in message for marker in _ARCH_MISMATCH_MARKERS)


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

    _LOGGER.info("pyannote runtime: %s", _runtime_summary())
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
        hint = ""
        if _is_compiler_arch_failure(exc):
            hint = (
                ". A CUDA kernel could not be compiled for this GPU's compute "
                "capability (NVRTC). TorchInductor is disabled by default "
                "(TORCH_COMPILE_DISABLE=1); if ALLOW_TORCH_COMPILE=1 or "
                "TORCH_COMPILE_DISABLE=0 is set, remove it. If not, the pod's "
                "PyTorch CUDA build is older than its GPU, so use the RunPod "
                f"PyTorch image matched to it. Runtime: {_runtime_summary()}"
            )
        raise DiarizationInferenceError(
            f"pyannote diarization failed for {audio_path}: "
            f"{type(exc).__name__}: {exc}{hint}"
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


def _extract_overlap_annotation(result: Any) -> Any | None:
    """Return the overlap-aware ``Annotation`` carried by ``result``, or ``None``.

    Unlike the exclusive annotation this one is optional: it is only used to flag
    crosstalk, so a result that does not carry it is a result with nothing to
    report rather than a failure.
    """

    annotation = getattr(result, _OVERLAPPING_ANNOTATION_ATTRIBUTE, None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation
    return None


def _to_crosstalk(annotation: Any) -> list[CrosstalkRegion]:
    """Return the intervals of ``annotation`` where two or more speakers overlap.

    The annotation is swept as a set of boundaries rather than compared pair by
    pair: at every boundary the set of simultaneously active speakers changes, and
    each stretch with two or more of them is an overlap. Adjacent stretches with
    the same speakers are merged, so a long crosstalk is one region instead of one
    region per turn boundary inside it.
    """

    boundaries: set[float] = set()
    spans: list[tuple[str, float, float]] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker = label.strip() if isinstance(label, str) else ""
        if not speaker:
            continue
        spans.append((speaker, float(turn.start), float(turn.end)))
        boundaries.add(float(turn.start))
        boundaries.add(float(turn.end))

    regions: list[CrosstalkRegion] = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        active = tuple(
            sorted(
                {
                    speaker
                    for speaker, span_start, span_end in spans
                    if span_start <= start and span_end >= end
                }
            )
        )
        if len(active) < 2:
            continue

        if regions and regions[-1].speaker_ids == active and regions[-1].end == start:
            merged = regions.pop()
            regions.append(CrosstalkRegion(speaker_ids=active, start=merged.start, end=end))
            continue
        regions.append(CrosstalkRegion(speaker_ids=active, start=start, end=end))

    return regions


def _to_overlap_turns(annotation: Any) -> list[SpeakerSegment]:
    """Return every speaker's turn from the overlap-aware annotation.

    Two speakers' turns may cover the same instant here, which is the whole point:
    this is the view in which the second person in a crosstalk is still visible.
    Each speaker's own turns are merged when they touch, because pyannote splits a
    single utterance wherever the *other* speaker's boundary crosses it - a split
    that carries no meaning of its own and would otherwise fragment the recovered
    text into pieces too short to transcribe. Merging is done per speaker, since the
    turns of two speakers interleave in the annotation.
    """

    spans: dict[str, list[tuple[float, float]]] = {}
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker = label.strip() if isinstance(label, str) else ""
        if not speaker:
            continue
        start, end = float(turn.start), float(turn.end)
        if end <= start:
            continue
        spans.setdefault(speaker, []).append((start, end))

    turns: list[SpeakerSegment] = []
    for speaker, intervals in spans.items():
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start - merged[-1][1] <= _OVERLAP_TURN_MERGE_SECONDS:
                merged[-1][1] = max(merged[-1][1], end)
                continue
            merged.append([start, end])
        turns.extend(
            SpeakerSegment(speaker_id=speaker, start=start, end=end)
            for start, end in merged
        )

    turns.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return turns


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


def _as_embedding(value: Any) -> tuple[float, ...] | None:
    """Convert a model embedding to a finite, non-zero Python vector."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(component) for component in vector):
        return None
    if sum(component * component for component in vector) <= 0:
        return None
    return vector


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return cosine similarity, or -1 for incompatible embedding dimensions."""

    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _speaker_centroids(result: Any) -> dict[str, tuple[float, ...]]:
    """Read Community-1's documented label-ordered speaker embeddings."""

    annotation = getattr(result, _OVERLAPPING_ANNOTATION_ATTRIBUTE, None)
    embeddings = getattr(result, "speaker_embeddings", None)
    if annotation is None or not hasattr(annotation, "labels") or embeddings is None:
        return {}

    labels = list(annotation.labels())
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    try:
        rows = list(embeddings)
    except TypeError:
        return {}
    if len(labels) != len(rows):
        return {}

    centroids: dict[str, tuple[float, ...]] = {}
    for label, row in zip(labels, rows):
        vector = _as_embedding(row)
        if isinstance(label, str) and label.strip() and vector is not None:
            centroids[label.strip()] = vector
    return centroids


def _is_crosstalk_turn(
    segment: SpeakerSegment, crosstalk: tuple[CrosstalkRegion, ...]
) -> bool:
    """Return whether any part of this turn overlaps another speaker."""

    return any(
        segment.start < region.end and segment.end > region.start
        for region in crosstalk
    )


def _verify_speaker_attributions(
    pipeline: Any,
    result: Any,
    audio_path: Path,
    segments: list[SpeakerSegment],
    crosstalk: tuple[CrosstalkRegion, ...],
) -> tuple[list[SpeakerSegment], tuple[SpeakerAttributionCorrection, ...], tuple[SpeakerSegment, ...]]:
    """Correct long, clean turns with strong agreement from speaker embeddings.

    Community-1's speaker embeddings are the same model-space prototypes used by
    its clustering. Comparing clean turn embeddings against those prototypes
    avoids a fixed SPEAKER_00/SPEAKER_01 assumption and requires no extra model.

    Returns the turns, the corrections made, and the turns whose label the
    embeddings *confirmed*. The last one is published so a later, weaker signal
    cannot overturn it: a pitch median is one acoustic dimension of a voice, while
    an embedding is a comparison against the voice itself.
    """

    centroids = _speaker_centroids(result)
    audio = getattr(pipeline, "_audio", None)
    embedder = getattr(pipeline, "_embedding", None)
    if len(centroids) < 2 or audio is None or embedder is None or torch is None:
        return segments, (), ()

    try:
        from pyannote.core import Segment
    except ImportError:
        _LOGGER.warning("pyannote.core is unavailable; preserving diarization labels")
        return segments, (), ()

    corrections: list[SpeakerAttributionCorrection] = []
    verified: list[SpeakerSegment] = []
    confirmed: list[SpeakerSegment] = []
    audio_file = {"audio": str(audio_path)}
    for index, segment in enumerate(segments):
        if (
            segment.duration < _MIN_VERIFICATION_TURN_SECONDS
            or _is_crosstalk_turn(segment, crosstalk)
        ):
            verified.append(segment)
            continue

        try:
            waveform, _ = audio.crop(
                audio_file,
                Segment(segment.start, segment.end),
                mode="pad",
            )
            with torch.inference_mode():
                raw_embedding = embedder(waveform.unsqueeze(0))
        except Exception as exc:
            _LOGGER.warning(
                "Speaker verification failed for %.3f-%.3fs; preserving %s: %s: %s",
                segment.start,
                segment.end,
                segment.speaker_id,
                type(exc).__name__,
                exc,
            )
            verified.extend(segments[index:])
            return verified, tuple(corrections), tuple(confirmed)

        embedding_rows = (
            raw_embedding.tolist()
            if hasattr(raw_embedding, "tolist")
            else raw_embedding
        )
        try:
            turn_embedding = _as_embedding(embedding_rows[0])
        except (IndexError, TypeError):
            turn_embedding = None
        if turn_embedding is None:
            verified.append(segment)
            continue

        scores = {
            speaker_id: _cosine_similarity(turn_embedding, centroid)
            for speaker_id, centroid in centroids.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            verified.append(segment)
            continue

        best_speaker, best_score = ranked[0]
        if segment.speaker_id not in scores:
            verified.append(segment)
            continue
        current_score = scores[segment.speaker_id]
        margin = best_score - current_score
        if (
            best_speaker != segment.speaker_id
            and best_score >= _MIN_REASSIGNMENT_SIMILARITY
            and margin >= _MIN_REASSIGNMENT_MARGIN
        ):
            verified.append(
                SpeakerSegment(best_speaker, segment.start, segment.end)
            )
            corrections.append(
                SpeakerAttributionCorrection(
                    start=segment.start,
                    end=segment.end,
                    original_speaker_id=segment.speaker_id,
                    corrected_speaker_id=best_speaker,
                    corrected_similarity=best_score,
                    similarity_margin=margin,
                )
            )
        else:
            verified.append(segment)
            confirmed.append(segment)

    return verified, tuple(corrections), tuple(confirmed)


def _resolve_speaker_counts(
    settings: Settings,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[int | None, int | None, int | None]:
    """Fill unset speaker-count hints from ``settings``.

    An explicit argument always wins, so a caller that knows better for one film
    is never overridden by the environment. Supplying the bounds at all is the
    single strongest guard against a character splitting into several clusters -
    which sounds, downstream, like one character being several people.
    """

    if min_speakers is None:
        min_speakers = settings.diarization_min_speakers
    if max_speakers is None:
        max_speakers = settings.diarization_max_speakers
    return num_speakers, min_speakers, max_speakers


def diarize_detailed(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationResult:
    """Diarize ``audio_path`` into turns **and** the crosstalk between them.

    Clean, sufficiently long exclusive turns are verified against Community-1's
    speaker embeddings before being returned. The overlap-aware view is also read
    twice: so simultaneous speech can be reported instead of silently collapsing
    onto whichever speaker the exclusive view kept, and so the speakers it *did*
    drop are identified - with their exact spans - for transcription to recover.

    Speaker-count hints that are left unset fall back to
    ``DIARIZATION_MIN_SPEAKERS`` / ``DIARIZATION_MAX_SPEAKERS``; an explicit
    argument always wins over the configured value.
    """

    resolved = settings if settings is not None else get_settings()

    source = _validate_audio_path(audio_path)
    token = _require_huggingface_token(resolved)
    num_speakers, min_speakers, max_speakers = _resolve_speaker_counts(
        resolved, num_speakers, min_speakers, max_speakers
    )
    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)

    pipeline = _load_pipeline(resolved, token)
    result = _apply_pipeline(
        pipeline,
        source,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    overlap_annotation = _extract_overlap_annotation(result)
    crosstalk = () if overlap_annotation is None else tuple(_to_crosstalk(overlap_annotation))
    overlap_turns = (
        () if overlap_annotation is None else tuple(_to_overlap_turns(overlap_annotation))
    )
    turns, corrections, verified = _verify_speaker_attributions(
        pipeline,
        result,
        source,
        _to_segments(result),
        crosstalk,
    )
    return DiarizationResult(
        turns=tuple(turns),
        crosstalk=crosstalk,
        corrections=corrections,
        overlap_turns=overlap_turns,
        verified=verified,
    )


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
        The model id, device, model cache, Hugging Face token and the fallback
        speaker-count bounds all come from here - nothing is hard-coded in this
        module.
    num_speakers, min_speakers, max_speakers:
        Optional hints for pyannote's clustering. Leave unset to fall back to the
        configured bounds, or to let pyannote estimate the count when those are
        unset too. ``num_speakers`` wins over the bounds.

    Returns
    -------
    list[SpeakerSegment]
        One immutable segment per speaker turn, sorted by ``start``, after
        high-confidence speaker-embedding corrections. An empty list means
        pyannote found no speech. Call :func:`diarize_detailed` instead to also
        retrieve crosstalk and correction details.

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

    return list(
        diarize_detailed(
            audio_path,
            settings=settings,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        ).turns
    )


__all__ = [
    "CrosstalkRegion",
    "DiarizationError",
    "DiarizationInferenceError",
    "DiarizationResult",
    "InvalidInputError",
    "InvalidSegmentError",
    "InvalidSpeakerCountError",
    "MissingHuggingFaceTokenError",
    "MissingInputError",
    "ModelInitializationError",
    "SpeakerAttributionCorrection",
    "SpeakerSegment",
    "UnsupportedOutputError",
    "diarize",
    "diarize_detailed",
]
