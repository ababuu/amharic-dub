"""Speech transcription with **faster-whisper** ``large-v3``.

This module is the only place in the pipeline that talks to ``faster_whisper``.
It consumes the dialogue stem written by :mod:`app.pipeline.separation` together
with the speaker turns produced by :mod:`app.pipeline.diarization`, and returns
one :class:`TranscriptSegment` per transcribed utterance::

    speech.wav + [SpeakerSegment, ...] -> transcribe() -> [TranscriptSegment, ...]

Each diarized region is transcribed on its own, so every line is attributed to
the speaker who actually said it instead of being guessed afterwards from a
whole-movie transcript.

Timing model
------------
faster-whisper is handed the samples of a single diarized region, so the
timestamps it reports are relative to the *start of that region*. They are
translated back onto the movie timeline with::

    absolute = region.start + whisper_local

and then clamped to ``[region.start, region.end]`` so a transcript segment can
never claim speech outside the diarized turn it came from. The diarization
timestamps themselves are never replaced by Whisper's, only used as the offset.

Contract with the ``faster_whisper`` package (checked against 1.2.1)
------------------------------------------------------------------
* ``WhisperModel(model, device=..., device_index=..., compute_type=...,
  download_root=...)`` loads the model.
* ``model.transcribe(audio, language=..., task=..., word_timestamps=...)``
  accepts a 16 kHz mono float32 NumPy array and returns
  ``(Iterable[Segment], TranscriptionInfo)``; the iterator is lazy, so inference
  only really runs when it is consumed.
* ``Segment`` exposes ``start``, ``end`` and ``text`` in seconds, relative to the
  audio that was passed in.
* ``decode_audio(path, sampling_rate=16000)`` decodes a file to that NumPy array.
* ``model.detect_language(audio)`` returns ``(language, probability, all_probs)``
  and only inspects the beginning of the audio, so it is cheap.

Deliberate non-goals
--------------------
* **No CUDA -> CPU fallback.** The device comes from the project settings, and an
  explicit ``cuda`` request that is unavailable must fail loudly. ``auto`` is
  never used, because that is exactly the silent fallback we must avoid.
* **No translation.** ``task`` is always ``"transcribe"``; adaptation into
  Amharic belongs to :mod:`app.pipeline.translation`.
* **No model downloads at import time** (and none in the test suite).
* **No faster-whisper objects in the public API.** Callers only ever see
  :class:`TranscriptSegment`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.pipeline.diarization import SpeakerSegment

try:  # Heavy runtime dependency. Guarded so that this module stays importable
    # (and testable) where the CUDA/PyTorch stack is not installed.
    from faster_whisper import WhisperModel, decode_audio
except ImportError:  # pragma: no cover - only without the AI runtime installed
    WhisperModel = None  # type: ignore[assignment]
    decode_audio = None  # type: ignore[assignment]


#: faster-whisper works at 16 kHz mono; the source audio is decoded once there.
SAMPLE_RATE = 16_000

#: Word-level timestamps make Whisper tighten every segment to the spoken words.
#: That matters here because a diarized region usually contains a little silence
#: before and after the words, and the words are what has to be re-timed later.
WORD_TIMESTAMPS = True

#: A diarized turn shorter than this is treated as an artefact and skipped.
#: Speaker diarization regularly emits a handful of such turns at boundaries -
#: the run that prompted this constant had one of 0.02 seconds. Two reasons not
#: to transcribe them: no word fits in a window that short, so there is nothing to
#: lose, and faster-whisper cannot word-align them anyway. Its decoder still emits
#: a token for such a sliver, but none survives filtering, so the alignment step
#: indexes an empty timestamp array and raises ``IndexError``, taking the whole
#: film down with it.
MIN_REGION_SECONDS = 0.1


class TranscriptionError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(TranscriptionError):
    """The requested input audio file does not exist."""


class InvalidInputError(TranscriptionError):
    """The input path exists but is not a regular file."""


class AudioDecodeError(TranscriptionError):
    """The input file could not be decoded into audio samples."""


class InvalidSpeakerSegmentsError(TranscriptionError, ValueError):
    """The supplied diarization input is not an iterable of SpeakerSegment."""


class ModelInitializationError(TranscriptionError):
    """The faster-whisper model could not be loaded on the configured device."""


class TranscriptionInferenceError(TranscriptionError):
    """faster-whisper failed while transcribing a diarized region."""


class UnsupportedOutputError(TranscriptionError):
    """faster-whisper returned a segment this module cannot convert."""


class InvalidSegmentError(TranscriptionError, ValueError):
    """A transcript segment has an empty speaker id/text or bad timestamps."""


def _strict_seconds(name: str, value: object) -> float:
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
class TranscriptSegment:
    """One transcribed utterance, attributed to a diarized speaker, in seconds.

    Instances are immutable and always valid, so the later stages (translation,
    timing, mixing) never have to defend themselves against a malformed line.
    ``text`` is stored stripped: leading and trailing whitespace never leaks
    downstream.
    """

    speaker_id: str
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidSegmentError("speaker_id must be a non-empty string")

        start = _strict_seconds("start", self.start)
        end = _strict_seconds("end", self.end)

        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        if not isinstance(self.text, str) or not self.text.strip():
            raise InvalidSegmentError("text must be a non-empty string")

        # Normalise so the declared types are guaranteed.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", self.text.strip())

    @property
    def duration(self) -> float:
        """Length of the utterance in seconds."""

        return self.end - self.start


def _validate_audio_path(path: str | Path) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path` pointing at an existing file."""

    source = Path(path)
    if not source.exists():
        raise MissingInputError(f"input audio file not found: {source}")
    if not source.is_file():
        raise InvalidInputError(f"input audio path is not a file: {source}")
    return source


def _validate_speaker_segments(speaker_segments: Iterable[SpeakerSegment]) -> list[SpeakerSegment]:
    """Return the diarization turns as a chronological list, validated.

    Only :class:`~app.pipeline.diarization.SpeakerSegment` instances are accepted;
    they validate their own timestamps and speaker ids at construction time.
    """

    if isinstance(speaker_segments, (str, bytes)) or not isinstance(speaker_segments, Iterable):
        raise InvalidSpeakerSegmentsError(
            "speaker_segments must be an iterable of SpeakerSegment objects, got "
            f"{type(speaker_segments).__name__}"
        )

    segments = list(speaker_segments)
    for index, segment in enumerate(segments):
        if not isinstance(segment, SpeakerSegment):
            raise InvalidSpeakerSegmentsError(
                f"speaker_segments[{index}] is {type(segment).__name__}, "
                "expected a SpeakerSegment"
            )

    return sorted(segments, key=lambda item: (item.start, item.end, item.speaker_id))


def _require_runtime() -> None:
    """Fail clearly when the faster-whisper runtime is not installed."""

    if WhisperModel is None or decode_audio is None:  # pragma: no cover - no runtime
        raise ModelInitializationError(
            "faster-whisper is not installed; install the runtime dependencies "
            "(faster-whisper) before running transcription"
        )


def _parse_device(device: str) -> tuple[str, int]:
    """Split a ``cuda:N`` style device into CTranslate2's device + index."""

    name, _, index = device.partition(":")
    if not index:
        return device, 0
    if not index.isdigit():
        raise ModelInitializationError(
            f"DEVICE must look like 'cpu', 'cuda' or 'cuda:N', got {device!r}"
        )
    return name, int(index)


def _load_model(settings: Settings) -> Any:
    """Load the configured faster-whisper model onto the configured device."""

    _require_runtime()

    device, device_index = _parse_device(settings.device)

    try:
        return WhisperModel(
            settings.transcription_model,
            device=device,
            device_index=device_index,
            compute_type=settings.transcription_compute_type,
            download_root=str(settings.model_cache_dir),
        )
    except Exception as exc:
        raise ModelInitializationError(
            f"could not load the faster-whisper model "
            f"{settings.transcription_model!r} on device {settings.device!r} with "
            f"compute type {settings.transcription_compute_type!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _decode_audio(source: Path) -> Any:
    """Decode the whole file once into a 16 kHz mono float32 NumPy array."""

    try:
        audio = decode_audio(str(source), sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        raise AudioDecodeError(
            f"could not decode audio {source}: {type(exc).__name__}: {exc}"
        ) from exc

    if audio is None or len(audio) == 0:
        raise AudioDecodeError(f"audio file {source} contains no samples")
    return audio


def _resolve_language(model: Any, audio: Any, settings: Settings) -> str | None:
    """Return the language to transcribe in, detecting it once when unset.

    Detecting per region is unreliable: a single diarized turn is often only a
    couple of seconds long, far too short for a confident language decision. The
    detection therefore runs once over the decoded dialogue stem and is reused
    for every region. ``None`` still means "let Whisper decide per region", which
    is what we fall back to when detection yields nothing usable.
    """

    if settings.transcription_language:
        return settings.transcription_language

    try:
        detected = model.detect_language(audio)
    except Exception as exc:
        raise TranscriptionInferenceError(
            f"faster-whisper could not detect the source language: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    language = detected[0] if isinstance(detected, tuple) and detected else detected
    return language if isinstance(language, str) and language.strip() else None


def _region_samples(audio: Any, region: SpeakerSegment) -> Any:
    """Return the decoded samples covering ``region`` (possibly empty).

    Floor/ceil are used so the slice can never cut a sample away from the edges
    of the diarized turn.
    """

    total = int(len(audio))
    start = min(max(int(math.floor(region.start * SAMPLE_RATE)), 0), total)
    end = min(max(int(math.ceil(region.end * SAMPLE_RATE)), 0), total)
    if end <= start:
        return audio[0:0]
    return audio[start:end]


def _whisper_timestamp(value: object, field: str) -> float:
    """Return a Whisper timestamp as a finite ``float``, or fail clearly."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedOutputError(
            f"faster-whisper returned a non-numeric {field}: {value!r}"
        )

    seconds = float(value)
    if not math.isfinite(seconds):
        raise UnsupportedOutputError(
            f"faster-whisper returned a non-finite {field}: {value!r}"
        )
    return seconds


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into ``[lower, upper]``."""

    return min(max(value, lower), upper)


def _convert(
    raw_segments: Iterable[Any],
    region: SpeakerSegment,
) -> list[TranscriptSegment]:
    """Turn Whisper's region-local segments into absolute transcript segments."""

    produced: list[TranscriptSegment] = []
    for raw in raw_segments:
        text = raw.text if isinstance(raw.text, str) else ""
        text = text.strip()
        if not text:
            continue  # silence or noise: nothing was actually said here

        local_start = _whisper_timestamp(raw.start, "start")
        local_end = _whisper_timestamp(raw.end, "end")

        # Whisper timestamps are relative to the region: move them back onto the
        # movie timeline, then keep them inside the diarized turn.
        absolute_start = _clamp(region.start + local_start, region.start, region.end)
        absolute_end = _clamp(region.start + local_end, region.start, region.end)
        if absolute_end <= absolute_start:
            continue  # degenerate turn, nothing usable to keep

        produced.append(
            TranscriptSegment(
                speaker_id=region.speaker_id,
                start=absolute_start,
                end=absolute_end,
                text=text,
            )
        )

    return produced


def _transcribe_region(
    model: Any,
    samples: Any,
    region: SpeakerSegment,
    language: str | None,
) -> list[TranscriptSegment]:
    """Transcribe one diarized region and return absolute transcript segments."""

    try:
        # ``transcribe`` is lazy, so the iterator is materialised inside the try
        # to make sure inference failures surface here and are wrapped.
        raw_segments, _info = model.transcribe(
            samples,
            language=language,
            task="transcribe",  # this stage never translates
            word_timestamps=WORD_TIMESTAMPS,
        )
        raw_segments = list(raw_segments)
    except Exception as exc:
        raise TranscriptionInferenceError(
            f"faster-whisper failed for speaker {region.speaker_id} "
            f"[{region.start:.2f}s - {region.end:.2f}s]: {type(exc).__name__}: {exc}"
        ) from exc

    return _convert(raw_segments, region)


def build_amharic_transcriber(
    settings: Settings | None = None,
    *,
    language: str = "am",
) -> Callable[[Path], str]:
    """Return a callable that transcribes one audio file as Amharic.

    This is the model half of :func:`app.pipeline.qc.measure_pronunciation`, which
    stays free of any model dependency by taking a callable. Handing it this one turns
    the quality report's pronunciation term from "not measured" into a measurement of
    whether the delivered audio says what the text says.

    It is the only automated check that can see content the *speech model* invented -
    an extra word at the start of a line, a fragment of the English prompt the voice was
    cloned from, or a mispronunciation. Comparing the audio with the film's original, as
    the bleed measurements do, cannot find any of those: they are not the original.

    The language is fixed rather than detected. Detection on a single short line is
    unreliable, and the answer is known - the text handed to the engine was Amharic.

    Notes
    -----
    The model is loaded once, on the first call, and reused. Failures return an empty
    string so that one unreadable line is counted as unmeasured rather than ending a run.
    """

    resolved = settings if settings is not None else get_settings()
    model: Any | None = None
    lock = threading.Lock()

    def transcribe(path: Path) -> str:
        nonlocal model
        with lock:
            if model is None:
                model = _load_model(resolved)
            current = model
        try:
            segments, _info = current.transcribe(
                str(path), language=language, beam_size=1, vad_filter=False
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            return ""

    return transcribe


def transcribe(
    audio_path: str | Path,
    speaker_segments: Iterable[SpeakerSegment],
    *,
    settings: Settings | None = None,
) -> list[TranscriptSegment]:
    """Transcribe the speech of each diarized region of ``audio_path``.

    Parameters
    ----------
    audio_path:
        Dialogue audio (typically the ``*_speech.wav`` stem written by
        :func:`app.pipeline.separation.separate_stems`).
    speaker_segments:
        The turns returned by :func:`app.pipeline.diarization.diarize`. Each one
        is transcribed independently and keeps its speaker id verbatim.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        The model, compute type, device, model cache, and language all come from
        here - nothing is hard-coded in this module.

    Returns
    -------
    list[TranscriptSegment]
        Chronologically sorted utterances. An empty list means either nothing was
        diarized or nothing intelligible was found.

    Raises
    ------
    MissingInputError, InvalidInputError
        The input is not an existing regular file.
    InvalidSpeakerSegmentsError
        ``speaker_segments`` is not an iterable of ``SpeakerSegment``.
    AudioDecodeError
        The input could not be decoded into audio samples.
    ModelInitializationError
        faster-whisper is missing, or the model could not be loaded on the
        configured device.
    TranscriptionInferenceError
        faster-whisper failed while transcribing.
    UnsupportedOutputError
        faster-whisper returned something this module cannot convert.
    """

    settings = settings if settings is not None else get_settings()

    source = _validate_audio_path(audio_path)
    regions = _validate_speaker_segments(speaker_segments)
    if not regions:
        # Nothing was diarized, so there is nothing to transcribe. Returning
        # early also avoids loading a multi-gigabyte model for nothing.
        return []

    _require_runtime()
    audio = _decode_audio(source)
    model = _load_model(settings)
    language = _resolve_language(model, audio, settings)

    transcribed: list[TranscriptSegment] = []
    for region in regions:
        if region.end - region.start < MIN_REGION_SECONDS:
            # A diarization artefact: too short to hold a word, and short enough
            # to break faster-whisper's word alignment (see MIN_REGION_SECONDS).
            continue
        samples = _region_samples(audio, region)
        if len(samples) == 0:
            continue  # the turn lies outside the decoded audio
        transcribed.extend(_transcribe_region(model, samples, region, language))

    transcribed.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return transcribed


__all__ = [
    "MIN_REGION_SECONDS",
    "SAMPLE_RATE",
    "WORD_TIMESTAMPS",
    "AudioDecodeError",
    "InvalidInputError",
    "InvalidSegmentError",
    "InvalidSpeakerSegmentsError",
    "MissingInputError",
    "ModelInitializationError",
    "TranscriptSegment",
    "TranscriptionError",
    "TranscriptionInferenceError",
    "UnsupportedOutputError",
    "build_amharic_transcriber",
    "transcribe",
]
