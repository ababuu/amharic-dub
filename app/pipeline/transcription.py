"""Speech transcription with **faster-whisper** ``large-v3``.

This module is the only place in the pipeline that talks to ``faster_whisper``.
It consumes the dialogue stem written by :mod:`app.pipeline.separation` together
with the speaker turns produced by :mod:`app.pipeline.diarization`, and returns
one :class:`TranscriptSegment` per transcribed utterance::

    speech.wav + [SpeakerSegment, ...] -> transcribe() -> [TranscriptSegment, ...]

Each diarized region is transcribed on its own, so every line is attributed to
the speaker who actually said it instead of being guessed afterwards from a
whole-movie transcript.

Those segments are pause-delimited, not sentence-delimited, so
:func:`assemble_sentences` turns them into the :class:`DialogueSentence` units the
rest of the pipeline adapts and speaks. A sentence keeps every timestamp of its
parts, and the parts stay available for the performance metadata that genuinely
changes inside one thought.

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

import difflib
import io
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.prosody import ProsodyError, estimate_f0

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

#: Shortest stretch of a crosstalk region that is worth listening to for the speaker
#: the exclusive view dropped. Below this the audio is a syllable or two of a
#: mixture: no word fits, and any text a decoder produces there is invented.
MIN_RECOVERED_OVERLAP_SECONDS = 0.30

#: How much of the recovered text has to match text already credited to another
#: speaker in the same crosstalk region before it is treated as *the same speech
#: heard twice*. A crosstalk region is one mixture, so a word-level assignment can
#: legitimately recover only part of it; anything the other speaker already has is
#: a duplicate, and emitting it again would have two characters say one line.
RECOVERED_DUPLICATE_RATIO = 0.6

#: Words from the recovered stretch shorter than this carry no evidence on their own
#: (a stray token Whisper emits between two real words).
MIN_RECOVERED_WORD_SECONDS = 0.02

#: Whisper's own hallucination heuristics, applied to a *recovered* line only.
#:
#: A recovered line is decoded from a mixture that the film itself made unintelligible,
#: which is exactly the input Whisper fills in from its language model. On the run that
#: prompted this it produced "RUN!" from 0.24s of two people talking at once - text
#: nobody said, in a character's voice, inside a scene. The three thresholds are the
#: ones Whisper's own reference implementation uses to spot that: a segment the model
#: considered mostly non-speech, one it was not confident about, or one whose text
#: compresses implausibly well (a repetition loop). They are applied only here, so a
#: normal line that happens to be quiet is never discarded.
MIN_RECOVERED_AVG_LOGPROB = -1.0
MAX_RECOVERED_NO_SPEECH = 0.6
MAX_RECOVERED_COMPRESSION = 2.4

#: How much pitched speech a recovered stretch needs before its register can be used to
#: check *who* is speaking in it. The same floor the register stage uses for a turn:
#: below it autocorrelation reports the room rather than the speaker, and the check
#: would be a coin flip. A stretch that cannot be checked is not recovered.
MIN_RECOVERED_VOICED_FRAMES = 8


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


def _validate_crosstalk(crosstalk: Iterable[CrosstalkRegion]) -> list[CrosstalkRegion]:
    """Return the crosstalk regions as a chronological list, validated.

    An empty input is normal: a pyannote result without an overlap-aware annotation
    reports no crosstalk, and a run then behaves exactly as it did before recovery
    existed.
    """

    if isinstance(crosstalk, (str, bytes)) or not isinstance(crosstalk, Iterable):
        raise InvalidSpeakerSegmentsError(
            "crosstalk must be an iterable of CrosstalkRegion objects, got "
            f"{type(crosstalk).__name__}"
        )

    regions = list(crosstalk)
    for index, region in enumerate(regions):
        if not isinstance(region, CrosstalkRegion):
            raise InvalidSpeakerSegmentsError(
                f"crosstalk[{index}] is {type(region).__name__}, "
                "expected a CrosstalkRegion"
            )
    return sorted(regions, key=lambda item: (item.start, item.end))


def _validate_overlap_turns(
    overlap_turns: Iterable[SpeakerSegment],
) -> list[SpeakerSegment]:
    """Return the overlap-aware turns as a chronological list, validated."""

    if isinstance(overlap_turns, (str, bytes)) or not isinstance(
        overlap_turns, Iterable
    ):
        raise InvalidSpeakerSegmentsError(
            "overlap_turns must be an iterable of SpeakerSegment objects, got "
            f"{type(overlap_turns).__name__}"
        )

    turns = list(overlap_turns)
    for index, turn in enumerate(turns):
        if not isinstance(turn, SpeakerSegment):
            raise InvalidSpeakerSegmentsError(
                f"overlap_turns[{index}] is {type(turn).__name__}, "
                "expected a SpeakerSegment"
            )
    return sorted(turns, key=lambda item: (item.start, item.end, item.speaker_id))


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

    return _convert(_decode(model, samples, region, language), region)


# ---------------------------------------------------------------------------
# Speech the exclusive view dropped
# ---------------------------------------------------------------------------


def _overlap_seconds(
    start: float, end: float, other_start: float, other_end: float
) -> float:
    """Return how much of ``start``..``end`` falls inside another interval."""

    return max(0.0, min(end, other_end) - max(start, other_start))


def _clip_span(
    start: float, end: float, lower: float, upper: float
) -> tuple[float, float] | None:
    """Return ``start``..``end`` intersected with ``lower``..``upper``, or ``None``."""

    clipped_start = max(start, lower)
    clipped_end = min(end, upper)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _subtract_spans(
    spans: Iterable[tuple[float, float]],
    covered: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the parts of ``spans`` that no interval in ``covered`` covers.

    Used to find the audio of a dropped speaker that the exclusive view did not
    already transcribe for them: a speaker can be missing from *part* of a
    crosstalk while having a turn of their own elsewhere in it.
    """

    remaining: list[tuple[float, float]] = []
    covered_list = list(covered)
    for start, end in spans:
        pieces = [(start, end)]
        for cover_start, cover_end in covered_list:
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if cover_end <= piece_start or cover_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < cover_start:
                    next_pieces.append((piece_start, cover_start))
                if cover_end < piece_end:
                    next_pieces.append((cover_end, piece_end))
            pieces = next_pieces
        remaining.extend(pieces)
    remaining.sort()
    return remaining


def _speaker_spans(
    speaker_id: str,
    region: CrosstalkRegion,
    overlap_turns: Iterable[SpeakerSegment],
) -> list[tuple[float, float]]:
    """Return when ``speaker_id`` was speaking inside ``region``.

    The overlap-aware turns are the only place this is recorded: the exclusive
    turns cannot say it, because that view had to give the instant to someone
    else. When the annotation has no turn for the speaker inside this region the
    whole region is used, because the region itself already asserts they were
    talking - refusing to look would lose the words the region exists to find.
    """

    spans: list[tuple[float, float]] = []
    for turn in overlap_turns:
        if turn.speaker_id != speaker_id:
            continue
        clipped = _clip_span(turn.start, turn.end, region.start, region.end)
        if clipped is not None:
            spans.append(clipped)
    if not spans:
        return [(region.start, region.end)]
    return spans


def _plausible_speech(raw: Any) -> bool:
    """Return whether a decoded segment looks like speech rather than a guess.

    Only used on recovered crosstalk audio, which is the one place this stage decodes a
    mixture the film itself made unintelligible. A segment that carries none of the
    three diagnostics is accepted: a decoder that does not report them is not evidence
    against itself.
    """

    average = getattr(raw, "avg_logprob", None)
    if isinstance(average, (int, float)) and not isinstance(average, bool):
        if float(average) < MIN_RECOVERED_AVG_LOGPROB:
            return False

    silent = getattr(raw, "no_speech_prob", None)
    if isinstance(silent, (int, float)) and not isinstance(silent, bool):
        if float(silent) > MAX_RECOVERED_NO_SPEECH:
            return False

    compression = getattr(raw, "compression_ratio", None)
    if isinstance(compression, (int, float)) and not isinstance(compression, bool):
        if float(compression) > MAX_RECOVERED_COMPRESSION:
            return False

    return True


def _timed_words(
    raw_segments: Iterable[Any], region: SpeakerSegment
) -> list[tuple[str, float, float]]:
    """Return ``(word, start, end)`` on the movie timeline for one decode.

    Word timings are what make a mixture separable: two speakers inside one
    crosstalk can only be told apart by *when* each word was spoken. A segment that
    carries no words falls back to its own text over its own span, so a decoder
    configured without word timestamps still contributes something.
    """

    words: list[tuple[str, float, float]] = []
    for raw in raw_segments:
        if not _plausible_speech(raw):
            continue
        entries = getattr(raw, "words", None)
        if entries:
            for entry in entries:
                text = getattr(entry, "word", "")
                text = text.strip() if isinstance(text, str) else ""
                if not text:
                    continue
                local_start = _whisper_timestamp(getattr(entry, "start", None), "start")
                local_end = _whisper_timestamp(getattr(entry, "end", None), "end")
                start = _clamp(region.start + local_start, region.start, region.end)
                end = _clamp(region.start + local_end, region.start, region.end)
                if end - start < MIN_RECOVERED_WORD_SECONDS:
                    continue
                words.append((text, start, end))
            continue

        text = raw.text if isinstance(raw.text, str) else ""
        text = text.strip()
        if not text:
            continue
        local_start = _whisper_timestamp(raw.start, "start")
        local_end = _whisper_timestamp(raw.end, "end")
        start = _clamp(region.start + local_start, region.start, region.end)
        end = _clamp(region.start + local_end, region.start, region.end)
        if end <= start:
            continue
        words.append((text, start, end))
    return words


def _assign_words(
    words: Iterable[tuple[str, float, float]],
    spans: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[str, float, float]]]:
    """Give every word to the speaker whose own turns cover most of it.

    ``spans`` holds only the speakers being recovered. That is deliberate: the
    speaker the exclusive view kept already has a text that came from a clean decode
    of their own turn, so a word in the mixture belongs to them only if their own
    line does not already contain it - which the caller checks, because comparing
    words is more reliable than comparing spans inside one region, where every span
    covers the whole region by construction.

    A word is short, so its span is the strongest evidence available about who said
    it: the speaker diarization recorded as active through it. Ties - a boundary
    where both were talking - are broken by speaker id, so the same input always
    produces the same assignment.
    """

    assigned: dict[str, list[tuple[str, float, float]]] = {name: [] for name in spans}
    for text, start, end in words:
        best_speaker: str | None = None
        best_cover = 0.0
        for speaker_id in sorted(spans):
            cover = sum(
                _overlap_seconds(start, end, span_start, span_end)
                for span_start, span_end in spans[speaker_id]
            )
            if cover > best_cover:
                best_speaker, best_cover = speaker_id, cover
        if best_speaker is not None and best_cover > 0.0:
            assigned[best_speaker].append((text, start, end))
    return assigned


def _words_to_segments(
    speaker_id: str, words: Sequence[tuple[str, float, float]]
) -> list[TranscriptSegment]:
    """Join one speaker's recovered words into contiguous transcript segments."""

    segments: list[TranscriptSegment] = []
    run: list[tuple[str, float, float]] = []
    for entry in words:
        if run and entry[1] - run[-1][2] > MAX_SENTENCE_GAP_SECONDS:
            segments.append(_run_to_segment(speaker_id, run))
            run = []
        run.append(entry)
    if run:
        segments.append(_run_to_segment(speaker_id, run))
    return [segment for segment in segments if segment is not None]


def _run_to_segment(
    speaker_id: str, run: Sequence[tuple[str, float, float]]
) -> TranscriptSegment | None:
    """Return one transcript segment for a run of recovered words, or ``None``."""

    text = " ".join(word for word, _start, _end in run).strip()
    if not text:
        return None
    start = min(entry[1] for entry in run)
    end = max(entry[2] for entry in run)
    if end <= start:
        return None
    try:
        return TranscriptSegment(speaker_id=speaker_id, start=start, end=end, text=text)
    except InvalidSegmentError:
        return None


def _normalise_for_comparison(text: str) -> str:
    """Return ``text`` lowercased with punctuation and spacing reduced to words."""

    letters = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(letters.lower().split())


def _duplicates(candidate: str, existing: Iterable[str]) -> bool:
    """Return whether ``candidate`` restates text already credited in the region."""

    normalised = _normalise_for_comparison(candidate)
    if not normalised:
        return True

    for text in existing:
        other = _normalise_for_comparison(text)
        if not other:
            continue
        if normalised == other:
            return True
        # One line wholly inside the other is the same speech, not new speech.
        if normalised in other or other in normalised:
            return True
        if difflib.SequenceMatcher(None, normalised, other).ratio() >= RECOVERED_DUPLICATE_RATIO:
            return True
    return False


def _register_of(
    registers: Mapping[str, float] | None, speaker_id: str
) -> float | None:
    """Return a speaker's median pitch in Hz, when it is known."""

    if not registers:
        return None
    value = registers.get(speaker_id)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        median = float(value)
        if math.isfinite(median) and median > 0:
            return median
    return None


def _sounds_like(
    samples: Any,
    rate: int,
    *,
    speaker_id: str,
    other_id: str,
    registers: Mapping[str, float] | None,
) -> bool:
    """Return whether this audio measures as ``speaker_id`` rather than ``other_id``.

    This is the check that stops a recovered line from being voiced by the character
    who was not talking. Whisper decodes a mixture to whichever voice dominates it, so
    a re-decode of the overlap usually returns the *kept* speaker's words - and handing
    those to the other character is heard as one character saying the other's line in
    the wrong voice, which is the defect this whole pass exists to avoid.

    Decisive means: enough pitched speech to measure, and the measurement falls nearer
    the missing speaker's own register than the kept speaker's. Anything less - too
    little voice, or a reading that sits between the two - is *not* evidence, and a
    line is not created from it. Leaving an unintelligible interjection out is a smaller
    defect than putting words in the wrong character's mouth, and the quality report
    names the region either way.
    """

    wanted = _register_of(registers, speaker_id)
    avoided = _register_of(registers, other_id)
    if wanted is None or avoided is None or wanted == avoided:
        # Without two known registers there is nothing to compare against, so the
        # acoustic gate cannot add anything and the structural checks stand alone.
        return True

    try:
        profile = estimate_f0(samples, rate)
    except ProsodyError:
        return False
    if profile.median_hz is None or profile.voiced_frames < MIN_RECOVERED_VOICED_FRAMES:
        return False

    measured = float(profile.median_hz)
    return abs(measured - wanted) < abs(measured - avoided)


def _recover_dropped_speakers(
    model: Any,
    audio: Any,
    regions: Sequence[SpeakerSegment],
    crosstalk: Sequence[CrosstalkRegion],
    overlap_turns: Sequence[SpeakerSegment],
    language: str | None,
    transcribed: Sequence[TranscriptSegment],
    registers: Mapping[str, float] | None = None,
) -> list[TranscriptSegment]:
    """Return lines for speakers the exclusive view dropped inside crosstalk.

    The exclusive diarization gives every instant to exactly one speaker, so when
    two people talk at once one of them loses the turn - and with it everything they
    said. Downstream that is heard as a scene where only one character ever speaks.

    Recovering it is only worth doing when the recovered words can be *attributed*.
    A mixture is decoded once, and Whisper returns whichever voice dominates it, so
    the naive version of this - re-decode the overlap, hand the text to the missing
    speaker - put the kept speaker's own words into the other character's mouth, in the
    other character's voice. That is worse than the silence it replaces, so a recovery
    has to clear three gates:

    1. **Exclusive time.** The missing speaker must have been active inside the region
       for at least :data:`MIN_RECOVERED_OVERLAP_SECONDS` beyond what their own turns
       already cover. A region where they had nothing of their own has nothing to
       recover.
    2. **The audio belongs to them.** The decoded stretch must measure nearer the
       missing speaker's register than the kept speaker's, with enough pitched speech
       for the comparison to mean something. A fragment too short to measure is not
       evidence, so it is refused rather than guessed at.
    3. **It is not what the other speaker already said.** A mixture transcribed twice
       tends to reproduce the dominant voice, and two characters saying one line is
       also wrong.

    A region the exclusive view did not fill at all is skipped entirely: with no
    authoritative text there is nothing to check a recovery against.
    """

    if not overlap_turns:
        return []

    recovered: list[TranscriptSegment] = []
    for region in crosstalk:
        if region.duration < MIN_RECOVERED_OVERLAP_SECONDS:
            continue

        inside = [
            segment
            for segment in transcribed
            if _overlap_seconds(
                segment.start, segment.end, region.start, region.end
            )
            > 0.0
        ]
        if not inside:
            # The exclusive view produced nothing here at all, so there is no
            # authoritative text to compare a recovery against and no evidence about
            # which of the two speakers a word belongs to. Guessing would invent
            # dialogue; the gap is reported instead by the quality report.
            continue
        present = {segment.speaker_id for segment in inside}
        missing = [
            speaker_id
            for speaker_id in region.speaker_ids
            if speaker_id not in present
        ]
        if not missing:
            continue
        spans = {
            speaker_id: _speaker_spans(speaker_id, region, overlap_turns)
            for speaker_id in region.speaker_ids
        }
        # Gate 1: the missing speaker's own activity, minus the part their *own*
        # exclusive turns already cover - a speaker can be missing from one stretch of a
        # crosstalk and present in another. The kept speaker's turn is deliberately not
        # subtracted: the exclusive view gives them the whole instant by construction, so
        # subtracting it would leave nothing at every genuine overlap and recovery could
        # never happen. Whether the decoded audio is really the missing speaker's is
        # decided by the acoustic gate below instead.
        already_theirs = [
            clipped
            for turn in regions
            if turn.speaker_id in missing
            for clipped in [_clip_span(turn.start, turn.end, region.start, region.end)]
            if clipped is not None
        ]
        uncovered = _subtract_spans(
            [span for speaker_id in missing for span in spans[speaker_id]],
            already_theirs,
        )
        if not uncovered:
            continue

        region_text = [segment.text for segment in inside]
        for start, end in uncovered:
            if end - start < MIN_RECOVERED_OVERLAP_SECONDS:
                continue
            window = SpeakerSegment(
                speaker_id=missing[0], start=start, end=end
            )
            samples = _region_samples(audio, window)
            if len(samples) == 0:
                continue
            if not _sounds_like(
                samples,
                SAMPLE_RATE,
                speaker_id=missing[0],
                other_id=sorted(present)[0],
                registers=registers,
            ):
                continue
            words = _timed_words(_decode(model, samples, window, language), window)
            if not words:
                continue
            # Every word in this stretch was spoken while nobody else was, so it is
            # the missing speaker's by construction - no timing contest is needed.
            for segment in _words_to_segments(missing[0], words):
                if _duplicates(segment.text, region_text):
                    continue
                recovered.append(segment)
                region_text.append(segment.text)

    return recovered


def _decode(
    model: Any, samples: Any, region: SpeakerSegment, language: str | None
) -> list[Any]:
    """Run one faster-whisper decode and materialise its segments."""

    try:
        raw_segments, _info = model.transcribe(
            samples,
            language=language,
            task="transcribe",
            word_timestamps=WORD_TIMESTAMPS,
        )
        return list(raw_segments)
    except Exception as exc:
        raise TranscriptionInferenceError(
            f"faster-whisper failed for speaker {region.speaker_id} "
            f"[{region.start:.2f}s - {region.end:.2f}s]: {type(exc).__name__}: {exc}"
        ) from exc


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
    crosstalk: Iterable[CrosstalkRegion] = (),
    overlap_turns: Iterable[SpeakerSegment] = (),
    registers: Mapping[str, float] | None = None,
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
    crosstalk:
        Regions where two or more speakers talk at once, from
        :func:`app.pipeline.diarization.diarize_detailed`. Inside one, the
        *exclusive* turn list can only hold one speaker, so the other speaker's
        words are absent from the transcript entirely - and a dub built from that
        transcript has one character talking while another one visibly is. Each
        region is therefore re-listened to once for the speakers the exclusive
        view dropped; see :func:`_recover_dropped_speakers`.
    overlap_turns:
        The overlap-aware turns from the same call. They say *when* each speaker
        was talking inside a crosstalk, which is the only way to tell two voices in
        one mixture apart. Without them nothing can be recovered, so a result that
        carries no overlap-aware annotation simply never recovers anything - it is
        not an error.
    registers:
        Each speaker's median pitch in Hz, when it has been measured. Recovery uses it
        to check that the audio it is about to credit to one character actually sounds
        like that character: a mixture decoded once returns whichever voice dominates
        it, and attributing that to the other speaker is heard as one character saying
        the other's line. Without registers the check cannot run, and the structural
        checks stand alone.

    Returns
    -------
    list[TranscriptSegment]
        Chronologically sorted utterances. An empty list means either nothing was
        diarized or nothing intelligible was found. Recovered simultaneous speech
        is included, attributed to the speaker who was actually saying it, so the
        timeline downstream holds two characters where the film had two.

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

    recovered = _recover_dropped_speakers(
        model,
        audio,
        regions,
        _validate_crosstalk(crosstalk),
        _validate_overlap_turns(overlap_turns),
        language,
        transcribed,
        registers,
    )
    if recovered:
        transcribed.extend(recovered)
        transcribed.sort(
            key=lambda segment: (segment.start, segment.end, segment.speaker_id)
        )
    return transcribed


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

#: Longest pause between two transcribed segments that can still belong to the
#: same sentence, in seconds.
#:
#: Whisper segments on pauses, not on grammar, so one spoken sentence routinely
#: arrives as several segments. A pause longer than this is treated as the end of
#: the thought whatever the words look like, so a beat in the scene is never
#: swallowed into one run-on utterance.
MAX_SENTENCE_GAP_SECONDS = 2.0

#: A pause at least this long, *together with* terminal punctuation, is a real
#: boundary. Below it a full stop is the transcriber's habit rather than the
#: speaker's intent: Whisper punctuates every pause-delimited fragment, so
#: punctuation alone split "No, you clearly." from "Don't know who you're talking
#: to" - one sentence, one emotion, spoken as two clips. Measured on a real scene,
#: 1.2s turned 41 fragments into 20 lines whose longest is 11.6s (under the cap),
#: where the old punctuation rule produced 38 lines of which most were fragments.
SENTENCE_BREAK_PAUSE_SECONDS = 1.2

#: Shortest completed sentence worth breaking on. A fragment shorter than this
#: absorbs what follows rather than becoming a line of its own, because a
#: 0.2-second clip cannot carry an intonation contour and is heard as a cut.
MIN_SENTENCE_SECONDS = 0.8

#: Longest span one assembled sentence may cover, and the largest number of
#: transcribed segments it may join. Both are safety valves for a transcript with
#: sparse punctuation: without them a monologue would collapse into one unit that
#: no longer fits the timing model.
MAX_SENTENCE_SECONDS = 12.0
MAX_SENTENCE_PARTS = 6

#: Characters that end a sentence in the languages this pipeline transcribes.
_TERMINAL_PUNCTUATION = (".", "!", "?", "\u2026", "\u1362", "\u1367")

#: Decoration that can sit outside a sentence's own punctuation.
_WRAPPING = "\"'\u201d\u2019\u2018\u201c\u00ab\u00bb()[]"


def _without_wrapping(text: str) -> str:
    """Return ``text`` stripped of surrounding whitespace and quotation marks."""

    return text.strip().strip(_WRAPPING)


def _ends_sentence(text: str) -> bool:
    """Return whether ``text`` already ends with terminal punctuation."""

    stripped = _without_wrapping(text)
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCTUATION


def _starts_mid_sentence(text: str) -> bool:
    """Return whether ``text`` begins as a continuation rather than a new sentence.

    A lowercase first letter is what a transcriber writes when it believes the
    previous segment was still the same sentence, so it is the one cue that
    survives even when every segment carries a full stop.
    """

    stripped = text.strip().lstrip(_WRAPPING)
    return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()


def _last_written_index(text: str) -> int:
    """Return the index of the last character that is not wrapping decoration."""

    index = len(text) - 1
    while index >= 0 and text[index] in _WRAPPING:
        index -= 1
    return index


def _join_sentence_parts(parts: Sequence[TranscriptSegment]) -> str:
    """Return the text of ``parts`` as one sentence.

    The boundary between two parts of one sentence is not a sentence end, so a
    full stop a transcriber put there becomes a comma. Question and exclamation
    marks are left alone: they carry the speaker's intent, not the segmenter's
    opinion.
    """

    joined: list[str] = []
    for part in parts:
        text = part.text.strip()
        if joined and _starts_mid_sentence(text):
            previous = joined[-1]
            index = _last_written_index(previous)
            if index >= 0 and previous[index] == ".":
                joined[-1] = previous[:index] + "," + previous[index + 1 :]
        joined.append(text)
    return " ".join(joined)


@dataclass(frozen=True, slots=True)
class DialogueSentence:
    """One complete spoken thought, assembled from transcribed segments.

    A :class:`TranscriptSegment` is a *pause-delimited* unit, not a sentence, and
    the difference is audible: translating and synthesizing each segment on its
    own turns one thought into several utterances, each with its own final
    intonation and its own full stop. This type is the sentence.

    Its ``parts`` are kept because the performance genuinely varies inside a
    sentence - a line can start neutral and end accusing - and that is metadata
    the dialogue model reports per part. ``start`` and ``end`` span the first and
    last part, so the original timing is preserved exactly; the gaps between parts
    are part of the span and are what the sentence may use to be spoken.
    """

    speaker_id: str
    start: float
    end: float
    text: str
    parts: tuple[TranscriptSegment, ...]

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

        parts = tuple(self.parts)
        if not parts:
            raise InvalidSegmentError("a sentence needs at least one part")
        for index, part in enumerate(parts):
            if not isinstance(part, TranscriptSegment):
                raise InvalidSegmentError(
                    f"parts[{index}] is {type(part).__name__}, expected a TranscriptSegment"
                )
            if part.speaker_id != self.speaker_id:
                raise InvalidSegmentError(
                    f"parts[{index}] belongs to {part.speaker_id!r}, not {self.speaker_id!r}"
                )

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "parts", parts)

    @property
    def duration(self) -> float:
        """Length of the sentence's span in seconds, gaps included."""

        return self.end - self.start


def assemble_sentences(
    segments: Iterable[TranscriptSegment],
) -> list[DialogueSentence]:
    """Group transcribed segments into complete sentences.

    The rules are deliberately textual first and temporal second, because a pause
    is a performance choice while a sentence boundary is a grammatical one:

    * segments of different speakers never join;
    * a segment that does not end with terminal punctuation continues;
    * a segment whose text starts mid-sentence (lowercase) continues, which is
      how a transcriber marks "the same sentence, split at a pause";
    * otherwise the sentence ends;
    * a pause longer than :data:`MAX_SENTENCE_GAP_SECONDS`, a span longer than
      :data:`MAX_SENTENCE_SECONDS` or more than :data:`MAX_SENTENCE_PARTS` parts
      ends it regardless.

    The result is chronological and keeps every timestamp of the input: the
    first part's ``start`` and the last part's ``end`` become the sentence's.
    """

    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise InvalidSegmentError(
            "segments must be an iterable of TranscriptSegment objects, got "
            f"{type(segments).__name__}"
        )

    ordered = list(segments)
    for index, segment in enumerate(ordered):
        if not isinstance(segment, TranscriptSegment):
            raise InvalidSegmentError(
                f"segments[{index}] is {type(segment).__name__}, expected a TranscriptSegment"
            )
    ordered.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))

    sentences: list[DialogueSentence] = []
    parts: list[TranscriptSegment] = []

    def flush() -> None:
        if parts:
            sentences.append(
                DialogueSentence(
                    speaker_id=parts[0].speaker_id,
                    start=parts[0].start,
                    end=parts[-1].end,
                    text=_join_sentence_parts(parts),
                    parts=tuple(parts),
                )
            )
            parts.clear()

    for segment in ordered:
        if not parts:
            parts.append(segment)
            continue
        previous = parts[-1]
        gap = segment.start - previous.end
        # A boundary needs both signals: the transcriber said the sentence ended
        # *and* the speaker actually paused, and what has accumulated is already a
        # sentence rather than a stray fragment.
        boundary = (
            _ends_sentence(previous.text)
            and gap >= SENTENCE_BREAK_PAUSE_SECONDS
            and previous.end - parts[0].start >= MIN_SENTENCE_SECONDS
        )
        continues = (
            segment.speaker_id == previous.speaker_id
            and gap <= MAX_SENTENCE_GAP_SECONDS
            and len(parts) < MAX_SENTENCE_PARTS
            and segment.end - parts[0].start <= MAX_SENTENCE_SECONDS
            and not boundary
        )
        if continues:
            parts.append(segment)
        else:
            flush()
            parts.append(segment)
    flush()
    return sentences


__all__ = [
    "MAX_SENTENCE_GAP_SECONDS",
    "MAX_SENTENCE_PARTS",
    "MAX_SENTENCE_SECONDS",
    "MIN_REGION_SECONDS",
    "SAMPLE_RATE",
    "WORD_TIMESTAMPS",
    "AudioDecodeError",
    "DialogueSentence",
    "InvalidInputError",
    "InvalidSegmentError",
    "InvalidSpeakerSegmentsError",
    "MissingInputError",
    "ModelInitializationError",
    "TranscriptSegment",
    "TranscriptionError",
    "TranscriptionInferenceError",
    "UnsupportedOutputError",
    "assemble_sentences",
    "build_amharic_transcriber",
    "transcribe",
]
