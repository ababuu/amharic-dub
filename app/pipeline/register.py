"""Who is speaking, measured from the voice itself.

Why this module exists
----------------------
Diarization answers "which turns belong together?" from speaker embeddings, and on
real film audio it is usually right about the *clusters* while being wrong about a
few individual turns. That minority is what breaks a dub: the reference recording
for a character is cut from one of their turns, so a single mislabelled turn is
enough to hand Walter a female voice for the whole film.

A voice's register - the fundamental frequency it sits at - is an independent
signal that costs nothing to measure. On the film that prompted this module the
clusters were 200 Hz and 120 Hz apart, a 1.66x separation, while eight of the
forty-one turns sat clearly on the wrong side of that gap. Re-measuring a few
outliers and reporting the separation turns a silent casting error into either a
correct cast or an explicit "this is not two people".

What this is not
----------------
* **Not a diarizer.** It never invents speakers, never merges clusters and never
  changes how many there are. It only reassigns a turn that its own cluster cannot
  explain and another cluster plainly can.
* **Not a gender classifier.** The words are ``masculine`` and ``feminine`` for a
  *register*, because pitch is all that is measured. A voice that sits between the
  two bands is reported ``uncertain`` rather than guessed, and casting then prefers
  an unused voice over a wrong one.
* **Not a model.** It reuses :func:`app.pipeline.prosody.estimate_f0`, so it needs
  no GPU, no download, and no network, and it stays testable on a synthetic clip.

Contract
--------
Register is aggregated over *every* turn a speaker has, weighted by voiced audio,
rather than read from a single window. That is the fix for the defect that
motivated this module: one 5-second window can be one of the mislabelled turns,
while the weighted median of thirty turns cannot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from app.pipeline.prosody import PitchProfile, ProsodyError, estimate_f0
from app.pipeline.transcription import TranscriptSegment

#: Two intervals closer than this are not sharing time. Same value as the other
#: stages use for a timestamp comparison, so "overlapping" means one thing everywhere.
TIMESTAMP_TOLERANCE = 1e-3

#: Below this median a voice reads as masculine, in Hz. Set from measured film
#: dialogue: male turns clustered at 94-160 Hz and female turns at 171-300 Hz, so
#: the boundary sits in the empty band between them rather than in either cluster.
MASCULINE_MAX_HZ = 165.0

#: Above this median a voice reads as feminine, in Hz. The gap between the two
#: constants is deliberate: a voice inside it is reported ``uncertain``.
FEMININE_MIN_HZ = 175.0

#: A turn needs this much voiced audio before its pitch is worth using. Shorter
#: than a syllable or two, autocorrelation reports the room, not the speaker.
MIN_VOICED_FRAMES = 8

#: How far (as a ratio) a turn may sit from its speaker's register before it is
#: treated as an outlier rather than expressive delivery. 1.35 is roughly five
#: semitones, which is a large but real expressive range for one voice.
OUTLIER_RATIO = 1.35

#: How much closer another speaker's register has to explain a turn before the turn
#: is moved. Half the outlier distance, so a turn that sits *between* two clusters
#: is left alone instead of being assigned on a coin flip.
REASSIGNMENT_MARGIN = 0.5

#: Minimum separation between two speakers' registers before pitch can arbitrate
#: between them. Below this the two voices are too alike for a register test to be
#: evidence of anything, and the diarization labels are kept.
MIN_SEPARATION_RATIO = 1.25

#: Passes of measure-and-reassign. Each pass can move a turn, which changes the
#: registers the next pass compares against; two passes settle on the film this was
#: built for and the loop is bounded so a pathological input cannot spin.
MAX_PASSES = 3

#: Shortest turn whose pitch may move it to another speaker, in seconds.
#:
#: A pitch median from a fragment is not a measurement of a voice. Two things go wrong
#: below this length, and both were heard on a real film: autocorrelation can lock onto
#: twice the period and report an octave high, and a fragment cut from inside one
#: person's sentence is not an utterance at all, so there is no reason for it to be a
#: different speaker. Moving such a turn puts the other character's voice *inside* one
#: sentence - on the run that prompted this, five short fragments of a single
#: continuous monologue were handed to the other character. The threshold matches
#: :data:`app.pipeline.diarization`'s own verification floor, so a turn is only ever
#: moved by a measurement that is long enough for both stages to agree on.
MIN_REASSIGNMENT_TURN_SECONDS = 1.5

#: Longest turn that may be treated as a fragment of its neighbours, in seconds.
SHORT_TURN_SECONDS = 1.5

#: How close a fragment has to sit to one of its neighbours, in seconds, for that
#: neighbour to explain it. At least one side must be this close: a fragment glued to
#: the end of somebody's sentence is part of it, while an utterance separated from both
#: neighbours by a beat is a reply and keeps its own label even when it is short.
CONTINUITY_GAP_SECONDS = 1.0

#: How wide one speaker's own turns may spread, as a ratio, before the cluster is
#: reported as probably holding more than one voice. A single voice varies by a quarter
#: either way between a hushed line and a shouted one; 2.0x is a full octave across
#: per-utterance medians, which no single speaker produces and two do easily.
CLUSTER_SPREAD_LIMIT = 2.0

_PRESENTATIONS = ("unknown", "uncertain", "masculine", "feminine")


class RegisterError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidSegmentsError(RegisterError, ValueError):
    """The segments handed in are not transcript segments."""


class AudioReadError(RegisterError):
    """The dialogue audio could not be read."""


def presentation_of(median_hz: float | None) -> str:
    """Return the register a median fundamental frequency falls in.

    ``uncertain`` is a real answer, not a failure: film dialogue includes
    whispering, shouting and children, and a voice measured between the bands is
    better reported as unknown than filed under the wrong one.
    """

    if median_hz is None:
        return "unknown"
    if median_hz <= MASCULINE_MAX_HZ:
        return "masculine"
    if median_hz >= FEMININE_MIN_HZ:
        return "feminine"
    return "uncertain"


@dataclass(frozen=True, slots=True)
class TurnPitch:
    """One turn's measured pitch, with the weight it carries in an aggregate."""

    start: float
    end: float
    median_hz: float | None
    voiced_frames: int
    speech_seconds: float

    @property
    def usable(self) -> bool:
        """``True`` when this turn has enough voiced audio to describe a voice."""

        return self.median_hz is not None and self.voiced_frames >= MIN_VOICED_FRAMES


@dataclass(frozen=True, slots=True)
class SpeakerRegister:
    """A speaker's register, aggregated over every turn they speak."""

    speaker_id: str
    median_hz: float | None
    low_hz: float | None
    high_hz: float | None
    turns: int
    measured_turns: int
    speech_seconds: float

    @property
    def presentation(self) -> str:
        """Return ``masculine``, ``feminine``, ``uncertain`` or ``unknown``."""

        return presentation_of(self.median_hz)

    @property
    def confident(self) -> bool:
        """``True`` when enough turns were measured for the aggregate to be trusted.

        Three turns is the floor: it survives one misattributed turn, which is the
        failure this module exists to absorb.
        """

        if self.median_hz is None:
            return False
        return self.measured_turns >= 3 and self.speech_seconds >= 1.5

    def describe(self) -> str:
        """Return a one-line human-readable summary."""

        if self.median_hz is None:
            return f"{self.speaker_id}: no measurable pitch ({self.turns} turn(s))"
        span = (
            f"{self.low_hz:.0f}-{self.high_hz:.0f} Hz"
            if self.low_hz is not None and self.high_hz is not None
            else "range unknown"
        )
        return (
            f"{self.speaker_id}: {self.median_hz:.0f} Hz median ({span}), "
            f"{self.presentation}, {self.measured_turns}/{self.turns} turn(s) measured"
        )


@dataclass(frozen=True, slots=True)
class AttributionCorrection:
    """A turn whose speaker changed after its register was measured."""

    start: float
    end: float
    from_speaker: str
    to_speaker: str
    median_hz: float
    distance_ratio: float


@dataclass(frozen=True, slots=True)
class ContinuityCorrection:
    """A short turn relabelled because the speaker around it explains it.

    Recorded separately from a pitch correction because the evidence is different in
    kind: this one comes from the shape of the dialogue (a fragment glued to one
    character's sentence, with nobody else talking over it), not from a frequency
    measurement of the fragment itself.
    """

    start: float
    end: float
    from_speaker: str
    to_speaker: str
    duration: float

    def describe(self) -> str:
        """Return a one-line explanation for a report."""

        return (
            f"{self.start:.2f}-{self.end:.2f}s ({self.duration:.2f}s) "
            f"{self.from_speaker} -> {self.to_speaker}"
        )


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """Registers before and after repair, plus what was reassigned."""

    before: Mapping[str, SpeakerRegister]
    after: Mapping[str, SpeakerRegister]
    corrections: tuple[AttributionCorrection, ...]
    #: Short fragments whose label was taken from the speaker around them.
    continuity: tuple[ContinuityCorrection, ...] = ()

    @property
    def reassigned_fraction(self) -> float:
        """Return the share of turns whose speaker had to change."""

        total = sum(register.turns for register in self.after.values())
        moved = len(self.corrections) + len(self.continuity)
        return moved / total if total else 0.0


def _read_speech(path: str | Path) -> tuple[np.ndarray, int]:
    """Return the dialogue stem as mono samples and its rate."""

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise AudioReadError(f"could not read the dialogue audio {path}: {exc}") from exc
    if data.size == 0:
        raise AudioReadError(f"the dialogue audio {path} contains no samples")
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(rate)


def _validate(segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
    """Return ``segments`` as a list, failing on anything else."""

    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise InvalidSegmentsError(
            f"segments must be an iterable of TranscriptSegment, got {type(segments).__name__}"
        )
    ordered = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, TranscriptSegment):
            raise InvalidSegmentsError(
                f"segments[{index}] is {type(segment).__name__}, expected a TranscriptSegment"
            )
        ordered.append(segment)
    return ordered


def measure_turn_pitch(
    segments: Sequence[TranscriptSegment],
    samples: np.ndarray,
    rate: int,
) -> list[TurnPitch]:
    """Measure the pitch of every segment against a decoded dialogue stem."""

    measured: list[TurnPitch] = []
    for segment in segments:
        first = max(0, int(round(segment.start * rate)))
        last = min(samples.size, int(round(segment.end * rate)))
        if last <= first:
            measured.append(TurnPitch(segment.start, segment.end, None, 0, 0.0))
            continue
        try:
            profile: PitchProfile = estimate_f0(samples[first:last], rate)
        except (ProsodyError, ValueError):
            measured.append(TurnPitch(segment.start, segment.end, None, 0, 0.0))
            continue
        measured.append(
            TurnPitch(
                start=segment.start,
                end=segment.end,
                median_hz=profile.median_hz,
                voiced_frames=profile.voiced_frames,
                speech_seconds=max(0.0, segment.end - segment.start),
            )
        )
    return measured


class _Spanned(Protocol):
    """Anything the pitch measurement needs: who, and when."""

    speaker_id: str
    start: float
    end: float


def measure_registers(
    spans: Iterable[_Spanned],
    audio: str | Path,
    *,
    minimum_turn_seconds: float = MIN_REASSIGNMENT_TURN_SECONDS,
) -> dict[str, float]:
    """Return each speaker's median pitch in Hz, measured before any transcript exists.

    The transcript stage needs this to tell *who* is talking inside a mixture - a
    crosstalk has to be decoded to be heard at all, and the decode returns whichever
    voice dominates it, so a recovered line is only trustworthy if the audio it came
    from measures as the character it is about to be attributed to. Waiting for the
    register stage is not possible: that stage needs the transcript this one produces.

    Only turns long enough for their own pitch to mean something are used, which is the
    same floor the repair pass applies. Unmeasurable speakers simply do not appear in
    the result, and recovery then falls back to its structural checks alone.

    Parameters
    ----------
    spans:
        Diarized turns: anything carrying ``speaker_id``, ``start`` and ``end``.
    audio:
        The dialogue stem the turns were diarized from.
    """

    ordered = sorted(
        (turn for turn in spans if getattr(turn, "speaker_id", None)),
        key=lambda turn: (turn.start, turn.end),
    )
    if not ordered:
        return {}

    samples, rate = _read_speech(audio)
    pitches = measure_turn_pitch(ordered, samples, rate)  # type: ignore[arg-type]
    registers = aggregate_registers(
        [
            TranscriptSegment(turn.speaker_id, turn.start, turn.end, "-")
            for turn in ordered
        ],
        pitches,
    )

    by_speaker: dict[str, list[float]] = {}
    for turn, pitch in zip(ordered, pitches):
        if not pitch.usable or pitch.median_hz is None:
            continue
        if pitch.end - pitch.start < minimum_turn_seconds:
            continue
        by_speaker.setdefault(turn.speaker_id, []).append(pitch.median_hz)

    measured: dict[str, float] = {}
    for speaker_id, values in by_speaker.items():
        register = registers.get(speaker_id)
        measured[speaker_id] = (
            float(register.median_hz)
            if register is not None and register.median_hz is not None
            else float(np.median(values))
        )
    return measured


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float | None:
    """Return the weight-weighted median of ``values``, or ``None``."""

    if not values:
        return None
    order = np.argsort(np.asarray(values))
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    total = float(ordered_weights.sum())
    if total <= 0:
        return float(np.median(ordered_values))
    cumulative = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cumulative, total / 2.0))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def aggregate_registers(
    segments: Sequence[TranscriptSegment],
    pitches: Sequence[TurnPitch],
) -> dict[str, SpeakerRegister]:
    """Return one :class:`SpeakerRegister` per speaker, over all of their turns.

    The aggregate is a median weighted by voiced frames - not the median of
    medians - so a long, clearly measured turn carries more weight than a three
    frame interjection, while a single misattributed turn still cannot move it.
    """

    by_speaker: dict[str, list[TurnPitch]] = {}
    counts: dict[str, int] = {}
    for segment, pitch in zip(segments, pitches):
        by_speaker.setdefault(segment.speaker_id, []).append(pitch)
        counts[segment.speaker_id] = counts.get(segment.speaker_id, 0) + 1

    registers: dict[str, SpeakerRegister] = {}
    for speaker_id, turns in by_speaker.items():
        usable = [turn for turn in turns if turn.usable]
        if not usable:
            registers[speaker_id] = SpeakerRegister(
                speaker_id=speaker_id,
                median_hz=None,
                low_hz=None,
                high_hz=None,
                turns=counts[speaker_id],
                measured_turns=0,
                speech_seconds=0.0,
            )
            continue

        values = [turn.median_hz for turn in usable if turn.median_hz is not None]
        weights = [float(turn.voiced_frames) for turn in usable]
        median = _weighted_median(values, weights)
        registers[speaker_id] = SpeakerRegister(
            speaker_id=speaker_id,
            median_hz=median,
            low_hz=float(np.percentile(values, 10)),
            high_hz=float(np.percentile(values, 90)),
            turns=counts[speaker_id],
            measured_turns=len(usable),
            speech_seconds=float(sum(turn.speech_seconds for turn in usable)),
        )
    return registers


def _separable(registers: Mapping[str, SpeakerRegister]) -> bool:
    """Return whether any two speakers differ enough in register to arbitrate."""

    medians = [
        register.median_hz
        for register in registers.values()
        if register.median_hz is not None
    ]
    if len(medians) < 2:
        return False
    return max(medians) / min(medians) >= MIN_SEPARATION_RATIO


def _nearest(
    median_hz: float, registers: Mapping[str, SpeakerRegister], *, exclude: str
) -> tuple[str, float] | None:
    """Return the closest *other* speaker and its log distance, or ``None``."""

    best: tuple[str, float] | None = None
    for speaker_id, register in registers.items():
        if speaker_id == exclude or register.median_hz is None:
            continue
        distance = abs(math.log(median_hz / register.median_hz))
        if best is None or distance < best[1]:
            best = (speaker_id, distance)
    return best


def repair_attribution(
    segments: Iterable[TranscriptSegment],
    audio: str | Path,
    *,
    max_passes: int = MAX_PASSES,
    overlaps: Iterable[tuple[float, float]] = (),
    trusted: Iterable[tuple[float, float]] = (),
) -> tuple[list[TranscriptSegment], AttributionReport]:
    """Reassign turns their own speaker's register cannot explain.

    Two corrections happen, in this order, and they use different evidence.

    **Continuity first.** A short fragment glued to one character's sentence, with
    nobody talking over it, belongs to that character - see
    :func:`smooth_short_turns`. This runs before anything is measured so that a
    fragment cannot pollute the register it would then be compared against.

    **Then the register.** Only *decisive* evidence moves a turn: it has to be long
    enough for its own pitch to mean something
    (:data:`MIN_REASSIGNMENT_TURN_SECONDS`), sit at least :data:`OUTLIER_RATIO` away
    from its speaker's register, and be at least twice as close to another speaker's.
    A turn that sits between two speakers is left where diarization put it, because a
    coin flip is not an improvement.

    ``trusted`` is where a **stronger** measurement already settled the question: spans
    whose label was confirmed by comparing the turn against the diarization model's own
    speaker embeddings. Pitch never overrides those. A pitch median measures one
    acoustic dimension, and a trained actor's expressive range covers most of an octave
    either way - one hushed line and one shouted line from the same person can sit an
    octave apart - while an embedding compares the voice itself. Ranking the two the
    other way round is how a real film ended up with fragments of one character's
    monologue voiced by the other character's clone.

    Parameters
    ----------
    segments:
        Transcript segments, each carrying the speaker diarization assigned.
    audio:
        The dialogue stem the segments were transcribed from.
    overlaps:
        Regions where two or more speakers talk at once. A short turn inside one keeps
        its diarized label: the other character really was talking there.
    trusted:
        Spans, in seconds, whose speaker was confirmed by speaker embeddings. Pitch may
        not move a turn that falls inside one.

    Returns
    -------
    tuple[list[TranscriptSegment], AttributionReport]
        The segments with corrections applied, in input order, and the registers and
        corrections that explain what changed.
    """

    ordered = _validate(segments)
    if not ordered:
        return [], AttributionReport(before={}, after={}, corrections=())

    ordered, continuity = smooth_short_turns(ordered, overlaps=overlaps)

    confirmed = [
        (start, end)
        for start, end in trusted
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
    ]

    samples, rate = _read_speech(audio)
    pitches = measure_turn_pitch(ordered, samples, rate)
    before = aggregate_registers(ordered, pitches)

    labels = [segment.speaker_id for segment in ordered]
    corrections: list[AttributionCorrection] = []
    after = before

    for _pass in range(max(1, max_passes)):
        registers = aggregate_registers(ordered, pitches)
        if not _separable(registers):
            after = registers
            break

        moved = 0
        for index, pitch in enumerate(pitches):
            own = registers.get(labels[index])
            if own is None or own.median_hz is None or not pitch.usable:
                continue
            if pitch.end - pitch.start < MIN_REASSIGNMENT_TURN_SECONDS:
                # Too short for its own pitch to identify a voice. Continuity has
                # already had its say; a fragment is not evidence of a stranger.
                continue
            if any(
                pitch.start < end - TIMESTAMP_TOLERANCE
                and pitch.end > start + TIMESTAMP_TOLERANCE
                for start, end in confirmed
            ):
                # Speaker embeddings already compared this turn with the voice
                # itself. Pitch is the weaker measurement and does not overturn it.
                continue
            assert pitch.median_hz is not None
            own_distance = abs(math.log(pitch.median_hz / own.median_hz))
            if own_distance < math.log(OUTLIER_RATIO):
                continue
            candidate = _nearest(pitch.median_hz, registers, exclude=labels[index])
            if candidate is None:
                continue
            other_id, other_distance = candidate
            if other_distance >= own_distance * REASSIGNMENT_MARGIN:
                continue

            corrections.append(
                AttributionCorrection(
                    start=pitch.start,
                    end=pitch.end,
                    from_speaker=labels[index],
                    to_speaker=other_id,
                    median_hz=pitch.median_hz,
                    distance_ratio=pitch.median_hz / own.median_hz,
                )
            )
            labels[index] = other_id
            moved += 1

        after = aggregate_registers(
            [
                TranscriptSegment(label, segment.start, segment.end, segment.text)
                for label, segment in zip(labels, ordered)
            ],
            pitches,
        )
        if moved == 0:
            break

    repaired = [
        segment if label == segment.speaker_id
        else TranscriptSegment(label, segment.start, segment.end, segment.text)
        for label, segment in zip(labels, ordered)
    ]
    return repaired, AttributionReport(
        before=before,
        after=after,
        corrections=tuple(corrections),
        continuity=continuity,
    )


def smooth_short_turns(
    segments: Iterable[TranscriptSegment],
    *,
    overlaps: Iterable[tuple[float, float]] = (),
    short_seconds: float = SHORT_TURN_SECONDS,
    max_gap: float = CONTINUITY_GAP_SECONDS,
) -> tuple[list[TranscriptSegment], tuple[ContinuityCorrection, ...]]:
    """Give every short fragment its neighbours' speaker, when nothing says otherwise.

    Diarization decides *who spoke when* from embeddings computed over a window, and a
    window that holds a third of a second of speech is not enough of a voice to decide
    anything: the exclusive view flip-flops at the pauses inside a single sentence. On
    the film that prompted this, one character's continuous monologue came back with
    five of its fragments labelled as the other character - each one voiced by the
    other character's clone, in the middle of a sentence.

    The evidence that fixes it is the shape of the dialogue rather than the sound of
    the fragment:

    * the turn is **short** - longer than that and its own audio is worth trusting, so
      it is left for :func:`repair_attribution` to judge;
    * the turns **immediately before and after** it belong to the same *other*
      speaker, so a single voice is running through this stretch of the film;
    * it is **glued to at least one side** - within :data:`CONTINUITY_GAP_SECONDS` -
      so it sits inside somebody's utterance rather than being a reply with a beat
      either side of it;
    * it is **not simultaneous** with another speaker. A real interjection is normally
      talked over, and that is exactly what the overlap-aware diarization reports. If
      two voices are really there, the label is kept.

    A relabel is only made next to a *fuller* neighbour: at least one of the two must
    be longer than the fragment itself. A run of fragments all the same length is
    therefore never evidence for itself, which is what stops a stretch of short turns
    relabelling one another in a circle.

    Parameters
    ----------
    segments:
        Transcribed turns, in chronological order, each with its diarized speaker.
    overlaps:
        Regions where two or more speakers talk at once, from
        :func:`app.pipeline.diarization.diarize_detailed`. A turn inside one keeps its
        label: somebody else really was there.

    Returns
    -------
    tuple[list[TranscriptSegment], tuple[ContinuityCorrection, ...]]
        The segments in input order and the fragments that were relabelled.
    """

    ordered = _validate(segments)
    regions = [
        (start, end)
        for start, end in overlaps
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
    ]
    if len(ordered) < 3:
        return ordered, ()

    corrections: list[ContinuityCorrection] = []
    labels = [segment.speaker_id for segment in ordered]
    for index in range(1, len(ordered) - 1):
        segment = ordered[index]
        duration = segment.end - segment.start
        if duration > short_seconds:
            continue
        if any(
            min(segment.end, end) - max(segment.start, start) > TIMESTAMP_TOLERANCE
            for start, end in regions
        ):
            continue

        previous = ordered[index - 1]
        following = ordered[index + 1]
        surrounding = previous.speaker_id
        if surrounding != following.speaker_id or surrounding == segment.speaker_id:
            continue
        previous_gap = segment.start - previous.end
        following_gap = following.start - segment.end
        if min(previous_gap, following_gap) > max_gap:
            continue
        if (
            previous.end - previous.start <= duration
            and following.end - following.start <= duration
        ):
            continue

        labels[index] = surrounding
        corrections.append(
            ContinuityCorrection(
                start=segment.start,
                end=segment.end,
                from_speaker=segment.speaker_id,
                to_speaker=surrounding,
                duration=duration,
            )
        )

    repaired = [
        segment
        if label == segment.speaker_id
        else TranscriptSegment(label, segment.start, segment.end, segment.text)
        for label, segment in zip(labels, ordered)
    ]
    return repaired, tuple(corrections)


def contiguous_turns(
    segments: Iterable[TranscriptSegment],
    *,
    max_gap: float = 0.5,
) -> list[tuple[str, float, float]]:
    """Return ``(speaker_id, start, end)`` runs of one speaker.

    Voice references are cut from continuous stretches of one person speaking, so
    this turns repaired transcript segments into the runs the profile stage wants.
    Segments are joined when the same speaker continues within ``max_gap``.
    """

    runs: list[list[object]] = []
    for segment in _validate(segments):
        if (
            runs
            and runs[-1][0] == segment.speaker_id
            and segment.start - float(runs[-1][2]) <= max_gap  # type: ignore[arg-type]
        ):
            runs[-1][2] = segment.end
        else:
            runs.append([segment.speaker_id, segment.start, segment.end])
    return [(str(a), float(b), float(c)) for a, b, c in runs]


def disagreement(
    report: AttributionReport,
    *,
    minimum_purity: float = 0.8,
) -> str | None:
    """Return why a cast cannot be trusted, or ``None`` when it can.

    A run should not ship a voice it cannot justify. The checks are deliberately
    about *separation* rather than about any specific pitch: two speakers whose
    registers overlap, or one speaker whose turns are spread across both bands, mean
    the material does not support an automatic cast.
    """

    measured = [r for r in report.after.values() if r.median_hz is not None]
    if len(measured) >= 2:
        medians = [r.median_hz for r in measured if r.median_hz is not None]
        if max(medians) / min(medians) < MIN_SEPARATION_RATIO:
            described = ", ".join(r.describe() for r in report.after.values())
            return (
                "the speakers' registers are too close for pitch to tell them apart "
                f"({described}); the cast would be a guess"
            )

    unconfident = [r for r in report.after.values() if not r.confident and r.turns > 0]
    if unconfident and len(report.after) > 1:
        described = ", ".join(r.describe() for r in unconfident)
        return f"too little measurable speech to place a voice ({described})"

    total = sum(r.turns for r in report.after.values())
    moved = len(report.corrections)
    if total and moved / total > (1.0 - minimum_purity):
        return (
            f"{moved} of {total} turn(s) had to be reassigned, so the diarization "
            "labels describe the timeline rather than the people"
        )

    spread = register_spread(report)
    if spread:
        described = ", ".join(spread)
        return (
            "a speaker's own turns do not look like one voice "
            f"({described}); check the separation before trusting the cast"
        )
    return None


def register_spread(
    report: AttributionReport,
    *,
    limit: float = CLUSTER_SPREAD_LIMIT,
) -> tuple[str, ...]:
    """Return one line per speaker whose own turns span an implausible range.

    A single voice, measured once per utterance across a film, varies by maybe a
    quarter either way: a hushed line and a shouted one sit a few semitones apart, not
    an octave. When the turns a cluster kept span more than
    :data:`CLUSTER_SPREAD_LIMIT`, the cluster is more likely to hold two voices than to
    hold one unusually expressive one - which is heard, downstream, as a character
    whose voice changes between scenes because their turns were split across two voice
    references.

    Reported rather than acted on: nothing here knows which halves belong together, and
    guessing would be a coin flip. It is the signal that the *separation* needs a look.
    """

    described: list[str] = []
    for speaker_id in sorted(report.before):
        before = report.before[speaker_id]
        if (
            before.low_hz is None
            or before.high_hz is None
            or before.median_hz is None
            or before.measured_turns < 3
        ):
            continue
        spread = before.high_hz / before.low_hz
        if spread >= limit:
            described.append(
                f"{speaker_id} spans {before.low_hz:.0f}-{before.high_hz:.0f} Hz "
                f"({spread:.2f}x around {before.median_hz:.0f} Hz over "
                f"{before.measured_turns} turn(s))"
            )
    return tuple(described)


__all__ = [
    "AttributionCorrection",
    "AttributionReport",
    "AudioReadError",
    "CONTINUITY_GAP_SECONDS",
    "CLUSTER_SPREAD_LIMIT",
    "ContinuityCorrection",
    "FEMININE_MIN_HZ",
    "InvalidSegmentsError",
    "MASCULINE_MAX_HZ",
    "MAX_PASSES",
    "MIN_REASSIGNMENT_TURN_SECONDS",
    "MIN_SEPARATION_RATIO",
    "MIN_VOICED_FRAMES",
    "OUTLIER_RATIO",
    "REASSIGNMENT_MARGIN",
    "RegisterError",
    "SHORT_TURN_SECONDS",
    "SpeakerRegister",
    "TurnPitch",
    "aggregate_registers",
    "contiguous_turns",
    "disagreement",
    "measure_registers",
    "measure_turn_pitch",
    "presentation_of",
    "register_spread",
    "repair_attribution",
    "smooth_short_turns",
]
