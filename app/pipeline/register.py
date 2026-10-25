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

import numpy as np
import soundfile as sf

from app.pipeline.prosody import PitchProfile, ProsodyError, estimate_f0
from app.pipeline.transcription import TranscriptSegment

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
class AttributionReport:
    """Registers before and after repair, plus what was reassigned."""

    before: Mapping[str, SpeakerRegister]
    after: Mapping[str, SpeakerRegister]
    corrections: tuple[AttributionCorrection, ...]

    @property
    def reassigned_fraction(self) -> float:
        """Return the share of turns whose speaker had to change."""

        total = sum(register.turns for register in self.after.values())
        return len(self.corrections) / total if total else 0.0


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
) -> tuple[list[TranscriptSegment], AttributionReport]:
    """Reassign turns their own speaker's register cannot explain.

    Only *decisive* evidence moves a turn: it has to sit at least
    :data:`OUTLIER_RATIO` away from its speaker's register and at least twice as
    close to another speaker's. A turn that sits between two speakers is left where
    diarization put it, because a coin flip is not an improvement.

    Parameters
    ----------
    segments:
        Transcript segments, each carrying the speaker diarization assigned.
    audio:
        The dialogue stem the segments were transcribed from.

    Returns
    -------
    tuple[list[TranscriptSegment], AttributionReport]
        The segments with decisive corrections applied, in input order, and the
        registers and corrections that explain what changed.
    """

    ordered = _validate(segments)
    if not ordered:
        return [], AttributionReport(before={}, after={}, corrections=())

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
        before=before, after=after, corrections=tuple(corrections)
    )


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
    return None


__all__ = [
    "AttributionCorrection",
    "AttributionReport",
    "AudioReadError",
    "FEMININE_MIN_HZ",
    "InvalidSegmentsError",
    "MASCULINE_MAX_HZ",
    "MAX_PASSES",
    "MIN_SEPARATION_RATIO",
    "MIN_VOICED_FRAMES",
    "OUTLIER_RATIO",
    "REASSIGNMENT_MARGIN",
    "RegisterError",
    "SpeakerRegister",
    "TurnPitch",
    "aggregate_registers",
    "contiguous_turns",
    "disagreement",
    "measure_turn_pitch",
    "presentation_of",
    "repair_attribution",
]
