"""Put an authored plan onto the audio's own clock, without transcribing anything.

Why this exists
---------------
A plan states where every line sits, so the only question left is whether its clock
agrees with the file. Answering that from a transcript costs a diarization pass and a
transcription pass - about 21 minutes on a feature episode - and, worse, it answers a
different question: when the transcript cannot recognise the lines it is given (46 of
595 on a real episode), the answer is noise and the retiming that follows is invention.

The dialogue stem already holds the answer. Inside a plan cue somebody is speaking and
outside one they are not, so the plan *is* a speech mask, and shifting that mask against
the stem's energy envelope finds the offset the same way an editor lines up two takes.
No model, no token, no GPU: a frame envelope, one FFT, and a bounded search.

Deliberate limits
-----------------
* **It only ever reports a shift it can justify.** The best lag must beat the median
  lag clearly, or the plan is left alone: a wrong shift is worse than none.
* **Per-line snapping is conservative.** A line moves only when its own window holds
  almost no speech *and* another nearby window holds substantially more.
* **Nothing here edits the dialogue.** It returns times; which lines exist, who says
  them and what they say stay exactly as authored.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.pipeline.dialogue_plan import PlanCue, PlanError

logger = logging.getLogger(__name__)

#: Frame step of the envelope, in seconds. Long enough that a frame spans a syllable
#: rather than a waveform cycle, short enough to place a line without audible error.
DEFAULT_HOP_SECONDS = 0.05

#: How far the plan may be out before it is treated as a different cut of the film.
DEFAULT_MAX_OFFSET_SECONDS = 300.0

#: How much better the best lag must be than the median lag before it is believed.
#: 1.25 is roughly "clear winner" for a mask scored against an energy envelope; a
#: flat profile - the plan belonging to another episode - scores near 1.0 everywhere.
MIN_OFFSET_CONFIDENCE = 1.25

#: A cue whose window carries less than this share of the speech energy a placed line
#: typically carries is "silent", and is a candidate for snapping.
SILENT_WINDOW_RATIO = 0.35

#: A snap has to find this much more energy before the line is moved.
MIN_SNAP_GAIN = 1.5

#: How far a single line may be moved to find its own speech.
DEFAULT_SNAP_RADIUS_SECONDS = 1.5

#: Length of the energy ramp used to turn a frame envelope into an onset signal.
ONSET_FRAMES = 3


class PlanAlignError(RuntimeError):
    """The plan could not be compared with the audio at all."""


@dataclass(frozen=True, slots=True)
class PlanAlignment:
    """What aligning a plan against the audio found."""

    #: Seconds to add to every authored time.
    offset: float
    #: Mean speech energy under the plan mask at ``offset``, on a 0-1 scale.
    score: float
    #: The same measurement at zero offset, for comparison.
    baseline: float
    #: Whether ``offset`` was applied. False means the evidence was too weak.
    applied: bool
    #: Cues moved individually after the global shift.
    snapped: int
    #: Share of cues whose window holds speech at their final times.
    placed: float
    #: One line, for the run log.
    reason: str

    @property
    def confident(self) -> bool:
        return self.applied or abs(self.offset) < 1e-6

    def as_dict(self) -> dict[str, object]:
        return {
            "offset_seconds": round(self.offset, 3),
            "applied": self.applied,
            "score": round(self.score, 4),
            "baseline": round(self.baseline, 4),
            "snapped": self.snapped,
            "placed": round(self.placed, 4),
            "reason": self.reason,
        }


def speech_envelope(
    path: str | np.ndarray,
    *,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(envelope, hop)``: per-frame speech energy, scaled to 0-1.

    ``path`` is any audio file (the dialogue stem). An array is accepted so a test can
    hand in a synthesised signal and so a caller that has already decoded the audio
    need not have it written to disk.
    """

    if isinstance(path, np.ndarray):
        samples = np.asarray(path, dtype=np.float64)
        rate = int(sample_rate or 0)
        if rate <= 0:
            raise PlanAlignError("an in-memory envelope needs its sample rate")
    else:
        try:
            data, rate = sf.read(str(path), dtype="float64", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise PlanAlignError(f"could not read {path} for alignment: {exc}") from exc
        samples = np.asarray(data, dtype=np.float64).mean(axis=1)

    if samples.size == 0:
        raise PlanAlignError("the dialogue stem is empty; nothing to align against")

    hop = max(1, int(round(float(hop_seconds) * rate)))
    frames = 1 + max(0, (samples.size - 1) // hop)
    usable = frames * hop
    if samples.size < usable:
        samples = np.pad(samples, (0, usable - samples.size))
    blocks = samples[:usable].reshape(frames, hop)
    energy = np.sqrt(np.mean(np.square(blocks), axis=1))

    # Loudness, not amplitude: a whisper and a shout are both speech, and only their
    # distance from the room's own floor says which frames carry any.
    decibels = 20.0 * np.log10(np.maximum(energy, 1e-8))
    floor = float(np.percentile(decibels, 30.0))
    above = np.maximum(0.0, decibels - floor)
    peak = float(np.percentile(above, 95.0))
    if peak <= 0.0:
        return np.zeros(frames, dtype=np.float64), float(hop) / rate
    return np.clip(above / peak, 0.0, 1.0), float(hop) / rate


def plan_mask(
    cues: Sequence[PlanCue],
    *,
    frames: int,
    hop: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Return a 0/1 mask that is 1 wherever a plan cue says somebody is speaking."""

    mask = np.zeros(frames, dtype=np.float64)
    for cue in cues:
        start = int(round((cue.start + offset) / hop))
        end = int(round((cue.end + offset) / hop))
        start = max(0, min(frames, start))
        end = max(start, min(frames, end))
        if end > start:
            mask[start:end] = 1.0
    return mask


def _correlate(envelope: np.ndarray, mask: np.ndarray, *, pad: int = 0) -> np.ndarray:
    """Return the circular cross-correlation, indexed by signed lag.

    ``result[k]`` is ``sum_f envelope[f] * mask[f - k]`` for a lag of ``k`` frames,
    with negative lags wrapped to the end of the array - the layout
    :func:`_lag_value` reads. Direct evaluation would be O(frames x lags) - hundreds
    of millions of operations for a feature episode - so it is done in the frequency
    domain, which is a few milliseconds for the same answer.

    ``pad`` reserves room for that many frames of lag in each direction. Without it a
    search wider than the signal wraps around and reports a lag that is really the
    opposite one, which is how a 5-second shift came out as 261.
    """

    needed = envelope.size + mask.size - 1 + 2 * max(0, int(pad))
    size = 1 << max(1, (needed - 1).bit_length())
    spectrum = np.fft.rfft(envelope, size) * np.conj(np.fft.rfft(mask, size))
    return np.fft.irfft(spectrum, size)


def _lag_value(correlation: np.ndarray, lag: int) -> float:
    """Return the score at a signed ``lag`` from a circular correlation."""

    return float(correlation[lag % correlation.size])


def estimate_offset(
    envelope: np.ndarray,
    hop: float,
    cues: Sequence[PlanCue],
    *,
    max_offset_seconds: float = DEFAULT_MAX_OFFSET_SECONDS,
    min_confidence: float = MIN_OFFSET_CONFIDENCE,
) -> PlanAlignment:
    """Return the shift that best puts ``cues`` over the speech in ``envelope``.

    The mask's own weight is constant across lags, so a lag's score is simply the mean
    envelope value under the mask: "how much speech is there where the plan says
    somebody is talking". The best lag is believed only when it beats the median lag,
    which is what stops an unrelated plan - matching nothing - from being shifted by
    whatever noise happens to peak.
    """

    if not cues:
        return PlanAlignment(
            offset=0.0,
            score=0.0,
            baseline=0.0,
            applied=False,
            snapped=0,
            placed=0.0,
            reason="the plan has no cues to align",
        )
    if envelope.size == 0:
        raise PlanAlignError("the envelope is empty")

    mask = plan_mask(cues, frames=envelope.size, hop=hop)
    weight = float(mask.sum())
    if weight <= 0.0:
        return PlanAlignment(
            offset=0.0,
            score=0.0,
            baseline=0.0,
            applied=False,
            snapped=0,
            placed=0.0,
            reason="no plan line falls inside this audio",
        )

    radius = int(round(float(max_offset_seconds) / hop))
    # A lag is only meaningful inside the audio that was measured, and the FFT needs
    # room for the whole search or it wraps and reports the opposite lag.
    radius = max(1, min(radius, envelope.size - 1))
    correlation = _correlate(envelope, mask, pad=radius) / weight
    lags = np.arange(-radius, radius + 1, dtype=np.int64)
    values = np.array(
        [_lag_value(correlation, int(lag)) for lag in lags], dtype=np.float64
    )
    if values.size == 0:
        return PlanAlignment(
            offset=0.0,
            score=_lag_value(correlation, 0),
            baseline=_lag_value(correlation, 0),
            applied=False,
            snapped=0,
            placed=0.0,
            reason="the plan lies outside this audio",
        )

    best_index = int(np.argmax(values))
    best_lag = int(lags[best_index])
    best = float(values[best_index])
    median = float(np.median(values))
    baseline = _lag_value(correlation, 0)
    score = max(best, 1e-9)
    confidence = best / max(abs(median), 1e-9)
    offset = best_lag * hop

    if abs(offset) < hop / 2.0:
        return PlanAlignment(
            offset=0.0,
            score=baseline,
            baseline=baseline,
            applied=False,
            snapped=0,
            placed=_placed_share(envelope, mask),
            reason="the plan already sits on this audio (best shift is under one frame)",
        )
    if confidence < min_confidence:
        return PlanAlignment(
            offset=0.0,
            score=baseline,
            baseline=baseline,
            applied=False,
            snapped=0,
            placed=_placed_share(envelope, mask),
            reason=(
                f"the best shift (+{offset:.2f}s) is only {confidence:.2f}x the typical "
                f"one, which is too weak to act on; the plan is used as authored. Check "
                "that it belongs to this video"
            ),
        )
    shifted = plan_mask(cues, frames=envelope.size, hop=hop, offset=offset)
    return PlanAlignment(
        offset=offset,
        score=best,
        baseline=baseline,
        applied=True,
        snapped=0,
        placed=_placed_share(envelope, shifted),
        reason=(
            f"shifted the plan {offset:+.2f}s, where {score:.2f} of the scale carries "
            f"speech against {baseline:.2f} as authored ({confidence:.2f}x the typical lag)"
        ),
    )


def _placed_share(envelope: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean envelope value under ``mask``, on its 0-1 scale."""

    weight = float(mask.sum())
    if weight <= 0.0:
        return 0.0
    return float(np.dot(envelope, mask) / weight)


def _window_energy(envelope: np.ndarray, hop: float, start: float, end: float) -> float:
    """Return the mean speech energy inside ``start``-``end``."""

    first = max(0, min(envelope.size, int(round(start / hop))))
    last = max(first, min(envelope.size, int(round(end / hop))))
    if last <= first:
        return 0.0
    return float(envelope[first:last].mean())


def snap_cues(
    cues: Sequence[PlanCue],
    envelope: np.ndarray,
    hop: float,
    *,
    radius_seconds: float = DEFAULT_SNAP_RADIUS_SECONDS,
    silent_ratio: float = SILENT_WINDOW_RATIO,
    min_gain: float = MIN_SNAP_GAIN,
) -> tuple[tuple[PlanCue, ...], int]:
    """Move a cue onto its own speech when its authored window holds none.

    Only a line that is essentially silent where it was authored is considered, and it
    only moves when another window within ``radius_seconds`` carries substantially more
    speech. That is the case the transcript used to handle - a line authored a little
    early, or one whose neighbour was retimed - without the cost or the invention.
    """

    if not cues:
        return (), 0
    mask = plan_mask(cues, frames=envelope.size, hop=hop)
    typical = _placed_share(envelope, mask)
    if typical <= 0.0:
        return tuple(cues), 0

    probe = silent_ratio * typical
    radius = max(hop, float(radius_seconds))
    moved = 0
    out: list[PlanCue] = []
    for cue in cues:
        window = cue.duration
        best_start = cue.start
        best_energy = _window_energy(envelope, hop, cue.start, cue.end)
        if best_energy < probe:
            # Search a coarse grid of candidate starts, then keep the best. A line is
            # never moved onto another line's speech: overlap is how two characters end
            # up sharing one window, which the mixer cannot undo.
            for step in np.arange(-radius, radius + 1e-9, hop):
                start = max(0.0, cue.start + float(step))
                end = start + window
                if end * 1.0 > envelope.size * hop:
                    break
                if any(
                    other is not cue
                    and start < other.end + hop
                    and end > other.start - hop
                    for other in cues
                ):
                    continue
                energy = _window_energy(envelope, hop, start, end)
                if energy > best_energy:
                    best_energy = energy
                    best_start = start
            if best_start != cue.start:
                before = _window_energy(envelope, hop, cue.start, cue.end)
                if before <= 0.0 or best_energy >= before * min_gain:
                    out.append(cue.moved_to(best_start, best_start + window))
                    moved += 1
                    continue
        out.append(cue)
    return tuple(out), moved


def align_plan(
    cues: Sequence[PlanCue],
    audio: str | np.ndarray,
    *,
    sample_rate: int | None = None,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    max_offset_seconds: float = DEFAULT_MAX_OFFSET_SECONDS,
    snap_radius_seconds: float = DEFAULT_SNAP_RADIUS_SECONDS,
) -> tuple[tuple[PlanCue, ...], PlanAlignment]:
    """Align ``cues`` to ``audio``, returning the cues and what was measured."""

    envelope, hop = speech_envelope(
        audio, hop_seconds=hop_seconds, sample_rate=sample_rate
    )
    found = estimate_offset(
        envelope, hop, cues, max_offset_seconds=max_offset_seconds
    )
    if not found.applied:
        return tuple(cues), found

    shifted = [cue.shifted(found.offset) for cue in cues]
    snapped, moved = snap_cues(
        shifted, envelope, hop, radius_seconds=snap_radius_seconds
    )
    mask = plan_mask(snapped, frames=envelope.size, hop=hop)
    final = _placed_share(envelope, mask)
    return snapped, PlanAlignment(
        offset=found.offset,
        score=found.score,
        baseline=found.baseline,
        applied=True,
        snapped=moved,
        placed=final,
        reason=found.reason
        + (f"; {moved} line(s) snapped onto their own speech" if moved else ""),
    )


__all__ = [
    "DEFAULT_HOP_SECONDS",
    "DEFAULT_MAX_OFFSET_SECONDS",
    "DEFAULT_SNAP_RADIUS_SECONDS",
    "MIN_OFFSET_CONFIDENCE",
    "PlanAlignError",
    "PlanAlignment",
    "align_plan",
    "estimate_offset",
    "plan_mask",
    "snap_cues",
    "speech_envelope",
]
