"""Prosody measurement: did the dub keep the actor's delivery?

Preserving the original actor's performance is one of the project's top-level goals,
and the pipeline's mechanism for it is unusual enough that it needs its own
measurement: the Chatterbox prompt is the *original actor's own audio* for that
line, so the generated Amharic take starts out carrying that delivery, and the
Seed-VC stage is supposed to replace the timbre without touching it. Nothing in the
pipeline checked that it actually survived - this module does.

Why pitch tracking rather than an embedding
------------------------------------------
A speaker-embedding model would answer "does this sound like the same voice", which
:mod:`app.pipeline.evaluation` asks separately. What has to be checked here is the
*delivery*: how high the voice sat, how far it moved, and how much of the line was
voiced at all. A shouted line has a wide, high pitch range; a whisper has almost no
periodicity; a flat read has a narrow one. Autocorrelation recovers exactly those
quantities from the waveform, needs no model, downloads nothing, and behaves
identically on every machine.

Bias cancels
------------
An autocorrelation tracker has known biases - it octave-errors on very low
utterances, and it misses creaky voice. Those biases matter for absolute pitch and
much less for the comparison this module exists to make: the original prompt and
the generated take are measured with the *same* tracker, so a systematic bias
appears on both sides and largely cancels in the ratio. The numbers are therefore
useful as relative measurements, which is how the reports use them, and should not
be quoted as absolute pitches.

Known limitations, stated rather than discovered later
------------------------------------------------------
* **Above the range.** A pitch higher than :data:`F0_MAX_HZ` has a period shorter
  than the shortest lag searched, so it cannot be found; a sub-multiple of it that
  falls inside the range is reported instead. No film dialogue reaches the ceiling.
* **Sustained musical tones.** A held violin note is steady and periodic, exactly
  like a sustained vowel, and no property of the waveform separates them. The module
  is meant for speech stems and generated takes, not for music.
* **Creaky voice and whispers.** Neither has reliable periodicity, so both are
  reported as unvoiced. That is a measurement, not an error: a comparison that
  cannot be made says so instead of inventing a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

#: Analysis sample rate. 16 kHz resolves every pitch this module cares about, and
#: decimating to it keeps the frame matrix small enough for a feature film.
ANALYSIS_RATE = 16_000

#: Pitch search range, in Hz. The floor is below a male speaking voice and the
#: ceiling above a child's, so the range covers film dialogue with headroom.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0

#: Frame geometry. 40 ms is long enough to hold several periods at 60 Hz, and a
#: 10 ms hop tracks a pitch movement within a syllable.
FRAME_SECONDS = 0.040
HOP_SECONDS = 0.010

#: A frame counts as speech when it rises this far above the signal's own noise
#: floor, estimated from its quietest frames. Using the *local* floor rather than an
#: absolute level is what lets the same measurement work on a clean stem and on a
#: quiet take - the same reasoning the voice-profile selector uses.
SPEECH_MARGIN_DB = 15.0

#: How strong the *first* autocorrelation peak must be, relative to the strongest
#: one, to be taken as the fundamental. Choosing the strongest peak outright is the
#: classic subharmonic error: a lag of two or three periods correlates just as well
#: as one, so a plain maximum reports a pitch an octave or a twelfth too low. Taking
#: the earliest peak that is nearly as strong as the best one finds the fundamental.
FIRST_PEAK_FRACTION = 0.85

#: Minimum normalised autocorrelation for a frame to be called voiced. Below this
#: the frame is aperiodic - a fricative, a plosive, or noise - and has no pitch.
VOICED_CORRELATION = 0.30

#: A frame also counts as speech when it is within this fraction of the loudest
#: frame **and** is clearly periodic. The relative-to-floor test above is what
#: separates speech from a music bed, because speech is bursty and steady material
#: is not - but a *sustained* delivery, a long shout or a held vowel, has no dynamic
#: range at all and would fail it. This second test keeps such lines measurable.
STEADY_PEAK_FRACTION = 0.15

#: How periodic a steady frame must be to be accepted by the path above. Set high on
#: purpose so that only clearly tonal material qualifies.
#:
#: Known limitation, stated rather than hidden: a *sustained musical tone* - a held
#: violin note, a synth pad - is periodic and steady, exactly like a sustained
#: vowel. Nothing about a waveform separates those two, so such a frame is accepted
#: and the caller is expected to know which stem it handed over. On the pipeline's
#: own inputs (the clean speech stem and a generated take) this does not arise.
STEADY_PERIODICITY = 0.55

#: A cap on how many frames are analysed, so measuring a feature-length file stays
#: bounded. The frames are spread across the whole signal rather than taken from the
#: front, so the description stays representative.
MAX_FRAMES = 4_000


#: A range this small is a monotone delivery rather than an expressive one. A held
#: note or a flat read has no range to be a fraction of, so the ratio is undefined
#: for it and :attr:`ProsodyComparison.preserved` decides that case directly.
FLAT_RANGE_SEMITONES = 1.5

#: A delivery counts as preserved when the pitch centre moved no further than this,
#: and the range did not collapse past :data:`PRESERVED_RANGE_RATIO`. Two semitones
#: is a whole tone: further than that and the audience hears a different performance.
PRESERVED_PITCH_SHIFT_SEMITONES = 2.0

#: The smallest generated-to-original range ratio that still counts as preserved.
#: Half is deliberately generous - it allows some flattening - because the failure
#: this catches is the near-total loss of expressiveness that neutral re-synthesis
#: produces, not ordinary variation.
PRESERVED_RANGE_RATIO = 0.5

#: A monotone original may become this much less monotone before the take counts as
#: over-articulated rather than preserved.
FLAT_TOLERANCE_FACTOR = 2.0


class ProsodyError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidAudioError(ProsodyError, ValueError):
    """The audio cannot be measured."""


@dataclass(frozen=True, slots=True)
class PitchProfile:
    """A description of how a stretch of speech was delivered.

    Every field is ``None``/``0.0`` when nothing voiced was found, which is a real
    result - a whisper, a silence, or a line buried under music - rather than an
    error to raise.
    """

    frames: int
    speech_frames: int
    voiced_frames: int
    median_hz: float | None
    low_hz: float | None
    high_hz: float | None

    @property
    def voiced_ratio(self) -> float:
        """Fraction of *speech* frames that carried a pitch."""

        if not self.speech_frames:
            return 0.0
        return self.voiced_frames / self.speech_frames

    @property
    def range_semitones(self) -> float | None:
        """How far the pitch moved, in semitones, between its 5th and 95th percentile.

        Percentiles rather than min/max, because a single octave error would
        otherwise dominate the range of a whole line.
        """

        if self.low_hz is None or self.high_hz is None:
            return None
        if self.low_hz <= 0 or self.high_hz <= 0:
            return None
        return 12.0 * math.log2(self.high_hz / self.low_hz)

    @property
    def is_voiced(self) -> bool:
        """``True`` when at least some of the speech carried a pitch."""

        return self.voiced_frames > 0 and self.median_hz is not None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the profile."""

        return {
            "frames": self.frames,
            "speech_frames": self.speech_frames,
            "voiced_frames": self.voiced_frames,
            "voiced_ratio": round(self.voiced_ratio, 4),
            "median_hz": None if self.median_hz is None else round(self.median_hz, 2),
            "low_hz": None if self.low_hz is None else round(self.low_hz, 2),
            "high_hz": None if self.high_hz is None else round(self.high_hz, 2),
            "range_semitones": (
                None if self.range_semitones is None else round(self.range_semitones, 3)
            ),
        }


def _mono(samples: np.ndarray) -> np.ndarray:
    """Return ``samples`` as a 1-D float32 array, averaging channels."""

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim != 1:
        raise InvalidAudioError(
            f"audio must be 1-D or 2-D, got {data.ndim} dimensions"
        )
    return np.ascontiguousarray(data, dtype=np.float32)


def _to_analysis_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    """Resample ``samples`` to :data:`ANALYSIS_RATE`."""

    if not isinstance(rate, int) or isinstance(rate, bool) or rate < 1:
        raise InvalidAudioError(f"sample rate must be a positive integer, got {rate!r}")
    if rate == ANALYSIS_RATE:
        return samples

    divisor = math.gcd(int(rate), ANALYSIS_RATE)
    up = ANALYSIS_RATE // divisor
    down = int(rate) // divisor
    return resample_poly(samples, up, down).astype(np.float32)


def _frames(samples: np.ndarray, *, frame: int, hop: int) -> np.ndarray:
    """Return a ``(frame_count, frame)`` view of the signal's overlapping frames."""

    if samples.size < frame:
        return np.empty((0, frame), dtype=np.float32)

    count = 1 + (samples.size - frame) // hop
    offsets = np.arange(count)[:, None] * hop + np.arange(frame)[None, :]
    return samples[offsets]


def estimate_f0(samples: np.ndarray, rate: int) -> PitchProfile:
    """Describe the pitch of ``samples`` using autocorrelation.

    Returns an all-``None`` profile when the signal holds no voiced speech, which is
    the honest answer for silence, a whisper or noise - not a failure.
    """

    data = _to_analysis_rate(_mono(samples), rate)
    frame = int(round(FRAME_SECONDS * ANALYSIS_RATE))
    hop = int(round(HOP_SECONDS * ANALYSIS_RATE))
    window = np.hanning(frame).astype(np.float32)

    blocks = _frames(data, frame=frame, hop=hop)
    if blocks.shape[0] == 0:
        return PitchProfile(0, 0, 0, None, None, None)

    if blocks.shape[0] > MAX_FRAMES:
        # Spread the sample across the whole signal instead of truncating it.
        picks = np.linspace(0, blocks.shape[0] - 1, MAX_FRAMES).astype(int)
        blocks = blocks[picks]

    windowed = blocks * window

    # Autocorrelation by FFT: the inverse transform of the power spectrum.
    size = 1 << int(math.ceil(math.log2(2 * frame)))

    def _autocorrelate(signal: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(signal, n=size, axis=1)
        return np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)

    correlation = _autocorrelate(windowed)

    # Normalising by the *window's* own autocorrelation removes the lag-dependent
    # taper: without it a long lag is penalised simply because the two windowed
    # copies overlap less, which is what makes a low-pitched voice - a lag near the
    # top of the search range - look unvoiced. What is left is a correlation
    # coefficient that is comparable across the whole range.
    window_correlation = _autocorrelate(np.tile(window, (1, 1)))[0]
    window_correlation = np.where(window_correlation > 1e-12, window_correlation, np.inf)
    correlation = correlation / window_correlation[None, :]

    zero_lag = correlation[:, :1]
    correlation = np.divide(
        correlation,
        zero_lag,
        out=np.zeros_like(correlation),
        where=zero_lag > 1e-12,
    )

    minimum_lag = max(2, int(ANALYSIS_RATE / F0_MAX_HZ))
    maximum_lag = min(frame - 1, int(ANALYSIS_RATE / F0_MIN_HZ))
    search = correlation[:, minimum_lag : maximum_lag + 1]
    if search.shape[1] == 0:
        return PitchProfile(int(blocks.shape[0]), 0, 0, None, None, None)

    # Take the *earliest* peak that is nearly as strong as the strongest, not the
    # strongest itself: a lag of two or three periods correlates as well as one, so
    # a plain maximum would report the pitch an octave low.
    rows = search.shape[0]
    strongest = np.max(search, axis=1)
    threshold = np.maximum(VOICED_CORRELATION, FIRST_PEAK_FRACTION * strongest)

    interior = np.zeros_like(search, dtype=bool)
    if search.shape[1] > 2:
        interior[:, 1:-1] = (search[:, 1:-1] >= search[:, :-2]) & (
            search[:, 1:-1] > search[:, 2:]
        )
    candidates = interior & (search >= threshold[:, None])

    no_candidate = search.shape[1]
    lags = np.where(candidates, np.arange(search.shape[1])[None, :], no_candidate)
    peaks = np.min(lags, axis=1)

    found = peaks < no_candidate
    strengths = np.where(found, search[np.arange(rows), np.clip(peaks, 0, search.shape[1] - 1)], 0.0)
    peak_hz = np.where(found, ANALYSIS_RATE / np.maximum(peaks + minimum_lag, 1), 0.0)

    # A frame is speech when it stands above the signal's own noise floor. Steady
    # material - music, hum, room tone - sits near its own floor whatever its level,
    # so this is the test that rejects a bed. A sustained delivery has no dynamic
    # range to stand above, so being near the loudest frame is accepted as well.
    energy = np.sqrt(np.mean(np.square(windowed), axis=1))
    if energy.size == 0 or float(np.max(energy)) <= 0.0:
        return PitchProfile(int(blocks.shape[0]), 0, 0, None, None, None)

    noise_floor = float(np.percentile(energy, 10))
    loud = float(np.max(energy))
    speech = (energy > noise_floor * (10.0 ** (SPEECH_MARGIN_DB / 20.0))) | (
        (energy > loud * STEADY_PEAK_FRACTION) & (strengths >= STEADY_PERIODICITY)
    )
    voiced = speech & (strengths >= VOICED_CORRELATION)

    pitches = peak_hz[voiced]
    speech_frames = int(np.count_nonzero(speech))
    if pitches.size == 0:
        return PitchProfile(int(blocks.shape[0]), speech_frames, 0, None, None, None)

    return PitchProfile(
        frames=int(blocks.shape[0]),
        speech_frames=speech_frames,
        voiced_frames=int(pitches.size),
        median_hz=float(np.median(pitches)),
        low_hz=float(np.percentile(pitches, 5)),
        high_hz=float(np.percentile(pitches, 95)),
    )


def profile_of_file(path: str | Path) -> PitchProfile:
    """Measure the pitch profile of an audio file.

    The file is read as it is, with no resampling to the pipeline rate or any other
    normalisation: the two sides of a comparison are measured identically, so a
    difference between them is a difference in delivery rather than in plumbing.
    """

    source = Path(path)
    if not source.is_file():
        raise InvalidAudioError(f"no audio to measure at {source}")

    try:
        samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
    except Exception as exc:
        raise InvalidAudioError(
            f"could not read {source.name}: {type(exc).__name__}: {exc}"
        ) from exc

    return estimate_f0(samples, int(rate))


@dataclass(frozen=True, slots=True)
class ProsodyComparison:
    """How much of a performance's delivery survived into the generated take."""

    original: PitchProfile
    generated: PitchProfile
    label: str = ""

    @property
    def usable(self) -> bool:
        """``True`` when both sides carried a pitch, so the ratios mean something."""

        return self.original.is_voiced and self.generated.is_voiced

    @property
    def median_shift_semitones(self) -> float | None:
        """How far the pitch centre moved, in semitones. ``0`` is an exact match."""

        if not self.usable:
            return None
        assert self.original.median_hz is not None and self.generated.median_hz is not None
        return 12.0 * math.log2(self.generated.median_hz / self.original.median_hz)

    @property
    def range_ratio(self) -> float | None:
        """Generated pitch range as a fraction of the original's.

        Near ``1.0`` means the delivery's expressiveness survived. Well below it -
        the common failure - means the take was flattened toward a neutral read,
        which is what losing emotion sounds like numerically.

        ``None`` when there is nothing to divide by: a monotone original has no range
        to be a fraction of, and reporting a huge or tiny ratio for one would be
        meaningless. :attr:`preserved` handles that case on its own terms.
        """

        if not self.usable:
            return None
        original = self.original.range_semitones
        generated = self.generated.range_semitones
        if original is None or generated is None or original <= FLAT_RANGE_SEMITONES:
            return None
        return generated / original

    @property
    def voiced_ratio_delta(self) -> float | None:
        """Change in the fraction of speech that carried a pitch.

        A large drop means the take lost periodicity - whispers flattened into
        ordinary speech, or a shouted line smoothed out.
        """

        if not self.original.speech_frames or not self.generated.speech_frames:
            return None
        return self.generated.voiced_ratio - self.original.voiced_ratio

    @property
    def preserved(self) -> bool:
        """``True`` when the delivery is recognisably the same performance.

        A monotone original - a held note, a flat read - is judged on its pitch
        centre alone, because it has no range to preserve. Judging it on a ratio
        would score a faithfully reproduced monotone line as a failure, which is
        exactly backwards.
        """

        shift = self.median_shift_semitones
        if shift is None or abs(shift) > PRESERVED_PITCH_SHIFT_SEMITONES:
            return False

        original = self.original.range_semitones
        generated = self.generated.range_semitones
        if original is None or generated is None:
            return False
        if original <= FLAT_RANGE_SEMITONES:
            return generated <= FLAT_RANGE_SEMITONES * FLAT_TOLERANCE_FACTOR
        return generated / original >= PRESERVED_RANGE_RATIO

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the comparison."""

        return {
            "label": self.label,
            "usable": self.usable,
            "median_shift_semitones": (
                None if self.median_shift_semitones is None
                else round(self.median_shift_semitones, 3)
            ),
            "range_ratio": (
                None if self.range_ratio is None else round(self.range_ratio, 4)
            ),
            "voiced_ratio_delta": (
                None if self.voiced_ratio_delta is None
                else round(self.voiced_ratio_delta, 4)
            ),
            "preserved": self.preserved,
            "original": self.original.as_dict(),
            "generated": self.generated.as_dict(),
        }


def compare_profiles(
    original: PitchProfile,
    generated: PitchProfile,
    *,
    label: str = "",
) -> ProsodyComparison:
    """Compare two measured pitch profiles."""

    return ProsodyComparison(original=original, generated=generated, label=label)


__all__ = [
    "ANALYSIS_RATE",
    "F0_MAX_HZ",
    "F0_MIN_HZ",
    "FIRST_PEAK_FRACTION",
    "FLAT_RANGE_SEMITONES",
    "FLAT_TOLERANCE_FACTOR",
    "FRAME_SECONDS",
    "HOP_SECONDS",
    "MAX_FRAMES",
    "PRESERVED_PITCH_SHIFT_SEMITONES",
    "PRESERVED_RANGE_RATIO",
    "SPEECH_MARGIN_DB",
    "STEADY_PEAK_FRACTION",
    "STEADY_PERIODICITY",
    "VOICED_CORRELATION",
    "InvalidAudioError",
    "PitchProfile",
    "ProsodyComparison",
    "ProsodyError",
    "compare_profiles",
    "estimate_f0",
    "profile_of_file",
]
