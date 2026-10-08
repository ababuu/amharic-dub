"""Quality control for a dubbed run: measure it, do not just listen to it.

This module answers one question about a run's output - *how good is it, and
where exactly is it worst?* - so that a change to any stage can be judged instead
of guessed at. It sits beside the pipeline rather than in it, consuming what the
stages produced::

    [AlignedClip] + [AdaptedDialogue] + [CrosstalkRegion]
        -> build_qc_report() -> QcReport

What it measures, and why each one
----------------------------------
* **Duration fit.** The dominant complaint about machine-dubbed audio in the one
  large human study of professional dubbing is an unnatural speaking rate - "too
  slow, too fast, or too uneven" - so the report carries the distribution of fit
  residuals and tempo factors, not a single average that would hide a bad tail.
* **Speaking rate, in syllables per second.** One Fidel character is one syllable,
  which is what the script encodes, so syllable counts are available without a
  phonemiser or a grapheme-to-phoneme model - neither of which exists for Amharic.
  A rate outside the plausible band for speech is a line no performer could
  deliver, however well it "fits".
* **Crosstalk.** Simultaneous speech is exactly what exclusive diarization cannot
  describe, so it is reported here as a known uncertainty rather than discovered
  later by listening.
* **Pronunciation (opt-in).** A round trip through an Amharic ASR model gives a
  character error rate against the text that was synthesized, which is the only
  automated check that the dub is *intelligible* rather than merely present. It
  needs a model, so it is an injected callable and is simply absent when no model
  is available - reported as not measured, never as a pass.

Deliberate non-goals
--------------------
* **No models of its own.** Nothing here imports torch, ASR or an embedding
  model. Pronunciation and speaker similarity are injected callables, so the whole
  model-free report runs anywhere, including in the test suite.
* **No thresholds that fail a run.** The report states what was measured. Deciding
  what is good enough is a project decision, not a module's.
* **No audio decoding.** Everything numeric comes from the metadata the stages
  already produced; only an injected transcriber reads a file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from app.pipeline.amharic_text import (
    HOMOPHONE_FAMILIES,
    SYLLABLE_RANGES,
    InvalidTextError,
    count_syllables,
    has_latin,
    has_pronounceable_text,
    latin_spans,
    syllable_sequence,
)
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.timing import EXACT_FIT_TOLERANCE_SECONDS, AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import TtsClip

#: A line whose residual is inside this fraction of its window is called a *close*
#: fit. Ten percent is the tolerance the dubbing literature quotes when it reports
#: "how many lines are within tolerance", so the figure is comparable to published
#: work rather than invented here.
CLOSE_FIT_RATIO = 0.10

#: A spoken delivery lives in roughly this band. Outside it a line is either
#: impossible to perform or unnaturally drawn out, so those lines are counted and
#: listed instead of being averaged away. The band is deliberately wide: it exists
#: to catch the pathological, not to grade ordinary variation.
PLAUSIBLE_SYLLABLES_PER_SECOND = (1.5, 9.0)

#: The name this module has always exposed for the comparison form of a string.
normalise_for_comparison = syllable_sequence


class QcError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidMeasurementError(QcError, ValueError):
    """An injected measurement returned something unusable."""


def character_error_rate(expected: str, actual: str) -> float:
    """Return the character error rate between two syllable sequences.

    The rate is the Levenshtein distance divided by the length of ``expected``, so
    ``0.0`` is an exact match and ``1.0`` means as many edits as there were
    characters. An empty ``expected`` has no rate and raises.
    """

    source = syllable_sequence(expected)
    target = syllable_sequence(actual)
    if not source:
        raise InvalidMeasurementError(
            "a character error rate needs expected text with at least one syllable"
        )
    if not target and source:
        return 1.0

    previous = list(range(len(target) + 1))
    for row, source_character in enumerate(source, start=1):
        current = [row]
        for column, target_character in enumerate(target, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (source_character != target_character),
                )
            )
        previous = current

    return previous[-1] / len(source)


@dataclass(frozen=True, slots=True)
class LineMetrics:
    """What quality control can say about one delivered line.

    Every value comes from the metadata the earlier stages already recorded, so
    building one costs no audio decoding and no model.
    """

    index: int
    speaker_id: str
    amharic: str
    syllables: int
    original_window: float
    speech_duration: float
    tempo: float
    required_tempo: float
    residual: float
    notes: tuple[str, ...] = ()

    @property
    def speaker(self) -> str:
        """Alias for the diarized speaker id."""

        return self.speaker_id

    @property
    def fits(self) -> bool:
        """``True`` when the delivered speech matches its window exactly."""

        return abs(self.residual) <= EXACT_FIT_TOLERANCE_SECONDS

    @property
    def residual_ratio(self) -> float:
        """The residual as a fraction of the window, or ``0.0`` for an empty window."""

        if self.original_window <= 0:
            return 0.0
        return self.residual / self.original_window

    @property
    def close_fit(self) -> bool:
        """``True`` when the line landed within :data:`CLOSE_FIT_RATIO` of its window."""

        return abs(self.residual_ratio) <= CLOSE_FIT_RATIO

    @property
    def syllables_per_second(self) -> float:
        """How fast the line is *delivered*, in syllables per second."""

        if self.speech_duration <= 0:
            return 0.0
        return self.syllables / self.speech_duration

    @property
    def natural_speech_duration(self) -> float:
        """How long the line would have run before it was stretched to fit.

        Dividing the applied tempo back out recovers the delivery the engine
        produced, whatever the fitter did to it afterwards - which is what makes the
        natural speaking rate measurable from a run that had to stretch its lines.
        """

        return self.speech_duration * self.tempo

    @property
    def natural_syllables_per_second(self) -> float:
        """The rate the engine actually speaks at, before any stretching.

        This is the figure the syllable budget should be computed from. Computing it
        from the delivered duration instead would read a stretched line as evidence
        that the engine speaks slower than it does, and the error compounds.
        """

        natural = self.natural_speech_duration
        if natural <= 0:
            return 0.0
        return self.syllables / natural

    @property
    def plausible_rate(self) -> bool:
        """``True`` when the delivery rate is inside the plausible band."""

        low, high = PLAUSIBLE_SYLLABLES_PER_SECOND
        return low <= self.syllables_per_second <= high

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of this line's measurements."""

        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "syllables": self.syllables,
            "original_window": round(self.original_window, 3),
            "speech_duration": round(self.speech_duration, 3),
            "tempo": round(self.tempo, 4),
            "required_tempo": round(self.required_tempo, 4),
            "residual": round(self.residual, 3),
            "residual_ratio": round(self.residual_ratio, 4),
            "fits": self.fits,
            "close_fit": self.close_fit,
            "syllables_per_second": round(self.syllables_per_second, 3),
            "natural_syllables_per_second": round(
                self.natural_syllables_per_second, 3
            ),
            "plausible_rate": self.plausible_rate,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class DurationFitReport:
    """The distribution of how well the delivered lines fit their windows."""

    lines: int
    fitted: int
    close_fits: int
    unfitted: int
    stretched: int
    implausible_rate: int
    mean_absolute_residual: float
    worst_residual: float
    mean_tempo: float
    slowest_tempo: float
    fastest_tempo: float

    @property
    def close_fit_ratio(self) -> float:
        """Fraction of lines inside :data:`CLOSE_FIT_RATIO` of their window."""

        if not self.lines:
            return 0.0
        return self.close_fits / self.lines

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the distribution."""

        return {
            "lines": self.lines,
            "fitted": self.fitted,
            "close_fits": self.close_fits,
            "close_fit_ratio": round(self.close_fit_ratio, 4),
            "unfitted": self.unfitted,
            "stretched": self.stretched,
            "implausible_rate": self.implausible_rate,
            "mean_absolute_residual": round(self.mean_absolute_residual, 3),
            "worst_residual": round(self.worst_residual, 3),
            "mean_tempo": round(self.mean_tempo, 4),
            "slowest_tempo": round(self.slowest_tempo, 4),
            "fastest_tempo": round(self.fastest_tempo, 4),
            "close_fit_tolerance": CLOSE_FIT_RATIO,
            "exact_fit_tolerance_seconds": EXACT_FIT_TOLERANCE_SECONDS,
        }


@dataclass(frozen=True, slots=True)
class PronunciationReport:
    """How intelligible the dubbed lines are, judged by an ASR round trip."""

    measured: int
    unmeasured: int
    mean_cer: float
    worst_cer: float
    worst_index: int | None
    model: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the measurement."""

        return {
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "mean_cer": round(self.mean_cer, 4),
            "worst_cer": round(self.worst_cer, 4),
            "worst_index": self.worst_index,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class QcReport:
    """Everything quality control could say about one run."""

    duration: DurationFitReport
    lines: tuple[LineMetrics, ...]
    crosstalk_regions: int
    crosstalk_seconds: float
    pronunciation: PronunciationReport | None = None

    @property
    def worst_lines(self) -> tuple[LineMetrics, ...]:
        """The lines furthest from their window, worst first."""

        return tuple(
            sorted(self.lines, key=lambda line: abs(line.residual), reverse=True)
        )

    @property
    def implausible_lines(self) -> tuple[LineMetrics, ...]:
        """Lines whose delivery rate falls outside the plausible band."""

        return tuple(line for line in self.lines if not line.plausible_rate)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the whole report."""

        return {
            "duration": self.duration.as_dict(),
            "pronunciation": (
                None if self.pronunciation is None else self.pronunciation.as_dict()
            ),
            "crosstalk": {
                "regions": self.crosstalk_regions,
                "seconds": round(self.crosstalk_seconds, 3),
            },
            "lines": [line.as_dict() for line in self.lines],
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        duration = self.duration
        parts = [
            f"{duration.close_fits}/{duration.lines} line(s) within "
            f"{CLOSE_FIT_RATIO:.0%} of their window",
        ]
        if duration.unfitted:
            parts.append(f"{duration.unfitted} not fitted")
        if duration.implausible_rate:
            parts.append(f"{duration.implausible_rate} at an implausible rate")
        if duration.stretched:
            parts.append(f"{duration.stretched} stretched")
        if self.crosstalk_regions:
            parts.append(
                f"{self.crosstalk_regions} crosstalk region(s) "
                f"({self.crosstalk_seconds:.2f}s)"
            )
        if self.pronunciation is None:
            parts.append("pronunciation not measured")
        else:
            parts.append(
                f"pronunciation CER {self.pronunciation.mean_cer:.3f} over "
                f"{self.pronunciation.measured} line(s)"
            )
        return ", ".join(parts)


def measure_lines(alignment: Iterable[AlignedClip]) -> tuple[LineMetrics, ...]:
    """Measure every aligned line, in the order they were aligned."""

    metrics: list[LineMetrics] = []
    for aligned in alignment:
        dialogue = aligned.clip.dialogue
        metrics.append(
            LineMetrics(
                index=aligned.index,
                speaker_id=dialogue.speaker_id,
                amharic=dialogue.amharic,
                syllables=count_syllables(dialogue.amharic),
                original_window=aligned.original_window,
                speech_duration=aligned.speech_duration,
                tempo=aligned.tempo,
                required_tempo=aligned.required_tempo,
                residual=aligned.residual,
                notes=tuple(aligned.notes),
            )
        )
    return tuple(metrics)


def summarise_duration(lines: Iterable[LineMetrics]) -> DurationFitReport:
    """Summarise the fit of the measured lines."""

    measured = tuple(lines)
    if not measured:
        return DurationFitReport(
            lines=0,
            fitted=0,
            close_fits=0,
            unfitted=0,
            stretched=0,
            implausible_rate=0,
            mean_absolute_residual=0.0,
            worst_residual=0.0,
            mean_tempo=0.0,
            slowest_tempo=0.0,
            fastest_tempo=0.0,
        )

    residuals = [line.residual for line in measured]
    tempos = [line.tempo for line in measured]
    return DurationFitReport(
        lines=len(measured),
        fitted=sum(1 for line in measured if line.fits),
        close_fits=sum(1 for line in measured if line.close_fit),
        unfitted=sum(1 for line in measured if not line.fits),
        stretched=sum(1 for line in measured if line.tempo != 1.0),
        implausible_rate=sum(1 for line in measured if not line.plausible_rate),
        mean_absolute_residual=fmean(abs(residual) for residual in residuals),
        worst_residual=max(residuals, key=abs),
        mean_tempo=fmean(tempos),
        slowest_tempo=min(tempos),
        fastest_tempo=max(tempos),
    )


def measure_pronunciation(
    dialogue: Iterable[AdaptedDialogue],
    clips: Iterable[TtsClip],
    *,
    transcribe: Callable[[Path], str],
    model: str = "unknown",
) -> PronunciationReport:
    """Measure the round-trip character error rate of the delivered lines.

    ``transcribe`` is handed the path of a *converted* clip - the audio that was
    actually voiced - and must return what it heard. Injecting it keeps this
    module free of any model dependency: an Amharic ASR model in production, a
    fixed string in a test.

    Clips are matched to dialogue by index. A clip that cannot be measured - a
    transcriber that fails, or an empty transcript - is counted as unmeasured
    rather than as an error, so one bad line cannot make the run look better or
    worse than it is.
    """

    expected = dict(enumerate(dialogue))
    rates: list[tuple[int, float]] = []
    unmeasured = 0

    for clip in clips:
        index = clip.index
        line = expected.get(index)
        if line is None:
            unmeasured += 1
            continue

        path = Path(clip.audio_path)
        try:
            heard = transcribe(path)
        except Exception:  # a measurement must never fail a run
            unmeasured += 1
            continue

        if not isinstance(heard, str):
            raise InvalidMeasurementError(
                f"the transcriber returned {type(heard).__name__} for line "
                f"{index}, expected the transcript as a string"
            )
        if not heard.strip():
            unmeasured += 1
            continue

        rates.append((index, character_error_rate(line.amharic, heard)))

    if not rates:
        return PronunciationReport(
            measured=0,
            unmeasured=unmeasured,
            mean_cer=0.0,
            worst_cer=0.0,
            worst_index=None,
            model=model,
        )

    worst_index, worst = max(rates, key=lambda item: item[1])
    return PronunciationReport(
        measured=len(rates),
        unmeasured=unmeasured,
        mean_cer=fmean(rate for _, rate in rates),
        worst_cer=worst,
        worst_index=worst_index,
        model=model,
    )


def build_qc_report(
    alignment: Iterable[AlignedClip],
    *,
    dialogue: Iterable[AdaptedDialogue] = (),
    clips: Iterable[TtsClip] = (),
    crosstalk: Iterable[CrosstalkRegion] = (),
    transcribe: Callable[[Path], str] | None = None,
    transcription_model: str = "unknown",
) -> QcReport:
    """Build the model-free report, measuring pronunciation when asked to.

    ``transcribe`` is the only thing that needs a model; without it the report is
    complete except for pronunciation, which is then reported as not measured
    rather than silently omitted.
    """

    lines = measure_lines(alignment)
    regions = tuple(crosstalk)

    pronunciation: PronunciationReport | None = None
    if transcribe is not None:
        pronunciation = measure_pronunciation(
            dialogue, clips, transcribe=transcribe, model=transcription_model
        )

    return QcReport(
        duration=summarise_duration(lines),
        lines=lines,
        crosstalk_regions=len(regions),
        crosstalk_seconds=sum(region.duration for region in regions),
        pronunciation=pronunciation,
    )


__all__ = [
    "CLOSE_FIT_RATIO",
    # Re-exported from ``amharic_text``: the syllable is the unit of *both* a
    # delivery rate here and a timing budget in ``dialogue_context``, and it would
    # be surprising for a caller measuring a run to reach into another module for it.
    # The Roman-script helpers come with it, because whether a borrowed word can be
    # read by the engine is a measurement's business too.
    "HOMOPHONE_FAMILIES",
    "PLAUSIBLE_SYLLABLES_PER_SECOND",
    "SYLLABLE_RANGES",
    "DurationFitReport",
    "InvalidMeasurementError",
    "InvalidTextError",
    "LineMetrics",
    "PronunciationReport",
    "QcError",
    "QcReport",
    "build_qc_report",
    "character_error_rate",
    "count_syllables",
    "has_latin",
    "has_pronounceable_text",
    "latin_spans",
    "measure_lines",
    "measure_pronunciation",
    "normalise_for_comparison",
    "summarise_duration",
]
