"""Evaluation: does this run actually dub well, and did anything regress?

:mod:`app.pipeline.qc` measures one run. This module answers the two questions that
follow, which are the ones that decide whether the system is getting better:

* **Is the evaluation material representative?** A run can score well because it was
  easy - a quiet scene between two characters - so the coverage check reports which
  of the hard cases the material actually contains. Improving the system on scenes
  it has never seen is guesswork, and this is what stops the evaluation from being
  an easy set that always passes.
* **Did a change make it worse?** A baseline of metrics can be saved and compared,
  so a stage change is judged by measurement rather than by memory.

The three quality signals map onto the project's own priorities:

===========================  ==========================================
Priority                     How it is measured here
===========================  ==========================================
Natural timing and pacing     fit distribution and delivery rate (``qc``)
Consistent character voices   speaker embedding similarity to the character's own
                              reference, and the *spread* across the whole film
Preservation of performance   pitch centring, pitch range and periodicity of the
                              generated take against the original actor's prompt
                              (:mod:`app.pipeline.prosody`)
===========================  ==========================================

Everything model-backed is an **injected callable**. Nothing here downloads
anything, so the coverage check and the baseline comparison run anywhere, and the
embedding and transcriber measurements appear only when a caller supplies them -
reported as not measured otherwise, never as a pass.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from app.pipeline.prosody import (
    PitchProfile,
    ProsodyComparison,
    compare_profiles,
    profile_of_file,
)
from app.pipeline.qc import QcReport, build_qc_report
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.timing import AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import TtsClip
from app.pipeline.voice_profiles import VoiceProfile

#: Categories the evaluation material should contain, because each one breaks a
#: different part of the pipeline. The names are what a coverage report prints.
GOLDEN_CATEGORIES: tuple[str, ...] = (
    "whisper",
    "shout",
    "short_line",
    "long_line",
    "rapid_turn_taking",
    "overlapping_speech",
    "low_intensity",
    "high_intensity",
    "code_switching",
    "recurring_speaker",
    "crowded_scene",
)

#: A line at or below this length is a one-word interjection or a clipped retort,
#: which is where a duration-aware adaptation has least room to work.
SHORT_LINE_SECONDS = 1.0

#: A line at or above this length is a monologue, which is where a delivery rate
#: that has drifted becomes audible.
LONG_LINE_SECONDS = 8.0

#: A gap at or below this between consecutive lines is rapid turn-taking, which is
#: where pauses and pacing matter most and where dubbing most often feels wrong.
RAPID_TURN_SECONDS = 0.2

#: Intensity at or below this is a restrained reading; at or above the high mark it
#: is an outburst. Both extremes are where the sampling controls and the conversion
#: are most likely to flatten a performance.
LOW_INTENSITY = 0.2
HIGH_INTENSITY = 0.8

#: A speaker appearing at least this many times is a recurring character, so the
#: material can say something about consistency across a film.
RECURRING_SPEAKER_LINES = 5

#: A scene with at least this many distinct speakers is where crosstalk and the
#: crowd side of diarization are actually exercised.
CROWDED_SCENE_SPEAKERS = 3

#: Words that indicate a whispered delivery. Matched case-insensitively against the
#: emotion and delivery text the adaptation stage produced, because there is no
#: loudness measurement to lean on and inventing one would be worse than reading what
#: the dialogue model already said.
WHISPER_MARKERS = ("whisper", "hushed", "murmur", "under their breath")

#: Words that indicate a shouted delivery.
SHOUT_MARKERS = ("shout", "shouting", "yell", "scream", "bellow", "roar")


class EvaluationError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidMeasurementError(EvaluationError, ValueError):
    """An injected measurement returned something unusable."""


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Which hard cases the material behind an evaluation actually contains."""

    counts: Mapping[str, int]
    lines: int

    @property
    def missing(self) -> tuple[str, ...]:
        """Categories with no representative in the material, in declaration order."""

        return tuple(
            category for category in GOLDEN_CATEGORIES if not self.counts.get(category)
        )

    @property
    def covered(self) -> tuple[str, ...]:
        """Categories that are represented."""

        return tuple(
            category for category in GOLDEN_CATEGORIES if self.counts.get(category)
        )

    @property
    def covered_ratio(self) -> float:
        """Fraction of the categories that are represented."""

        return len(self.covered) / len(GOLDEN_CATEGORIES)

    @property
    def representative(self) -> bool:
        """``True`` when every category the pipeline is judged on is present.

        A run that is not representative is not evidence: it may pass simply
        because the hard cases were never in it.
        """

        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the coverage."""

        return {
            "lines": self.lines,
            "counts": {name: int(count) for name, count in sorted(self.counts.items())},
            "covered": list(self.covered),
            "missing": list(self.missing),
            "covered_ratio": round(self.covered_ratio, 4),
            "representative": self.representative,
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if self.representative:
            return (
                f"coverage: all {len(GOLDEN_CATEGORIES)} categories represented "
                f"across {self.lines} line(s)"
            )
        return (
            f"coverage: {len(self.covered)}/{len(GOLDEN_CATEGORIES)} categories; "
            f"missing {', '.join(self.missing)}"
        )


def _text_of(line: AdaptedDialogue) -> str:
    return f"{line.emotion} {line.delivery}".lower()


def _has_latin(text: str) -> bool:
    """``True`` when the line contains Latin letters, i.e. a borrowed English word."""

    return any("a" <= character.lower() <= "z" for character in text)


def measure_coverage(
    dialogue: Sequence[AdaptedDialogue],
    *,
    crosstalk: Sequence[CrosstalkRegion] = (),
) -> CoverageReport:
    """Report which of :data:`GOLDEN_CATEGORIES` the material contains.

    Everything is derived from metadata the pipeline already produced - line
    timings, the performance direction, the speakers and the reported crosstalk - so
    a coverage report costs nothing and can be printed for every run.
    """

    counts = {category: 0 for category in GOLDEN_CATEGORIES}
    if not dialogue:
        return CoverageReport(counts=counts, lines=0)

    windows = [line.duration for line in dialogue]
    speakers = [line.speaker_id for line in dialogue]
    speech_per_speaker: dict[str, int] = {}
    for speaker in speakers:
        speech_per_speaker[speaker] = speech_per_speaker.get(speaker, 0) + 1

    for line, window in zip(dialogue, windows):
        if window <= SHORT_LINE_SECONDS:
            counts["short_line"] += 1
        if window >= LONG_LINE_SECONDS:
            counts["long_line"] += 1
        if line.intensity <= LOW_INTENSITY:
            counts["low_intensity"] += 1
        if line.intensity >= HIGH_INTENSITY:
            counts["high_intensity"] += 1
        if _has_latin(line.amharic):
            counts["code_switching"] += 1

        text = _text_of(line)
        if any(marker in text for marker in WHISPER_MARKERS):
            counts["whisper"] += 1
        if any(marker in text for marker in SHOUT_MARKERS):
            counts["shout"] += 1

    for previous, following in zip(dialogue, dialogue[1:]):
        gap = following.start - previous.end
        if (
            previous.speaker_id != following.speaker_id
            and 0.0 <= gap <= RAPID_TURN_SECONDS
        ):
            counts["rapid_turn_taking"] += 1
            break

    if crosstalk:
        counts["overlapping_speech"] = len(crosstalk)

    if any(count >= RECURRING_SPEAKER_LINES for count in speech_per_speaker.values()):
        counts["recurring_speaker"] = max(speech_per_speaker.values())

    if len(set(speakers)) >= CROWDED_SCENE_SPEAKERS:
        counts["crowded_scene"] = len(set(speakers))

    return CoverageReport(counts=counts, lines=len(dialogue))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or ``0.0`` if either is empty."""

    if len(left) != len(right):
        raise InvalidMeasurementError(
            f"embeddings must have the same width, got {len(left)} and {len(right)}"
        )
    if not left:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class SpeakerIdentity:
    """How well one character held their own voice across the material."""

    speaker_id: str
    samples: int
    mean_similarity: float
    minimum_similarity: float
    spread: float

    @property
    def consistent(self) -> bool:
        """``True`` when every sample held the character's identity well.

        The floor matters more than the mean here: a character who sounds right four
        times and like somebody else once is the failure an audience notices, and an
        average would hide it.
        """

        return self.minimum_similarity >= 0.60 and self.samples >= 2

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of this character's identity."""

        return {
            "speaker_id": self.speaker_id,
            "samples": self.samples,
            "mean_similarity": round(self.mean_similarity, 4),
            "minimum_similarity": round(self.minimum_similarity, 4),
            "spread": round(self.spread, 4),
            "consistent": self.consistent,
        }


@dataclass(frozen=True, slots=True)
class SpeakerIdentityReport:
    """Identity consistency for every character that could be measured."""

    characters: tuple[SpeakerIdentity, ...]
    unmeasured: int
    embedder: str

    @property
    def measured(self) -> int:
        """How many clips were measured."""

        return sum(character.samples for character in self.characters)

    @property
    def overall_mean_similarity(self) -> float:
        """Mean similarity over every clip, weighted by samples."""

        if not self.measured:
            return 0.0
        return (
            sum(character.mean_similarity * character.samples for character in self.characters)
            / self.measured
        )

    @property
    def worst(self) -> SpeakerIdentity | None:
        """The character with the lowest worst-case similarity."""

        if not self.characters:
            return None
        return min(self.characters, key=lambda character: character.minimum_similarity)

    @property
    def all_consistent(self) -> bool:
        """``True`` when every measured character held their identity."""

        return bool(self.characters) and all(
            character.consistent for character in self.characters
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the report."""

        return {
            "embedder": self.embedder,
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "overall_mean_similarity": round(self.overall_mean_similarity, 4),
            "all_consistent": self.all_consistent,
            "characters": [character.as_dict() for character in self.characters],
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if not self.characters:
            return "voice identity: not measured"
        return (
            f"voice identity: {len(self.characters)} character(s), mean similarity "
            f"{self.overall_mean_similarity:.3f}, worst "
            f"{min(c.minimum_similarity for c in self.characters):.3f}"
        )


def measure_speaker_identity(
    clips: Iterable[TtsClip],
    *,
    profiles: Mapping[str, VoiceProfile] | None = None,
    embed: Callable[[Path], Sequence[float]],
    embedder: str = "unknown",
) -> SpeakerIdentityReport:
    """Measure how far each generated line drifted from its character's identity.

    Each clip is compared with the reference the character was converted *towards*,
    so the measurement is of the pipeline's own contract - "this should sound like
    that character" - rather than of an external notion of voice quality.

    ``embed`` is handed the path of an audio file and must return a speaker
    embedding. Injecting it keeps this module free of a model dependency, and a clip
    whose embedding cannot be measured is counted as unmeasured rather than as a
    failure, so one unreadable file cannot make a film look good or bad.
    """

    by_speaker: dict[str, list[float]] = {}
    unmeasured = 0

    for clip in clips:
        speaker_id = clip.dialogue.speaker_id
        reference = clip.voice_reference_path
        if profiles is not None:
            profile = profiles.get(speaker_id)
            if profile is None:
                unmeasured += 1
                continue
            reference = profile.reference_audio

        try:
            generated = embed(Path(clip.audio_path))
            target = embed(Path(reference))
        except Exception:  # a measurement must never fail a run
            unmeasured += 1
            continue

        if not isinstance(generated, Sequence) or not isinstance(target, Sequence):
            raise InvalidMeasurementError(
                f"the embedder returned {type(generated).__name__} for {clip.audio_path}"
            )

        by_speaker.setdefault(speaker_id, []).append(
            cosine_similarity(target, generated)
        )

    characters = tuple(
        SpeakerIdentity(
            speaker_id=speaker_id,
            samples=len(values),
            mean_similarity=fmean(values),
            minimum_similarity=min(values),
            spread=pstdev(values) if len(values) > 1 else 0.0,
        )
        for speaker_id, values in sorted(by_speaker.items())
    )
    return SpeakerIdentityReport(
        characters=characters, unmeasured=unmeasured, embedder=embedder
    )


@dataclass(frozen=True, slots=True)
class PerformancePreservationReport:
    """Whether the generated takes kept the original actors' deliveries."""

    comparisons: tuple[ProsodyComparison, ...]
    unmeasured: int

    @property
    def measured(self) -> int:
        """How many lines could be compared."""

        return len(self.comparisons)

    @property
    def unusable(self) -> int:
        """Lines where one side carried no pitch, so nothing could be concluded."""

        return sum(1 for comparison in self.comparisons if not comparison.usable)

    @property
    def usable_comparisons(self) -> tuple[ProsodyComparison, ...]:
        """The comparisons a ratio can actually be computed from."""

        return tuple(
            comparison for comparison in self.comparisons if comparison.usable
        )

    @property
    def preserved_ratio(self) -> float:
        """Fraction of measurable lines whose delivery was recognisably kept."""

        usable = self.usable_comparisons
        if not usable:
            return 0.0
        return sum(1 for comparison in usable if comparison.preserved) / len(usable)

    @property
    def mean_pitch_range_ratio(self) -> float | None:
        """Mean generated pitch range as a fraction of the original's.

        Below ``1.0`` is the failure that matters: the take was flattened toward a
        neutral read, which is what losing the performance sounds like numerically.
        """

        ratios = [
            comparison.range_ratio
            for comparison in self.usable_comparisons
            if comparison.range_ratio is not None
        ]
        return fmean(ratios) if ratios else None

    @property
    def mean_absolute_pitch_shift_semitones(self) -> float | None:
        """Mean absolute move of the pitch centre, in semitones."""

        shifts = [
            comparison.median_shift_semitones
            for comparison in self.usable_comparisons
            if comparison.median_shift_semitones is not None
        ]
        return fmean(abs(shift) for shift in shifts) if shifts else None

    @property
    def worst(self) -> ProsodyComparison | None:
        """The line whose delivery was distorted most."""

        usable = [
            comparison
            for comparison in self.usable_comparisons
            if comparison.range_ratio is not None
        ]
        if not usable:
            return None
        return min(usable, key=lambda comparison: comparison.range_ratio or 0.0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the report."""

        range_ratio = self.mean_pitch_range_ratio
        shift = self.mean_absolute_pitch_shift_semitones
        return {
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "unusable": self.unusable,
            "preserved_ratio": round(self.preserved_ratio, 4),
            "mean_pitch_range_ratio": (
                None if range_ratio is None else round(range_ratio, 4)
            ),
            "mean_absolute_pitch_shift_semitones": (
                None if shift is None else round(shift, 3)
            ),
            "lines": [comparison.as_dict() for comparison in self.comparisons],
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if not self.usable_comparisons:
            return "performance: not measured"
        range_ratio = self.mean_pitch_range_ratio or 0.0
        shift = self.mean_absolute_pitch_shift_semitones or 0.0
        return (
            f"performance: {self.preserved_ratio:.0%} of {len(self.usable_comparisons)} "
            f"line(s) preserved, pitch range {range_ratio:.0%} of the original, "
            f"centre moved {shift:.2f} semitone(s)"
        )


def measure_performance_preservation(
    clips: Iterable[TtsClip],
    *,
    profiler: Callable[[Path], PitchProfile] = profile_of_file,
) -> PerformancePreservationReport:
    """Compare each generated take with the original performance it was prompted on.

    ``clip.performance_reference_path`` is the actor's own audio for that line, cut
    from the clean speech stem, and ``clip.audio_path`` is the take the character's
    voice was converted into. Comparing the two is the only direct check that the
    pipeline did what its design claims: carry the performance, replace the timbre.

    This reads audio, so it is opt-in rather than part of every run.
    """

    comparisons: list[ProsodyComparison] = []
    unmeasured = 0

    for clip in clips:
        label = f"{clip.dialogue.speaker_id}#{clip.index}"
        try:
            original = profiler(Path(clip.performance_reference_path))
            generated = profiler(Path(clip.audio_path))
        except Exception:  # a measurement must never fail a run
            unmeasured += 1
            continue
        comparisons.append(compare_profiles(original, generated, label=label))

    return PerformancePreservationReport(
        comparisons=tuple(comparisons), unmeasured=unmeasured
    )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Everything an evaluation run could establish about one dub."""

    qc: QcReport
    coverage: CoverageReport
    performance: PerformancePreservationReport | None = None
    identity: SpeakerIdentityReport | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the whole evaluation."""

        return {
            "qc": self.qc.as_dict(),
            "coverage": self.coverage.as_dict(),
            "performance": None if self.performance is None else self.performance.as_dict(),
            "identity": None if self.identity is None else self.identity.as_dict(),
        }

    def summary(self) -> str:
        """Return the multi-line summary a run prints."""

        lines = [self.qc.summary(), self.coverage.summary()]
        if self.performance is not None:
            lines.append(self.performance.summary())
        if self.identity is not None:
            lines.append(self.identity.summary())
        return "\n".join(lines)


def build_evaluation_report(
    alignment: Iterable[AlignedClip],
    *,
    dialogue: Iterable[AdaptedDialogue] = (),
    clips: Iterable[TtsClip] = (),
    crosstalk: Iterable[CrosstalkRegion] = (),
    profiles: Mapping[str, VoiceProfile] | None = None,
    transcribe: Callable[[Path], str] | None = None,
    transcription_model: str = "unknown",
    embed: Callable[[Path], Sequence[float]] | None = None,
    embedder: str = "unknown",
    measure_performance: bool = False,
    profiler: Callable[[Path], PitchProfile] = profile_of_file,
) -> EvaluationReport:
    """Build the evaluation report, measuring whatever the caller supplied.

    ``measure_performance`` reads audio for every line, so it is off by default: it
    belongs to an evaluation run, not to every run.
    """

    lines = tuple(dialogue)
    voiced_clips = tuple(clips)

    qc = build_qc_report(
        alignment,
        dialogue=lines,
        clips=voiced_clips,
        crosstalk=crosstalk,
        transcribe=transcribe,
        transcription_model=transcription_model,
    )
    coverage = measure_coverage(lines, crosstalk=tuple(crosstalk))

    identity = None
    if embed is not None:
        identity = measure_speaker_identity(
            voiced_clips, profiles=profiles, embed=embed, embedder=embedder
        )

    performance = None
    if measure_performance:
        performance = measure_performance_preservation(voiced_clips, profiler=profiler)

    return EvaluationReport(
        qc=qc, coverage=coverage, performance=performance, identity=identity
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

#: Metrics where a larger value is better. Everything numeric that is not named here
#: is treated as "smaller is better" when it is in :data:`LOWER_IS_BETTER`, and as
#: directionless otherwise - a directionless metric that moves is reported as a
#: change, not as a regression, because nothing here can say which way is better.
HIGHER_IS_BETTER = frozenset(
    {
        "close_fit_ratio",
        "close_fits",
        "fitted",
        "covered_ratio",
        "measured",
        "overall_mean_similarity",
        "preserved_ratio",
        "range_ratio",
        "mean_pitch_range_ratio",
        "mean_similarity",
        "minimum_similarity",
        "voiced_ratio",
    }
)

LOWER_IS_BETTER = frozenset(
    {
        "crosstalk_seconds",
        "implausible_rate",
        "mean_absolute_residual",
        "mean_cer",
        "mean_absolute_pitch_shift_semitones",
        "unfitted",
        "unmeasured",
        "unusable",
        "worst_cer",
        "worst_residual",
    }
)

#: How far a metric may move, as a fraction of its baseline, before it is reported.
#: Loose on purpose: this is a tripwire for a real regression, not a noise detector,
#: and a metric that legitimately moves more can be given its own tolerance.
DEFAULT_TOLERANCE = 0.10

#: Per-metric overrides, matched on the metric's leaf name.
TOLERANCES: Mapping[str, float] = {
    # Identity and delivery are the qualities least tolerant of drift.
    "overall_mean_similarity": 0.02,
    "mean_similarity": 0.02,
    "minimum_similarity": 0.05,
    "preserved_ratio": 0.05,
    "mean_pitch_range_ratio": 0.05,
    "close_fit_ratio": 0.05,
}


@dataclass(frozen=True, slots=True)
class MetricChange:
    """One metric compared against its baseline."""

    path: str
    baseline: float
    current: float

    @property
    def name(self) -> str:
        """The metric's leaf name, without its path."""

        return self.path.rsplit(".", 1)[-1]

    @property
    def delta(self) -> float:
        """The absolute change."""

        return self.current - self.baseline

    @property
    def relative(self) -> float:
        """The change as a fraction of the baseline's magnitude."""

        scale = abs(self.baseline)
        if scale <= 1e-12:
            return 0.0 if abs(self.delta) <= 1e-12 else math.inf
        return self.delta / scale

    @property
    def tolerance(self) -> float:
        """How far this metric may move before it is reported."""

        return TOLERANCES.get(self.name, DEFAULT_TOLERANCE)

    @property
    def direction(self) -> str:
        """``"higher"``, ``"lower"`` or ``"neutral"``."""

        if self.name in HIGHER_IS_BETTER:
            return "higher"
        if self.name in LOWER_IS_BETTER:
            return "lower"
        return "neutral"

    @property
    def regression(self) -> bool:
        """``True`` when the metric moved the wrong way by more than its tolerance."""

        if abs(self.relative) <= self.tolerance:
            return False
        if self.direction == "higher":
            return self.delta < 0
        if self.direction == "lower":
            return self.delta > 0
        return False

    @property
    def improvement(self) -> bool:
        """``True`` when the metric moved the right way by more than its tolerance."""

        if abs(self.relative) <= self.tolerance:
            return False
        if self.direction == "higher":
            return self.delta > 0
        if self.direction == "lower":
            return self.delta < 0
        return False

    @property
    def changed(self) -> bool:
        """``True`` when the metric moved more than its tolerance, either way."""

        return abs(self.relative) > self.tolerance and not self.regression

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the comparison."""

        return {
            "path": self.path,
            "baseline": round(self.baseline, 6),
            "current": round(self.current, 6),
            "delta": round(self.delta, 6),
            "relative": (
                None if math.isinf(self.relative) else round(self.relative, 6)
            ),
            "tolerance": self.tolerance,
            "direction": self.direction,
            "verdict": (
                "regression"
                if self.regression
                else "improvement"
                if self.improvement
                else "changed"
                if self.changed
                else "unchanged"
            ),
        }


def _numeric_leaves(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a nested report into ``{path: number}``, skipping everything else."""

    leaves: dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            leaves[path] = float(value)
        elif isinstance(value, Mapping):
            leaves.update(_numeric_leaves(value, prefix=f"{path}."))
        elif isinstance(value, list):
            # Per-line detail is not compared: a line added or removed would shift
            # every index, so only the aggregates are useful as a baseline.
            continue
    return leaves


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """What changed between a saved baseline and the current evaluation."""

    changes: tuple[MetricChange, ...]

    @property
    def regressions(self) -> tuple[MetricChange, ...]:
        """The metrics that moved the wrong way beyond their tolerance."""

        return tuple(change for change in self.changes if change.regression)

    @property
    def improvements(self) -> tuple[MetricChange, ...]:
        """The metrics that moved the right way beyond their tolerance."""

        return tuple(change for change in self.changes if change.improvement)

    @property
    def ok(self) -> bool:
        """``True`` when nothing regressed."""

        return not self.regressions

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the comparison."""

        return {
            "ok": self.ok,
            "regressions": [change.as_dict() for change in self.regressions],
            "improvements": [change.as_dict() for change in self.improvements],
            "changes": [change.as_dict() for change in self.changes],
        }

    def summary(self) -> str:
        """Return the multi-line summary a run prints."""

        if not self.changes:
            return "baseline: no comparable metrics"

        if self.ok:
            lines = [
                f"baseline: no regression across {len(self.changes)} metric(s)"
                + (f", {len(self.improvements)} improved" if self.improvements else "")
            ]
        else:
            lines = [
                f"baseline: {len(self.regressions)} REGRESSION(s) across "
                f"{len(self.changes)} metric(s)"
            ]
        for change in self.regressions:
            lines.append(
                f"  {change.path}: {change.baseline:.4g} -> {change.current:.4g} "
                f"({change.relative:+.1%}, tolerance {change.tolerance:.0%})"
            )
        return "\n".join(lines)


def compare_to_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> BaselineComparison:
    """Compare two evaluation reports metric by metric.

    Metrics present on only one side are skipped rather than treated as changes: a
    run that gained a measurement has not regressed, and a report shape that changed
    should not raise a false alarm.
    """

    current_leaves = _numeric_leaves(current)
    baseline_leaves = _numeric_leaves(baseline)
    shared = sorted(set(current_leaves) & set(baseline_leaves))

    return BaselineComparison(
        changes=tuple(
            MetricChange(
                path=path,
                baseline=baseline_leaves[path],
                current=current_leaves[path],
            )
            for path in shared
        )
    )


def save_baseline(report: EvaluationReport | Mapping[str, Any], path: str | Path) -> Path:
    """Write an evaluation report as a baseline that later runs can be compared to."""

    payload = report.as_dict() if isinstance(report, EvaluationReport) else dict(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Read a baseline written by :func:`save_baseline`.

    A missing file raises: unlike the dialogue bible, a baseline that is not there
    means the comparison cannot be made, and silently comparing against nothing is
    exactly the false confidence this module exists to prevent.
    """

    source = Path(path)
    if not source.is_file():
        raise EvaluationError(f"no baseline at {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationError(
            f"{source} must hold a JSON object, got {type(payload).__name__}"
        )
    return payload


__all__ = [
    "CROWDED_SCENE_SPEAKERS",
    "DEFAULT_TOLERANCE",
    "GOLDEN_CATEGORIES",
    "HIGHER_IS_BETTER",
    "LONG_LINE_SECONDS",
    "LOWER_IS_BETTER",
    "RAPID_TURN_SECONDS",
    "RECURRING_SPEAKER_LINES",
    "SHORT_LINE_SECONDS",
    "TOLERANCES",
    "BaselineComparison",
    "CoverageReport",
    "EvaluationError",
    "EvaluationReport",
    "InvalidMeasurementError",
    "MetricChange",
    "PerformancePreservationReport",
    "SpeakerIdentity",
    "SpeakerIdentityReport",
    "build_evaluation_report",
    "compare_to_baseline",
    "cosine_similarity",
    "load_baseline",
    "measure_coverage",
    "measure_performance_preservation",
    "measure_speaker_identity",
    "save_baseline",
]
