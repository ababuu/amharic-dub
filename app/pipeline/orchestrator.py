"""Run every stage of the pipeline, in order, from one call.

The stage order is the one ``app/pipeline/__init__.py`` documents::

    source video
      -> video.extract_audio            48 kHz stereo PCM WAV
      -> separation.separate_stems      speech / music / effects
      -> diarization.diarize            speaker turns
      -> transcription.transcribe       spoken lines, per speaker
      -> translation.adapt_dialogue     spoken Amharic (+ performance metadata when
                                        the backend produces it)
      -> voice_profiles.build_voice_profiles   one identity reference per speaker
                                        (per-character engines only)
      -> voice_bank.apply_voice_bank    fallback only: a character whose film
                                        window cannot supply a clean reference is
                                        cast from a native-Amharic bank
      -> tts.synthesize_dialogue        Amharic speech clips
      -> timing.align_dialogue          each line fitted to its original window
      -> mixing.mix_track              dialogue over ducked music + effects
      -> video.mux_dub                  the Amharic mix back into the video

and the deliverable is a dubbed MP4: the original picture, untouched, with the
Amharic dialogue in place of the English.

Command line::

    python -m app.pipeline.orchestrator data/input/movie.mp4
    python -m app.pipeline.orchestrator movie.mp4 --out runs/movie --max-lines 5

What it deliberately does not do
--------------------------------
* **No caching of its own.** Every stage runs on every call. The expensive
  per-line work in :mod:`app.pipeline.tts` is already content-addressed and skips
  artifacts that are on disk, which is where the real cost is; resuming whole
  stages from a manifest is a later concern.
* **No stage settings of its own beyond passing ``settings`` through**, so the
  orchestrator can never disagree with a stage about a path or a limit.
* **No hiding of failures.** A stage error is wrapped only to say which stage it
  came from; the original exception and traceback are preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import soundfile

from app.config import Settings, get_settings
from app.pipeline import (
    context_ai,
    dialogue_context,
    dialogue_plan,
    diarization,
    evaluation,
    mixing,
    qc,
    register,
    script_reconciliation,
    separation,
    target_extraction,
    timing,
    transcription,
    translation,
    tts,
    video,
    voice_bank,
    voice_profiles,
)
from app.pipeline.amharic_text import count_syllables
from app.pipeline.dialogue_context import placement_windows
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.mixing import MixResult
from app.pipeline.timing import AlignedClip
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import SkippedLine, TtsClip
from app.pipeline.video import MuxResult
from app.pipeline.voice_profiles import VoiceProfile, portable_path
from app.utils import drive, rclone

#: Manifest written next to a run's artifacts.
MANIFEST_FILENAME = "manifest.json"

#: Sub-directory holding the extracted track and the separated stems: the large,
#: per-source intermediates, kept with the run rather than in ``WORK_DIR``.
STAGES_DIRNAME = "stages"

#: Stage names, in execution order, as recorded in the manifest.
STAGE_ORDER: tuple[str, ...] = (
    "extract",
    "separation",
    "diarization",
    "transcription",
    "translation",
    "voice_profiles",
    "tts",
    "timing",
    "mixing",
    "mux",
)


class OrchestrationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingSourceError(OrchestrationError):
    """The source video does not exist."""


class EmptyStageError(OrchestrationError):
    """A stage ran but produced nothing the next stage could use."""


class StageError(OrchestrationError):
    """A stage failed; the original exception is chained onto this one."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything the stages produced for one source video."""

    source: Path
    output_dir: Path
    track: Path
    stems: separation.StemPaths
    turns: tuple[SpeakerSegment, ...]
    lines: tuple[TranscriptSegment, ...]
    crosstalk: tuple[CrosstalkRegion, ...]
    dialogue: tuple[AdaptedDialogue, ...]
    profiles: Mapping[str, VoiceProfile]
    clips: tuple[TtsClip, ...]
    skipped: tuple[SkippedLine, ...]
    alignment: tuple[AlignedClip, ...]
    mix: MixResult
    dub: MuxResult
    manifest_path: Path
    stage_seconds: Mapping[str, float]
    elapsed_seconds: float
    synthesized_lines: int
    #: How much of the film's original dialogue survived separation into the bed.
    bleed: separation.BleedReport | None = None
    #: How much of the film's original audio survived into the dubbed dialogue track.
    dub_bleed: separation.BleedReport | None = None
    #: Anything the TTS stage wanted the run to know without treating it as a failure.
    notes: tuple[str, ...] = ()
    #: Turns the diarization model's own speaker embeddings settled.
    verified: tuple[SpeakerSegment, ...] = ()
    speaker_corrections: tuple[diarization.SpeakerAttributionCorrection, ...] = ()
    #: Per-speaker register measured from the voice itself, and the turns whose
    #: label it contradicted. The audit trail for "why was this character cast with
    #: this voice?".
    attribution: register.AttributionReport | None = None
    #: What crosstalk target extraction did: which regions got per-speaker tracks,
    #: which characters were refused and why. ``None`` when the stage never ran.
    separation: "target_extraction.ExtractionReport | None" = None
    #: Lines that came from a separated per-speaker track rather than the mixture.
    separated_lines: int = 0
    #: Lines still recovered by the previous mixture path (normally zero now that
    #: the register gate refuses unattributable mixture decodes).
    legacy_separated_lines: int = 0
    #: Recoveries the acoustic gate refused; the reason is in the run's report.
    refused_recoveries: int = 0
    #: Sourced research about the film's cast, passed to the translation prompt.
    character_guide: dialogue_context.CharacterGuide | None = None
    #: Speaker-labelled episode script used to reconcile ASR lines with who says
    #: them, and the result of that reconciliation.
    episode_script: script_reconciliation.EpisodeScript | None = None
    script_reconciliation: script_reconciliation.ScriptReconciliation | None = None
    #: The authored dialogue plan a run was given, how much of it belongs to this
    #: audio, and which measured voice each character was cast with.
    plan: dialogue_plan.PlanReport | None = None
    #: The AI speaker-attribution review: what was corrected and what was left for
    #: QC. ``None`` when the review was disabled or never had evidence to review.
    attribution_review: context_ai.AttributionReview | None = None

    @property
    def final_video(self) -> Path:
        """The deliverable: the source video with the Amharic dialogue track."""

        return self.dub.path

    @property
    def partial(self) -> bool:
        """``True`` when only some lines were synthesized, so the dub is partial."""

        return self.synthesized_lines < len(self.dialogue)

    @property
    def unfitted_lines(self) -> tuple[AlignedClip, ...]:
        """Lines whose delivery could not be fitted to their original window."""

        return tuple(line for line in self.alignment if not line.fits)

    def _attribution_dict(self) -> dict[str, Any]:
        """Return the register audit: what each voice measured, and what moved."""

        if self.attribution is None:
            # A review can still have run without register measurement, and a QC
            # case is worth recording even when the pitch audit is missing.
            payload: dict[str, Any] = {}
            if self.attribution_review is not None:
                payload["ai_review"] = self.attribution_review.as_dict()
            if self.script_reconciliation is not None:
                payload["script_reconciliation"] = self.script_reconciliation.as_dict()
            return payload

        def registers(items: Mapping[str, register.SpeakerRegister]) -> dict[str, Any]:
            return {
                speaker_id: {
                    "median_hz": measured.median_hz,
                    "low_hz": measured.low_hz,
                    "high_hz": measured.high_hz,
                    "presentation": measured.presentation,
                    "turns": measured.turns,
                    "measured_turns": measured.measured_turns,
                    "speech_seconds": round(measured.speech_seconds, 3),
                    "confident": measured.confident,
                }
                for speaker_id, measured in items.items()
            }

        payload = {
            "before": registers(self.attribution.before),
            "after": registers(self.attribution.after),
            "reassigned_fraction": round(self.attribution.reassigned_fraction, 4),
            "corrections": [
                {
                    "start": correction.start,
                    "end": correction.end,
                    "from": correction.from_speaker,
                    "to": correction.to_speaker,
                    "median_hz": correction.median_hz,
                }
                for correction in self.attribution.corrections
            ],
            "warning": register.disagreement(self.attribution),
        }
        if self.attribution_review is not None:
            payload["ai_review"] = self.attribution_review.as_dict()
        if self.script_reconciliation is not None:
            payload["script_reconciliation"] = self.script_reconciliation.as_dict()
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the run, for a manifest or a report."""

        return {
            "source": portable_path(self.source),
            "output_dir": portable_path(self.output_dir),
            "final_video": portable_path(self.final_video),
            "partial": self.partial,
            "synthesized_lines": self.synthesized_lines,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "stage_seconds": {
                name: round(seconds, 3) for name, seconds in self.stage_seconds.items()
            },
            "track": portable_path(self.track),
            "stems": {
                name: portable_path(path) for name, path in self.stems.as_dict().items()
            },
            "diarization": {
                "turns": len(self.turns),
                "speakers": sorted({turn.speaker_id for turn in self.turns}),
                # The turns themselves, so a wrong line can be traced to the turn
                # that produced it without re-running the model.
                "detail": [
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker_id": turn.speaker_id,
                    }
                    for turn in self.turns
                ],
                "verified": [
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker_id": turn.speaker_id,
                    }
                    for turn in self.verified
                ],
                "speaker_corrections": [
                    {
                        "start": correction.start,
                        "end": correction.end,
                        "from": correction.original_speaker_id,
                        "to": correction.corrected_speaker_id,
                        "similarity": round(correction.corrected_similarity, 4),
                        "margin": round(correction.similarity_margin, 4),
                    }
                    for correction in self.speaker_corrections
                ],
                # Simultaneous speech the exclusive attribution could not carry.
                # Reported so a run can be audited for it without re-listening.
                "crosstalk": {
                    "regions": len(self.crosstalk),
                    "seconds": round(
                        sum(region.duration for region in self.crosstalk), 3
                    ),
                    "detail": [
                        {
                            "start": region.start,
                            "end": region.end,
                            "speakers": list(region.speaker_ids),
                        }
                        for region in self.crosstalk
                    ],
                },
            },
            "transcription": {
                "lines": len(self.lines),
                "sentences": len(self.dialogue),
                "segments": [_segment_dict(segment) for segment in self.lines],
                # What the optional per-speaker split did, including the reasons a
                # region was still single-voiced. Absent means the stage was off or
                # no crosstalk dropped a speaker.
                "crosstalk_separation": (
                    None if self.separation is None else self.separation.as_dict()
                ),
                "separated_lines": self.separated_lines,
                "legacy_separated_lines": self.legacy_separated_lines,
                "refused_recoveries": self.refused_recoveries,
            },
            "translation": {
                "lines": len(self.dialogue),
                "segments": [_dialogue_dict(line) for line in self.dialogue],
            },
            "voice_profiles": {
                speaker_id: {
                    "reference_audio": portable_path(profile.reference_audio),
                    "reference_start": profile.reference_start,
                    "reference_end": profile.reference_end,
                    "reference_duration": profile.reference_duration,
                    "reference_text": profile.reference_text,
                    "clone_prompt_path": (
                        None
                        if profile.clone_prompt_path is None
                        else portable_path(profile.clone_prompt_path)
                    ),
                    "quality_score": profile.quality_score,
                    "selection_reason": profile.selection_reason,
                    "estimated_vocal_presentation": profile.estimated_vocal_presentation,
                    "estimated_pitch_hz": profile.estimated_pitch_hz,
                    # What the film itself contained, kept even after a bank voice
                    # replaced it, so a casting decision can be audited.
                    "source_vocal_presentation": profile.source_vocal_presentation,
                    "source_pitch_hz": profile.source_pitch_hz,
                    "reference_source": profile.reference_source,
                    "reference_voice_id": profile.reference_voice_id,
                }
                for speaker_id, profile in self.profiles.items()
            },
            "speakers": self._attribution_dict(),
            # The researched cast is evidence for translation, not a casting
            # decision, so it is recorded with its sources for audit.
            "characters": (
                None if self.character_guide is None else self.character_guide.as_dict()
            ),
            "script": (
                None if self.episode_script is None else self.episode_script.summary()
            ),
            "plan": None if self.plan is None else self.plan.as_dict(),
            "tts": {
                "clips": [clip.to_dict() for clip in self.clips],
                # Lines the stage could not voice. Recorded because a line missing
                # from the dub is audible, so a run has to be able to say which
                # ones and why without re-listening to the film.
                "skipped": [line.as_dict() for line in self.skipped],
            },
            "timing": {
                "lines": len(self.alignment),
                "unfitted": len(self.unfitted_lines),
                "segments": [line.to_dict() for line in self.alignment],
            },
            "mixing": self.mix.to_dict(),
            "mux": self.dub.to_dict(),
            "bleed": None if self.bleed is None else self.bleed.as_dict(),
            "dub_bleed": None if self.dub_bleed is None else self.dub_bleed.as_dict(),
        }

    def report(self) -> str:
        """Return a short human-readable summary of the run."""

        speakers = sorted({turn.speaker_id for turn in self.turns})
        stretched = sum(1 for line in self.alignment if line.tempo != 1.0)
        drifted = [line for line in self.alignment if line.drift > 0.005]
        crosstalk_summary = (
            "not separated" if self.separation is None else self.separation.summary()
        )
        if self.separation is not None and self.separation.available:
            crosstalk_summary += (
                f"; {self.separated_lines} line(s) from separated track(s), "
                f"{self.legacy_separated_lines} from the mixture, "
                f"{self.refused_recoveries} refused"
            )
        lines = [
            f"source          {self.source}",
            f"output          {self.output_dir}",
            f"stems           {', '.join(path.name for path in self.stems.as_dict().values())}",
            f"diarization     {len(self.turns)} turn(s), {len(speakers)} speaker(s)",
            f"transcription   {len(self.lines)} line(s)",
            f"crosstalk       {crosstalk_summary}",
            f"adaptation      {len(self.dialogue)} Amharic line(s)",
            f"voice profiles  {len(self.profiles)} profile(s)",
            f"tts             {len(self.clips)} clip(s)",
            f"timing          {len(self.alignment)} aligned, {stretched} stretched, "
            f"{len(self.unfitted_lines)} not fitted",
            f"placement       {len(drifted)} line(s) moved later to avoid speaking over "
            f"the line before them, worst "
            f"{max((line.drift for line in drifted), default=0.0):.2f} s",
            f"bleed           "
            f"{'not measured' if self.bleed is None else self.bleed.summary()}",
            f"dub bleed       "
            f"{'not measured' if self.dub_bleed is None else self.dub_bleed.summary()}",
            f"mixing          {self.mix.duration:.1f} s track, "
            f"{len(self.mix.overlaps)} overlapping line(s), "
            f"peak {self.mix.peak:.3f}",
            f"dub             {self.dub.path.name} "
            f"({self.dub.video_codec} video, {self.dub.audio_codec} "
            f"{self.dub.audio_sample_rate} Hz audio)",
            f"manifest        {self.manifest_path}",
            f"elapsed         {self.elapsed_seconds:.1f} s",
        ]
        for note in self.notes:
            lines.append(f"note            {note}")
        if self.partial or self.skipped:
            reasons: list[str] = []
            if self.skipped:
                reasons.append(f"{len(self.skipped)} line(s) could not be voiced")
            if self.synthesized_lines + len(self.skipped) < len(self.dialogue):
                reasons.append("--max-lines")
            detail = f" ({'; '.join(reasons)})" if reasons else ""
            lines.insert(
                6,
                f"PARTIAL DUB     only {self.synthesized_lines} of "
                f"{len(self.dialogue)} line(s) were synthesized{detail}",
            )
        return "\n".join(lines)


def _print_progress(message: str) -> None:
    """Default progress sink: one flushed line per stage."""

    print(message, flush=True)


def _audio_seconds(path: str | Path) -> float | None:
    """Return the duration of an audio file, or ``None`` when it cannot be read."""

    try:
        info = soundfile.info(str(path))
    except (OSError, RuntimeError, ValueError):
        return None
    return float(info.duration) if info.frames else None


def _plan_profile_turns(
    scope: dialogue_plan.PlanScope, casting: dialogue_plan.PlanCasting
) -> list[SpeakerSegment]:
    """Return each cast character's lines as the runs a voice reference is cut from.

    The plan's own timings are the most precise statement available about when a
    character was speaking, so they are what the reference window is chosen from.
    """

    mapping = casting.mapping
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for cue in scope.cues:
        speaker_id = mapping.get(cue.speaker)
        if speaker_id is None:
            continue
        by_speaker.setdefault(speaker_id, []).append((cue.start, cue.end))

    turns: list[SpeakerSegment] = []
    for speaker_id, spans in by_speaker.items():
        spans.sort()
        start, end = spans[0]
        for next_start, next_end in spans[1:]:
            if next_start - end <= 0.5:
                end = max(end, next_end)
            else:
                turns.append(SpeakerSegment(speaker_id, start, end))
                start, end = next_start, next_end
        turns.append(SpeakerSegment(speaker_id, start, end))
    return turns


def _stage(
    name: str,
    seconds: dict[str, float],
    progress: Callable[[str], None],
    work: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run one stage, timing it and naming it when it fails."""

    progress(f"  -> {name}")
    started = time.perf_counter()
    try:
        result = work(*args, **kwargs)
    except Exception as exc:
        raise StageError(
            f"the {name} stage failed with {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        seconds[name] = time.perf_counter() - started
    return result


def _segment_dict(segment: Any) -> dict[str, Any]:
    """Return the timing and text fields of a diarized or transcribed segment.

    The segment types are frozen dataclasses without a JSON view, and the
    orchestrator only needs the fields a manifest reader would look for.
    """

    return {
        "speaker_id": segment.speaker_id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


def _dialogue_dict(line: AdaptedDialogue) -> dict[str, Any]:
    """Return the full content of one adapted Amharic line."""

    entry: dict[str, Any] = {
        "speaker_id": line.speaker_id,
        "start": line.start,
        "end": line.end,
        "source_text": line.source_text,
        "amharic": line.amharic,
        "emotion": line.emotion,
        "intensity": line.intensity,
        "delivery": line.delivery,
        "pause_before": line.pause_before,
        "pause_after": line.pause_after,
    }
    if line.beats:
        # One sentence, several performances: the direction and the moment it
        # applies to, so nothing the dialogue model saw is lost in the manifest.
        entry["beats"] = [beat.as_dict() for beat in line.beats]
    return entry


def _report_dialogue_fit(
    dialogue: Sequence[Any],
    settings: Any,
    report: Callable[[str], None],
) -> None:
    """Say whether the Amharic can be spoken in the time the film gives it.

    This is the measurement that decides what the rest of the run can do. Amharic does not
    express the same idea in the same number of syllables as English, so a faithful
    translation of film dialogue is routinely longer than the lines it replaces - and if
    the total is longer than the time available, *something* has to give: the delivery is
    sped up, or lines are cut short, or two voices end up overlapping. Stating the ratio
    here means that is a known quantity from the adaptation stage onward rather than
    something inferred from an audio artefact at the end.
    """

    spans = [(line.start, line.end) for line in dialogue]
    if not spans:
        return
    # Overlap-aware: a line the film played at the same time as another is not given
    # the other character's speaking time to fill, because both voices are heard.
    rooms = placement_windows(spans, gap=settings.timing_min_line_gap, overlap_aware=True)
    rate = settings.translation_syllables_per_second
    needed = sum(count_syllables(line.amharic) / rate for line in dialogue)
    available = sum(rooms)
    if available <= 0:
        return

    over = sum(
        1
        for line, room in zip(dialogue, rooms)
        if count_syllables(line.amharic) > max(1, round(room * rate))
    )
    ratio = needed / available
    report(
        f"  dialogue fit: {needed:.1f}s of Amharic in {available:.1f}s available "
        f"({ratio:.2f}x), {over} line(s) over their own time"
    )
    if ratio > 1.0:
        report(
            "        - the Amharic is longer than the film's dialogue time; the timing "
            "stage will use the silence between lines first, then speed lines up to "
            f"{settings.timing_max_tempo:g}x, and only then cut a line short"
        )


def _report_plan_fit(
    dialogue: Sequence[Any],
    settings: Any,
    report: Callable[[str], None],
) -> None:
    """Name the plan lines that cannot be spoken in the time the video gives them.

    A plan is authored, so an overrun cannot be fixed by asking a model for a
    shorter line - it is a line the operator has to edit. Naming the worst offenders
    here means that edit is made against a measurement instead of by listening to a
    finished dub and guessing which line sounded rushed.
    """

    rate = settings.translation_syllables_per_second
    if rate <= 0 or not dialogue:
        return
    overruns: list[tuple[float, int, str]] = []
    for index, line in enumerate(dialogue):
        window = line.end - line.start
        if window <= 0:
            continue
        over = count_syllables(line.amharic) / rate - window
        if over > 0.25:
            overruns.append((over, index, line.source_text))
    if not overruns:
        report("  dialogue fit: every plan line fits the time the video gives it")
        return
    overruns.sort(reverse=True)
    total = sum(over for over, _, _ in overruns)
    report(
        f"  dialogue fit: {len(overruns)}/{len(dialogue)} plan line(s) need more time "
        f"than the video gives them ({total:.1f}s over in total); the timing stage "
        f"will speed them to {settings.timing_max_tempo:g}x and then let them run late"
    )
    for over, index, text in overruns[:5]:
        report(f"        +{over:.1f}s line {index}: {text[:60]}")
    if len(overruns) > 5:
        report("        shorten these in the plan to keep the dub in sync")


def run_pipeline(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    max_lines: int | None = None,
    plan_file: str | Path | None = None,
    plan_keep_timing: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run every implemented stage on ``source`` and return what they produced.

    Parameters
    ----------
    source:
        Source video. Its audio track is extracted first, so the stages never see
        the container format.
    output_dir:
        Directory for the run: the extracted track, the stems and the manifest.
        Defaults to ``<OUTPUT_DIR>/<source stem>``.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
    max_lines:
        Synthesize only the first ``N`` adapted lines. A cost control for the
        first run of a long film; every earlier stage still runs in full.
    plan_file:
        An authored dialogue plan (``<video>.plan.json`` in ``INPUT_DIR`` by
        default) whose lines already carry the speaker, timing and Amharic. When
        one is used the sentence and translation stages are skipped, and each
        character is cast to a measured voice from the diarization.
    plan_keep_timing:
        Keep the plan's own timings instead of taking each line's timing from the
        transcribed speech. Only useful when the plan was timed against this exact
        video; otherwise the video is the source of truth for when a line is spoken.
    progress:
        Callable receiving one line per stage. Defaults to printing.

    Returns
    -------
    PipelineResult
        Including :attr:`PipelineResult.final_video`, the dubbed MP4.

    Raises
    ------
    MissingSourceError
        The source video does not exist.
    EmptyStageError
        A stage produced nothing to pass on.
    StageError
        A stage failed. The stage's own exception is chained, so the original
        cause and traceback are still available.
    """

    resolved = settings if settings is not None else get_settings()

    video_path = Path(source)
    if not video_path.is_file():
        raise MissingSourceError(f"source video not found: {video_path}")

    # Only after the source is known to exist, so a typo never leaves empty
    # directories behind in the working tree.
    resolved.ensure_directories()

    run_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(resolved.output_dir) / video_path.stem
    )
    stages_dir = run_dir / STAGES_DIRNAME
    stages_dir.mkdir(parents=True, exist_ok=True)

    report = _print_progress if progress is None else progress
    seconds: dict[str, float] = {}
    started = time.perf_counter()

    report(f"{video_path.name}: running stages into {run_dir}")

    track = _stage(
        "extract",
        seconds,
        report,
        video.extract_audio,
        video_path,
        stages_dir / f"{video_path.stem}{video.DEFAULT_TRACK_SUFFIX}",
        settings=resolved,
    )
    stems = _stage(
        "separation",
        seconds,
        report,
        separation.separate_stems,
        track,
        output_dir=stages_dir,
        settings=resolved,
        on_progress=report,
    )

    diarized = _stage(
        "diarization",
        seconds,
        report,
        diarization.diarize_detailed,
        stems.speech,
        settings=resolved,
    )
    turns = tuple(diarized.turns)
    if not turns:
        raise EmptyStageError(
            f"diarization found no speech in {stems.speech}; nothing downstream can run"
        )
    crosstalk = tuple(diarized.crosstalk)
    speaker_corrections = tuple(diarized.corrections)
    if speaker_corrections:
        report(
            f"  speaker verification: corrected {len(speaker_corrections)} "
            "high-confidence turn attribution(s)"
        )
    if diarized.verified:
        # Said out loud because it changes what a later stage may do: these labels
        # were compared with the model's own speaker embeddings, so no pitch
        # measurement is allowed to overturn them.
        report(
            f"  speaker verification: {len(diarized.verified)} turn(s) settled by "
            "the speaker embeddings"
        )
    if crosstalk:
        report(
            f"  crosstalk: {len(crosstalk)} region(s), "
            f"{diarized.crosstalk_seconds:.2f}s of simultaneous speech"
        )

    # Which voice belongs to which cluster, measured before any text exists. Needs the
    # stem and nothing else, so it is available to the transcription stage - which must
    # check that a recovered crosstalk line actually sounds like the character it is
    # about to be credited to. Never fatal: without it the structural checks still run.
    speaker_registers: dict[str, float] = {}
    try:
        speaker_registers = register.measure_registers(turns, stems.speech)
    except register.RegisterError as exc:
        report(
            f"  register: unmeasured before transcription ({exc}); crosstalk "
            "recovery will rely on timing alone"
        )
    else:
        if speaker_registers:
            report(
                "  registers: "
                + ", ".join(
                    f"{speaker_id} {median:.0f} Hz"
                    for speaker_id, median in sorted(speaker_registers.items())
                )
            )

    transcription_result = _stage(
        "transcription",
        seconds,
        report,
        transcription.transcribe_detailed,
        stems.speech,
        turns,
        settings=resolved,
        # Simultaneous speech is re-listened to for the speaker the exclusive
        # turn list had to drop: without this, a two-character argument is
        # transcribed as one character talking and the dub follows it.
        crosstalk=crosstalk,
        overlap_turns=diarized.overlap_turns,
        # Which voice is which, measured from the diarized turns before any text
        # exists. A crosstalk can only be heard by decoding a mixture, and the
        # decode returns whichever voice dominates it - so a recovered line has to
        # be checked against the register of the character it is about to be given
        # to, or the other character ends up saying it in the wrong voice.
        registers=speaker_registers,
        # Target-speaker extraction prints what it did (or why it could not run)
        # through the same reporter as every other stage.
        report=report,
    )
    lines = tuple(transcription_result.lines)
    separation_report = transcription_result.separation
    if not lines:
        raise EmptyStageError("transcription produced no lines; nothing to adapt")
    if crosstalk:
        # Two voices in the film and one in the transcript is the defect that makes a
        # dub sound like a monologue, so what matters is not how many lines touch a
        # crosstalk region but how many *characters* it now has.
        doubled = sum(
            1
            for region in crosstalk
            if len(
                {
                    line.speaker_id
                    for line in lines
                    if line.start < region.end and line.end > region.start
                }
            )
            >= 2
        )
        report(
            f"  simultaneous speech: {doubled}/{len(crosstalk)} crosstalk region(s) "
            "now carry lines for both characters (the exclusive view keeps one "
            "speaker per instant, so the other is recovered from the overlap-aware "
            "diarization)"
        )

    # An authored plan already contains the sentences, their speakers and the
    # Amharic, so it replaces the sentence and translation stages outright. What it
    # cannot contain is which measured voice each character has: that is a property
    # of the audio and is cast from the diarization below.
    plan: dialogue_plan.DialoguePlan | None = None
    plan_scope = dialogue_plan.PlanScope()
    joined = (
        Path(plan_file)
        if plan_file is not None
        else dialogue_plan.discover_plan(resolved.input_dir, video_path)
    )
    if joined is not None:
        try:
            plan = dialogue_plan.load_plan(joined)
        except dialogue_plan.PlanError as exc:
            report(f"  plan: ignored {joined.name} ({exc})")
            plan = None
        else:
            plan_scope = dialogue_plan.scope_plan(
                plan,
                transcript=lines,
                audio_duration=_audio_seconds(stems.speech),
                retime=not plan_keep_timing,
            )
            report(
                f"  plan: {joined.name}: {len(plan)} authored line(s), "
                f"{len(plan_scope)} in this audio ({plan_scope.method})"
            )
            if plan_scope.dropped_outside:
                report(
                    f"  plan: {plan_scope.dropped_outside} line(s) fall outside this "
                    "audio and belong to another part of the episode"
                )
            if plan_scope.dropped_unmatched:
                report(
                    f"  plan: {plan_scope.dropped_unmatched} line(s) have no matching "
                    "speech in this audio"
                )
            if not plan_scope.cues:
                report("  plan: nothing in it applies to this audio; ignoring it")
                plan = None
    if plan is not None and plan_scope.interpolated:
        report(
            f"  plan: {plan_scope.interpolated} line(s) the transcript did not "
            "recognise were placed between their neighbours"
        )

    # With a plan the run owes a model nothing: the Amharic is already written and
    # every character's name is already stated. Research stays available for its
    # free, local inputs only.
    research_settings = (
        resolved if plan is None else replace(resolved, context_ai_enabled=False)
    )

    # The film's own title is the only dependable way to know who its characters
    # are. It is read from a .txt file discovered in the input directory (never a
    # hard-coded name), researched through DeepSeek's Search and Fetch endpoints,
    # and cached so the whole run - and the next run of the same episode - shares
    # one sourced guide. Missing title, missing key or an unavailable endpoint is a
    # note, never a failed run.
    # A script the operator drops into data/input is free and authoritative. It is
    # discovered before any research, so the paid Search/Fetch/extraction path is
    # skipped entirely when one is present.
    local_script: script_reconciliation.EpisodeScript | None = None
    try:
        local_script = context_ai.local_script_for_run(
            video_path, settings=research_settings, report=report
        )
    except Exception as exc:
        report(f"  script: local file unreadable ({type(exc).__name__}: {exc})")

    guide: dialogue_context.CharacterGuide | None = None
    try:
        guide = context_ai.character_guide_for_run(
            video_path,
            settings=research_settings,
            script=local_script,
            report=report,
        )
    except Exception as exc:  # research must never be the reason a film fails
        report(f"  characters: unavailable ({type(exc).__name__}: {exc}); continuing")

    # The script is the other half of the research: a sourced, speaker-labelled
    # transcript lets the reconciliation below recover the lines diarization merged
    # into one character's long turn. A local file is used as-is; otherwise it is
    # fetched once and cached per episode.
    script = local_script
    if script is None:
        try:
            script = context_ai.episode_script_for_run(
                video_path,
                settings=research_settings,
                guide=guide,
                # The clip's own words validate a fetched transcript before it is
                # used: a page for the wrong episode shares almost none of its phrases.
                transcript=[line.text for line in lines],
                report=report,
            )
        except Exception as exc:  # a missing script degrades, never fails
            report(f"  script: unavailable ({type(exc).__name__}: {exc}); continuing")

    # Who is speaking, measured from the voice itself. Diarization usually gets the
    # clusters right and a handful of turns wrong, and one wrong turn is enough to
    # hand a character the wrong voice for the whole film - the reference recording
    # is cut from a single turn. It also has to run *before* sentences are assembled:
    # a stretch where two people were labelled as one speaker can only split into
    # two sentences once the labels are right.
    try:
        lines, attribution = register.repair_attribution(
            lines,
            stems.speech,
            # A short turn inside a crosstalk keeps its label: the other character
            # really was talking there, so it is not a diarization fragment.
            overlaps=[(region.start, region.end) for region in crosstalk],
            # And a turn the embeddings confirmed keeps its label too: pitch is a
            # weaker measurement of identity and must not overturn a stronger one.
            trusted=[(turn.start, turn.end) for turn in diarized.verified],
        )
    except register.RegisterError as exc:
        # Loud, never silent: this stage is what stops a character being cast with the
        # wrong gender, so a run that loses it has to say so rather than quietly
        # falling back to raw diarization labels.
        report(f"  register: unavailable ({exc}); keeping diarization labels")
        lines, attribution = list(lines), None

    if attribution is not None:
        for speaker_id, measured in attribution.after.items():
            report(f"  register: {measured.describe()}")
        if attribution.continuity:
            # Named rather than counted, because each one is a place where the dub
            # would otherwise have had the other character's voice inside a sentence.
            report(
                f"  speaker continuity: {len(attribution.continuity)} short "
                "fragment(s) took the speaker of the sentence around them"
            )
            for correction in attribution.continuity:
                report(f"        {correction.describe()}")
        if attribution.corrections:
            report(
                f"  speaker verification: reassigned {len(attribution.corrections)} "
                f"turn(s) whose pitch contradicted their label "
                f"({attribution.reassigned_fraction:.0%} of the transcript)"
            )
        untrustworthy = register.disagreement(attribution)
        if untrustworthy:
            report(f"  cast warning: {untrustworthy}")

    # A long diarization turn is verified as one voice, but a short interjection
    # from the other character inside it belongs to the interjection. The episode
    # script, aligned word by word, is the evidence that separates them. It may only
    # move a label when the match is strong and the character can be tied to a
    # diarized voice; a label embedding-verified at the line's own timescale stays.
    script_review: script_reconciliation.ScriptReconciliation | None = None
    if script is not None:
        pitches: list[float | None] | None = None
        try:
            pitches = register.measure_line_pitches(lines, stems.speech)
        except register.RegisterError as exc:
            report(f"  script: line pitch unmeasured ({exc})")
        try:
            lines, script_review = script_reconciliation.reconcile_speakers(
                lines,
                script=script,
                guide=guide,
                registers=(
                    attribution.after if attribution is not None else speaker_registers
                ),
                pitches=pitches,
                verified=diarized.verified,
                threshold=resolved.context_ai_confidence_threshold,
            )
        except Exception as exc:  # reconciliation is an improvement, not a prerequisite
            report(f"  script: reconciliation unavailable ({type(exc).__name__}: {exc})")
            script_review = None
        else:
            if script_review.applied:
                report(
                    f"  script: corrected {len(script_review.applied)} speaker "
                    f"attribution(s) against {script_review.matched_lines} matched "
                    "line(s)"
                )
            for change in script_review.applied:
                report(
                    f"        {change.start:.2f}-{change.end:.2f}s "
                    f"{change.from_speaker} -> {change.to_speaker} "
                    f"({change.character})"
                )
            if script_review.uncertain:
                report(
                    f"  script: {len(script_review.uncertain)} line(s) left for QC in "
                    "run.speakers.script_reconciliation.uncertain"
                )

    # Without a script, fall back to the diarization-only review. With one, the
    # script alignment is the stronger evidence and a second, weaker review would
    # only reintroduce the coarse-turn mistake.
    speaker_review: context_ai.AttributionReview | None = None
    if script_review is None:
        try:
            lines, speaker_review = context_ai.review_speaker_attribution(
                lines,
                settings=research_settings,
                turns=turns,
                verified=diarized.verified,
                crosstalk=crosstalk,
                registers=(
                    attribution.after if attribution is not None else speaker_registers
                ),
                guide=guide,
                report=report,
            )
        except Exception as exc:  # the review is an improvement, not a prerequisite
            report(f"  ai review: unavailable ({type(exc).__name__}: {exc})")
            speaker_review = None
        else:
            if speaker_review is not None and speaker_review.skipped:
                report(f"  ai review: {speaker_review.skipped}")
            if speaker_review is not None and speaker_review.uncertain:
                report(
                    "  ai review: uncertain attributions are recorded in "
                    "run.speakers.ai_review.uncertain for QC"
                )

    lines = tuple(lines)

    # A transcriber segments on pauses, not on grammar, so one spoken sentence
    # routinely arrives as several segments. Adapting and speaking each of them
    # separately produces exactly what a listener hears as broken delivery: the
    # same thought becomes several utterances, each with its own final intonation.
    # The sentence is therefore the unit from here on; the raw segments stay the
    # record of what was transcribed, and the clauses inside a sentence stay
    # available as performance beats.
    #
    # Punctuation alone cannot do this reliably - "Do you know?" and "how much I make
    # a year." are one question - so the adaptation model is asked where the sentences
    # are. If it is not configured or its answer is unusable, the punctuation rules
    # decide exactly as before.
    # What each voice measured as, for the adaptation prompt. Amharic marks gender on
    # the verb, so a line spoken by Skyler and a line spoken by Walt need different
    # forms even when the English is identical.
    speaker_genders: dict[str, str] = {}
    if attribution is not None:
        speaker_genders = {
            speaker_id: measured.presentation
            for speaker_id, measured in attribution.after.items()
        }
    elif speaker_registers:
        speaker_genders = {
            speaker_id: register.presentation_of(median)
            for speaker_id, median in speaker_registers.items()
        }

    sentences: tuple[Any, ...] = ()
    if plan is None:
        def _group_sentences(fragments):
            return translation.group_transcript_sentences(
                fragments, settings=resolved, on_progress=report
            )

        sentences = tuple(
            transcription.assemble_sentences(lines, grouper=_group_sentences)
        )
        if sentences and len(sentences) != len(lines):
            report(
                f"  dialogue: {len(lines)} transcribed segment(s) assembled into "
                f"{len(sentences)} sentence(s)"
            )

    # A bank is a fallback, never the default: a character keeps the actor's own
    # voice unless the film cannot supply a reference that is safe to clone. It is
    # loaded here because whether one exists decides what happens to a plan
    # character the film never lets speak alone: voiced from the bank, or reported.
    bank = voice_bank.load_voice_bank(settings=resolved)

    # With a plan the characters are already named and the Amharic is already
    # written. The one thing left to decide is which measured voice belongs to each
    # of them, and that is casting rather than translation - so no request is made.
    plan_casting: dialogue_plan.PlanCasting | None = None
    bible: dialogue_context.CharacterBible | None = None
    if plan is not None:
        plan_casting = dialogue_plan.cast_plan(
            plan,
            plan_scope.cues,
            turns=turns,
            registers=(
                attribution.after if attribution is not None else speaker_registers
            ),
            crosstalk=crosstalk,
            verified=diarized.verified,
        )
        if plan_casting.skipped:
            report(f"  casting: {plan_casting.skipped}")
        if plan_casting.unassigned and bank is not None:
            # The plan has lines for these characters, so they are voiced from the
            # native-Amharic bank - matched by the gender the plan states - rather
            # than being dropped from the dub.
            plan_casting = dialogue_plan.assign_bank_voices(plan_casting)
        for cast in plan_casting.casts:
            if cast.speaker_id is None:
                report(f"  casting: {cast.character}: unassigned ({cast.reason})")
            else:
                report(
                    f"  casting: {cast.character} -> {cast.speaker_id} "
                    f"[{cast.source}] ({cast.speaker_gender}; {cast.reason})"
                )
        dialogue = dialogue_plan.plan_dialogue(
            plan,
            plan_scope,
            plan_casting,
            # The plan's Amharic is meant to be spoken, so each line may use the
            # silence before the next one; the configured gap is what keeps two
            # voices from ever sounding at once.
            min_gap=resolved.timing_min_line_gap,
        )
        unvoiced = dialogue_plan.dropped_cues(plan_scope, plan_casting)
        if unvoiced:
            report(
                f"  casting: {unvoiced} plan line(s) left unvoiced because their "
                "character has no measured voice in this audio"
            )
        report("  adaptation: authored plan; no translation request")
        seconds["translation"] = 0.0
        _report_plan_fit(dialogue, resolved, report)
    else:
        # Consistency state, not a guess: an absent bible yields an empty one, so a
        # first run works and a second run inherits whatever was filled in by hand.
        # The researched guide travels beside the operator's cast so every translation
        # request in the run sees the same sourced names and genders.
        bible = dialogue_context.CharacterBible.load(
            resolved.dialogue_bible_path
        ).with_guide(guide)
        # The script reconciliation also established which diarized voice is which
        # character. That is exactly the state translation needs for gendered forms and
        # address, so it fills any speaker the operator's bible did not already name.
        if script_review is not None and script_review.mapping:
            named = dict(bible.characters)
            for found in script_review.mapping:
                if not found.speaker_id or found.speaker_id in named:
                    continue
                named[found.speaker_id] = dialogue_context.Character(
                    speaker_id=found.speaker_id,
                    name=found.character,
                    gender=found.gender if found.gender != "unknown" else "",
                    sources=tuple(script.sources[:1]) if script is not None else (),
                )
            bible = dialogue_context.CharacterBible(characters=named, guide=guide)

        dialogue = tuple(
            _stage(
                "translation",
                seconds,
                report,
                translation.adapt_dialogue,
                sentences,
                settings=resolved,
                bible=bible,
                # Measured from the film's own voices: Amharic marks gender on verbs
                # and pronouns, and the model must not have to guess it from a name.
                speaker_genders=speaker_genders,
                enforce_budget=resolved.translation_enforce_budget,
                enforce_fidel_loanwords=resolved.translation_enforce_fidel_loanwords,
                maximum_reduction_attempts=resolved.translation_max_reduction_attempts,
                on_progress=report,
            )
        )
    if plan is None and resolved.translation_backend == "nllb":
        # NLLB is a sentence-level translation model: it is handed one line at a time
        # and cannot be asked for a shorter rewrite, so the scene/character context,
        # the syllable-budget re-ask and the script check have nothing to act on here.
        # They are not silently skipped - the run says so.
        report(
            "  adaptation: NLLB translates line by line; scene/character context and "
            "length enforcement are not applied"
        )
        # Whether the Amharic can be spoken in the time the film gives it is the number
        # that decides whether the dub fits or has to be squashed, so it is measured and
        # stated here rather than left to be discovered from the timing report later.
        _report_dialogue_fit(dialogue, resolved, report)
    elif plan is None:
        # Which model produced the Amharic, and how hard it was asked to think, are the
        # first things to know when comparing two runs. The provider also decides whether
        # the requested thinking level was even available, which a bare model name would
        # leave a reader to guess at.
        report(
            f"  adaptation: {resolved.translation_provider} "
            f"{resolved.translation_model} "
            f"({translation.describe_thinking(resolved)})"
        )
        report(f"  adaptation fallback: {translation.describe_fallback(resolved)}")
        report(
            "  adaptation requests: one per batch; no rewrites, retries, or failover"
        )
        # Only the instruction-following backend reads the bible, so only it can report
        # having applied one.
        if bible:
            report(f"  dialogue bible: {len(bible)} character(s) applied")

    if not dialogue:
        raise EmptyStageError("adaptation produced no lines; nothing to synthesize")
    # an engine that reads a per-speaker reference can act on. A single-voice engine
    # speaks everyone the same way, so building identities for it would be work whose
    # only result is a set of references nothing reads. Which engines those are is the
    # TTS stage's own contract, asked rather than assumed here.
    needs_profiles = tts.engine_needs_profiles(resolved.tts_engine)
    profiles: Mapping[str, VoiceProfile] = {}
    if not needs_profiles:
        seconds["voice_profiles"] = 0.0
        report(f"  voice profiles: skipped ({resolved.tts_engine} is single-voice)")
        if bank is not None:
            # Said rather than silently ignored: a configured bank that changes nothing
            # is the one case where the operator cannot tell from the output that
            # fallback casting is off.
            report(
                f"  voice bank: not used ({len(bank)} voice(s) available, but "
                f"{resolved.tts_engine} speaks every character in one voice)"
            )
    else:
        # Runs measured from the repaired transcript rather than from the raw
        # diarization: a turn that contains two people cannot describe one voice, and
        # the transcript's own word timings are the more precise boundary. With a
        # plan, the plan's own line timings are more precise still, so they define
        # the window each reference is cut from.
        if plan_casting is not None:
            profile_turns = _plan_profile_turns(plan_scope, plan_casting)
        else:
            profile_turns = [
                SpeakerSegment(speaker_id, start, end)
                for speaker_id, start, end in register.contiguous_turns(lines)
            ]
        unusable: dict[str, voice_profiles.NoUsableReferenceError] = {}
        # Built from every transcribed line, not from a shortened run: a profile is a
        # character's identity and is reused across runs, so it is never truncated.
        profiles = _stage(
            "voice_profiles",
            seconds,
            report,
            voice_profiles.build_voice_profiles,
            profile_turns,
            stems.speech,
            # Simultaneous speech is passed in because it is the one thing the
            # per-line overlap check cannot see: the turns it compares are exclusive,
            # so a window full of crosstalk still reports no overlap.
            crosstalk=crosstalk,
            transcript=lines,
            settings=resolved,
            # The actor is the voice identity, so the reference sent to the engine
            # must be one voice and its text must describe its audio exactly; that is
            # what stops the model speaking the English prompt before the Amharic
            # line. A clean but undescribed window is the one safe step down - the
            # engine transcribes that window itself, so audio and text stay aligned.
            require_reference_transcript=True,
            allow_clean_fallback=True,
            # A character with no usable window is reported, not given a worse
            # reference; whether that report becomes a bank fallback or a failure is
            # decided below, per character.
            on_unusable=lambda speaker_id, error: unusable.__setitem__(
                speaker_id, error
            ),
            registers=attribution.after if attribution is not None else None,
        )
        if not profiles and not unusable:
            # A plan whose characters are all voiced from the bank legitimately has
            # no film profile, so only an empty result with nothing to fall back to
            # is an empty stage.
            bank_cast_ids = [
                cast.speaker_id
                for cast in (
                    plan_casting.bank_casts if plan_casting is not None else ()
                )
                if cast.speaker_id
            ]
            if not bank_cast_ids or bank is None:
                raise EmptyStageError(
                    "no voice profile could be built from the dialogue stem; the TTS "
                    "stage needs one identity reference per speaker"
                )
        if unusable and bank is None:
            # Every character has to sound like someone, and an unusable window is
            # not a voice. Failing here names the character and says how to fix it;
            # cloning from the bad window instead is how a second person ends up
            # inside a character's mouth for the whole film.
            reasons = "; ".join(
                f"{speaker_id}: {error}"
                for speaker_id, error in unusable.items()
            )
            raise StageError(
                "the voice_profiles stage found no usable actor reference for "
                f"{len(unusable)} character(s) and no voice bank is configured to "
                f"fall back to ({reasons}). Build a bank with "
                "`python scripts/build_voice_bank.py` and point VOICE_BANK_DIR at it, "
                "or fix the dialogue stem those characters were isolated from."
            ) from next(iter(unusable.values()))
        if unusable:
            assert bank is not None
            # A character with no usable reference may still have been *measured*:
            # the register stage aggregates every turn before any window is cut. That
            # measurement is what keeps a fallback voice in the actor's own register
            # instead of handing a low-pitched part the highest voice in the bank.
            measured = attribution.after if attribution is not None else {}
            try:
                profiles = voice_bank.apply_voice_bank(
                    profiles,
                    settings=resolved,
                    bank=bank,
                    requests=[
                        voice_bank.VoiceRequest(
                            speaker_id=speaker_id,
                            presentation=(
                                measured[speaker_id].presentation
                                if speaker_id in measured
                                else "unknown"
                            ),
                            pitch_hz=(
                                measured[speaker_id].median_hz
                                if speaker_id in measured
                                else None
                            ),
                        )
                        for speaker_id in unusable
                    ],
                    reasons={
                        speaker_id: str(error)
                        for speaker_id, error in unusable.items()
                    },
                )
            except voice_bank.VoiceBankError as exc:
                raise StageError(
                    "the voice_profiles stage failed while casting "
                    f"{len(unusable)} character(s) from the voice bank: {exc}"
                ) from exc

        # A plan character the film never let speak alone has no measured voice; the
        # bank supplies one, chosen by the gender the plan states for that character.
        bank_requests = [
            voice_bank.VoiceRequest(
                speaker_id=cast.speaker_id, presentation=cast.gender
            )
            for cast in (plan_casting.bank_casts if plan_casting is not None else ())
            if cast.speaker_id
        ]
        if bank_requests:
            if bank is None:
                raise StageError(
                    f"{len(bank_requests)} plan character(s) have no measurable voice "
                    "in this footage and no voice bank is configured to voice them "
                    "(build one with `python scripts/build_voice_bank.py`). Add a "
                    "gender to every character in the plan so the fallback can pick "
                    "the right register."
                )
            try:
                profiles = voice_bank.apply_voice_bank(
                    profiles,
                    settings=resolved,
                    bank=bank,
                    requests=bank_requests,
                    reasons={
                        cast.speaker_id: "the film never lets this character speak alone"
                        for cast in plan_casting.bank_casts  # type: ignore[union-attr]
                        if cast.speaker_id
                    },
                )
            except voice_bank.VoiceBankError as exc:
                raise StageError(
                    "the voice_profiles stage failed while casting "
                    f"{len(bank_requests)} plan character(s) from the voice bank: {exc}"
                ) from exc

        film_ids = [
            speaker_id
            for speaker_id, profile in profiles.items()
            if profile.reference_source == "film"
        ]
        bank_ids = [
            speaker_id
            for speaker_id, profile in profiles.items()
            if profile.reference_source == "voice_bank"
        ]
        if film_ids:
            report(
                f"  voice references: {len(film_ids)} character(s) cloned from the "
                "film's own dialogue (clean, single-voice actor audio; the actor's "
                "voice is the identity)"
            )
        if bank_ids:
            assert bank is not None
            report(
                f"  voice bank: {len(bank_ids)} of {len(profiles)} character(s) cast "
                f"from the native-Amharic fallback ({len(bank)} voice(s), "
                f"{bank.dataset}"
                f"{'@' + bank.revision[:8] if bank.revision else ''}) because the film "
                "supplied no reference that is safe to clone"
            )
            for speaker_id in bank_ids:
                profile = profiles[speaker_id]
                reason = unusable.get(speaker_id)
                report(
                    f"  cast: {speaker_id} -> {profile.reference_voice_id} "
                    f"({profile.estimated_vocal_presentation}, "
                    f"{profile.reference_duration:.1f}s)"
                    + (f"; {reason}" if reason is not None else "")
                )
                film_presentation = profile.source_vocal_presentation
                if (
                    film_presentation in {"masculine", "feminine"}
                    and profile.estimated_vocal_presentation
                    in {"masculine", "feminine"}
                    and film_presentation != profile.estimated_vocal_presentation
                ):
                    # A fallback in the wrong register is heard as a different
                    # character, so it is called out rather than left in a manifest.
                    report(
                        f"        WARNING: {speaker_id} measured {film_presentation} "
                        "in the film but the bank supplied a "
                        f"{profile.estimated_vocal_presentation} voice; the bank has "
                        "no unused voice in that register"
                    )
        elif bank is not None:
            # Stated so an operator who uploaded a bank can see that it was not
            # needed, instead of wondering why the actors still have their accents.
            report(
                f"  voice bank: available at {resolved.voice_bank_dir} but unused - "
                "every character kept the actor's own reference"
            )
        else:
            report(
                f"  voice bank: none at {resolved.voice_bank_dir} (fallback only; a "
                f"{voice_bank.BANK_MANIFEST_FILENAME} there would be used for a "
                "character whose film reference is unusable)"
            )
        for speaker_id, profile in profiles.items():
            if profile.reference_source != "film":
                continue
            pitch = (
                f", median pitch {profile.estimated_pitch_hz:.0f} Hz"
                if profile.estimated_pitch_hz is not None
                else ""
            )
            report(
                f"  voice presentation estimate: {speaker_id} "
                f"{profile.estimated_vocal_presentation}{pitch}; actor reference "
                "is the voice identity"
            )

    spoken = dialogue
    if max_lines is not None and max_lines < len(dialogue):
        if max_lines < 0:
            raise OrchestrationError(f"max_lines must be >= 0, got {max_lines}")
        report(
            f"  max-lines: synthesizing the first {max_lines} of {len(dialogue)} line(s)"
        )
        spoken = dialogue[:max_lines]

    synthesized = _stage(
        "tts",
        seconds,
        report,
        tts.synthesize_dialogue_detailed,
        spoken,
        stems.speech,
        profiles or None,
        settings=resolved,
    )
    clips = tuple(synthesized.clips)
    skipped = tuple(synthesized.skipped)
    if synthesized.pacing is not None:
        # Reported, not silent: "the whole film is being delivered 8% faster" is a
        # decision about the performance, and it should not have to be inferred.
        report(f"  pacing: film rate {synthesized.pacing.film_rate:.3f}")
    if skipped:
        report(
            f"  skipped: {len(skipped)} line(s) could not be voiced "
            f"({len(synthesized.failed)} engine failure(s))"
        )
    for note in synthesized.notes:
        report(f"  note: {note}")
    if not clips:
        raise EmptyStageError(
            "speech synthesis produced no clips; there is nothing to place or mix"
        )

    alignment = tuple(
        _stage(
            "timing",
            seconds,
            report,
            timing.align_dialogue,
            clips,
            output_dir=run_dir,
            settings=resolved,
        )
    )

    # Measured, because the one way the source language can survive into a finished
    # dub is the separator leaving it in the music or effects, which are re-mixed
    # untouched. Cheap next to the stages above, and the alternative is finding out
    # from a listener.
    try:
        bleed = separation.measure_bed_bleed(
            stems, [(line.start, line.end) for line in lines]
        )
    except separation.SeparationError as exc:
        # A diagnostic must never be the reason a finished dub is lost, so an
        # unreadable stem is reported as unmeasured rather than raised.
        bleed = None
        report(f"  bleed: not measured ({exc})")
    else:
        report(f"  bleed: {bleed.summary()}")
        if not bleed.clean:
            report(
                "  bleed: WARNING the original dialogue is still in the bed under "
                f"{bleed.leaked} line(s); it will be heard under the Amharic"
            )

    mixed = _stage(
        "mixing",
        seconds,
        report,
        mixing.mix_track,
        alignment,
        stems.music,
        stems.effects,
        output_dir=run_dir,
        settings=resolved,
    )

    # The bed check above cannot see the finished dub: it inspects the music and
    # effects, and the dubbed dialogue is not part of them. This is the other half -
    # the original audio surviving into the track a listener actually hears - and it
    # is what catches a mixing, placement or caching fault that reintroduces English.
    try:
        dub_bleed = separation.measure_dub_bleed(
            track,
            mixed.dialogue_path,
            [(line.start, line.end) for line in alignment],
        )
    except separation.SeparationError as exc:
        dub_bleed = None
        report(f"  dub bleed: not measured ({exc})")
    else:
        report(f"  dub bleed: {dub_bleed.summary()}")
        if not dub_bleed.clean:
            report(
                "  dub bleed: WARNING the original audio is audible inside the dubbed "
                f"dialogue on {dub_bleed.leaked} line(s)"
            )

    dub = _stage(
        "mux",
        seconds,
        report,
        video.mux_dub,
        video_path,
        mixed.mixed_path,
        run_dir / f"{video_path.stem}{video.DEFAULT_DUB_SUFFIX}",
        settings=resolved,
    )

    result = PipelineResult(
        source=video_path,
        output_dir=run_dir,
        track=track,
        stems=stems,
        turns=turns,
        lines=lines,
        crosstalk=crosstalk,
        verified=tuple(diarized.verified),
        speaker_corrections=speaker_corrections,
        attribution=attribution,
        dialogue=dialogue,
        profiles=dict(profiles),
        clips=clips,
        skipped=skipped,
        notes=tuple(synthesized.notes),
        bleed=bleed,
        dub_bleed=dub_bleed,
        alignment=alignment,
        mix=mixed,
        dub=dub,
        manifest_path=run_dir / MANIFEST_FILENAME,
        stage_seconds=dict(seconds),
        elapsed_seconds=time.perf_counter() - started,
        synthesized_lines=len(clips),
        separation=separation_report,
        separated_lines=transcription_result.separated_lines,
        legacy_separated_lines=transcription_result.legacy_lines,
        refused_recoveries=transcription_result.refused_recoveries,
        character_guide=guide,
        episode_script=script,
        script_reconciliation=script_review,
        attribution_review=speaker_review,
        plan=(
            None
            if plan is None
            else dialogue_plan.PlanReport(
                plan=plan,
                scope=plan_scope,
                casting=plan_casting or dialogue_plan.PlanCasting(),
            )
        ),
    )

    # Measured, not assumed: the model-free quality report is part of every run.
    # Pronunciation needs an Amharic ASR model and is off unless asked for: it loads a
    # second model and transcribes every clip, which is real time on a feature film.
    # When it *is* asked for, it is the only check that can see content the speech model
    # invented - a stray word, a fragment of the English prompt it was cloned from, a
    # mispronunciation. None of that is the original audio, so comparing the output with
    # the original, as the bleed measurements do, can never find it.
    transcriber: Callable[[Path], str] | None = None
    if resolved.qc_pronunciation:
        try:
            transcriber = transcription.build_amharic_transcriber(resolved)
        except Exception as exc:
            report(f"  qc: pronunciation unavailable ({type(exc).__name__}: {exc})")

    quality = qc.build_qc_report(
        result.alignment,
        dialogue=result.dialogue,
        clips=result.clips,
        crosstalk=result.crosstalk,
        transcribe=transcriber,
        transcription_model=resolved.transcription_model,
    )
    # Which hard cases the material actually contains. A run can score well simply
    # because it was easy, so this says what was *not* exercised.
    coverage = evaluation.measure_coverage(result.dialogue, crosstalk=result.crosstalk)

    _write_manifest(
        result,
        settings=resolved,
        quality=quality,
        coverage=coverage,
    )
    report(result.report())
    report(f"qc              {quality.summary()}")
    report(f"coverage        {coverage.summary()}")
    return result


def _provenance(settings: Settings) -> dict[str, Any]:
    """Describe what actually produced a run, for reproducibility.

    Which model adapted the dialogue, which speech engine spoke it, and how hard the
    model was asked to think are the first things two runs are compared on - and the
    terminal that printed them is gone as soon as the Pod stops, while the manifest
    travels home with the dub.
    """

    provenance: dict[str, Any] = {
        "tts_engine": settings.tts_engine,
        "translation_backend": settings.translation_backend,
    }

    if settings.translation_backend == "openai":
        provenance["translation_provider"] = settings.translation_provider
        provenance["translation_model"] = settings.translation_model
        # What was *sent*, not what was configured: the provider decides whether the
        # requested level was available, and a manifest must not claim reasoning the
        # request never carried.
        provenance["translation_thinking"] = translation.describe_thinking(settings)
        provenance["translation_fallback"] = translation.describe_fallback(settings)
    else:
        # NLLB is a local checkpoint, so it is the path that identifies it.
        provenance["translation_model"] = settings.translation_model

    return provenance


def _engine_note(settings: Settings) -> str:
    """Describe the synthesis chain this run actually used.

    What produced a dub has to be readable from the manifest alone, so the note names
    the engine and says whether it cloned a per-character voice or spoke everyone in
    one.
    """

    if tts.engine_needs_profiles(settings.tts_engine):
        return (
            f"{settings.tts_engine} runs voice cloning: each character is spoken in "
            "their own voice, cloned from their isolated performance in the film, so "
            "the dub carries that performance's accent as well as its timbre. A "
            "character whose film audio cannot supply a clean reference is cast from "
            "a native-Amharic bank instead, and the profile records which source "
            "each voice came from"
        )
    return (
        f"{settings.tts_engine} is a single-voice engine: one voice speaks every part"
    )


def _write_manifest(
    result: PipelineResult,
    *,
    settings: Settings,
    quality: qc.QcReport | None = None,
    coverage: evaluation.CoverageReport | None = None,
) -> Path:
    """Write the run's manifest, atomically, and return its path."""

    report = quality if quality is not None else qc.build_qc_report(
        result.alignment,
        dialogue=result.dialogue,
        clips=result.clips,
        crosstalk=result.crosstalk,
    )
    exercised = (
        coverage
        if coverage is not None
        else evaluation.measure_coverage(result.dialogue, crosstalk=result.crosstalk)
    )

    payload: dict[str, Any] = {
        "run": result.to_dict(),
        "settings": settings.as_dict(),
        "provenance": _provenance(settings),
        "qc": report.as_dict(),
        "coverage": exercised.as_dict(),
        "notes": [
            "video streams are copied, never re-encoded; the source's own audio "
            "streams are replaced by the Amharic mix",
            _engine_note(settings),
            "loudness normalization (EBU R128) is not applied: the mix is placed "
            "at a defined peak with defined dialogue and bed levels",
            "clip paths are content-addressed artifacts of the tts stage, so later "
            "runs of the same lines reuse them instead of re-synthesizing",
            "a line whose Amharic needs more time than the film gave it is moved "
            "later rather than being spoken over the next line, because two voices "
            "at once cannot be understood; the shift is recorded per line as 'drift'. "
            "The exception is a pair the film itself overlapped: that simultaneity is "
            "the performance, so it is kept rather than serialized, and "
            "qc.overlap measures how much of it survived",
            "speech the film played simultaneously is recovered for the character the "
            "exclusive diarization had to drop, written as a short competing line, "
            "and placed at the same instant as the line it overlaps; when "
            "CROSSTALK_SEPARATION is enabled the two voices are separated first and "
            "the missing character's own track is transcribed, and the register gate "
            "still refuses any track that does not measure as that character; "
            "qc.overlap.unpaired names any crosstalk where only one character is heard",
            "the qc block is measured, not assumed; pronunciation is left "
            "unmeasured because it needs an Amharic ASR model, which is injected "
            "by the caller rather than downloaded by a run",
        ],
    }
    if result.partial or result.skipped:
        why = (
            f"{len(result.skipped)} line(s) could not be voiced"
            if result.skipped
            else "the run was limited with --max-lines"
        )
        payload["notes"].insert(
            0,
            f"INCOMPLETE DUB: only {result.synthesized_lines} of "
            f"{len(result.dialogue)} line(s) were synthesized ({why}); see run.tts "
            "for the lines that are missing and why",
        )
    payload["notes"].extend(result.notes)

    target = result.manifest_path
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


#: How many pages of corpus metadata an automatic bank build scans. The full split is
#: ~140 pages and takes minutes; 40 is already thousands of rows from dozens of
#: speakers, which is far more than the dozen voices a film needs, and keeps the
#: pre-flight short on a pod that is billing by the minute.
AUTO_BANK_PAGES = 40

#: Voices an automatic bank build keeps.
AUTO_BANK_VOICES = 12


def ensure_voice_bank(
    directory: str | Path,
    *,
    settings: Settings,
    voices: int = AUTO_BANK_VOICES,
    max_pages: int = AUTO_BANK_PAGES,
    report: Callable[[str], None] | None = None,
) -> "voice_bank.VoiceBank":
    """Return the bank in ``directory``, building one there first if it is empty.

    Casting needs a bank, and a bank is a build artifact rather than a committed file,
    so a fresh checkout has nothing to cast from. Building it is cheap (no model, no GPU)
    and is the only step the pipeline does implicitly - and only because the caller asked
    for casting on the command line.

    **Own recordings win over a public corpus.** When ``voice_sample_dir`` holds
    recording/transcript pairs, the bank is built from those: an operator who took the
    trouble to supply voices does not want a download to replace them. The corpus is the
    fallback for an installation that supplied nothing.

    Raises
    ------
    voice_bank.VoiceBankError
        A bank exists but cannot be read, or the build failed.
    """

    base = Path(directory)
    say = report if report is not None else (lambda _message: None)

    existing = voice_bank.load_voice_bank(directory=base)
    if existing is not None:
        say(f"voice bank: using the {len(existing)} voice(s) already in {base}")
        return existing

    samples = Path(settings.voice_sample_dir)
    supplied = (
        sorted(
            path
            for path in samples.iterdir()
            if path.is_file() and path.suffix.lower() in voice_bank.SAMPLE_AUDIO_SUFFIXES
        )
        if samples.is_dir()
        else []
    )

    if supplied:
        say(
            f"voice bank: none in {base}; building one from the {len(supplied)} "
            f"recording(s) in {samples}"
        )
        built = voice_bank.build_voice_bank_from_samples(
            samples,
            destination=base,
            settings=settings,
            report=say,
        )
    else:
        say(f"voice bank: none in {base}; building one from the public corpus")
        built = voice_bank.build_voice_bank(
            destination=base,
            settings=settings,
            voices=voices,
            max_pages=max_pages,
            report=say,
        )
    say(f"voice bank: built {len(built)} voice(s) in {base}")
    return built


def _fetch_source_link(
    link: str, *, settings: Settings, name: str | None = None
) -> Path:
    """Download a share link into ``INPUT_DIR`` and return the local path."""

    settings.ensure_directories()
    filename = drive.safe_filename(name) if name else "episode.mp4"
    destination = Path(settings.input_dir) / filename

    def progress(written: int, total: int) -> None:
        if total > 0:
            share = written / total
            _print_progress(
                f"  download: {written / 1e6:.0f} of {total / 1e6:.0f} MB "
                f"({share:.0%})"
            )
        else:
            _print_progress(f"  download: {written / 1e6:.0f} MB")

    _print_progress(f"fetching {link} -> {destination}")
    path = drive.download_file(link, destination, progress=progress)
    _print_progress(f"  download: saved {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def _resolve_source(
    source: str,
    *,
    settings: Settings,
    name: str | None = None,
    report: Callable[[str], None],
) -> Path:
    """Return a local path for ``source``, fetching it from Drive when it is remote.

    Three forms are accepted, so the episode never has to be in the repository:

    * a local path, used as it is;
    * ``remote:folder/file``, fetched through rclone;
    * an ``https://drive.google.com/...`` share link, downloaded directly.

    In every case the file lands in ``INPUT_DIR``, which is what the plan, the title
    file and any later run are keyed to.
    """

    text = source.strip()
    if rclone.looks_like_remote_path(text):
        if name:
            report(
                "  source: --source-name applies to share links; a remote path keeps "
                "the file's own name"
            )
        return rclone.fetch_remote_file(settings, text, report=report)
    if text.lower().startswith(("http://", "https://")):
        return _fetch_source_link(text, settings=settings, name=name)
    return Path(text)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m app.pipeline.orchestrator SOURCE``."""

    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline.orchestrator",
        description=(
            "Run every dubbing stage on a source video and write a dubbed MP4: the "
            "original picture with an Amharic dialogue track."
        ),
    )
    parser.add_argument(
        "source",
        type=str,
        help=(
            "source video to dub: a local path, an rclone path such as "
            "gdrive:amharic-dub/input/S04E06.mkv, or a Google Drive share link. A "
            "remote one is fetched into INPUT_DIR first, so the episode never has to "
            "be committed to git."
        ),
    )
    parser.add_argument(
        "--source-name",
        default=None,
        metavar="NAME",
        help=(
            "filename to save a downloaded share link as (default: episode.mp4). Use "
            "it when the plan is named after the video."
        ),
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "do not upload the finished run to DRIVE_OUTPUT_FOLDER. Without this flag "
            "a run is pushed whenever the Drive remote is configured, so the dub "
            "reaches Drive without a second command."
        ),
    )
    parser.add_argument(
        "--rclone-config",
        type=Path,
        default=None,
        metavar="FILE",
        help="rclone config to use for this run (default: RCLONE_CONFIG)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the run (default: <OUTPUT_DIR>/<source stem>)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help=(
            "synthesize only the first N adapted sentences (first-run cost control). "
            "The result is a partial dub: the remaining sentences are not voiced."
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "authored dialogue plan (JSON) that is the source of truth for the lines, "
            "their speakers and the Amharic. Timing is taken from the video's own "
            "transcription, and each character is cast to a measured voice from the "
            "film. Default: <INPUT_DIR>/<video>.plan.json when that file exists."
        ),
    )
    parser.add_argument(
        "--plan-keep-timing",
        action="store_true",
        help=(
            "keep the plan's own timings instead of taking each line's timing from "
            "the transcribed speech. Only use it when the plan was timed against this "
            "exact video; by default the video decides when each line is spoken."
        ),
    )
    parser.add_argument(
        "--cast-from-bank",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "make a native-Amharic voice bank available as the fallback for "
            "characters whose film audio cannot supply a clean, usable reference. "
            "The actor's own reference stays the identity for everyone else. Uses "
            "DIR (default: VOICE_BANK_DIR), building a bank there first if none "
            "exists - which is what a fresh checkout needs, since a bank is a build "
            "artifact and is not committed."
        ),
    )
    args = parser.parse_args(argv)

    settings = rclone.with_config(get_settings(), args.rclone_config)

    # The episode itself is far too large to live in git, so it usually arrives from
    # Google Drive. Accepting a Drive path or share link as the source keeps that to
    # one command, and it lands in INPUT_DIR so every later step - the plan, the title
    # file, a second run - finds it exactly where a copied file would be.
    try:
        video_path = _resolve_source(
            str(args.source),
            settings=settings,
            name=args.source_name,
            report=_print_progress,
        )
    except (drive.DriveError, rclone.TransferError) as exc:
        print(f"\nFAILED: could not fetch the source video: {exc}", file=sys.stderr)
        return 1

    if args.cast_from_bank is not None:
        directory = Path(args.cast_from_bank) if args.cast_from_bank else Path(
            settings.voice_bank_dir
        )
        try:
            ensure_voice_bank(directory, settings=settings, report=print)
        except voice_bank.VoiceBankError as exc:
            print(f"\nFAILED: could not prepare a voice bank: {exc}", file=sys.stderr)
            return 1
        settings = replace(settings, voice_bank_dir=directory)

    try:
        result = run_pipeline(
            video_path,
            output_dir=args.out,
            settings=settings,
            max_lines=args.max_lines,
            plan_file=args.plan,
            plan_keep_timing=args.plan_keep_timing,
        )
    except OrchestrationError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    if result.skipped:
        # A dub missing lines is a deliverable that is wrong rather than a run that
        # failed, so it still exits non-zero: a caller that only checks the exit code
        # must not be able to ship a film with dialogue missing from it.
        print(
            f"\nINCOMPLETE DUB: {len(result.skipped)} of {len(result.dialogue)} "
            f"line(s) were not voiced. {result.manifest_path} records the reason for "
            "each one under run.tts.skipped_lines.",
            file=sys.stderr,
        )
        code = 2
    else:
        code = 0

    # A dub that never leaves the pod has not been delivered, and the pod is
    # disposable. Pushing here means one command does the whole job; --no-push is
    # for a run whose artifact you want to inspect locally first, and an
    # unconfigured remote simply skips it.
    if args.no_push:
        _print_progress("  drive: not pushing (--no-push)")
    elif not rclone.drive_ready(settings):
        _print_progress(
            f"  drive: not pushing (Drive remote {rclone.describe(settings)})"
        )
    else:
        try:
            _print_progress(f"  drive: pushing {result.output_dir.name} "
                            f"({rclone.describe(settings)})")
            rclone.push_run(settings, result.output_dir, report=_print_progress)
        except rclone.TransferError as exc:
            # The dub exists on the pod and in the manifest; a failed upload must not
            # turn a finished run into a failed one, but it must be loud.
            print(f"\nUPLOAD FAILED: the dub is at {result.final_video} but was not "
                  f"pushed to Drive: {exc}", file=sys.stderr)
            return 3
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_FILENAME",
    "STAGES_DIRNAME",
    "STAGE_ORDER",
    "EmptyStageError",
    "MissingSourceError",
    "OrchestrationError",
    "PipelineResult",
    "StageError",
    "ensure_voice_bank",
    "main",
    "run_pipeline",
]
