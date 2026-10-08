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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.pipeline import (
    dialogue_context,
    diarization,
    evaluation,
    mixing,
    qc,
    separation,
    timing,
    transcription,
    translation,
    tts,
    video,
    voice_profiles,
)
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.mixing import MixResult
from app.pipeline.timing import AlignedClip
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import SkippedLine, TtsClip
from app.pipeline.video import MuxResult
from app.pipeline.voice_profiles import VoiceProfile, portable_path

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
                "segments": [_segment_dict(segment) for segment in self.lines],
            },
            "translation": {
                "lines": len(self.dialogue),
                "segments": [_dialogue_dict(line) for line in self.dialogue],
            },
            "voice_profiles": {                speaker_id: {
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
                }
                for speaker_id, profile in self.profiles.items()
            },
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
        }

    def report(self) -> str:
        """Return a short human-readable summary of the run."""

        speakers = sorted({turn.speaker_id for turn in self.turns})
        stretched = sum(1 for line in self.alignment if line.tempo != 1.0)
        lines = [
            f"source          {self.source}",
            f"output          {self.output_dir}",
            f"stems           {', '.join(path.name for path in self.stems.as_dict().values())}",
            f"diarization     {len(self.turns)} turn(s), {len(speakers)} speaker(s)",
            f"transcription   {len(self.lines)} line(s)",
            f"adaptation      {len(self.dialogue)} Amharic line(s)",
            f"voice profiles  {len(self.profiles)} profile(s)",
            f"tts             {len(self.clips)} clip(s)",
            f"timing          {len(self.alignment)} aligned, {stretched} stretched, "
            f"{len(self.unfitted_lines)} not fitted",
            f"mixing          {self.mix.duration:.1f} s track, "
            f"{len(self.mix.overlaps)} overlapping line(s), "
            f"peak {self.mix.peak:.3f}",
            f"dub             {self.dub.path.name} "
            f"({self.dub.video_codec} video, {self.dub.audio_codec} "
            f"{self.dub.audio_sample_rate} Hz audio)",
            f"manifest        {self.manifest_path}",
            f"elapsed         {self.elapsed_seconds:.1f} s",
        ]
        if self.partial:
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

    return {
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


def run_pipeline(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    max_lines: int | None = None,
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
    if crosstalk:
        report(
            f"  crosstalk: {len(crosstalk)} region(s), "
            f"{diarized.crosstalk_seconds:.2f}s of simultaneous speech"
        )

    lines = tuple(
        _stage(
            "transcription",
            seconds,
            report,
            transcription.transcribe,
            stems.speech,
            turns,
            settings=resolved,
        )
    )
    if not lines:
        raise EmptyStageError("transcription produced no lines; nothing to adapt")

    # Consistency state, not a guess: an absent bible yields an empty one, so a
    # first run works and a second run inherits whatever was filled in by hand.
    bible = dialogue_context.CharacterBible.load(resolved.dialogue_bible_path)

    dialogue = tuple(
        _stage(
            "translation",
            seconds,
            report,
            translation.adapt_dialogue,
            lines,
            settings=resolved,
            bible=bible,
            enforce_budget=resolved.translation_enforce_budget,
            enforce_fidel_loanwords=resolved.translation_enforce_fidel_loanwords,
        )
    )
    if resolved.translation_backend == "nllb":
        # NLLB is a sentence-level translation model: it is handed one line at a time
        # and cannot be asked for a shorter rewrite, so the scene/character context,
        # the syllable-budget re-ask and the script check have nothing to act on here.
        # They are not silently skipped - the run says so.
        report(
            "  adaptation: NLLB translates line by line; scene/character context and "
            "length enforcement are not applied"
        )
    elif bible:
        report(f"  dialogue bible: {len(bible)} character(s) applied")
    if not dialogue:
        raise EmptyStageError("adaptation produced no lines; nothing to synthesize")

    # Voice profiles answer "what should this character sound like?" - a question only
    # a per-character engine can act on. A single-voice engine speaks everyone the
    # same way, so building identities for it would be work whose only result is a
    # set of references nothing reads.
    single_voice = resolved.tts_engine != "chatterbox"
    profiles: Mapping[str, VoiceProfile] = {}
    if single_voice:
        seconds["voice_profiles"] = 0.0
        report(f"  voice profiles: skipped ({resolved.tts_engine} is single-voice)")
    else:
        # Built from every transcribed line, not from a shortened run: a profile is a
        # character's identity and is reused across runs, so it is never truncated.
        profiles = _stage(
            "voice_profiles",
            seconds,
            report,
            voice_profiles.build_voice_profiles,
            turns,
            stems.speech,
            transcript=lines,
            settings=resolved,
        )
        if not profiles:
            raise EmptyStageError(
                "no voice profile could be built from the dialogue stem; the TTS stage "
                "needs one identity reference per speaker"
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
    if skipped:
        report(
            f"  skipped: {len(skipped)} line(s) could not be voiced "
            f"({len(synthesized.failed)} engine failure(s))"
        )
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
        dialogue=dialogue,
        profiles=dict(profiles),
        clips=clips,
        skipped=skipped,
        alignment=alignment,
        mix=mixed,
        dub=dub,
        manifest_path=run_dir / MANIFEST_FILENAME,
        stage_seconds=dict(seconds),
        elapsed_seconds=time.perf_counter() - started,
        synthesized_lines=len(clips),
    )

    # Measured, not assumed: the model-free quality report is part of every run.
    # Pronunciation is deliberately left unmeasured here - it needs an Amharic ASR
    # model, and injecting one is the caller's decision (see the qc module).
    quality = qc.build_qc_report(
        result.alignment,
        dialogue=result.dialogue,
        clips=result.clips,
        crosstalk=result.crosstalk,
    )
    # Which hard cases the material actually contains. A run can score well simply
    # because it was easy, so this says what was *not* exercised.
    coverage = evaluation.measure_coverage(result.dialogue, crosstalk=result.crosstalk)

    _write_manifest(result, settings=resolved, quality=quality, coverage=coverage)
    report(result.report())
    report(f"qc              {quality.summary()}")
    report(f"coverage        {coverage.summary()}")
    return result


def _provenance(settings: Settings) -> dict[str, Any]:
    """Describe what actually produced a run, for reproducibility.

    Seed-VC is the one engine that is a *checkout* rather than a pinned dependency,
    so the commit it is sitting on is recorded here. A re-clone that silently moved
    the engine is otherwise invisible in a delivered dub.
    """

    return {
        "seed_vc_revision": tts.seed_vc_revision(settings.seed_vc_repo_path),
        "seed_vc_repo_path": portable_path(Path(settings.seed_vc_repo_path)),
        "seed_vc_convert_style": settings.seed_vc_convert_style,
        "seed_vc_diffusion_steps": settings.seed_vc_diffusion_steps,
    }


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
            "Seed-VC V2 runs in timbre-only mode: it replaces the character's voice "
            "and leaves the take's delivery alone (SEED_VC_CONVERT_STYLE=false)",
            "loudness normalization (EBU R128) is not applied: the mix is placed "
            "at a defined peak with defined dialogue and bed levels",
            "clip paths are content-addressed artifacts of the tts stage, so later "
            "runs of the same lines reuse them instead of re-synthesizing",
            "the qc block is measured, not assumed; pronunciation is left "
            "unmeasured because it needs an Amharic ASR model, which is injected "
            "by the caller rather than downloaded by a run",
        ],
    }

    target = result.manifest_path
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m app.pipeline.orchestrator SOURCE``."""

    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline.orchestrator",
        description=(
            "Run every dubbing stage on a source video and write a dubbed MP4: the "
            "original picture with an Amharic dialogue track."
        ),
    )
    parser.add_argument("source", type=Path, help="source video to dub")
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
            "synthesize only the first N adapted lines (first-run cost control). "
            "The result is a partial dub: the remaining lines are not voiced."
        ),
    )
    args = parser.parse_args(argv)

    try:
        run_pipeline(args.source, output_dir=args.out, max_lines=args.max_lines)
    except OrchestrationError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


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
    "main",
    "run_pipeline",
]
