"""End-to-end smoke run of the implemented stages on a real GPU.

One command drives the whole currently-implemented chain and prints what each
stage actually produced, so the pipeline can be validated on the A40::

    python scripts/test_gpu.py
    python scripts/test_gpu.py --video data/input/test.mp4 --max-lines 3

    data/input/test.mp4
        -> audio track extracted by this script (see note below)
        -> separation.separate_stems          (speech / music / effects)
        -> diarization.diarize                (speaker turns)
        -> transcription.transcribe           (spoken lines)
        -> translation.adapt_dialogue         (Amharic + performance metadata)
        -> voice_profiles.build_voice_profiles (per-speaker reference)
        -> tts.synthesize_dialogue            (Amharic speech clips)

Output goes to ``data/output/tts-test/``: ``takes/`` holds the raw engine output
per line, ``clips/`` the final clips to listen to (the take with the line's
pauses), plus ``performance/`` (the original-performance references cut from the
speech stem) and ``stages/`` (the extracted mix and the separated stems).

Notes
-----
* This is a runner, not a stage and not a test: it only calls the existing stage
  functions in their existing order, passing each stage's own outputs to the next
  one. No text, speaker id, timing or performance value is hard-coded here.
* Which models the translation and synthesis stages use is the configuration's
  decision, not this runner's: ``TRANSLATION_BACKEND`` picks the translator and
  ``TTS_ENGINE`` the synthesis engine, and both are reported before the run. The
  runner follows the same branch the orchestrator does, so a single-voice engine
  skips the voice-profile stage entirely rather than building references nothing
  reads.
* The audio track is extracted by ``app.pipeline.video.extract_audio``, at the
  48 kHz that BandIt v2 Multi requires, so the runner and the orchestrator share
  one extractor instead of each keeping its own copy.
* The run stops after speech synthesis **by design**: this runner reports what
  each GPU stage produced, and the stages after it need no GPU. To produce the
  deliverable, run ``python -m app.pipeline.orchestrator``, which continues
  through timing, mixing and muxing to a dubbed MP4 and reuses the clips this
  runner already wrote.
* Real weights are downloaded on first use; expect a long first run.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import soundfile as sf

# Invoking this file directly puts ``scripts/`` on ``sys.path`` rather than the
# project root, so the ``app`` package is made importable explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, Settings, get_settings  # noqa: E402
from app.pipeline import (  # noqa: E402
    diarization,
    separation,
    transcription,
    translation,
    tts,
    video,
    voice_profiles,
)
from app.pipeline.translation import AdaptedDialogue  # noqa: E402
from app.pipeline.tts import TtsClip  # noqa: E402
from app.pipeline.voice_profiles import VoiceProfile  # noqa: E402

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "input" / "test.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "output" / "tts-test"


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def _info(label: str, value: object) -> None:
    print(f"  {label:<26} {value}", flush=True)


def _shorten(text: str, limit: int = 88) -> str:
    """Collapse whitespace and truncate, for readable one-line previews."""

    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _audio_summary(path: Path) -> str:
    """Return ``duration / rate / channels`` of an audio file."""

    info = sf.info(str(path))
    return (
        f"{info.frames / info.samplerate:6.2f} s  "
        f"{info.samplerate:6d} Hz  {info.channels} ch  {info.subtype}"
    )


def report_settings(settings: Settings) -> None:
    """Print the resolved configuration and stop early on missing credentials."""

    _rule("Settings")
    _info("device", settings.device)
    _info("output dir", settings.output_dir)
    _info("work dir", settings.work_dir)
    _info("model cache", settings.model_cache_dir)
    _info("diarization model", settings.diarization_model)
    _info("transcription model", settings.transcription_model)
    _info("translation backend", settings.translation_backend)
    _info("translation model", settings.translation_model)
    _info("tts engine", settings.tts_engine)
    if settings.tts_engine == "omnivoice":
        _info("omnivoice model", settings.omnivoice_model)
        _info(
            "omnivoice steps",
            f"{settings.omnivoice_steps} @ guidance "
            f"{settings.omnivoice_guidance_scale:g}",
        )
    elif settings.tts_engine == "chatterbox":
        _info("chatterbox model", settings.chatterbox_model)
        _info("seed-vc checkout", settings.seed_vc_repo_path)
    else:
        _info("tts model", settings.tts_model)

    # The check is configuration-aware, so this names only what this run will
    # actually read: the token always, the DeepSeek key only under the
    # instruction-following backend.
    missing = settings.missing_credentials()
    if missing:
        raise RuntimeError(
            "missing credentials: "
            + ", ".join(missing)
            + " - the gated diarization pipeline needs HUGGINGFACE_TOKEN"
            + (
                ", and TRANSLATION_BACKEND=openai needs DEEPSEEK_API_KEY"
                if settings.translation_backend == "openai"
                else ""
            )
            + " (copy .env.example to .env and fill them in)"
        )


def report_stems(stems: separation.StemPaths) -> None:
    """Print the separated stems with their duration, rate and channel count."""

    _rule("1. Separation - BandIt v2 Multi")
    for name, path in stems.as_dict().items():
        _info(f"{name} stem", f"{path}")
        _info("", _audio_summary(path))


def report_turns(turns: list[diarization.SpeakerSegment]) -> None:
    """Print the diarized turns, grouped per speaker."""

    _rule("2. Diarization - pyannote Community-1")
    speakers: dict[str, list[diarization.SpeakerSegment]] = {}
    for turn in turns:
        speakers.setdefault(turn.speaker_id, []).append(turn)

    _info("turns", len(turns))
    _info("speakers", len(speakers))
    for speaker_id, own in speakers.items():
        spoken = sum(turn.end - turn.start for turn in own)
        first = min(turn.start for turn in own)
        _info(
            speaker_id,
            f"{len(own):3d} turns  {spoken:7.2f} s speech  first at {first:7.2f} s",
        )


def report_lines(lines: list[transcription.TranscriptSegment]) -> None:
    """Print every transcribed line."""

    _rule("3. Transcription - faster-whisper large-v3")
    _info("lines", len(lines))
    for index, line in enumerate(lines, start=1):
        print(
            f"  [{index:03d}] {line.start:7.2f}-{line.end:7.2f} s  "
            f"{line.speaker_id:<12} {_shorten(line.text)}",
            flush=True,
        )


def report_dialogue(dialogue: list[AdaptedDialogue], settings: Settings) -> None:
    """Print the adapted Amharic lines with the performance the model chose."""

    _rule(f"4. Adaptation - {settings.translation_backend} ({settings.translation_model})")
    _info("lines", len(dialogue))
    for index, line in enumerate(dialogue, start=1):
        print(
            f"  [{index:03d}] {line.start:7.2f}-{line.end:7.2f} s  "
            f"{line.speaker_id:<12} {_shorten(line.amharic)}",
            flush=True,
        )
        print(
            f"        emotion={line.emotion!r} intensity={line.intensity:.2f} "
            f"delivery={line.delivery!r} "
            f"pauses={line.pause_before:.2f}/{line.pause_after:.2f} s",
            flush=True,
        )


def report_profiles(profiles: dict[str, VoiceProfile]) -> None:
    """Print each speaker's selected reference - the voice identity for TTS."""

    _rule("5. Voice profiles - per-speaker reference")
    for speaker_id, profile in profiles.items():
        _info(speaker_id, f"reference for {profile.reference_duration:.2f} s")
        _info("  reference audio", profile.resolve_reference_audio())
        _info(
            "  window in stem",
            f"{profile.reference_start:7.2f}-{profile.reference_end:7.2f} s",
        )
        _info("  quality score", f"{profile.quality_score:.3f}")
        _info("  why chosen", _shorten(profile.selection_reason or "(not recorded)"))


def _engine_label(settings: Settings) -> str:
    """Return the synthesis chain a run with these settings uses, for reporting."""

    if settings.tts_engine == "chatterbox":
        return "Chatterbox Amharic -> Seed-VC V2"
    if settings.tts_engine == "omnivoice":
        return f"OmniVoice cloned voice ({settings.omnivoice_model})"
    return f"{settings.tts_engine} ({settings.tts_model})"


def report_clips(
    clips: list[TtsClip],
    output_dir: Path,
    *,
    settings: Settings,
) -> None:
    """Print every generated clip, its engine stages and its controls."""

    _rule(f"6. Speech synthesis - {_engine_label(settings)}")
    _info("clips", len(clips))
    for clip in clips:
        controls = clip.performance
        print(
            f"\n  [{clip.index:03d}] {clip.speaker_id}  "
            f"{clip.start:7.2f}-{clip.end:7.2f} s "
            f"(original {clip.original_duration:5.2f} s)",
            flush=True,
        )
        print(f"        amharic  {_shorten(clip.amharic)}", flush=True)
        print(f"        source   {_shorten(clip.dialogue.source_text)}", flush=True)
        # Both paths write a raw take and then a clip with the pauses rendered, so
        # the two files are reported the same way and the engines that produced them
        # are named: under a single-voice engine there is no conversion, and the
        # take is the clip's only input.
        _info(f"  take  [{clip.performance_engine}]", f"{clip.take_path}")
        _info("", _audio_summary(clip.take_path))
        _info(f"  clip  [{clip.style_engine}]", f"{clip.audio_path}")
        _info("", _audio_summary(clip.audio_path))
        _info(
            "  durations",
            f"speech {clip.speech_duration:.2f} s + pauses "
            f"{clip.rendered_pause_before:.2f}/{clip.rendered_pause_after:.2f} s "
            f"= {clip.duration:.2f} s",
        )
        _info("  sample rate", f"{clip.sample_rate} Hz")
        _info(
            "  performance",
            f"exaggeration={controls.exaggeration:.3f} "
            f"cfg_weight={controls.cfg_weight:.3f} "
            f"temperature={controls.temperature:.3f} seed={controls.seed}",
        )
        _info(
            "  mapped from",
            f"emotion={controls.emotion!r} intensity={controls.intensity:.2f} "
            f"delivery={controls.delivery!r}",
        )
        if clip.performance_reference_path is not None:
            _info("  performance prompt", f"{clip.performance_reference_path}")
        if clip.voice_reference_path is not None:
            _info("  identity reference", f"{clip.voice_reference_path}")

    _rule("Output")
    for directory in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        files = sorted(directory.iterdir())
        _info(directory.name, f"{len(files)} file(s)")
        for path in files:
            _info("", f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    _info("listen to", output_dir / "clips")


def run(source: Path, output_dir: Path, max_lines: int | None) -> int:
    """Run the implemented chain once. Returns a process exit code."""

    started = time.perf_counter()
    stage = "settings"
    try:
        settings = get_settings()
        settings.ensure_directories()
        report_settings(settings)

        work = output_dir / "stages"
        stage = "audio extraction"
        _rule("0. Audio track (video.extract_audio)")
        mix = video.extract_audio(
            source, work / f"{source.stem}_mix.wav", settings=settings
        )
        _info("video", f"{source}  ({source.stat().st_size / 1e6:.1f} MB)")
        _info("extracted mix", f"{mix}")
        _info("", _audio_summary(mix))

        stage = "separation"
        stems = separation.separate_stems(mix, output_dir=work, settings=settings)
        report_stems(stems)

        stage = "diarization"
        turns = diarization.diarize(stems.speech, settings=settings)
        report_turns(turns)
        if not turns:
            raise RuntimeError("diarization found no speech in the speech stem")

        stage = "transcription"
        lines = transcription.transcribe(stems.speech, turns, settings=settings)
        report_lines(lines)
        if not lines:
            raise RuntimeError("transcription produced no lines")

        stage = "adaptation"
        dialogue = translation.adapt_dialogue(lines, settings=settings)
        report_dialogue(dialogue, settings)
        if not dialogue:
            raise RuntimeError("adaptation produced no lines")

        stage = "voice profiles"
        # Only an engine that reads a per-speaker reference needs an identity built for
        # it. A single-voice engine speaks every line the same way, so building profiles
        # would burn minutes and a model download on references nothing reads - which is
        # also why the orchestrator skips the stage.
        needs_profiles = tts.engine_needs_profiles(settings.tts_engine)
        profiles: dict[str, VoiceProfile] = {}
        if not needs_profiles:
            _rule("5. Voice profiles - skipped")
            _info(
                "reason", f"{settings.tts_engine} is single-voice, one voice for all"
            )
        else:
            profiles = voice_profiles.build_voice_profiles(
                turns, stems.speech, transcript=lines, settings=settings
            )
            report_profiles(profiles)

        stage = "tts"
        if max_lines is not None and max_lines < len(dialogue):
            print(f"\n  --max-lines {max_lines}: synthesizing the first {max_lines} line(s)")
            dialogue = dialogue[:max_lines]
        clips = tts.synthesize_dialogue(
            dialogue, stems.speech, profiles, output_dir=output_dir, settings=settings
        )
        report_clips(clips, output_dir, settings=settings)
    except Exception as exc:  # noqa: BLE001 - the runner reports everything it hits
        print(f"\nFAILED during: {stage}")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _rule("Done")
    _info("stage", "validated every GPU stage up to speech synthesis")
    _info("clips", f"{len(clips)} in {output_dir / 'clips'}")
    _info("elapsed", f"{time.perf_counter() - started:.1f} s")
    _info(
        "next",
        "python -m app.pipeline.orchestrator <video>  -> dubbed MP4 "
        "(timing, mixing, muxing; reuses these clips)",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the implemented dubbing stages once, end to end, on a GPU."
    )
    parser.add_argument(
        "--video", type=Path, default=DEFAULT_VIDEO, help="source MP4 to dub"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for the generated audio",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="synthesize only the first N adapted lines (first-run cost control)",
    )
    args = parser.parse_args(argv)

    return run(args.video, args.out, args.max_lines)


if __name__ == "__main__":
    sys.exit(main())
