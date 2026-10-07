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
        -> tts.synthesize_dialogue            (Chatterbox -> Seed-VC V2 clips)

Output goes to ``data/output/tts-test/``: ``chatterbox/`` holds the intermediate
Amharic takes, ``converted/`` the Seed-VC output, ``clips/`` the final clips to
listen to (Seed-VC audio with the line's pauses), plus ``performance/`` (the
original-performance references cut from the speech stem) and ``stages/`` (the
extracted mix and the separated stems).

Notes
-----
* This is a runner, not a stage and not a test: it only calls the existing stage
  functions in their existing order, passing each stage's own outputs to the next
  one. No text, speaker id, timing or performance value is hard-coded here.
* ``app/pipeline/video.py`` is still a placeholder, so the audio track is pulled
  out of the MP4 here with FFmpeg, at the 48 kHz that BandIt v2 Multi requires.
* Voice identity comes from ``voice_profiles.VoiceProfile`` alone: the reference
  is selected from the speaker's own dialogue in the separated speech stem, and
  the TTS stage uses it as the Seed-VC target while the original actor audio of
  each line is the Chatterbox performance prompt.
* The run stops after TTS. ``timing`` and ``mixing`` are not implemented, so the
  clips are not yet aligned to the original timings nor muxed back.
* Real weights are downloaded on first use; expect a long first run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    voice_profiles,
)
from app.pipeline.translation import AdaptedDialogue  # noqa: E402
from app.pipeline.tts import TtsClip  # noqa: E402
from app.pipeline.voice_profiles import VoiceProfile  # noqa: E402

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "input" / "test.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "output" / "tts-test"

#: BandIt v2 Multi is a 48 kHz model and separation rejects every other rate.
MIX_SAMPLE_RATE = 48_000
MIX_CHANNELS = 2


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


def extract_track(video: Path, destination: Path) -> Path:
    """Extract the video's audio track as 48 kHz stereo PCM WAV.

    The ``video`` stage that will own this is still a placeholder, so the runner
    does it here; separation needs a 48 kHz file to accept the input at all.
    """

    if not video.is_file():
        raise FileNotFoundError(f"test video not found: {video}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        str(MIX_CHANNELS),
        "-ar",
        str(MIX_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no audio at {destination}")
    return destination


def report_settings(settings: Settings) -> None:
    """Print the resolved configuration and stop early on missing credentials."""

    _rule("Settings")
    _info("device", settings.device)
    _info("output dir", settings.output_dir)
    _info("work dir", settings.work_dir)
    _info("model cache", settings.model_cache_dir)
    _info("diarization model", settings.diarization_model)
    _info("transcription model", settings.transcription_model)
    _info("translation model", settings.translation_model)
    _info("tts model", settings.tts_model)
    _info("seed-vc checkout", settings.seed_vc_repo_path)

    missing = settings.missing_credentials()
    if missing:
        raise RuntimeError(
            "missing credentials: "
            + ", ".join(missing)
            + " - diarization needs HUGGINGFACE_TOKEN and adaptation needs "
            "DEEPSEEK_API_KEY (copy .env.example to .env and fill them in)"
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


def report_dialogue(dialogue: list[AdaptedDialogue]) -> None:
    """Print the adapted Amharic lines with the performance the model chose."""

    _rule("4. Adaptation - DeepSeek Amharic dialogue")
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


def report_clips(clips: list[TtsClip], output_dir: Path) -> None:
    """Print every generated clip, its engine stages and its controls."""

    _rule("6+7. Speech - Chatterbox Amharic -> Seed-VC V2")
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
        _info("  chatterbox take", f"{clip.take_path}")
        _info("", _audio_summary(clip.take_path))
        _info("  seed-vc clip", f"{clip.audio_path}")
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
        _info("  performance prompt", f"{clip.performance_reference_path}")
        _info("  identity reference", f"{clip.voice_reference_path}")

    _rule("Output")
    for directory in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        files = sorted(directory.iterdir())
        _info(directory.name, f"{len(files)} file(s)")
        for path in files:
            _info("", f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    _info("listen to", output_dir / "clips")


def run(video: Path, output_dir: Path, max_lines: int | None) -> int:
    """Run the implemented chain once. Returns a process exit code."""

    started = time.perf_counter()
    stage = "settings"
    try:
        settings = get_settings()
        settings.ensure_directories()
        report_settings(settings)

        work = output_dir / "stages"
        stage = "audio extraction"
        _rule("0. Audio track (video stage is still a placeholder)")
        mix = extract_track(video, work / f"{video.stem}_mix.wav")
        _info("video", f"{video}  ({video.stat().st_size / 1e6:.1f} MB)")
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
        report_dialogue(dialogue)
        if not dialogue:
            raise RuntimeError("adaptation produced no lines")

        stage = "voice profiles"
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
        report_clips(clips, output_dir)
    except Exception as exc:  # noqa: BLE001 - the runner reports everything it hits
        print(f"\nFAILED during: {stage}")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _rule("Done")
    _info("stage", "stopped after tts (timing and mixing are not implemented)")
    _info("clips", f"{len(clips)} in {output_dir / 'clips'}")
    _info("elapsed", f"{time.perf_counter() - started:.1f} s")
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
