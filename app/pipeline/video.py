"""Video input/output handling with FFmpeg.

Two ends of the pipeline, of which the first is implemented:

* **Demux/extract** - :func:`extract_audio` pulls the audio track out of the
  source video in the canonical pipeline format (48 kHz stereo PCM WAV), which is
  the only format :func:`app.pipeline.separation.separate_stems` accepts.
* **Mux/export** - :func:`mux_dub` puts the final mixed Amharic audio back into
  the video. The original video stream is copied, never re-encoded, and the
  original audio streams are **replaced**: carrying them over would keep the
  English dialogue in the deliverable, which is the whole point of the dub.

Why the extraction lives here rather than in the runner
------------------------------------------------------
The extraction was previously inlined in ``scripts/test_gpu.py``. It belongs to
this stage so the runner and the orchestrator share one implementation, one
sample rate and one set of errors instead of each spelling them out again.

Stream handling is explicit
---------------------------
Both ends inspect the actual streams with ``ffprobe`` rather than trusting file
names: an earlier attempt muxed a 48 kHz track into a video whose audio was
reported around 44.1 kHz and produced AAC decode errors, because matching
filenames were taken to mean matching formats. Extraction verifies what it wrote
and muxing verifies what it produced.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from app.config import Settings, get_settings
from app.pipeline.voice_profiles import portable_path

#: BandIt v2 Multi is a 48 kHz model, and the whole pipeline runs at that rate so
#: no stem ever has to be resampled.
#: :func:`app.pipeline.separation.separate_stems` rejects anything else, so the
#: extractor must produce exactly this. ``tests/test_video.py`` keeps the two
#: definitions in step.
PIPELINE_SAMPLE_RATE = 48_000

#: Stereo PCM WAV, the layout BandIt expects and mixing will keep.
PIPELINE_CHANNELS = 2

#: Codec of the extracted intermediate. PCM keeps the stage lossless and avoids
#: the AAC sample-rate mismatch that broke an earlier mux attempt.
PIPELINE_CODEC = "pcm_s16le"

#: Suffix given to the extracted track when no destination is passed.
DEFAULT_TRACK_SUFFIX = "_mix.wav"

#: Suffix given to the dubbed video when no destination is passed.
DEFAULT_DUB_SUFFIX = "_amharic.mp4"

#: The delivery audio codec, its bitrate, and how the stream is labelled. AAC at
#: 48 kHz stereo is what every player and streaming target accepts, and 192 kbit/s
#: is transparent for dialogue while keeping the file small. The language tag is
#: what makes players and players' subtitle logic treat the track as Amharic; a
#: title tag is deliberately not set, because the MP4 muxer does not carry one
#: through to the stream.
MUX_AUDIO_CODEC = "aac"
MUX_AUDIO_BITRATE = "192k"
MUX_AUDIO_LANGUAGE = "amh"

#: Containers that accept the MP4 fast-start flag, which moves the index to the
#: front of the file. Worth having for the streaming target, but the flag is not
#: valid for every container, so it is only passed for these.
FASTSTART_SUFFIXES = frozenset({".mp4", ".m4v", ".mov"})

#: How far the delivered audio may differ from the video's own length before the
#: mismatch is reported. The mix is built at the stems' length, which is the
#: source's length, so anything beyond a fraction of a second means an input was
#: not what the pipeline expects.
DURATION_TOLERANCE_SECONDS = 0.5


class VideoError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(VideoError):
    """A source video, or an audio track, does not exist."""


class MissingFfmpegError(VideoError):
    """FFmpeg is not on ``PATH``."""


class MissingFfprobeError(VideoError):
    """FFprobe is not on ``PATH``."""


class ExtractionError(VideoError):
    """FFmpeg failed, or produced nothing."""


class ProbeError(VideoError):
    """FFprobe failed, or returned something unusable."""


class MuxError(VideoError):
    """FFmpeg failed to mux the dub, or produced something unusable."""


class InvalidAudioError(VideoError):
    """An audio track is not the canonical pipeline format."""


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One stream as ``ffprobe`` reported it."""

    index: int
    codec_type: str
    codec_name: str
    sample_rate: int | None = None
    channels: int | None = None
    duration: float | None = None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What ``ffprobe`` found in a media file."""

    path: Path
    format_name: str
    duration: float
    streams: tuple[StreamInfo, ...] = ()

    @property
    def video_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.codec_type == "video")

    @property
    def audio_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.codec_type == "audio")


@dataclass(frozen=True, slots=True)
class MuxResult:
    """The delivered video, and what was verified about it."""

    path: Path
    video_codec: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    video_streams: int
    audio_streams: int
    duration: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the delivered file."""

        return {
            "path": portable_path(self.path),
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "video_streams": self.video_streams,
            "audio_streams": self.audio_streams,
            "duration": self.duration,
            "notes": list(self.notes),
        }


def _run_ffmpeg(arguments: list[str]) -> None:
    """Run ``ffmpeg`` with ``arguments`` and fail clearly when it does not work.

    A module-local seam on purpose: :mod:`app.pipeline.voice_profiles` has the
    same subprocess call under its own name, but each stage owns its error type
    and its wording, so replacing one in a test never affects the other.
    """

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MissingFfmpegError(
            "ffmpeg was not found on PATH; extracting a video's audio track needs "
            "FFmpeg (see the runtime requirements in README.md)"
        )

    process = subprocess.run(  # noqa: S603 - argv list, no shell involved
        [executable, *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip().splitlines()
        raise ExtractionError(
            f"ffmpeg failed to extract the audio track (exit code "
            f"{process.returncode}): {detail[-1] if detail else 'no output'}"
        )


def _verify_track(path: Path) -> None:
    """Check that the extraction produced the canonical 48 kHz stereo WAV.

    FFmpeg will happily write something else - a track that followed its source's
    44.1 kHz rate, or a mono downmix - so the result is inspected rather than
    assumed. Catching it here turns a confusing separation failure into an
    explicit one.
    """

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(
            f"could not read the extracted track {path}: {exc}"
        ) from exc

    if info.samplerate != PIPELINE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"the extracted track is {info.samplerate} Hz, not the "
            f"{PIPELINE_SAMPLE_RATE} Hz the pipeline requires"
        )
    if info.channels != PIPELINE_CHANNELS:
        raise InvalidAudioError(
            f"the extracted track has {info.channels} channel(s), not "
            f"{PIPELINE_CHANNELS}"
        )
    if info.frames <= 0:
        raise InvalidAudioError(f"the extracted track {path} contains no samples")


def extract_audio(
    video: str | Path,
    destination: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Extract ``video``'s audio track as 48 kHz stereo PCM WAV.

    Parameters
    ----------
    video:
        Source video in ``INPUT_DIR`` (``.mp4`` / ``.mkv`` / ...). Video streams
        are dropped; only the audio is kept.
    destination:
        File to write. Defaults to ``<WORK_DIR>/<video stem>_mix.wav``. Parent
        directories are created when missing.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.

    Returns
    -------
    pathlib.Path
        The written track, which is the input for
        :func:`app.pipeline.separation.separate_stems`.

    Raises
    ------
    MissingInputError
        The source video does not exist.
    MissingFfmpegError
        FFmpeg is not on ``PATH``.
    ExtractionError
        FFmpeg failed or wrote nothing.
    InvalidAudioError
        The result is not 48 kHz stereo PCM with samples in it.
    """

    source = Path(video)
    if not source.is_file():
        raise MissingInputError(f"source video not found: {source}")

    resolved = settings if settings is not None else get_settings()
    target = (
        Path(destination)
        if destination is not None
        else Path(resolved.work_dir) / f"{source.stem}{DEFAULT_TRACK_SUFFIX}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(PIPELINE_CHANNELS),
            "-ar",
            str(PIPELINE_SAMPLE_RATE),
            "-c:a",
            PIPELINE_CODEC,
            str(target),
        ]
    )

    if not target.is_file() or target.stat().st_size == 0:
        raise ExtractionError(f"ffmpeg produced no audio at {target}")

    _verify_track(target)
    return target


def _run_ffprobe(arguments: list[str]) -> dict[str, Any]:
    """Run ``ffprobe`` with ``arguments`` and return its JSON output."""

    executable = shutil.which("ffprobe")
    if executable is None:
        raise MissingFfprobeError(
            "ffprobe was not found on PATH; inspecting the streams of a video "
            "needs it (it ships with FFmpeg - see README.md)"
        )

    process = subprocess.run(  # noqa: S603 - argv list, no shell involved
        [executable, *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip().splitlines()
        raise ProbeError(
            f"ffprobe failed (exit code {process.returncode}): "
            f"{detail[-1] if detail else 'no output'}"
        )

    try:
        payload = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned output that is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProbeError("ffprobe returned an unexpected result")
    return payload


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: str | Path) -> MediaInfo:
    """Return the streams and duration of ``path``, as ``ffprobe`` sees them.

    Raises
    ------
    MissingInputError
        The file does not exist.
    MissingFfprobeError
        FFprobe is not on ``PATH``.
    ProbeError
        FFprobe failed or returned something unusable.
    """

    target = Path(path)
    if not target.is_file():
        raise MissingInputError(f"media file not found: {target}")

    payload = _run_ffprobe(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(target),
        ]
    )

    streams: list[StreamInfo] = []
    for position, raw in enumerate(payload.get("streams") or []):
        if not isinstance(raw, dict):
            continue
        streams.append(
            StreamInfo(
                index=_optional_int(raw.get("index")) or position,
                codec_type=str(raw.get("codec_type") or ""),
                codec_name=str(raw.get("codec_name") or ""),
                sample_rate=_optional_int(raw.get("sample_rate")),
                channels=_optional_int(raw.get("channels")),
                duration=_optional_float(raw.get("duration")),
            )
        )

    section = payload.get("format") or {}
    return MediaInfo(
        path=target,
        format_name=str(section.get("format_name") or ""),
        duration=_optional_float(section.get("duration")) or 0.0,
        streams=tuple(streams),
    )


def _verify_dub_track(path: Path) -> tuple[int, int, float]:
    """Check the audio that is about to be muxed, returning its shape."""

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read the mix {path}: {exc}") from exc

    if info.samplerate != PIPELINE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"the mix is {info.samplerate} Hz; the deliverable must be "
            f"{PIPELINE_SAMPLE_RATE} Hz"
        )
    if info.channels != PIPELINE_CHANNELS:
        raise InvalidAudioError(
            f"the mix has {info.channels} channel(s); the deliverable must be stereo"
        )
    if info.frames <= 0:
        raise InvalidAudioError(f"the mix {path} contains no samples")
    return info.samplerate, info.channels, info.frames / info.samplerate


def mux_dub(
    video: str | Path,
    audio: str | Path,
    destination: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> MuxResult:
    """Put ``audio`` into ``video`` as its only audio track.

    The video streams are copied bit-for-bit, so no picture is re-encoded and no
    quality is lost. The source's own audio streams are deliberately not mapped:
    the dubbed track replaces them, because keeping them would leave the original
    dialogue audible under the dub.

    Parameters
    ----------
    video:
        The source video that was extracted and processed.
    audio:
        The final mix from :func:`app.pipeline.mixing.mix_track`, at the pipeline
        sample rate and stereo.
    destination:
        File to write. Defaults to ``<OUTPUT_DIR>/<video stem>_amharic.mp4``.
        Parent directories are created when missing.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.

    Returns
    -------
    MuxResult
        The delivered file plus what ``ffprobe`` verified about it: the audio
        codec, rate and channel count, the stream counts, and any note. A note is
        not a failure - it is the record of something that had to be worked
        around, such as a length mismatch.

    Raises
    ------
    MissingInputError
        The video or the audio file does not exist.
    InvalidAudioError
        The audio is not the pipeline format, or the video has no video stream.
    MissingFfmpegError, MissingFfprobeError, MuxError, ProbeError
        The tools are missing or failed, or the result is not a usable dub.
    """

    source = Path(video)
    if not source.is_file():
        raise MissingInputError(f"source video not found: {source}")
    mix = Path(audio)
    if not mix.is_file():
        raise MissingInputError(f"the mix to mux was not found: {mix}")

    resolved = settings if settings is not None else get_settings()
    target = (
        Path(destination)
        if destination is not None
        else Path(resolved.output_dir) / f"{source.stem}{DEFAULT_DUB_SUFFIX}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    sample_rate, channels, audio_duration = _verify_dub_track(mix)

    origin = probe(source)
    if not origin.video_streams:
        raise InvalidAudioError(
            f"{source} has no video stream, so there is nothing to dub into"
        )

    notes: list[str] = []
    if origin.audio_streams:
        notes.append(
            f"the source's {len(origin.audio_streams)} audio stream(s) were replaced "
            "by the Amharic mix"
        )
    if len(origin.video_streams) > 1:
        notes.append(f"the source has {len(origin.video_streams)} video streams")

    arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(mix),
        # Every video stream, copied rather than re-encoded; the first audio
        # stream - ours - and nothing else.
        "-map",
        "0:v",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        MUX_AUDIO_CODEC,
        "-b:a",
        MUX_AUDIO_BITRATE,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-metadata:s:a:0",
        f"language={MUX_AUDIO_LANGUAGE}",
    ]
    if target.suffix.lower() in FASTSTART_SUFFIXES:
        arguments += ["-movflags", "+faststart"]
    arguments.append(str(target))

    _run_ffmpeg(arguments)

    if not target.is_file() or target.stat().st_size == 0:
        raise MuxError(f"ffmpeg produced no video at {target}")

    delivered = probe(target)
    video_streams = delivered.video_streams
    audio_streams = delivered.audio_streams
    if not video_streams:
        raise MuxError(f"the delivered file {target} has no video stream")
    if len(audio_streams) != 1:
        raise MuxError(
            f"the delivered file {target} has {len(audio_streams)} audio streams; "
            "exactly one dubbed track was expected"
        )

    track = audio_streams[0]
    if track.codec_name != MUX_AUDIO_CODEC:
        raise MuxError(
            f"the delivered audio is {track.codec_name!r}, not {MUX_AUDIO_CODEC!r}"
        )
    if track.sample_rate != PIPELINE_SAMPLE_RATE:
        raise MuxError(
            f"the delivered audio is {track.sample_rate} Hz, not "
            f"{PIPELINE_SAMPLE_RATE} Hz"
        )

    source_video_codec = origin.video_streams[0].codec_name
    if video_streams[0].codec_name != source_video_codec:
        notes.append(
            f"the video codec changed from {source_video_codec!r} to "
            f"{video_streams[0].codec_name!r}"
        )
    if len(video_streams) != len(origin.video_streams):
        notes.append(
            f"the delivered file has {len(video_streams)} video streams, the source "
            f"had {len(origin.video_streams)}"
        )

    if abs(delivered.duration - audio_duration) > DURATION_TOLERANCE_SECONDS:
        notes.append(
            f"the delivered file is {delivered.duration:.3f}s long but the mix is "
            f"{audio_duration:.3f}s; the picture and the dub may drift apart"
        )

    return MuxResult(
        path=target,
        video_codec=video_streams[0].codec_name,
        audio_codec=track.codec_name,
        audio_sample_rate=track.sample_rate or sample_rate,
        audio_channels=track.channels or channels,
        video_streams=len(video_streams),
        audio_streams=len(audio_streams),
        duration=delivered.duration,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_DUB_SUFFIX",
    "DEFAULT_TRACK_SUFFIX",
    "DURATION_TOLERANCE_SECONDS",
    "FASTSTART_SUFFIXES",
    "MUX_AUDIO_BITRATE",
    "MUX_AUDIO_CODEC",
    "MUX_AUDIO_LANGUAGE",
    "PIPELINE_CHANNELS",
    "PIPELINE_CODEC",
    "PIPELINE_SAMPLE_RATE",
    "ExtractionError",
    "InvalidAudioError",
    "MediaInfo",
    "MissingFfmpegError",
    "MissingFfprobeError",
    "MissingInputError",
    "MuxError",
    "MuxResult",
    "ProbeError",
    "StreamInfo",
    "VideoError",
    "extract_audio",
    "mux_dub",
    "probe",
]
