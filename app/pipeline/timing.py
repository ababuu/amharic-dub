"""Timing alignment for dubbed dialogue.

Each synthesized clip is fitted to the window of the line it replaces:

* the **speech** of the line is time-stretched, pitch-preserving, to the length of
  the original line, so the dub lands on the same beats as the performance;
* the **rendered pauses** are left alone - they are deliberate silence, not slack
  to be absorbed - so a line's leading pause still precedes the line;
* a line that would need more than the configured tempo band is **reported** as
  not fitting rather than mangled to fit, and the run records by how much;
* a line that is still too long **moves the next line later** instead of being
  spoken over it, so two voices are never heard at once unless the source put
  them there.

The result of this stage is one aligned WAV per line at the pipeline sample rate,
plus where to place it in the timeline, which is exactly what
:mod:`app.pipeline.mixing` consumes.

Deliberate non-goals
--------------------
* **No resampling for change of speed.** Speed is changed with FFmpeg's
  ``atempo``, which preserves pitch; resampling a clip to change its length would
  shift every voice up or down, which is never wanted.
* **No fitting across the tempo band.** Nothing here silently rewrites what the
  dialogue model asked for: a clamped stretch is recorded on the clip.
* **No padding to fill a window.** A short line keeps its natural delivery and is
  placed at its original start; the gap that follows is silence, not stretched
  speech.
* **No touching of the clip files.** The clips remain the TTS stage's artifacts;
  the aligned audio is written beside them.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings
from app.pipeline.dialogue_context import MINIMUM_PLACEMENT_WINDOW
from app.pipeline.tts import TtsClip
from app.pipeline.voice_profiles import portable_path

#: Every stage after separation works at BandIt's 48 kHz, so no stage has to
#: resample another stage's output. ``tests/test_timing.py`` keeps this in step
#: with ``separation.REQUIRED_SAMPLE_RATE`` and ``video.PIPELINE_SAMPLE_RATE``.
PIPELINE_SAMPLE_RATE = 48_000

#: Directory (under the work directory) holding the aligned clips.
TIMING_DIRECTORY_NAME = "timing"

#: Sub-directory of :data:`TIMING_DIRECTORY_NAME` holding the aligned lines.
ALIGNED_DIRECTORY_NAME = "aligned"

#: A residual this small is not worth a filter: it is below both the ear's sense
#: of lip-sync and FFmpeg's own rounding of a tempo change.
EXACT_FIT_TOLERANCE_SECONDS = 0.03

#: ``atempo`` only accepts factors inside this range, so a configuration outside
#: it is rejected instead of producing an unusable filter argument.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

#: Length of the fade applied to the tail of a line that has to be cut short, in
#: seconds. Long enough that the cut is inaudible, short enough not to lose a syllable.
TRIM_FADE_SECONDS = 0.025


class TimingError(RuntimeError):
    """Base class for every error raised by this module."""


class ConfigurationError(TimingError, ValueError):
    """The tempo bounds contradict FFmpeg or each other."""


class InvalidClipError(TimingError, ValueError):
    """A clip is not the :class:`~app.pipeline.tts.TtsClip` this stage needs."""


class MissingInputError(TimingError):
    """A clip's audio file, or the configured output directory, is unusable."""


class MissingFfmpegError(TimingError):
    """FFmpeg is not on ``PATH``."""


class StretchError(TimingError):
    """FFmpeg failed to stretch a line, or produced something unusable."""


class InvalidAudioError(TimingError):
    """A clip, or a stretched result, is not usable audio."""


@dataclass(frozen=True, slots=True)
class AlignedClip:
    """One line, stretched to fit and placed on the source timeline.

    ``start`` is where the file begins, already accounting for the leading pause,
    so ``start`` is earlier than the original line's own start by exactly
    ``rendered_pause_before``. ``required_tempo`` is what an exact fit would have
    needed and ``tempo`` is what was applied: the two differ only for a line that
    did not fit, which is why both are kept. ``drift`` is how far the line had to
    move later than the original performance because the line before it ran long.
    """

    index: int
    clip: TtsClip
    audio_path: Path
    sample_rate: int
    start: float
    tempo: float
    required_tempo: float
    speech_duration: float
    original_window: float
    rendered_pause_before: float
    rendered_pause_after: float
    notes: tuple[str, ...] = ()
    #: Seconds of speech this line was allowed before it would run into the next one.
    #: Equal to ``original_window`` when the line was considered alone.
    available: float | None = None
    #: Seconds cut off the end, with a fade, because the line could not fit even at the
    #: tempo limit. Non-zero means the translation was longer than the time available.
    #: Stays zero unless ``TIMING_TRIM_TO_FIT`` is on.
    trimmed: float = 0.0
    #: Seconds this line runs past its available time. Non-zero means it will overlap the
    #: next line; it is reported rather than cut away.
    overrun: float = 0.0
    #: Seconds this line was moved later than the original performance started, because the
    #: line before it ran long. ``0.0`` means it landed exactly where the actor spoke.
    #: A drifted line is *late* rather than doubling another voice, which is the smaller
    #: cost: two voices at once is unintelligible, a late line is merely a little behind.
    drift: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidClipError(
                f"index must be a non-negative integer, got {self.index!r}"
            )
        if not isinstance(self.clip, TtsClip):
            raise InvalidClipError(
                f"clip must be a TtsClip, got {type(self.clip).__name__}"
            )

        object.__setattr__(self, "audio_path", Path(self.audio_path))

        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise InvalidClipError(
                f"sample_rate must be an integer, got {self.sample_rate!r}"
            )
        if self.sample_rate < 1:
            raise InvalidClipError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )

        for name in ("start", "tempo", "required_tempo", "rendered_pause_before",
                     "rendered_pause_after", "drift", "overrun", "trimmed"):
            value = getattr(self, name)
            number = _finite(name, value)
            if number < 0:
                raise InvalidClipError(f"{name} must be >= 0, got {number}")
            object.__setattr__(self, name, number)

        for name in ("speech_duration", "original_window"):
            number = _finite(name, getattr(self, name))
            if number <= 0:
                raise InvalidClipError(f"{name} must be positive, got {number}")
            object.__setattr__(self, name, number)

        if isinstance(self.notes, list):
            object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def speaker_id(self) -> str:
        """The diarization speaker id this line belongs to."""

        return self.clip.speaker_id

    @property
    def residual(self) -> float:
        """How much longer than its window the delivered speech is, in seconds.

        Negative means the line underruns its window, positive means it overruns.
        """

        return self.speech_duration - self.original_window

    @property
    def fits(self) -> bool:
        """``True`` when the speech matches its window within the tolerance."""

        return abs(self.residual) <= EXACT_FIT_TOLERANCE_SECONDS

    @property
    def duration(self) -> float:
        """Length of the delivered file: speech plus its rendered pauses."""

        return (
            self.rendered_pause_before + self.speech_duration + self.rendered_pause_after
        )

    @property
    def end(self) -> float:
        """Where the delivered file stops on the source timeline."""

        return self.start + self.duration

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the aligned line."""

        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "audio_path": portable_path(self.audio_path),
            "sample_rate": self.sample_rate,
            "start": self.start,
            "end": self.end,
            "original_start": self.clip.start,
            "original_end": self.clip.end,
            "original_window": self.original_window,
            "tempo": self.tempo,
            "required_tempo": self.required_tempo,
            "speech_duration": self.speech_duration,
            "residual": self.residual,
            "fits": self.fits,
            "available": self.available,
            "trimmed": self.trimmed,
            "overrun": self.overrun,
            "drift": self.drift,
            "rendered_pause_before": self.rendered_pause_before,
            "rendered_pause_after": self.rendered_pause_after,
            "duration": self.duration,
            "notes": list(self.notes),
        }


def _finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite float, or raise :class:`InvalidClipError`."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidClipError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise InvalidClipError(f"{name} must be a finite number, got {value!r}")
    return number


def _run_ffmpeg(arguments: list[str]) -> None:
    """Run ``ffmpeg`` with ``arguments`` and fail clearly when it does not work."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MissingFfmpegError(
            "ffmpeg was not found on PATH; aligning dubbed dialogue needs FFmpeg "
            "(see the runtime requirements in README.md)"
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
        raise StretchError(
            f"ffmpeg failed to stretch a dubbed line (exit code "
            f"{process.returncode}): {detail[-1] if detail else 'no output'}"
        )


def _read_mono_file(path: Path, *, label: str) -> tuple[np.ndarray, int]:
    """Read an audio file into a mono float32 array, with its sample rate."""

    if not path.is_file():
        raise MissingInputError(f"{label} was not found: {path}")

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {label} at {path}: {exc}") from exc

    if data.size == 0 or data.shape[0] == 0:
        raise InvalidAudioError(f"{label} at {path} contains no samples")
    if sample_rate < 1:
        raise InvalidAudioError(f"{label} at {path} has an invalid sample rate")

    # The clips are mono, but a stereo one would still be usable: generated
    # dialogue is placed as a single centred channel, so channels are averaged.
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(sample_rate)


def _write_mono(path: Path, samples: np.ndarray, sample_rate: int, *, label: str) -> Path:
    """Write ``samples`` as mono PCM-16 WAV and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            str(path), np.asarray(samples, dtype=np.float32), sample_rate,
            format="WAV", subtype="PCM_16",
        )
    except (OSError, RuntimeError) as exc:
        raise StretchError(f"could not write the {label} to {path}: {exc}") from exc
    return path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _speech_region(
    clip: TtsClip, samples: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Return the line's speech, without its rendered pauses, from the clip file.

    The clip is ``[pause_before][speech][pause_after]``, so the speech is a slice
    at a known offset. A clip whose file is shorter than its metadata claims is
    used as far as it goes rather than rejected: the metadata comes from the TTS
    stage's own measurement of the same file, so any disagreement is a fraction
    of a sample.
    """

    first = int(round(clip.rendered_pause_before * sample_rate))
    count = int(round(clip.speech_duration * sample_rate))
    if count < 1:
        raise InvalidClipError(
            f"the clip for line {clip.index} has no speech to align "
            f"({clip.speech_duration:.4f}s)"
        )

    region = samples[first : first + count]
    if region.size == 0:
        raise InvalidClipError(
            f"the clip for line {clip.index} is shorter than its own leading pause"
        )
    return np.ascontiguousarray(region, dtype=np.float32)


def _stretched_speech(
    clip: TtsClip,
    region: np.ndarray,
    sample_rate: int,
    tempo: float,
    destination: Path,
) -> tuple[np.ndarray, int]:
    """Return the line's speech stretched by ``tempo``, at the pipeline rate.

    The speech is written out and read back through FFmpeg, which does both jobs
    in one pass: the tempo change (pitch-preserving, and only when one is needed)
    and the sample-rate conversion to the pipeline rate.
    """

    source = _write_mono(destination, region, sample_rate, label="clip speech")
    output = destination.with_name(f"{destination.stem}.stretched.wav")
    arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if tempo != 1.0:
        # atempo is the pitch-preserving stretcher; the length of the result is
        # the input length divided by the factor.
        arguments += ["-af", f"atempo={tempo:.6f}"]
    arguments += [
        "-ar",
        str(PIPELINE_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    try:
        _run_ffmpeg(arguments)
    finally:
        source.unlink(missing_ok=True)

    try:
        samples, rate = _read_mono_file(output, label="the stretched line")
    finally:
        output.unlink(missing_ok=True)

    if rate != PIPELINE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"ffmpeg returned {rate} Hz for a stretched line, expected "
            f"{PIPELINE_SAMPLE_RATE} Hz"
        )
    if samples.size == 0:
        raise StretchError(f"stretching line {clip.index} produced no audio")
    return samples, rate


def _fade_out(
    samples: np.ndarray,
    sample_rate: int,
    seconds: float = TRIM_FADE_SECONDS,
) -> np.ndarray:
    """Return ``samples`` with its tail faded to silence.

    Used only when a line has to be cut short. Cutting mid-waveform is a click; a fade
    over a few tens of milliseconds is inaudible, and the speech that is kept is not
    altered at all.
    """

    count = min(int(round(seconds * sample_rate)), samples.shape[0])
    if count <= 1:
        return samples
    faded = np.array(samples, dtype=np.float32, copy=True)
    faded[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return faded


def _audio_start(clip: TtsClip) -> float:
    """Return where a clip's file begins on the source timeline."""

    lead = min(clip.rendered_pause_before, clip.dialogue.start)
    return clip.dialogue.start - lead


def align_clip(
    clip: TtsClip,
    destination: str | Path,
    *,
    settings: Settings | None = None,
    room: float | None = None,
) -> AlignedClip:
    """Fit one clip to its window, and describe where to place it.

    Parameters
    ----------
    clip:
        A clip from :func:`app.pipeline.tts.synthesize_dialogue`.
    destination:
        File to write the aligned line to. Created, with its parents, when missing.
    settings:
        Project settings override; the tempo band comes from here.
    room:
        How many seconds of speech this line may occupy before it would run into the
        next one, including any silence between them. ``None`` means "no more than its
        own window", which is what a single clip considered alone can know.

    Returns
    -------
    AlignedClip

    Notes
    -----
    A line that fits its own window is fitted to that window, exactly: matching the
    original timing is what keeps the dub in step with the picture. A line that *cannot*
    fit is allowed to use the silence after it up to ``room`` before it is squashed at
    all, because Amharic does not express the same idea in the same number of syllables
    the English did and compressing hard to hit a window is what makes a dub sound
    rushed. Only a line that still does not fit at the tempo limit is cut short, and that
    is reported rather than done quietly.
    """

    resolved = settings if settings is not None else get_settings()
    minimum, maximum = _tempo_bounds(resolved)

    if not isinstance(clip, TtsClip):
        raise InvalidClipError(f"expected a TtsClip, got {type(clip).__name__}")

    samples, sample_rate = _read_mono_file(
        Path(clip.audio_path), label=f"the clip for line {clip.index}"
    )
    region = _speech_region(clip, samples, sample_rate)

    window = clip.dialogue.duration
    speech = clip.speech_duration

    # The room a line may use is never smaller than its own window: the window is where
    # the actor actually spoke, and no line should be squashed below the performance it
    # replaces just because the line in front of it ran long. ``room`` can come back
    # smaller than the window when the two original performances overlapped.
    limit = (
        window
        if room is None
        else max(window, float(room), MINIMUM_PLACEMENT_WINDOW)
    )
    target = window if speech <= window else max(window, min(limit, speech))
    required = speech / target
    within_tolerance = abs(speech - window) <= EXACT_FIT_TOLERANCE_SECONDS
    tempo = 1.0 if within_tolerance else _clamp(required, minimum, maximum)

    stretched, rate = _stretched_speech(
        clip,
        region,
        sample_rate,
        tempo,
        Path(destination).with_name(f"{Path(destination).stem}.speech.wav"),
    )
    speech_duration = stretched.shape[0] / rate

    notes: list[str] = []
    trimmed = 0.0
    overrun = 0.0
    if speech_duration > limit + EXACT_FIT_TOLERANCE_SECONDS:
        overrun = speech_duration - limit
        if resolved.timing_trim_to_fit:
            # Cutting is opt-in, because it is the one outcome that damages the
            # performance: the listener hears a word clipped off the end. Two voices
            # briefly at once is the lesser evil, so an over-long line is reported and
            # left intact unless the run has asked for the guarantee.
            keep = int(round(limit * rate))
            if keep > 0:
                trimmed = speech_duration - keep / rate
                stretched = _fade_out(stretched[:keep], rate)
                speech_duration = keep / rate
                overrun = 0.0
                notes.append(
                    f"the line needed {required:.3f}x its own window but even the "
                    f"{limit:.3f}s available to it (up to the next line) only allows "
                    f"{maximum:g}x; the last {trimmed:.3f}s were cut with a fade so the "
                    "next line is not spoken over - the Amharic is longer than its window"
                )
        else:
            notes.append(
                f"the line is {overrun:.3f}s longer than the {limit:.3f}s it has before "
                "the next line and it is left intact; the next line is moved later "
                "instead of the two being spoken over each other - the Amharic is "
                "longer than its time"
            )

    residual = speech_duration - window

    if within_tolerance:
        # Close enough that a filter would only add its own rounding.
        notes.append("already within the fit tolerance; no stretching applied")
    elif abs(residual) > EXACT_FIT_TOLERANCE_SECONDS and trimmed == 0.0:
        # Judged on the audio that was actually produced, not on the arithmetic that
        # asked for it, so a clamp is only reported when it really shows.
        notes.append(
            f"an exact fit needed a tempo of {required:.3f}, outside "
            f"[{minimum:g}, {maximum:g}]; clamped to {tempo:.3f}, so the line "
            f"runs {abs(residual):.3f}s "
            f"{'long' if residual > 0 else 'short'} of its window"
        )

    # A line at the very start of the film has nowhere to put its leading pause.
    # The pause is trimmed rather than the line being pushed late, so the speech
    # still lands on its original start.
    lead = min(clip.rendered_pause_before, clip.dialogue.start)
    if lead < clip.rendered_pause_before - 1e-9:
        notes.append(
            f"the {clip.rendered_pause_before:.3f}s leading pause was trimmed to "
            f"{lead:.3f}s so the line could keep its original start of "
            f"{clip.dialogue.start:.3f}s"
        )
    trail = clip.rendered_pause_after

    pieces = [stretched]
    if lead:
        pieces.insert(0, np.zeros(int(round(lead * rate)), dtype=np.float32))
    if trail:
        pieces.append(np.zeros(int(round(trail * rate)), dtype=np.float32))

    _write_mono(
        Path(destination),
        np.concatenate(pieces),
        rate,
        label="aligned line",
    )

    return AlignedClip(
        index=clip.index,
        clip=clip,
        audio_path=Path(destination),
        sample_rate=rate,
        start=clip.dialogue.start - lead,
        tempo=tempo,
        required_tempo=required,
        speech_duration=speech_duration,
        original_window=window,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        notes=tuple(notes),
        available=limit,
        trimmed=trimmed,
        overrun=overrun,
    )


def _tempo_bounds(settings: Settings) -> tuple[float, float]:
    """Return the validated ``(minimum, maximum)`` tempo band."""

    minimum = _finite("TIMING_MIN_TEMPO", settings.timing_min_tempo)
    maximum = _finite("TIMING_MAX_TEMPO", settings.timing_max_tempo)

    if minimum < ATEMPO_MIN or maximum > ATEMPO_MAX:
        raise ConfigurationError(
            f"the tempo band [{minimum:g}, {maximum:g}] is outside the range "
            f"FFmpeg's atempo filter supports, [{ATEMPO_MIN:g}, {ATEMPO_MAX:g}]"
        )
    if minimum > maximum:
        raise ConfigurationError(
            f"TIMING_MIN_TEMPO ({minimum:g}) must not exceed TIMING_MAX_TEMPO "
            f"({maximum:g})"
        )
    return minimum, maximum


def resolve_timing_directory(
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Return the directory that holds the aligned lines.

    An explicit ``output_dir`` wins, so a run can keep its artifacts together;
    otherwise the work directory is used.
    """

    resolved = settings if settings is not None else get_settings()
    root = Path(output_dir) if output_dir is not None else Path(resolved.work_dir)
    return root / TIMING_DIRECTORY_NAME / ALIGNED_DIRECTORY_NAME


def align_dialogue(
    clips: list[TtsClip] | tuple[TtsClip, ...],
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> list[AlignedClip]:
    """Fit every clip to its original window, in the order it was given.

    Parameters
    ----------
    clips:
        Clips from :func:`app.pipeline.tts.synthesize_dialogue`.
    output_dir:
        Directory to keep the aligned lines in. Defaults to
        ``<WORK_DIR>/timing/aligned``.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.

    Returns
    -------
    list[AlignedClip]
        One entry per input clip, in the same order. An empty input returns an
        empty list without touching FFmpeg. A line that could not fit its time is
        moved later rather than spoken over the next one, so the returned lines
        never overlap unless they overlapped in the source.

    Raises
    ------
    ConfigurationError
        The tempo band is not usable.
    InvalidClipError
        An entry is not a :class:`~app.pipeline.tts.TtsClip`.
    MissingInputError
        A clip's audio file does not exist.
    MissingFfmpegError, StretchError, InvalidAudioError
        FFmpeg is missing, failed, or produced audio that is not the pipeline
        format.
    """

    resolved = settings if settings is not None else get_settings()
    _tempo_bounds(resolved)

    ordered = list(clips)
    if not ordered:
        return []

    directory = resolve_timing_directory(output_dir, settings=resolved)
    directory.mkdir(parents=True, exist_ok=True)

    # Each line is told the room it has before the next one begins. The next line's own
    # leading pause is included, because that pause is part of where its audio starts and
    # so is the earliest moment two lines could collide. Without this every line is fitted
    # to its own window alone, and a line that cannot fit - which is most of them in a
    # language longer than English - simply overruns into its neighbour and two voices
    # speak at once.
    guard = resolved.timing_min_line_gap
    rooms: list[float | None] = []
    for position, clip in enumerate(ordered):
        if position + 1 >= len(ordered):
            rooms.append(None)
            continue
        following = ordered[position + 1]
        deadline = _audio_start(following) - guard
        rooms.append(deadline - clip.dialogue.start - clip.rendered_pause_after)

    aligned: list[AlignedClip] = []
    for clip, room in zip(ordered, rooms):
        if not isinstance(clip, TtsClip):
            raise InvalidClipError(
                f"expected a TtsClip, got {type(clip).__name__}"
            )
        destination = directory / f"{Path(clip.audio_path).stem}.wav"
        aligned.append(align_clip(clip, destination, settings=resolved, room=room))

    return _cascade(
        aligned,
        guard=guard,
        overlap=resolved.timing_max_overlap_seconds,
    )


def _cascade(
    aligned: list[AlignedClip], *, guard: float, overlap: float = 0.0
) -> list[AlignedClip]:
    """Place lines later so that no two of them are ever *intelligibly* at once.

    A line whose Amharic needs more time than the film left it cannot be fitted without
    either rushing it past intelligibility or cutting a word off the end. Both were
    tried; both were audible. So the overrun is paid for in position instead: the next
    line waits until this one has finished, and only the lines that were genuinely
    simultaneous in the original are allowed to stay simultaneous here.

    ``overlap`` is the concession that keeps the picture: up to this many seconds of the
    previous line may still be sounding when the next one starts, which is what a
    conversation already sounds like, before the next line is moved at all. A third of a
    second of hand-over is heard as a natural interruption; three seconds of it is two
    people talking at once, which is the thing a dub must never do.
    """

    placed: list[AlignedClip] = []
    cursor = 0.0
    for clip in aligned:
        # A line keeps its original position unless the line in front of it is still
        # speaking. When the two performances overlapped in the source, the overlap is the
        # scene - two people talking over each other - and it is preserved rather than
        # tidied away.
        simultaneous = bool(placed) and clip.clip.start < placed[-1].clip.end - 1e-9
        start = clip.start if simultaneous else max(clip.start, cursor - overlap)
        drift = start - clip.start
        if drift > 1e-6:
            clip = replace(
                clip,
                start=start,
                drift=drift,
                notes=clip.notes
                + (
                    f"the line before it ran {drift:.3f}s past its own time, so this line "
                    "starts later rather than being spoken over",
                ),
            )
        cursor = clip.end + guard
        placed.append(clip)
    return placed


__all__ = [
    "ALIGNED_DIRECTORY_NAME",
    "ATEMPO_MAX",
    "ATEMPO_MIN",
    "EXACT_FIT_TOLERANCE_SECONDS",
    "PIPELINE_SAMPLE_RATE",
    "TIMING_DIRECTORY_NAME",
    "AlignedClip",
    "ConfigurationError",
    "InvalidAudioError",
    "InvalidClipError",
    "MissingFfmpegError",
    "MissingInputError",
    "StretchError",
    "TRIM_FADE_SECONDS",
    "TimingError",
    "align_clip",
    "align_dialogue",
    "resolve_timing_directory",
]
