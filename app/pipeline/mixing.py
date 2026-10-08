"""Final audio re-mixing.

Rebuilds the film's soundtrack as the dubbed Amharic dialogue on top of the
**music and effects stems only**:

* every aligned line is placed at its original timestamp, so the dub stays in
  sync, and overlaps are summed and reported rather than silently resolved;
* the music and effects are **ducked** under the dialogue with a smoothed, gentle
  reduction - not a hard cut - so the bed breathes with the dialogue;
* the sum is kept under a peak ceiling with headroom for lossy encoding, and the
  reduction that was needed is reported.

The original English dialogue is never used: it was removed from the bed by
separation, and the dubbed dialogue replaces it. Mixing the full original mix
back in would reintroduce the language the pipeline exists to replace.

Output is two WAV files at the pipeline sample rate: the dialogue stem on its
own, which is what makes a mis-levelled dub diagnosable, and the final mix, which
is what :func:`app.pipeline.video.mux_dub` puts back into the video.

Deliberately not done here
--------------------------
* **No loudness normalization (EBU R128).** The mix is placed at a defined peak
  with defined dialogue and bed levels, which is reproducible and auditable; a
  programme-loudness pass is a delivery decision that belongs with the encode of
  the final file, not with the mix that has to be inspectable afterwards.
* **No compression, EQ or de-essing.** Nothing here second-guesses the engines.
* **No resampling of the bed.** The stems are already the pipeline rate; a stem
  at another rate is rejected rather than quietly converted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from app.config import Settings, get_settings
from app.pipeline.timing import PIPELINE_SAMPLE_RATE as _TIMING_SAMPLE_RATE
from app.pipeline.timing import AlignedClip
from app.pipeline.voice_profiles import portable_path

#: The mix, like every stage after separation, runs at BandIt's 48 kHz. The value
#: is taken from the timing stage rather than redeclared so the two cannot drift;
#: ``tests/test_mixing.py`` checks it against the other stages' declarations.
PIPELINE_SAMPLE_RATE = _TIMING_SAMPLE_RATE

#: Stereo, matching the source stems and the deliverable.
CHANNELS = 2

#: Directory (under the work directory) holding the mix artifacts.
MIX_DIRECTORY_NAME = "mix"

#: The continuous dubbed dialogue stem, before the bed is added.
DIALOGUE_STEM_FILENAME = "dialogue_amharic.wav"

#: The final Amharic mix, ready to be muxed into the video.
MIXED_FILENAME = "mix_amharic.wav"

#: Dialogue counts as present above this envelope level (about -34 dBFS), which is
#: below quiet speech and above a typical noise floor in a music stem.
DUCK_THRESHOLD = 0.02

#: The duck reduction is ramped over this long, so it cannot click. Using a
#: symmetric ramp also means the bed is already down when a line starts.
DUCK_FADE_SECONDS = 0.05

#: The final mix is kept at or below this peak (about -1 dBFS), leaving the
#: headroom a lossy encoder needs. Exceeding it scales the whole mix, bed
#: included, so the balance between dialogue and bed never changes.
PEAK_CEILING = 10.0 ** (-1.0 / 20.0)


class MixingError(RuntimeError):
    """Base class for every error raised by this module."""


class ConfigurationError(MixingError, ValueError):
    """A mix level is not usable."""


class InvalidClipError(MixingError, ValueError):
    """An entry is not an :class:`~app.pipeline.timing.AlignedClip`."""


class MissingInputError(MixingError):
    """A stem or an aligned line does not exist."""


class InvalidAudioError(MixingError):
    """A stem or an aligned line is not the pipeline format."""


class WriteError(MixingError):
    """The mix could not be written."""


@dataclass(frozen=True, slots=True)
class Overlap:
    """Two placed lines that play at the same time.

    Overlaps are kept, because the original performances overlapped too and
    shifting either line would break its sync. They are reported so a mix that
    sounds crowded can be traced to specific lines.
    """

    first_index: int
    second_index: int
    first_speaker: str
    second_speaker: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "first_speaker": self.first_speaker,
            "second_speaker": self.second_speaker,
            "seconds": self.seconds,
        }


@dataclass(frozen=True, slots=True)
class MixResult:
    """The rebuilt soundtrack: the dialogue stem and the final mix."""

    dialogue_path: Path
    mixed_path: Path
    sample_rate: int
    channels: int
    duration: float
    dialogue_gain_db: float
    duck_db: float
    peak: float
    peak_gain_db: float
    overlaps: tuple[Overlap, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def peak_gain_reduction(self) -> bool:
        """``True`` when the mix had to be scaled down to respect the ceiling."""

        return self.peak_gain_db < 0.0

    @property
    def fits(self) -> bool:
        """``True`` when every placed line fitted the soundtrack."""

        return not any("dropped" in note or "trimmed" in note for note in self.notes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the mix."""

        return {
            "dialogue_path": portable_path(self.dialogue_path),
            "mixed_path": portable_path(self.mixed_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": self.duration,
            "dialogue_gain_db": self.dialogue_gain_db,
            "duck_db": self.duck_db,
            "peak": self.peak,
            "peak_gain_db": self.peak_gain_db,
            "peak_gain_reduction": self.peak_gain_reduction,
            "overlaps": [overlap.to_dict() for overlap in self.overlaps],
            "notes": list(self.notes),
        }


def db_to_gain(decibels: float) -> float:
    """Return the linear amplitude ratio for ``decibels``."""

    return float(10.0 ** (decibels / 20.0))


def resolve_mix_directory(
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Return the directory that holds the mix artifacts."""

    resolved = settings if settings is not None else get_settings()
    root = Path(output_dir) if output_dir is not None else Path(resolved.work_dir)
    return root / MIX_DIRECTORY_NAME


def _validate_levels(settings: Settings) -> tuple[float, float]:
    """Return the validated ``(dialogue_gain_db, duck_db)``."""

    gain = settings.mix_dialogue_gain_db
    duck = settings.mix_duck_db
    for name, value in (("MIX_DIALOGUE_GAIN_DB", gain), ("MIX_DUCK_DB", duck)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{name} must be a number, got {value!r}")
        if not np.isfinite(float(value)):
            raise ConfigurationError(f"{name} must be a finite number, got {value!r}")

    if duck < 0:
        raise ConfigurationError(
            f"MIX_DUCK_DB is how far to duck the music and effects, so it cannot be "
            f"negative; got {duck}"
        )
    if duck > 60:
        raise ConfigurationError(
            f"MIX_DUCK_DB of {duck} would silence the music and effects entirely; "
            "the duck is meant to be gentle"
        )
    return float(gain), float(duck)


def _read_stereo(path: Path, *, label: str) -> tuple[np.ndarray, int]:
    """Read an audio file into a float32 ``(samples, channels)`` array."""

    if not path.is_file():
        raise MissingInputError(f"{label} was not found: {path}")

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {label} at {path}: {exc}") from exc

    if data.shape[0] == 0 or data.shape[1] == 0:
        raise InvalidAudioError(f"{label} at {path} contains no samples")
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def _as_channels(
    data: np.ndarray, *, label: str, notes: list[str]
) -> np.ndarray:
    """Return ``data`` as ``CHANNELS`` wide, noting any conversion."""

    if data.shape[1] == CHANNELS:
        return data
    if data.shape[1] == 1:
        notes.append(f"{label} is mono and was placed in both channels")
        return np.repeat(data, CHANNELS, axis=1)
    if data.shape[1] > CHANNELS:
        notes.append(
            f"{label} has {data.shape[1]} channels; the first {CHANNELS} were kept"
        )
        return data[:, :CHANNELS]
    raise InvalidAudioError(f"{label} has no usable channels")


def _check_rate(sample_rate: int, *, label: str) -> None:
    if sample_rate != PIPELINE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"{label} is {sample_rate} Hz, not the {PIPELINE_SAMPLE_RATE} Hz the "
            "pipeline works at; it must be resampled before mixing"
        )


def _duck_gain(
    dialogue: np.ndarray, *, duck_db: float, sample_rate: int
) -> np.ndarray:
    """Return the per-sample gain to apply to the bed, shaped ``(samples, 1)``.

    The dialogue's envelope is thresholded, then ramped, so the bed is reduced
    while a line plays and returns to full level in the gaps. A symmetric ramp is
    used: it cannot click, and it starts the reduction just before the line does,
    which keeps the line's first syllable from being masked by the bed.
    """

    if duck_db <= 0.0:
        return np.ones((dialogue.shape[0], 1), dtype=np.float32)

    depth = 1.0 - db_to_gain(-duck_db)
    envelope = np.abs(dialogue).max(axis=1)
    active = (envelope > DUCK_THRESHOLD).astype(np.float32)

    window = max(1, int(round(DUCK_FADE_SECONDS * sample_rate)))
    ramped = uniform_filter1d(active, size=window, mode="nearest")

    return (1.0 - depth * ramped).astype(np.float32).reshape(-1, 1)


def _find_overlaps(placed: list[tuple[int, str, float, float]]) -> tuple[Overlap, ...]:
    """Return every pair of placed lines that play at the same time."""

    ordered = sorted(placed, key=lambda entry: entry[2])
    found: list[Overlap] = []
    for position, first in enumerate(ordered):
        for second in ordered[position + 1 :]:
            if second[2] >= first[3]:
                # Sorted by start: once a line begins after this one ends, so do
                # all the rest.
                break
            found.append(
                Overlap(
                    first_index=first[0],
                    second_index=second[0],
                    first_speaker=first[1],
                    second_speaker=second[1],
                    seconds=float(first[3] - second[2]),
                )
            )
    return tuple(found)


def mix_track(
    aligned: list[AlignedClip] | tuple[AlignedClip, ...],
    music_stem: str | Path,
    effects_stem: str | Path,
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> MixResult:
    """Place the dubbed dialogue over the music and effects stems.

    Parameters
    ----------
    aligned:
        Lines from :func:`app.pipeline.timing.align_dialogue`.
    music_stem, effects_stem:
        The ``music`` and ``effects`` stems from
        :func:`app.pipeline.separation.separate_stems`. The speech stem is
        deliberately not accepted: the dub replaces it.
    output_dir:
        Directory to keep the mix artifacts in. Defaults to ``<WORK_DIR>/mix``.
    settings:
        Project settings override; the dialogue level and duck depth come from
        here.

    Returns
    -------
    MixResult
        With the dialogue stem, the final mix, and what had to be adjusted.

    Raises
    ------
    ConfigurationError
        A level is not usable.
    InvalidClipError
        An entry is not an :class:`~app.pipeline.timing.AlignedClip`.
    MissingInputError
        A stem or an aligned line does not exist.
    InvalidAudioError
        A stem is not the pipeline format, or a line is not the pipeline rate.
    WriteError
        An output file could not be written.
    """

    resolved = settings if settings is not None else get_settings()
    dialogue_gain_db, duck_db = _validate_levels(resolved)

    lines = list(aligned)
    for line in lines:
        if not isinstance(line, AlignedClip):
            raise InvalidClipError(
                f"expected an AlignedClip, got {type(line).__name__}"
            )

    notes: list[str] = []
    music, music_rate = _read_stereo(Path(music_stem), label="the music stem")
    effects, effects_rate = _read_stereo(Path(effects_stem), label="the effects stem")
    _check_rate(music_rate, label="the music stem")
    _check_rate(effects_rate, label="the effects stem")

    music = _as_channels(music, label="the music stem", notes=notes)
    effects = _as_channels(effects, label="the effects stem", notes=notes)

    if music.shape[0] != effects.shape[0]:
        # Separation writes both stems at the source length; a mismatch means one
        # was edited, so the longer of the two defines the film and the shorter is
        # padded rather than the mix being truncated to the shorter one.
        frames = max(music.shape[0], effects.shape[0])
        notes.append(
            f"the music and effects stems differ in length "
            f"({music.shape[0]} vs {effects.shape[0]} samples); the shorter one was "
            f"padded with silence to {frames} samples"
        )
        music = np.pad(music, ((0, frames - music.shape[0]), (0, 0)))
        effects = np.pad(effects, ((0, frames - effects.shape[0]), (0, 0)))

    total_frames = music.shape[0]
    duration = total_frames / PIPELINE_SAMPLE_RATE

    dialogue = np.zeros((total_frames, CHANNELS), dtype=np.float32)
    gain = db_to_gain(dialogue_gain_db)
    placed: list[tuple[int, str, float, float]] = []

    for line in sorted(lines, key=lambda entry: (entry.start, entry.index)):
        samples, sample_rate = _read_stereo(
            Path(line.audio_path), label=f"the aligned line {line.index}"
        )
        _check_rate(sample_rate, label=f"the aligned line {line.index}")

        mono = samples.mean(axis=1, dtype=np.float32) * gain
        offset = int(round(line.start * PIPELINE_SAMPLE_RATE))
        if offset >= total_frames:
            notes.append(
                f"line {line.index} starts at {line.start:.3f}s, after the end of "
                f"the soundtrack ({duration:.3f}s); it was dropped"
            )
            continue

        available = total_frames - offset
        if mono.shape[0] > available:
            notes.append(
                f"line {line.index} ran {(mono.shape[0] - available) / PIPELINE_SAMPLE_RATE:.3f}s "
                "past the end of the soundtrack; it was trimmed"
            )
            mono = mono[:available]

        stereo = np.repeat(mono[:, None], CHANNELS, axis=1)
        dialogue[offset : offset + mono.shape[0]] += stereo
        placed.append(
            (
                line.index,
                line.speaker_id,
                # Both in seconds: the offset is in samples, so the end of the
                # placed audio is derived from the line's start instead.
                line.start,
                line.start + mono.shape[0] / PIPELINE_SAMPLE_RATE,
            )
        )

    overlaps = _find_overlaps(placed)

    bed = music + effects
    bed *= _duck_gain(dialogue, duck_db=duck_db, sample_rate=PIPELINE_SAMPLE_RATE)
    mixed = dialogue + bed

    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    peak_gain_db = 0.0
    original_peak = peak
    if peak > PEAK_CEILING:
        reduction = PEAK_CEILING / peak
        mixed *= reduction
        peak = float(np.abs(mixed).max())
        peak_gain_db = float(20.0 * np.log10(reduction))
        notes.append(
            f"the mix peaked at {20.0 * np.log10(original_peak):+.2f} dBFS and was "
            f"scaled by {peak_gain_db:+.2f} dB to stay under the ceiling"
        )

    directory = resolve_mix_directory(output_dir, settings=resolved)
    directory.mkdir(parents=True, exist_ok=True)
    dialogue_path = directory / DIALOGUE_STEM_FILENAME
    mixed_path = directory / MIXED_FILENAME
    _write_stereo(dialogue_path, dialogue, label="the dialogue stem")
    _write_stereo(mixed_path, mixed, label="the final mix")

    return MixResult(
        dialogue_path=dialogue_path,
        mixed_path=mixed_path,
        sample_rate=PIPELINE_SAMPLE_RATE,
        channels=CHANNELS,
        duration=duration,
        dialogue_gain_db=dialogue_gain_db,
        duck_db=duck_db,
        peak=peak,
        peak_gain_db=peak_gain_db,
        overlaps=overlaps,
        notes=tuple(notes),
    )


def _write_stereo(path: Path, samples: np.ndarray, *, label: str) -> Path:
    """Write ``samples`` as PCM-16 WAV at the pipeline rate and return the path."""

    try:
        sf.write(
            str(path),
            np.asarray(samples, dtype=np.float32),
            PIPELINE_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
    except (OSError, RuntimeError) as exc:
        raise WriteError(f"could not write {label} to {path}: {exc}") from exc
    return path


__all__ = [
    "CHANNELS",
    "DIALOGUE_STEM_FILENAME",
    "DUCK_FADE_SECONDS",
    "DUCK_THRESHOLD",
    "MIXED_FILENAME",
    "MIX_DIRECTORY_NAME",
    "PEAK_CEILING",
    "PIPELINE_SAMPLE_RATE",
    "ConfigurationError",
    "InvalidAudioError",
    "InvalidClipError",
    "MixResult",
    "MissingInputError",
    "MixingError",
    "Overlap",
    "WriteError",
    "db_to_gain",
    "mix_track",
    "resolve_mix_directory",
]
