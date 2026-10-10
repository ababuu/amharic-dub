"""Cinematic audio source separation with **BandIt v2 Multi**.

This module is the only place in the pipeline that talks to BandIt. It splits a
movie's soundtrack into three stems - ``speech``, ``music`` and ``effects`` -
and writes each one as a PCM WAV. The ``speech`` stem feeds diarization and
transcription; ``music`` and ``effects`` are re-mixed untouched at the end.

Contract with the ``bandit_infer`` package
------------------------------------------
* ``BanditSession("v2-multi", device=..., weights_dir=...)`` then ``load()``.
* ``session.infer(audio, sample_rate=48000)`` where ``audio`` is a float32 NumPy
  array shaped ``(channels, samples)``.
* The result maps ``"speech"``, ``"music"`` and ``"effects"`` to arrays in the
  same ``(channels, samples)`` layout.

Deliberate non-goals
--------------------
* **No resampling.** v2 Multi is a 48 kHz model, so any other rate is rejected.
* **No manual chunking.** BandIt's v2 runtime already runs its own chunked,
  overlap-add inference handler.
* **No CUDA->CPU fallback.** The device comes from the project settings, and an
  explicit ``cuda`` request that is unavailable must fail loudly.
* **No weight downloads at import time** (and none in the test suite).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings

try:  # ``bandit_infer`` pulls in torch/CUDA; it is a runtime-only dependency.
    from bandit_infer import BanditSession
except ImportError:  # pragma: no cover - keeps the module importable for tests
    BanditSession = None  # type: ignore[assignment]


#: BandIt v2 Multi - the cinematic separation model verified by the package.
MODEL_NAME = "v2-multi"

#: v2 Multi is trained for 48 kHz audio; BandIt rejects every other rate.
REQUIRED_SAMPLE_RATE = 48_000

#: Stem names returned by the v2 Multi model, in output order.
STEM_NAMES: tuple[str, ...] = ("speech", "music", "effects")

#: Environment variable shared with ``bandit_infer`` to relocate its weight
#: cache. The GPU worker points it at a persistent path (e.g. ``/models/bandit``).
WEIGHTS_DIR_ENV_VAR = "BANDIT_INFER_WEIGHTS"


class SeparationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(SeparationError):
    """The requested input audio file does not exist."""


class UnsupportedSampleRateError(SeparationError):
    """The input is not the 48 kHz audio BandIt v2 Multi requires."""


class InvalidAudioError(SeparationError):
    """The input, or a produced stem, could not be used as audio."""


class SeparationInferenceError(SeparationError):
    """BandIt failed to load or run during separation."""


class MissingStemError(SeparationError):
    """BandIt returned a result without one of the expected stems."""


@dataclass(frozen=True)
class StemPaths:
    """Filesystem paths of the three stems produced by :func:`separate_stems`."""

    speech: Path
    music: Path
    effects: Path

    def as_dict(self) -> dict[str, Path]:
        """Return the stems as a ``{stem_name: path}`` mapping."""

        return {"speech": self.speech, "music": self.music, "effects": self.effects}


def resolve_weights_dir(settings: Settings | None = None) -> Path:
    """Return the BandIt weight-cache directory.

    The existing ``BANDIT_INFER_WEIGHTS`` configuration wins when it is set (the
    GPU worker points it at a persistent model directory); otherwise the
    project's ``MODEL_CACHE_DIR`` is used so weights stay inside the model cache.
    """

    settings = settings if settings is not None else get_settings()

    configured = os.environ.get(WEIGHTS_DIR_ENV_VAR)
    if configured and configured.strip():
        return Path(configured.strip()).expanduser()

    return Path(settings.model_cache_dir) / "bandit"


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file into a float32 ``(channels, samples)`` array.

    The sample rate is returned untouched; deciding whether it is acceptable is
    the caller's responsibility, so no implicit resampling ever happens here.
    """

    audio_path = Path(path)
    if not audio_path.is_file():
        raise MissingInputError(f"input audio file not found: {audio_path}")

    try:
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:  # soundfile errors subclass RuntimeError
        raise InvalidAudioError(f"could not read audio file {audio_path}: {exc}") from exc

    audio = np.ascontiguousarray(np.asarray(data, dtype=np.float32).T)
    if audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] == 0:
        raise InvalidAudioError(f"audio file {audio_path} contains no usable samples")

    return audio, int(sample_rate)


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise UnsupportedSampleRateError(
            f"BandIt {MODEL_NAME} requires {REQUIRED_SAMPLE_RATE} Hz audio, got "
            f"{sample_rate} Hz; resample the input before calling separate_stems()"
        )


def _run_bandit(
    audio: np.ndarray, settings: Settings, weights_dir: Path
) -> Mapping[str, np.ndarray]:
    """Load a BandIt session, run inference, and always close the session."""

    if BanditSession is None:  # pragma: no cover - only when the dep is missing
        raise SeparationInferenceError(
            "bandit-infer is not installed; install the runtime dependencies "
            "(torch, torchaudio, bandit-infer) before running separation"
        )

    try:
        session = BanditSession(MODEL_NAME, device=settings.device, weights_dir=weights_dir)
    except Exception as exc:
        raise SeparationInferenceError(
            f"could not create BandIt session (model={MODEL_NAME!r}, "
            f"device={settings.device!r}): {exc}"
        ) from exc

    try:
        session.load()
        stems = session.infer(audio, sample_rate=REQUIRED_SAMPLE_RATE)
    except Exception as exc:
        raise SeparationInferenceError(f"BandIt inference failed: {exc}") from exc
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    if not isinstance(stems, Mapping):
        raise MissingStemError(
            f"BandIt returned {type(stems).__name__}, expected a mapping of stems"
        )

    return stems


def _collect_stems(stems: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [name for name in STEM_NAMES if name not in stems]
    if missing:
        raise MissingStemError("BandIt result is missing expected stem(s): " + ", ".join(missing))

    collected: dict[str, np.ndarray] = {}
    for name in STEM_NAMES:
        array = np.asarray(stems[name], dtype=np.float32)
        if array.size == 0:
            raise InvalidAudioError(f"BandIt returned an empty '{name}' stem")
        collected[name] = array
    return collected


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim == 1:
        data = audio
    elif audio.ndim == 2:
        data = audio.T  # (channels, samples) -> (samples, channels)
    else:
        raise InvalidAudioError(f"cannot write a {audio.ndim}-dimensional stem array")

    try:
        sf.write(str(path), data, sample_rate, format="WAV", subtype="PCM_16")
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not write stem {path}: {exc}") from exc


def separate_stems(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> StemPaths:
    """Split ``input_path`` into speech/music/effects stems with BandIt v2 Multi.

    Parameters
    ----------
    input_path:
        A 48 kHz WAV/audio file. It is never resampled; other rates are rejected.
    output_dir:
        Directory for the produced ``<name>_speech.wav`` etc. Defaults to the
        project ``WORK_DIR``. Created when missing.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.

    Returns
    -------
    StemPaths
        Paths of the three written PCM-16 WAV stems at the original 48 kHz rate.
    """

    settings = settings if settings is not None else get_settings()
    source = Path(input_path)

    audio, sample_rate = read_audio(source)
    _validate_sample_rate(sample_rate)

    weights_dir = resolve_weights_dir(settings)
    stems = _run_bandit(audio, settings, weights_dir)
    collected = _collect_stems(stems)

    destination = Path(output_dir) if output_dir is not None else Path(settings.work_dir)
    destination.mkdir(parents=True, exist_ok=True)

    prefix = source.stem
    written: dict[str, Path] = {}
    for name in STEM_NAMES:
        stem_path = destination / f"{prefix}_{name}.wav"
        _write_stem(stem_path, collected[name], sample_rate)
        written[name] = stem_path

    return StemPaths(
        speech=written["speech"],
        music=written["music"],
        effects=written["effects"],
    )


#: How far below the original performance the bed has to sit for the English to be
#: inaudible under the dub. Speech stays intelligible about 15 dB under competing
#: sound, so a bed carrying the original that close to the mix is a leak, not a
#: level choice.
AUDIBLE_BLEED_DB = -15.0


@dataclass(frozen=True, slots=True)
class BleedReport:
    """How much of the film's **original dialogue** survived into the bed.

    The music and effects stems are re-mixed untouched, so anything the separator
    failed to take out of them is played under the Amharic. That is the one way the
    source language can still be heard in a finished dub, and it is worth measuring
    rather than assuming: a separator that works on ten minutes of dialogue can still
    misclassify a whispered line at minute 95 of a feature.

    ``leaked`` counts the windows where the bed is both *level with* and *correlated
    with* the isolated speech - two independent conditions, because loud music that
    happens to sit under a line is not a leak.
    """

    windows: int
    leaked: int
    worst_seconds: float
    worst_index: int | None
    worst_relative_db: float
    worst_correlation: float

    @property
    def clean(self) -> bool:
        """``True`` when no window carries the original dialogue audibly."""

        return self.leaked == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the measurement."""

        return {
            "windows": self.windows,
            "leaked": self.leaked,
            "worst_seconds": round(self.worst_seconds, 3),
            "worst_index": self.worst_index,
            "worst_relative_db": round(self.worst_relative_db, 2),
            "worst_correlation": round(self.worst_correlation, 4),
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if not self.windows:
            return "original-dialogue bleed not measured"
        if self.clean:
            return (
                f"no original dialogue left in the bed over {self.windows} line "
                f"window(s); worst {self.worst_relative_db:.1f} dB below the speech"
            )
        return (
            f"WARNING: the bed still carries the original dialogue in {self.leaked} "
            f"of {self.windows} window(s); worst {self.worst_relative_db:.1f} dB "
            "below the speech at {:.2f}s".format(self.worst_seconds)
        )


def _open_stem(path: Path, *, what: str) -> sf.SoundFile:
    """Open a stem for windowed reading, or raise the module's own error."""

    try:
        return sf.SoundFile(str(path))
    except (OSError, RuntimeError) as exc:  # soundfile errors subclass RuntimeError
        if not Path(path).is_file():
            raise MissingInputError(f"{what} not found: {path}") from exc
        raise InvalidAudioError(f"could not open {what} {path}: {exc}") from exc


def measure_bed_bleed(
    stems: StemPaths,
    windows: Iterable[tuple[float, float]],
    *,
    threshold_db: float = AUDIBLE_BLEED_DB,
) -> BleedReport:
    """Measure how much original dialogue is left in the music and effects stems.

    Parameters
    ----------
    stems:
        The three stems one run produced.
    windows:
        ``(start, end)`` of every moment the film has dialogue. The speech stem is
        what the separator thinks the dialogue is; these windows say when it happened.
    threshold_db:
        How close to the isolated speech the bed may come before the bleed is called
        audible.

    Returns
    -------
    BleedReport

    Notes
    -----
    Each window is read straight out of the files, so memory stays flat: three
    two-hour 48 kHz stems are about 8 GB of float32 in total, and an A40 doing the
    rest of the pipeline does not have that to spare. Nothing is ever loaded whole.

    Both stems are summed to one channel: the question is whether the *performance*
    is present, and a mono sum answers it without a channel-wise alignment
    assumption.
    """

    windows = list(windows)
    speech_file = _open_stem(Path(stems.speech), what="speech stem")
    with speech_file, _open_stem(Path(stems.music), what="music stem") as music_file, (
        _open_stem(Path(stems.effects), what="effects stem")
    ) as effects_file:
        rate = int(speech_file.samplerate)
        for other, name in ((music_file, "music"), (effects_file, "effects")):
            if int(other.samplerate) != rate:
                raise UnsupportedSampleRateError(
                    f"the {name} stem is {other.samplerate} Hz but the speech stem is "
                    f"{rate} Hz; the stems of one run must share a rate"
                )

        frames = min(len(speech_file), len(music_file), len(effects_file))
        floor = max(1, int(round(0.02 * rate)))

        leaked = 0
        counted = 0
        worst_index: int | None = None
        worst_relative = -math.inf
        worst_correlation = 0.0
        worst_seconds = 0.0

        for index, (start, end) in enumerate(windows):
            first = max(0, int(round(start * rate)))
            last = min(frames, int(round(end * rate)))
            if last - first < floor:
                continue
            counted += 1

            try:
                reference = _read_window(speech_file, first, last - first)
                candidate = _read_window(music_file, first, last - first) + _read_window(
                    effects_file, first, last - first
                )
            except (OSError, RuntimeError) as exc:
                raise InvalidAudioError(
                    f"could not read the stems at {start:.3f}s-{end:.3f}s: {exc}"
                ) from exc

            reference_rms = float(np.sqrt(np.mean(reference**2)))
            bed_rms = float(np.sqrt(np.mean(candidate**2)))
            if reference_rms <= 0.0 or bed_rms <= 0.0:
                continue
            relative_db = 20.0 * math.log10(bed_rms / reference_rms)

            if reference.std() > 0.0 and candidate.std() > 0.0:
                correlation = float(np.corrcoef(candidate, reference)[0, 1])
            else:
                correlation = 0.0

            if relative_db >= threshold_db and correlation >= 0.5:
                leaked += 1
            if relative_db > worst_relative:
                worst_relative = relative_db
                worst_correlation = correlation
                worst_seconds = start
                worst_index = index

    if not math.isfinite(worst_relative):
        worst_relative = -math.inf

    return BleedReport(
        windows=counted,
        leaked=leaked,
        worst_seconds=worst_seconds,
        worst_index=worst_index,
        worst_relative_db=worst_relative,
        worst_correlation=worst_correlation,
    )


def _read_window(handle: sf.SoundFile, first: int, frames: int) -> np.ndarray:
    """Read ``frames`` samples starting at ``first`` and return them as one channel.

    ``frames`` is always within every stem - the caller clamps to the shortest - so a
    short read is not a case that arises.
    """

    handle.seek(first)
    block = handle.read(frames, dtype="float32", always_2d=True)
    return np.asarray(block, dtype=np.float32).mean(axis=1)


__all__ = [
    "AUDIBLE_BLEED_DB",
    "MODEL_NAME",
    "REQUIRED_SAMPLE_RATE",
    "STEM_NAMES",
    "WEIGHTS_DIR_ENV_VAR",
    "BleedReport",
    "SeparationError",
    "MissingInputError",
    "UnsupportedSampleRateError",
    "InvalidAudioError",
    "SeparationInferenceError",
    "MissingStemError",
    "StemPaths",
    "measure_bed_bleed",
    "read_audio",
    "resolve_weights_dir",
    "separate_stems",
]
