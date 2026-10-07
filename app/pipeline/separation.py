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

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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


__all__ = [
    "MODEL_NAME",
    "REQUIRED_SAMPLE_RATE",
    "STEM_NAMES",
    "WEIGHTS_DIR_ENV_VAR",
    "SeparationError",
    "MissingInputError",
    "UnsupportedSampleRateError",
    "InvalidAudioError",
    "SeparationInferenceError",
    "MissingStemError",
    "StemPaths",
    "read_audio",
    "resolve_weights_dir",
    "separate_stems",
]
