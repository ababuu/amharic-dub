"""Central configuration for the Amharic dubbing pipeline.

All runtime configuration is read from environment variables so that the exact
same code runs locally and on the RunPod GPU worker without any changes.

Design rules
------------
* A local ``.env`` file is loaded when present, but it never overrides variables
  that are already defined in the real environment. This lets the RunPod worker
  environment inject values that win over a developer's local file.
* Secrets (``DEEPSEEK_API_KEY``, ``HUGGINGFACE_TOKEN``) are optional. The
  project must stay runnable without them so that the scaffold, the health
  check, and the tests work before any credentials exist.
* Secrets are *never* hard-coded and are *never* logged or printed.

Environment variables
---------------------
``DEEPSEEK_API_KEY``  API key for the DeepSeek dialogue adaptation/translation.
``HUGGINGFACE_TOKEN`` Token used to download gated Hugging Face models
                      (e.g. pyannote speaker diarization).
``INPUT_DIR``         Directory holding source videos. Defaults to ``data/input``.
``WORK_DIR``          Scratch directory for intermediate artifacts.
``OUTPUT_DIR``        Directory for final dubbed videos and manifests.
``MODEL_CACHE_DIR``   Where model weights are downloaded at runtime.
``DIARIZATION_MODEL`` Hugging Face pipeline id used for speaker diarization.
``TRANSCRIPTION_MODEL``
                      faster-whisper model used for transcription.
``TRANSCRIPTION_COMPUTE_TYPE``
                      CTranslate2 compute type (e.g. ``float16`` on GPU).
``TRANSCRIPTION_LANGUAGE``
                      Source language code; unset means detect automatically.
``TRANSLATION_MODEL`` DeepSeek model used for dialogue adaptation.
``TRANSLATION_BASE_URL``
                      DeepSeek API base URL (OpenAI-compatible endpoint).
``TRANSLATION_BATCH_SIZE``
                      Consecutive dialogue lines adapted in one request.
``TRANSLATION_DISABLE_THINKING``
                      Set ``false`` if the API rejects the thinking toggle.
``VOICE_PROFILE_DIR`` Directory holding per-speaker voice profiles.
``VOICE_REFERENCE_MIN_DURATION`` / ``..._TARGET_DURATION`` / ``..._MAX_DURATION``
                      Preferred length of a voice-cloning reference, in seconds.
``TTS_MODEL``         Amharic speech adapter used by the TTS stage.
``SEED_VC_REPO_PATH`` Seed-VC checkout used for the identity-conversion step.
``SEED_VC_DIFFUSION_STEPS``
                      Diffusion steps of the Seed-VC V2 converter.
``TTS_PERFORMANCE_REFERENCE_MIN_DURATION`` / ``..._MAX_DURATION``
                      Length of the original-performance prompt handed to
                      Chatterbox, in seconds.
``TTS_MAX_PAUSE_SECONDS``
                      Longest pause rendered around a synthesized line.
``DEVICE``            Compute device hint, ``cuda`` by default.
``LOG_LEVEL``         Logging verbosity, ``INFO`` by default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # ``python-dotenv`` is part of the base requirements.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the module importable without it
    load_dotenv = None  # type: ignore[assignment]


#: Repository root (the directory that contains ``app/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "working"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models_cache"

#: pyannote Community-1 speaker diarization pipeline on the Hugging Face Hub.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

#: faster-whisper model used to transcribe the dialogue stem.
DEFAULT_TRANSCRIPTION_MODEL = "large-v3"

#: CTranslate2 compute type. ``float16`` targets the A40 GPU; CPU runs need
#: ``int8`` or ``float32``, because CTranslate2 does not support fp16 on CPU.
DEFAULT_TRANSCRIPTION_COMPUTE_TYPE = "float16"

#: DeepSeek model used for dialogue adaptation into Amharic.
DEFAULT_TRANSLATION_MODEL = "deepseek-flash"

#: DeepSeek API endpoint. The official endpoint is OpenAI-compatible.
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"

#: How many consecutive dialogue lines are adapted in a single API request.
#: Small enough to stay well inside the context window and keep a rejection
#: cheap, large enough for the model to follow who is answering whom.
DEFAULT_TRANSLATION_BATCH_SIZE = 10

#: Directory holding the per-speaker voice profiles consumed by the TTS stage.
DEFAULT_VOICE_PROFILE_DIR = DEFAULT_WORK_DIR / "voices"

#: Preferred length of a voice-cloning reference, in seconds. The target is the
#: window the selector aims for, the minimum rejects snippets too short to clone
#: from, and the maximum caps how much audio a single profile keeps around.
DEFAULT_VOICE_REFERENCE_MIN_DURATION = 3.0
DEFAULT_VOICE_REFERENCE_TARGET_DURATION = 10.0
DEFAULT_VOICE_REFERENCE_MAX_DURATION = 15.0

#: Amharic speech adapter consumed by :mod:`app.pipeline.tts`. It is a LoRA
#: delta plus a Fidel tokenizer that is applied on top of Chatterbox
#: Multilingual v3 at runtime, so only the adapter is named here.
DEFAULT_TTS_MODEL = "gabar-tech/chatterbox-amharic"

#: Name of the Seed-VC checkout inside ``MODEL_CACHE_DIR``. Seed-VC is not
#: published as a package, so the identity-conversion step of
#: :mod:`app.pipeline.tts` runs it from a checkout of its repository.
DEFAULT_SEED_VC_REPO_NAME = "seed-vc"

#: Seed-VC V2 diffusion steps. The V2 inference script's own default is 30;
#: fewer steps trade quality for speed.
DEFAULT_SEED_VC_DIFFUSION_STEPS = 30

#: Preferred length of the original-performance prompt handed to Chatterbox, in
#: seconds. The Amharic adapter clones from roughly ten seconds of audio, so a
#: short line is padded with its own surrounding dialogue and a long monologue
#: is trimmed, both around the line that is being synthesized.
DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION = 6.0
DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION = 12.0

#: Longest pause rendered before and after a synthesized line, in seconds. The
#: dialogue model estimates the pauses of a scene and normally stays below 1.5
#: seconds; this bound keeps a wildly wrong estimate from becoming a hole in the
#: dub.
DEFAULT_TTS_MAX_PAUSE_SECONDS = 2.0


def load_env_file(path: Optional[Path] = None) -> None:
    """Load a ``.env`` file without overriding existing environment variables.

    Missing files and a missing ``python-dotenv`` installation are both ignored
    on purpose: the project has to start even when nothing is configured.
    """

    if load_dotenv is None:
        return

    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a trimmed environment value, falling back to ``default``."""

    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _read_path(name: str, default: Path) -> Path:
    """Return an environment value as a :class:`~pathlib.Path`.

    Relative paths are resolved against the project root so that behaviour does
    not depend on the current working directory.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_int(name: str, default: int) -> int:
    """Return an environment value as a positive integer, or ``default``.

    A malformed value is a configuration mistake and fails loudly rather than
    being silently replaced by the default.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


#: Values accepted (case-insensitively) for a boolean environment flag.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _read_bool(name: str, default: bool) -> bool:
    """Return an environment value as a boolean, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _read_float(name: str, default: float) -> float:
    """Return an environment value as a finite positive float, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved, immutable runtime settings for the dubbing pipeline."""

    input_dir: Path
    work_dir: Path
    output_dir: Path
    model_cache_dir: Path
    deepseek_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    device: str = "cuda"
    log_level: str = "INFO"
    #: Hugging Face pipeline id used by :mod:`app.pipeline.diarization`.
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    #: faster-whisper model and CTranslate2 compute type for transcription.
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    transcription_compute_type: str = DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
    #: Source language code for transcription; ``None`` detects it automatically.
    transcription_language: Optional[str] = None
    #: DeepSeek dialogue adaptation settings for :mod:`app.pipeline.translation`.
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = DEFAULT_TRANSLATION_BASE_URL
    translation_batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE
    translation_disable_thinking: bool = True
    #: Voice-profile settings for :mod:`app.pipeline.voice_profiles`.
    voice_profile_dir: Path = DEFAULT_VOICE_PROFILE_DIR
    voice_reference_min_duration: float = DEFAULT_VOICE_REFERENCE_MIN_DURATION
    voice_reference_target_duration: float = DEFAULT_VOICE_REFERENCE_TARGET_DURATION
    voice_reference_max_duration: float = DEFAULT_VOICE_REFERENCE_MAX_DURATION
    #: TTS settings for :mod:`app.pipeline.tts`: the Amharic speech adapter, the
    #: Seed-VC checkout that converts identity, and the performance-prompt and
    #: pause bounds of one synthesized line.
    tts_model: str = DEFAULT_TTS_MODEL
    seed_vc_repo_path: Path = DEFAULT_MODEL_CACHE_DIR / DEFAULT_SEED_VC_REPO_NAME
    seed_vc_diffusion_steps: int = DEFAULT_SEED_VC_DIFFUSION_STEPS
    tts_performance_reference_min_duration: float = (
        DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION
    )
    tts_performance_reference_max_duration: float = (
        DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION
    )
    tts_max_pause_seconds: float = DEFAULT_TTS_MAX_PAUSE_SECONDS

    # -- construction -------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from the environment."""

        load_env_file()
        work_dir = _read_path("WORK_DIR", DEFAULT_WORK_DIR)
        model_cache_dir = _read_path("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
        return cls(
            input_dir=_read_path("INPUT_DIR", DEFAULT_INPUT_DIR),
            work_dir=work_dir,
            output_dir=_read_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            model_cache_dir=model_cache_dir,
            deepseek_api_key=_read_env("DEEPSEEK_API_KEY"),
            huggingface_token=_read_env("HUGGINGFACE_TOKEN"),
            device=(_read_env("DEVICE", "cuda") or "cuda").lower(),
            log_level=(_read_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            diarization_model=(
                _read_env("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)
                or DEFAULT_DIARIZATION_MODEL
            ),
            transcription_model=(
                _read_env("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
                or DEFAULT_TRANSCRIPTION_MODEL
            ),
            transcription_compute_type=(
                _read_env("TRANSCRIPTION_COMPUTE_TYPE", DEFAULT_TRANSCRIPTION_COMPUTE_TYPE)
                or DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
            ),
            transcription_language=(
                (_read_env("TRANSCRIPTION_LANGUAGE") or "").lower() or None
            ),
            translation_model=(
                _read_env("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL)
                or DEFAULT_TRANSLATION_MODEL
            ),
            translation_base_url=(
                _read_env("TRANSLATION_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)
                or DEFAULT_TRANSLATION_BASE_URL
            ),
            translation_batch_size=_read_int(
                "TRANSLATION_BATCH_SIZE", DEFAULT_TRANSLATION_BATCH_SIZE
            ),
            translation_disable_thinking=_read_bool("TRANSLATION_DISABLE_THINKING", True),
            voice_profile_dir=_read_path("VOICE_PROFILE_DIR", work_dir / "voices"),
            voice_reference_min_duration=_read_float(
                "VOICE_REFERENCE_MIN_DURATION", DEFAULT_VOICE_REFERENCE_MIN_DURATION
            ),
            voice_reference_target_duration=_read_float(
                "VOICE_REFERENCE_TARGET_DURATION", DEFAULT_VOICE_REFERENCE_TARGET_DURATION
            ),
            voice_reference_max_duration=_read_float(
                "VOICE_REFERENCE_MAX_DURATION", DEFAULT_VOICE_REFERENCE_MAX_DURATION
            ),
            tts_model=(_read_env("TTS_MODEL", DEFAULT_TTS_MODEL) or DEFAULT_TTS_MODEL),
            seed_vc_repo_path=_read_path(
                "SEED_VC_REPO_PATH", model_cache_dir / DEFAULT_SEED_VC_REPO_NAME
            ),
            seed_vc_diffusion_steps=_read_int(
                "SEED_VC_DIFFUSION_STEPS", DEFAULT_SEED_VC_DIFFUSION_STEPS
            ),
            tts_performance_reference_min_duration=_read_float(
                "TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
                DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
            ),
            tts_performance_reference_max_duration=_read_float(
                "TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
                DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
            ),
            tts_max_pause_seconds=_read_float(
                "TTS_MAX_PAUSE_SECONDS", DEFAULT_TTS_MAX_PAUSE_SECONDS
            ),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def has_deepseek_credentials(self) -> bool:
        """``True`` when a DeepSeek API key is configured."""

        return bool(self.deepseek_api_key)

    @property
    def has_huggingface_credentials(self) -> bool:
        """``True`` when a Hugging Face token is configured."""

        return bool(self.huggingface_token)

    def missing_credentials(self) -> list[str]:
        """Return the names of the credentials that are not configured yet."""

        missing: list[str] = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.huggingface_token:
            missing.append("HUGGINGFACE_TOKEN")
        return missing

    def ensure_directories(self) -> None:
        """Create the input/working/output/model-cache directories if needed."""

        for directory in (
            self.input_dir,
            self.work_dir,
            self.output_dir,
            self.model_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, object]:
        """Return a redacted, JSON-safe view of the settings (never secrets)."""

        return {
            "input_dir": str(self.input_dir),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "model_cache_dir": str(self.model_cache_dir),
            "deepseek_api_key_set": self.has_deepseek_credentials,
            "huggingface_token_set": self.has_huggingface_credentials,
            "device": self.device,
            "log_level": self.log_level,
            "diarization_model": self.diarization_model,
            "transcription_model": self.transcription_model,
            "transcription_compute_type": self.transcription_compute_type,
            "transcription_language": self.transcription_language,
            "translation_model": self.translation_model,
            "translation_base_url": self.translation_base_url,
            "translation_batch_size": self.translation_batch_size,
            "translation_disable_thinking": self.translation_disable_thinking,
            "voice_profile_dir": str(self.voice_profile_dir),
            "voice_reference_min_duration": self.voice_reference_min_duration,
            "voice_reference_target_duration": self.voice_reference_target_duration,
            "voice_reference_max_duration": self.voice_reference_max_duration,
            "tts_model": self.tts_model,
            "seed_vc_repo_path": str(self.seed_vc_repo_path),
            "seed_vc_diffusion_steps": self.seed_vc_diffusion_steps,
            "tts_performance_reference_min_duration": (
                self.tts_performance_reference_min_duration
            ),
            "tts_performance_reference_max_duration": (
                self.tts_performance_reference_max_duration
            ),
            "tts_max_pause_seconds": self.tts_max_pause_seconds,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance."""

    return Settings.from_env()


__all__ = [
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_SEED_VC_DIFFUSION_STEPS",
    "DEFAULT_SEED_VC_REPO_NAME",
    "DEFAULT_TRANSCRIPTION_COMPUTE_TYPE",
    "DEFAULT_TRANSCRIPTION_MODEL",
    "DEFAULT_TRANSLATION_BASE_URL",
    "DEFAULT_TRANSLATION_BATCH_SIZE",
    "DEFAULT_TRANSLATION_MODEL",
    "DEFAULT_TTS_MAX_PAUSE_SECONDS",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
    "DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
    "DEFAULT_VOICE_PROFILE_DIR",
    "DEFAULT_VOICE_REFERENCE_MAX_DURATION",
    "DEFAULT_VOICE_REFERENCE_MIN_DURATION",
    "DEFAULT_VOICE_REFERENCE_TARGET_DURATION",
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "load_env_file",
]
