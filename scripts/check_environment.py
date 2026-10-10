#!/usr/bin/env python3
"""Environment health check for the Amharic dubbing pipeline.

Reports the runtime prerequisites needed on the GPU worker (a RunPod A40 pod
built from the official RunPod PyTorch image) or on a local machine:

* Python version
* PyTorch version (only if it is installed)
* whether CUDA is available
* GPU name and available VRAM (only if CUDA is available)
* FFmpeg availability

With ``--full`` it also runs the pre-flight checks that must pass before a paid
GPU session is started:

* the credentials the pipeline needs, including a warning when a value is still
  the placeholder copied from ``.env.example``
* the Seed-VC checkout and the ``configs/v2/vc_wrapper.yaml`` the TTS stage runs
* every third-party module the stages import at run time
* every stage module of this project, so a broken import in the checkout is caught
  here rather than halfway through a paid run

The script never calls an external API and never downloads a model, so it is safe
to run before anything is configured. Exit code is ``0`` when no blocking problem
is found and ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Invoking this file directly puts ``scripts/`` on ``sys.path`` rather than the
# project root, so the ``app`` package is made importable explicitly. It also
# lets ``scripts/install_dependencies.sh`` reuse the import list below.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON = (3, 12)
BYTES_PER_GIB = 1024**3
SEPARATOR = "-" * 60

#: Top-level import name -> the part of the pipeline that needs it. Kept here,
#: rather than in each script, so the installer's smoke check and this pre-flight
#: can never disagree about what has to be importable.
#: Packages every configuration needs, with what needs each one.
RUNTIME_MODULES: dict[str, str] = {
    "torch": "runtime",
    "torchaudio": "runtime",
    "numpy": "runtime",
    "soundfile": "audio I/O",
    "av": "faster-whisper decoding",
    "dotenv": "configuration",
    "faster_whisper": "transcription",
    "pyannote.audio": "diarization",
    "torchcodec": "pyannote.audio",
    "bandit_infer": "separation",
    "librosa": "audio analysis",
    "yaml": "configuration",
    "pydub": "audio editing",
    "einops": "audio modelling",
    "scipy": "audio analysis",
    "tqdm": "progress reporting",
    "transformers": "model loading",
    "huggingface_hub": "model downloads",
}

#: Packages only one translation backend or speech engine needs, keyed by the setting
#: that selects it. A configuration is checked against its own list and nothing else:
#: demanding a working Chatterbox for an OmniVoice run would report a failure over
#: something the run never imports, which sends someone chasing a red mark that does not
#: affect them. ``openai`` is required by the default backend, so it is listed here rather
#: than above; running ``TRANSLATION_BACKEND=nllb`` correctly stops requiring it.
BACKEND_MODULES: dict[str, tuple[str, dict[str, str]]] = {
    "openai": ("translation_backend", {"openai": "LLM adaptation"}),
    "nllb": (
        "translation_backend",
        {"sentencepiece": "NLLB-200 tokenizer"},
    ),
    "omnivoice": (
        "tts_engine",
        {"omnivoice": "OmniVoice synthesis", "accelerate": "OmniVoice device placement"},
    ),
    "mms": ("tts_engine", {"uroman": "MMS-TTS romanisation"}),
    "chatterbox": (
        "tts_engine",
        {
            "chatterbox": "Chatterbox synthesis",
            "peft": "Chatterbox Amharic adapter",
            "safetensors": "Chatterbox Amharic adapter",
            "hydra": "Seed-VC",
            "omegaconf": "Seed-VC",
            "munch": "Seed-VC",
            "matplotlib": "Seed-VC (BigVGAN)",
        },
    ),
}


def required_runtime_modules(settings: Any) -> dict[str, str]:
    """Return the packages this configuration actually needs, with what needs each."""

    required = dict(RUNTIME_MODULES)
    for name, (attribute, modules) in BACKEND_MODULES.items():
        if getattr(settings, attribute, None) == name:
            required.update(modules)
    return required

#: The project's own modules, imported as well. The third-party list above cannot
#: catch a broken import *inside* the project, so a typo or a bad import in a stage
#: would otherwise surface on the GPU worker in the middle of a paid run rather than
#: here. The stage modules guard their heavy dependencies, so this list imports
#: cleanly even on a machine where torch is not installed.
PROJECT_MODULES: tuple[str, ...] = (
    "app.config",
    "app.pipeline.amharic_text",
    "app.pipeline.dialogue_context",
    "app.pipeline.diarization",
    "app.pipeline.evaluation",
    "app.pipeline.mixing",
    "app.pipeline.nllb",
    "app.pipeline.orchestrator",
    "app.pipeline.prosody",
    "app.pipeline.qc",
    "app.pipeline.separation",
    "app.pipeline.timing",
    "app.pipeline.transcription",
    "app.pipeline.translation",
    "app.pipeline.tts",
    "app.pipeline.video",
    "app.pipeline.voice_profiles",
)

#: The Seed-VC file the TTS stage instantiates its converter from.
SEED_VC_CONFIG = Path("configs") / "v2" / "vc_wrapper.yaml"


def format_gib(num_bytes: float) -> str:
    """Format a byte count as gibibytes with one decimal place."""

    return f"{num_bytes / BYTES_PER_GIB:.1f} GiB"


def check_python() -> bool:
    """Report the Python version. Returns ``False`` if it is too old."""

    current = (sys.version_info.major, sys.version_info.minor)
    supported = current >= MIN_PYTHON
    status = "OK" if supported else "FAIL"
    required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    print(f"Python version: {platform.python_version()}  [{status}] (requires >= {required})")
    return supported


def check_torch() -> bool:
    """Report PyTorch/CUDA status.

    PyTorch is optional at this stage of the project, so a missing installation
    is reported and treated as non-blocking. Returns ``False`` only for real
    failures.
    """

    try:
        import torch
    except ImportError:
        print("PyTorch version: not installed  [SKIP] (required once AI stages are added)")
        return True

    print(f"PyTorch version: {torch.__version__}")

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - depends on broken CUDA installs
        print(f"CUDA available: unknown  [WARN] ({exc})")
        return True

    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print("GPU name: n/a")
        print("Available VRAM: n/a")
        return True

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover
        print(f"CUDA devices: unknown  [WARN] ({exc})")
        return True

    for index in range(device_count):
        try:
            gpu_name = torch.cuda.get_device_name(index)
        except Exception:  # pragma: no cover
            gpu_name = "unknown"

        free_bytes = None
        total_bytes = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:  # pragma: no cover - older torch or driver issues
            try:
                total_bytes = torch.cuda.get_device_properties(index).total_memory
            except Exception:
                total_bytes = None

        label = "GPU name" if index == 0 else f"GPU name [{index}]"
        print(f"{label}: {gpu_name}")

        if free_bytes is not None and total_bytes is not None:
            print(
                f"Available VRAM: {format_gib(free_bytes)} free of "
                f"{format_gib(total_bytes)}"
            )
        elif total_bytes is not None:
            print(f"Available VRAM: {format_gib(total_bytes)} total (free unknown)")
        else:
            print("Available VRAM: unknown")

    return True


def check_ffmpeg() -> bool:
    """Report whether FFmpeg is available on ``PATH``."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        print("FFmpeg: not found on PATH  [FAIL]")
        return False

    version = "unknown"
    try:
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        if output:
            version = output.splitlines()[0]
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FFmpeg: found at {executable} but could not run it  [WARN] ({exc})")
        return True

    print(f"FFmpeg: {executable}")
    print(f"        {version}")
    return True


def _env_example_placeholders() -> dict[str, str]:
    """Return the placeholder values shipped in ``.env.example``.

    A ``.env`` copied from the example but never filled in would otherwise look
    configured, so the pre-flight compares against these values instead of only
    checking that a credential is non-empty. Unreadable files yield no
    placeholders rather than an error: this is a helpful check, not a hard
    prerequisite.
    """

    example = PROJECT_ROOT / ".env.example"
    if not example.is_file():
        return {}

    placeholders: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        value = value.strip()
        if value:
            placeholders[name.strip()] = value
    return placeholders


def check_credentials() -> bool:
    """Report the pipeline credentials, flagging unedited ``.env`` placeholders.

    Only the credentials the configured run needs are blocking: the Hugging Face
    token always, because the gated diarization pipeline needs it, and the
    DeepSeek key only when ``TRANSLATION_BACKEND`` is ``openai``. Failing on a key
    the configured backend never reads would send someone hunting for a credential
    they do not need while the real problem waits.
    """

    from app.config import get_settings

    settings = get_settings()
    placeholders = _env_example_placeholders()
    deepseek_required = settings.translation_backend == "openai"
    configured = {
        "HUGGINGFACE_TOKEN": (settings.huggingface_token, True),
        "DEEPSEEK_API_KEY": (settings.deepseek_api_key, deepseek_required),
    }

    healthy = True
    for name, (value, required) in configured.items():
        if not required:
            print(
                f"{name}: not needed by TRANSLATION_BACKEND="
                f"{settings.translation_backend}  [SKIP]"
            )
            continue
        if not value:
            print(f"{name}: not set  [FAIL]")
            healthy = False
        elif value == placeholders.get(name):
            print(f"{name}: still the .env.example placeholder  [FAIL]")
            healthy = False
        else:
            print(f"{name}: set  [OK]")

    if settings.huggingface_token and settings.huggingface_token != placeholders.get(
        "HUGGINGFACE_TOKEN"
    ):
        print("        - a set token still fails if its account has not accepted the")
        print("          conditions on the pyannote/speaker-diarization-community-1")
        print("          model page")

    return healthy


def check_seed_vc() -> bool:
    """Report whether the Seed-VC checkout the TTS stage runs is usable.

    Seed-VC converts a synthesized take into a target voice, which only the
    prompt-and-convert engine does. Under a single-voice engine there is no
    conversion step, so a missing checkout is reported as unused rather than as a
    failure - the run does not touch it.
    """

    from app.config import get_settings

    settings = get_settings()
    if settings.tts_engine != "chatterbox":
        print(
            f"Seed-VC checkout: not used by TTS_ENGINE={settings.tts_engine}  [SKIP]"
        )
        return True
    repo = Path(settings.seed_vc_repo_path)
    if not repo.is_dir():
        print(f"Seed-VC checkout: {repo}  [FAIL] (not a directory)")
        print("        - git clone https://github.com/Plachtaa/seed-vc " + str(repo))
        return False

    config = repo / SEED_VC_CONFIG
    if not config.is_file():
        print(f"Seed-VC checkout: {repo}  [FAIL] (no {SEED_VC_CONFIG})")
        return False

    print(f"Seed-VC checkout: {repo}  [OK]")
    return True


def check_runtime_imports() -> bool:
    """Import every third-party module this configuration needs, and report the failures.

    Every import here is top level and none of them downloads a model, so this is
    the cheapest way to find a package that is missing *before* a stage reaches it
    halfway through a run. The list is built from the configuration - see
    :func:`required_runtime_modules` - so a run is never failed over a package its own
    backend or engine does not import.
    """

    import importlib

    from app.config import get_settings

    required = required_runtime_modules(get_settings())

    failures: list[tuple[str, str, str]] = []
    for name, needed_by in required.items():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failure, never stop
            failures.append((name, needed_by, f"{type(exc).__name__}: {exc}"))

    project_failures: list[tuple[str, str]] = []
    for name in PROJECT_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failure, never stop
            project_failures.append((name, f"{type(exc).__name__}: {exc}"))

    for name, error in project_failures:
        print(f"  FAIL  {name:<16} (project): {error}")

    if failures:
        for name, needed_by, error in failures:
            print(f"  FAIL  {name:<16} ({needed_by}): {error}")
        print(
            f"Runtime imports: {len(failures)} of {len(required)} failed  "
            "[FAIL]"
        )
        print("        - run: bash scripts/install_dependencies.sh")
    else:
        print(f"Runtime imports: all {len(required)} available  [OK]")

    if project_failures:
        print(
            f"Project modules: {len(project_failures)} of {len(PROJECT_MODULES)} "
            "failed  [FAIL]"
        )
        print("        - this is a bug in the checkout, not a missing package")
    else:
        print(f"Project modules: all {len(PROJECT_MODULES)} import cleanly  [OK]")

    return not failures and not project_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the runtime prerequisites of the dubbing pipeline."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also check credentials, the Seed-VC checkout and every runtime import",
    )
    args = parser.parse_args(argv)

    print("Amharic dubbing pipeline - environment health check")
    print(SEPARATOR)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    checks = {
        "Python": check_python(),
        "PyTorch": check_torch(),
        "FFmpeg": check_ffmpeg(),
    }

    if args.full:
        print(SEPARATOR)
        checks["Credentials"] = check_credentials()
        checks["Seed-VC"] = check_seed_vc()
        checks["Runtime imports"] = check_runtime_imports()

    print(SEPARATOR)
    failures = [name for name, healthy in checks.items() if not healthy]
    if failures:
        print(f"Result: FAIL ({', '.join(failures)})")
        return 1

    print("Result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
