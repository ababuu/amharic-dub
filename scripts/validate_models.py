#!/usr/bin/env python3
"""Initialise every model the pipeline needs, one at a time, and report.

The stages load their models lazily, so a missing checkpoint, a rejected Hugging
Face token or an incompatible package only shows up partway through a run - after
other models have already been downloaded. This script loads each model by itself,
prints what happened, and releases it before moving on, so a failure names exactly
one component.

::

    python scripts/validate_models.py
    python scripts/validate_models.py --only chatterbox
    python scripts/validate_models.py --only bandit,seed-vc

Each step drives the stage's *own* loader (``separation``'s session, ``diarization``'s
pipeline loader, ``transcription``'s model loader, ``translation``'s client,
``tts``'s engines), so what is validated here is the same code path a real run
takes. Nothing is synthesized and no stage is executed: this checks
initialisation only. ``scripts/test_gpu.py`` is the end-to-end run.

Models are dropped and CUDA's cache is emptied between steps, so the whole
validation fits on a single 48 GB card even though every model is loaded.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

# Invoking this file directly puts ``scripts/`` on ``sys.path`` rather than the
# project root, so the ``app`` package is made importable explicitly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEPARATOR = "-" * 60


def _gib(num_bytes: float) -> str:
    return f"{num_bytes / 1024**3:.1f} GiB"


def _vram() -> str:
    """Return the current CUDA allocation, or an empty string without CUDA."""

    try:
        import torch
    except ImportError:
        return ""

    if not torch.cuda.is_available():
        return ""
    free, total = torch.cuda.mem_get_info()
    return f"vram {_gib(total - free)} used of {_gib(total)}"


def _release() -> None:
    """Drop every cached engine and free what CUDA is holding."""

    from app.pipeline import tts

    tts.reset_engine_cache()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - no runtime installed
        pass


def _report(title: str, value: str) -> None:
    print(f"      {title:<18} {value}", flush=True)


# ---------------------------------------------------------------------------
# One loader per model
# ---------------------------------------------------------------------------


def load_torch() -> None:
    """Report the compute runtime every model below depends on."""

    import torch

    _report("torch", torch.__version__)
    _report("cuda available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        _report("device", torch.cuda.get_device_name(0))
        _report("cuda version", torch.version.cuda or "unknown")


def load_bandit() -> None:
    """Build the BandIt v2 Multi session and load its weights."""

    from app.pipeline import separation

    if separation.BanditSession is None:  # pragma: no cover - missing dependency
        raise RuntimeError("bandit-infer is not installed")

    settings = separation.get_settings()
    weights_dir = separation.resolve_weights_dir(settings)
    _report("model", separation.MODEL_NAME)
    _report("weights dir", str(weights_dir))

    session = separation.BanditSession(
        separation.MODEL_NAME, device=settings.device, weights_dir=weights_dir
    )
    try:
        session.load()
        _report("stems", ", ".join(separation.STEM_NAMES))
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def load_pyannote() -> None:
    """Load the gated pyannote Community-1 diarization pipeline."""

    from app.pipeline import diarization

    settings = diarization.get_settings()
    token = settings.huggingface_token
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set; the Community-1 checkpoint is gated, "
            "and the token's account must have accepted its conditions"
        )

    _report("model", settings.diarization_model)
    # The stage's own loader, so the device move and the "pipeline is None" case
    # are exercised exactly as a diarization run would exercise them.
    pipeline = diarization._load_pipeline(settings, token)  # noqa: SLF001
    _report("device", str(getattr(pipeline, "device", settings.device)))


def load_whisper() -> None:
    """Load the faster-whisper transcription model onto the configured device."""

    from app.pipeline import transcription

    settings = transcription.get_settings()
    _report("model", settings.transcription_model)
    _report("compute type", settings.transcription_compute_type)

    model = transcription._load_model(settings)  # noqa: SLF001
    _report("loaded", type(model).__name__)


def load_deepseek() -> None:
    """Build the DeepSeek client and confirm the key is accepted.

    ``client.models.list()`` is the smallest call that proves the credential works,
    which is worth knowing before a run reaches the adaptation stage.
    """

    from app.pipeline import translation

    settings = translation.get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    _report("base url", settings.translation_base_url)
    _report("model", settings.translation_model)

    client = translation._build_client(settings, settings.deepseek_api_key)  # noqa: SLF001
    models = client.models.list()
    _report("models listed", str(len(getattr(models, "data", []) or [])))


def load_chatterbox() -> None:
    """Load the Amharic adapter on top of its pinned Chatterbox base."""

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("adapter", settings.tts_model)
    _report("device", settings.device)

    engine = tts.load_chatterbox_engine(settings=settings)
    # The engine is lazy: ``_load()`` is what fetches the adapter and the base
    # checkpoint, which is the step worth validating.
    adapter = engine._load()  # noqa: SLF001
    _report("sample rate", str(engine.sample_rate))
    _report("loaded", type(adapter).__name__)


def load_seed_vc() -> None:
    """Build the Seed-VC V2 wrapper from its checkout and load its checkpoints."""

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("checkout", str(settings.seed_vc_repo_path))
    _report("diffusion steps", str(settings.seed_vc_diffusion_steps))

    engine = tts.load_seed_vc_engine(settings=settings)
    engine._load()  # noqa: SLF001 - downloads and builds the V2 wrapper
    _report("convert style", str(engine.convert_style))


#: Validation steps in pipeline order, cheapest first within reason.
STEPS: tuple[tuple[str, str, object], ...] = (
    ("torch", "compute runtime", load_torch),
    ("deepseek", "DeepSeek adaptation client", load_deepseek),
    ("bandit", "BandIt v2 Multi separation", load_bandit),
    ("pyannote", "pyannote Community-1 diarization", load_pyannote),
    ("whisper", "faster-whisper transcription", load_whisper),
    ("chatterbox", "Chatterbox Amharic synthesis", load_chatterbox),
    ("seed-vc", "Seed-VC V2 voice conversion", load_seed_vc),
)


def _cuda_available() -> bool | None:
    """Return CUDA availability, or ``None`` when PyTorch is not installed."""

    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.is_available()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load each pipeline model on its own and report the result."
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated subset to validate "
            f"(one or more of: {', '.join(name for name, _, _ in STEPS)})"
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "run the model-loading steps even without CUDA. They download real "
            "weights and are meant for the GPU worker, not a development machine."
        ),
    )
    args = parser.parse_args(argv)

    selected = {name for name, _, _ in STEPS}
    if args.only:
        requested = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = sorted(requested - selected)
        if unknown:
            print(f"unknown step(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"known steps: {', '.join(sorted(selected))}", file=sys.stderr)
            return 2
        selected = requested

    # Everything but the runtime report fetches weights, so refuse to do that on a
    # machine without a GPU: this is the check that stops a development box from
    # quietly filling its disk with checkpoints.
    if (selected - {"torch"}) and not args.allow_cpu and _cuda_available() is not True:
        print(
            "refusing to load models: torch reports no CUDA device on this machine.",
            file=sys.stderr,
        )
        print(
            "The steps below download real weights and are meant for the GPU worker.",
            file=sys.stderr,
        )
        print(
            "Run this on the pod, or pass --allow-cpu if you really mean to do it here.",
            file=sys.stderr,
        )
        return 2

    print("Amharic dubbing pipeline - model initialisation check")
    print(SEPARATOR)
    print("Models are loaded one at a time and released before the next one.\n")

    results: list[tuple[str, bool, float, str]] = []
    for name, description, loader in STEPS:
        if name not in selected:
            continue
        print(f"[{name}] {description}", flush=True)
        started = time.perf_counter()
        error = ""
        healthy = True
        try:
            loader()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            healthy = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        vram = _vram()
        if vram:
            _report("memory", vram)
        if healthy:
            print(f"      OK ({elapsed:.1f} s)", flush=True)
        else:
            print(f"      FAIL ({elapsed:.1f} s)  {error}", flush=True)
        results.append((name, healthy, elapsed, error))
        _release()

    print(SEPARATOR)
    failed = [name for name, healthy, _, _ in results if not healthy]
    print(f"validated {len(results)} of {len(STEPS)} step(s) in {sum(r[2] for r in results):.1f} s")
    if failed:
        print(f"Result: FAIL ({', '.join(failed)})")
        print(
            "\nFix the first failure before starting the end-to-end run: see the "
            "component named above, not the model that happened to load after it."
        )
        return 1

    print("Result: OK - every selected model initialised")
    print("\nNext: python scripts/test_gpu.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
