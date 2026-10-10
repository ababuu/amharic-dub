#!/usr/bin/env python3
"""Initialise every model the pipeline needs, one at a time, and report.

The stages load their models lazily, so a missing checkpoint, a rejected Hugging
Face token or an incompatible package only shows up partway through a run - after
other models have already been downloaded. This script loads each model by itself,
prints what happened, and releases it before moving on, so a failure names exactly
one component.

Which translation and synthesis models are checked follows the configuration:

* ``TRANSLATION_BACKEND`` selects a local ``nllb`` checkpoint, or ``openai`` for a
  served instruction-following model (``TRANSLATION_PROVIDER`` picks the vendor).
* ``TTS_ENGINE`` selects ``mms`` (the default), or ``chatterbox`` plus ``seed-vc``.

Only the selected ones are loaded. Checking the other pair would download many
gigabytes a run never touches and, worse, would leave the models it *does* use
unchecked, so the pre-flight would pass and the run would still fail. To check the
other configuration, set the environment variables::

    TRANSLATION_BACKEND=openai TTS_ENGINE=chatterbox python scripts/validate_models.py

::

    python scripts/validate_models.py
    python scripts/validate_models.py --only nllb
    python scripts/validate_models.py --only bandit,mms

Each step drives the stage's *own* loader (``separation``'s session, ``diarization``'s
pipeline loader, ``transcription``'s model loader, ``translation``'s backend,
``tts``'s engines), so what is validated here is the same code path a real run
takes. The translation and synthesis steps go one step further than loading and run
one probe line through the model, because a checkpoint that loads can still reject
every input. ``scripts/test_gpu.py`` is the end-to-end run.

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


def load_nllb() -> None:
    """Load the NLLB translation model and translate one short line.

    Loading the weights only proves the download worked; translating proves the
    tokenizer, the language codes and ``generate`` all agree, which is the part that
    actually fails when a checkpoint and a library version disagree.
    """

    from app.pipeline import nllb

    settings = nllb.get_settings()
    _report("model", settings.translation_model)
    _report("device", settings.device)
    _report("languages", f"{nllb.SOURCE_LANGUAGE} -> {nllb.TARGET_LANGUAGE}")

    translator = nllb.load_translator(settings=settings)
    result = translator.translate("Hello.")
    if not result.text.strip():
        raise RuntimeError("NLLB returned an empty translation")
    _report("loaded", "translated a probe line")


def load_mms() -> None:
    """Load MMS-TTS Amharic and synthesize one word.

    Synthesis is the step worth validating here, because it exercises the two things
    a load alone would not: the romanisation the checkpoint requires, and the
    duration predictor producing a waveform at all.
    """

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("model", settings.tts_model)
    _report("device", settings.device)
    _report("seed / rate", f"{settings.mms_seed} / {settings.mms_speaking_rate:g}")

    if settings.tts_engine != "mms":
        raise RuntimeError(
            f"TTS_ENGINE is {settings.tts_engine!r}, so the MMS engine is not what a "
            "run would use; validate 'chatterbox' and 'seed-vc' instead"
        )

    engine = tts.load_mms_engine(settings=settings)
    romanized = engine.romanize("ሰላም")
    _report("romanised", romanized)

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        written = engine.synthesize(
            text="ሰላም", destination=_Path(directory) / "probe.wav"
        )
        if not written.is_file() or written.stat().st_size == 0:
            raise RuntimeError("MMS-TTS wrote no audio for the probe line")
    _report("loaded", f"synthesized at {engine.sample_rate} Hz")


def load_omnivoice() -> None:
    """Load OmniVoice and synthesize one word in a cloned voice.

    Cloning is exercised, not just loading: the project's whole reason for choosing
    this engine is that it speaks as a specific voice, and a load alone would not show
    whether the reference was accepted.
    """

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("model", settings.omnivoice_model)
    _report("device", settings.device)
    _report("steps / guidance", f"{settings.omnivoice_steps} / {settings.omnivoice_guidance_scale:g}")

    if settings.tts_engine != "omnivoice":
        raise RuntimeError(
            f"TTS_ENGINE is {settings.tts_engine!r}, so the OmniVoice engine is not "
            "what a run would use; validate that engine instead"
        )

    engine = tts.load_omnivoice_engine(settings=settings)

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        import numpy as _np
        import soundfile as _sf

        root = _Path(directory)
        # A reference is required: this engine has no voice of its own to fall back on,
        # which is precisely the difference from the single-voice engines.
        reference = root / "reference.wav"
        rate = 24_000
        seconds = 5.0
        tone = 0.4 * _np.sin(
            2.0 * _np.pi * 150.0 * _np.arange(int(seconds * rate)) / rate
        )
        _sf.write(str(reference), tone.astype(_np.float32), rate, subtype="PCM_16")

        written = engine.synthesize(
            text="ሰላም",
            voice_reference=reference,
            destination=root / "probe.wav",
        )
        if not written.is_file() or written.stat().st_size == 0:
            raise RuntimeError("OmniVoice wrote no audio for the probe line")

    _report("loaded", f"synthesized a cloned line at {engine.sample_rate} Hz")


def load_served_model() -> None:
    """Build the served-model client and confirm the key is accepted.

    ``client.models.list()`` is the smallest call that proves the credential works,
    which is worth knowing before a run reaches the adaptation stage. The provider
    decides which key is read and which endpoint is called, so this validates whichever
    one the run would actually use.
    """

    from app.pipeline import translation

    settings = translation.get_settings()
    if settings.translation_backend != "openai":
        raise RuntimeError(
            f"TRANSLATION_BACKEND is {settings.translation_backend!r}, so a served "
            "endpoint is not what a run would use; validate 'nllb' instead"
        )
    if not settings.translation_api_key:
        raise RuntimeError(f"{settings.translation_api_key_env} is not set")

    _report("provider", settings.translation_provider)
    _report("base url", settings.translation_base_url)
    _report("model", settings.translation_model)
    _report("thinking", translation.describe_thinking(settings))

    client = translation._build_client(  # noqa: SLF001
        settings, settings.translation_api_key
    )
    models = client.models.list()
    _report("models listed", str(len(getattr(models, "data", []) or [])))


def load_chatterbox() -> None:
    """Load the Amharic adapter on top of its pinned Chatterbox base."""

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("engine", settings.tts_engine)
    _report("adapter", settings.chatterbox_model)
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


#: Steps that every configuration needs, in pipeline order.
STEPS: tuple[tuple[str, str, object], ...] = (
    ("torch", "compute runtime", load_torch),
    ("bandit", "BandIt v2 Multi separation", load_bandit),
    ("pyannote", "pyannote Community-1 diarization", load_pyannote),
    ("whisper", "faster-whisper transcription", load_whisper),
)

#: Steps that depend on the configured backend and engine. Only the ones a run will
#: actually use are loaded: validating the other pair would download many gigabytes
#: the run never touches, and - worse - would leave the models it *does* use
#: unchecked, so the pre-flight would pass and the run would still fail.
CONFIGURED_STEPS: tuple[tuple[str, str, object, str], ...] = (
    ("nllb", "NLLB-200 translation (TRANSLATION_BACKEND=nllb)", load_nllb, "nllb"),
    (
        "served",
        "instruction-following adaptation (TRANSLATION_BACKEND=openai)",
        load_served_model,
        "openai",
    ),
    (
        "omnivoice",
        "OmniVoice cloned-voice synthesis (TTS_ENGINE=omnivoice)",
        load_omnivoice,
        "omnivoice",
    ),
    ("mms", "MMS-TTS Amharic synthesis (TTS_ENGINE=mms)", load_mms, "mms"),
    (
        "chatterbox",
        "Chatterbox Amharic synthesis (TTS_ENGINE=chatterbox)",
        load_chatterbox,
        "chatterbox",
    ),
    (
        "seed-vc",
        "Seed-VC V2 voice conversion (TTS_ENGINE=chatterbox)",
        load_seed_vc,
        "chatterbox",
    ),
)


#: How to make a step that the current configuration does not select part of it.
ACTIVATION: dict[str, str] = {
    "nllb": "set TRANSLATION_BACKEND=nllb",
    "served": "set TRANSLATION_BACKEND=openai",
    "omnivoice": "set TTS_ENGINE=omnivoice",
    "mms": "set TTS_ENGINE=mms",
    "chatterbox": "set TTS_ENGINE=chatterbox",
    "seed-vc": "set TTS_ENGINE=chatterbox",
}


def configured_steps() -> tuple[tuple[str, str, object], ...]:
    """Return every step this configuration needs, in run order.

    The selection is taken from the settings themselves - ``TRANSLATION_BACKEND`` and
    ``TTS_ENGINE`` - so what is validated is what a run would load.
    """

    from app.pipeline import translation, tts

    settings = translation.get_settings()
    backend = settings.translation_backend
    engine = tts.get_settings().tts_engine

    selected: list[tuple[str, str, object]] = list(STEPS)
    for name, description, loader, wanted in CONFIGURED_STEPS:
        if name in ("nllb", "served") and wanted != backend:
            continue
        if name in ("mms", "omnivoice", "chatterbox", "seed-vc") and wanted != engine:
            continue
        selected.append((name, description, loader))
    return tuple(selected)


def all_step_names() -> set[str]:
    """Return every step name that exists, configured or not."""

    return {name for name, _, _ in STEPS} | {
        name for name, _, _, _ in CONFIGURED_STEPS
    }


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
            f"(one or more of: {', '.join(name for name, _, _ in configured_steps())})"
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

    steps = configured_steps()

    selected = {name for name, _, _ in steps}
    if args.only:
        requested = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = sorted(requested - selected - all_step_names())
        if unknown:
            print(f"unknown step(s): {', '.join(unknown)}", file=sys.stderr)
            print(
                f"known steps: {', '.join(sorted(all_step_names()))}",
                file=sys.stderr,
            )
            return 2
        # A step that exists but belongs to the other configuration is worth
        # distinguishing from a typo: the names are in this file's help and in the
        # README, so being told *how* to select one is the difference between a dead
        # end and a one-line fix.
        other = sorted(requested - selected)
        if other:
            print(
                f"not part of this configuration: {', '.join(other)}", file=sys.stderr
            )
            for name in other:
                print(f"  {name}: {ACTIVATION[name]}", file=sys.stderr)
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
    for name, description, loader in steps:
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
