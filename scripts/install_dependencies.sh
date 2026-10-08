#!/usr/bin/env bash
#
# Install the pipeline's runtime dependencies on the GPU worker (RunPod A40).
#
# Why this is not just `pip install -r requirements.txt`:
#
#   chatterbox-tts 0.1.7 - the revision this project pins - declares
#   torch==2.6.0, torchaudio==2.6.0, numpy<2 and transformers==5.2.0. Letting pip
#   resolve it rewrites the CUDA-matched torch/torchaudio that the RunPod PyTorch
#   image already provides, and downgrades numpy to 1.x, which the pyannote stack
#   cannot use. So chatterbox-tts is installed last with --no-deps, and every
#   package it actually imports is declared in requirements.txt instead.
#
#   The result is deliberately not `pip check`-clean: chatterbox-tts's metadata
#   still asks for torch 2.6 / numpy 1.x. That conflict is known and expected -
#   this script reports it and does not fail on it.
#
# Usage:
#   bash scripts/install_dependencies.sh
#
# Environment:
#   PYTHON   interpreter to install into (default: python)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"
PYTHON="${PYTHON:-python}"

#: Installed separately, with --no-deps; everything else comes from requirements.
DEFERRED_PACKAGE="chatterbox-tts"

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "error: ${REQUIREMENTS} not found" >&2
    exit 1
fi

echo "==> Interpreter"
"${PYTHON}" -c 'import sys; print(sys.version)'

echo "==> Upgrading pip"
"${PYTHON}" -m pip install --upgrade pip

# Everything except the deferred package, so pip never sees chatterbox-tts's
# torch 2.6 / numpy 1.x requirements while it resolves the rest.
FILTERED="$(mktemp)"
trap 'rm -f "${FILTERED}"' EXIT
grep -v -E "^[[:space:]]*${DEFERRED_PACKAGE}[[:space:]=@]" "${REQUIREMENTS}" > "${FILTERED}"

echo "==> Installing requirements (${DEFERRED_PACKAGE} deferred)"
"${PYTHON}" -m pip install -r "${FILTERED}"

# The deferred package's own spec line, taken from requirements.txt so the
# pinned revision lives in exactly one place.
DEFERRED_SPEC="$(grep -E "^[[:space:]]*${DEFERRED_PACKAGE}[[:space:]=@]" "${REQUIREMENTS}" | head -n 1 || true)"
if [[ -z "${DEFERRED_SPEC}" ]]; then
    echo "error: no ${DEFERRED_PACKAGE} entry in ${REQUIREMENTS}" >&2
    exit 1
fi

echo "==> Installing ${DEFERRED_PACKAGE} without dependency resolution"
echo "    ${DEFERRED_SPEC}"
"${PYTHON}" -m pip install --no-deps "${DEFERRED_SPEC}"

echo "==> Import smoke check"
# The module list lives in scripts/check_environment.py so the installer and the
# pre-flight can never disagree about what has to be importable.
"${PYTHON}" -c "
import sys
sys.path.insert(0, r'${PROJECT_ROOT}')
from scripts.check_environment import check_runtime_imports
raise SystemExit(0 if check_runtime_imports() else 1)
"

echo "==> Checking the pinned torch line survived"
"${PYTHON}" - <<'PY'
"""Fail loudly if the CUDA-matched torch was replaced by the chatterbox install."""

import torch

print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.__version__.startswith("2.8"):
    raise SystemExit(
        f"  torch is {torch.__version__}, not the pinned 2.8 line; "
        "the CUDA-matched build was replaced"
    )
PY
PY

# Informational only: the chatterbox-tts metadata conflict is intentional.
echo "==> pip check (expected to report chatterbox-tts only)"
if ! "${PYTHON}" -m pip check; then
    echo "    note: pip check complaints about chatterbox-tts are expected."
fi

echo
echo "Done. Next:"
echo "  export HF_HOME=${PROJECT_ROOT}/.cache/huggingface"
echo "  python scripts/check_environment.py --full"
echo "  python scripts/validate_models.py"
echo "  python scripts/test_gpu.py"
