#!/usr/bin/env bash
#
# Install the pipeline's runtime dependencies on the GPU worker (RunPod A40).
#
# Usage:
#   bash scripts/install_dependencies.sh
#
# Environment:
#   PYTHON   interpreter to install into (default: python)
#
# The script does more than pip does: it imports every runtime module the pipeline
# needs - the list lives in scripts/check_environment.py, so the installer and the
# pre-flight can never disagree about what has to be importable - and it fails
# loudly if the CUDA-matched torch the worker image provides was replaced.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"
PYTHON="${PYTHON:-python}"

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "error: ${REQUIREMENTS} not found" >&2
    exit 1
fi

echo "==> Interpreter"
"${PYTHON}" -c 'import sys; print(sys.version)'

echo "==> Upgrading pip"
"${PYTHON}" -m pip install --upgrade pip

echo "==> Installing requirements"
"${PYTHON}" -m pip install -r "${REQUIREMENTS}"

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
"""Fail loudly if the CUDA-matched torch was replaced during the install."""

import torch

print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.__version__.startswith("2.8"):
    raise SystemExit(
        f"  torch is {torch.__version__}, not the pinned 2.8 line; "
        "the CUDA-matched build was replaced"
    )
PY

echo "==> pip check"
"${PYTHON}" -m pip check

echo
echo "Done. Next:"
echo "  python scripts/check_environment.py --full                  # credentials, imports"
echo "  python scripts/validate_models.py                           # load each model once"
echo "  python -m app.pipeline.orchestrator data/input/test.mp4     # -> dubbed MP4"
