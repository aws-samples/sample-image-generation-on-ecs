#!/usr/bin/env bash
set -euo pipefail
cd "$CODEBUILD_SRC_DIR"
pip install --quiet uv==0.7.12
uv sync --group dev --quiet
source .venv/bin/activate

echo "[lint] Running make ${MAKE_TARGET:-lint}..."
make "${MAKE_TARGET:-lint}"
