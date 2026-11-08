#!/usr/bin/env bash
set -euo pipefail
cd "$CODEBUILD_SRC_DIR"
pip install --quiet uv==0.7.12
uv sync --group dev --quiet
source .venv/bin/activate

export ZIMAGE_GALLERY_BUCKET="${ZIMAGE_GALLERY_BUCKET:-test-bucket}"
export ZIMAGE_LOCAL="${ZIMAGE_LOCAL:-true}"

echo "[test] Running make ${MAKE_TARGET:-test}..."
make "${MAKE_TARGET:-test}"
