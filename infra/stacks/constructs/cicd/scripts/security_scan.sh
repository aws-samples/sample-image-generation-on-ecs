#!/usr/bin/env bash
set -euo pipefail
cd "$CODEBUILD_SRC_DIR"
pip install --quiet uv==0.7.12
uv sync --group dev --quiet
source .venv/bin/activate

echo "[security] Running make ${MAKE_TARGET:-security}..."
if ! make "${MAKE_TARGET:-security}"; then
  echo "[security] FAILED — dumping findings:"
  # Output dir must match the --output-dir the make target used (see Makefile).
  case "${MAKE_TARGET:-security}" in
    security-web) ash_out=".ash/out-web" ;;
    *) ash_out=".ash/out" ;;
  esac
  cat "${ash_out}/reports/ash.summary.txt" 2>/dev/null || echo "No summary found in ${ash_out}"
  exit 2
fi
