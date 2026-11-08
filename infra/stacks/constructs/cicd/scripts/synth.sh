#!/usr/bin/env bash
set -euo pipefail
cd "$CODEBUILD_SRC_DIR"
pip install --quiet uv==0.7.12
uv sync --group dev --quiet
source .venv/bin/activate

npm install -g aws-cdk@2.1130.0
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION="${AWS_REGION:-us-east-1}"

echo "[synth] Running make ${MAKE_TARGET:-synth}..."
make "${MAKE_TARGET:-synth}"
