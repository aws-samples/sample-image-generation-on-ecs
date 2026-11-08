#!/usr/bin/env bash
# Deploy the inference-ecs stack with image tags resolved from ECR.
#
# The Build stage pushes images tagged with both "latest" and a short SHA.
# We resolve the latest image tag (non-"latest") from each ECR repo and
# pass it as CDK context. This makes CDK detect the new image and trigger
# a rolling ECS deployment automatically — no force-restart needed.
#
# Required env: ZIMG_ENDPOINT_STACK — short stack name to deploy
set -euo pipefail

echo "[deploy-endpoint] === start $(date) ==="
node --version
python3 --version

npm install -g aws-cdk@2.1130.0
pip install --quiet uv==0.7.12
uv sync --quiet
source .venv/bin/activate

cd "$CODEBUILD_SRC_DIR"

export CDK_DEFAULT_ACCOUNT="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
export CDK_DEFAULT_REGION="${DEPLOY_REGION:-${AWS_REGION:-us-east-1}}"

PREFIX="${ZIMG_PREFIX:-dev-zimg}"
SHORT_NAME="${ZIMG_ENDPOINT_STACK:?ZIMG_ENDPOINT_STACK must be set (e.g. endpoint-512)}"
STACK_NAME="${PREFIX}-${SHORT_NAME}"

# Resolve the latest non-"latest" tag from each inference ECR repo.
# This tag changes on every Build stage, so CDK detects the diff.
resolve_tag() {
  local repo="$1"
  aws ecr describe-images --repository-name "$repo" \
    --query 'imageDetails | sort_by(@, &imagePushedAt) | [-1].imageTags[?@ != `latest`] | [0]' \
    --output text 2>/dev/null || echo "latest"
}

TAG_512=$(resolve_tag "${PREFIX}-serving-512")
TAG_768=$(resolve_tag "${PREFIX}-serving-768")
TAG_1024=$(resolve_tag "${PREFIX}-serving-1024")
TAG_GPU=$(resolve_tag "${PREFIX}-serving-gpu")

echo "[deploy-endpoint] image tags: 512=${TAG_512} 768=${TAG_768} 1024=${TAG_1024} gpu=${TAG_GPU}"

echo "[deploy-endpoint] running cdk deploy for ${STACK_NAME}..."
cdk deploy "${STACK_NAME}" \
  --require-approval never \
  --exclusively \
  --context "image_tag_512=${TAG_512}" \
  --context "image_tag_768=${TAG_768}" \
  --context "image_tag_1024=${TAG_1024}" \
  --context "image_tag_gpu=${TAG_GPU}" \
  --outputs-file "/tmp/deploy-${SHORT_NAME}-outputs.json"

echo "[deploy-endpoint] forcing ECS rolling update to pull new images..."
CLUSTER="${PREFIX}-inference-cluster"
for svc in "${PREFIX}-inference-512" "${PREFIX}-inference-768" "${PREFIX}-inference-1024" "${PREFIX}-inference-gpu"; do
  aws ecs update-service --cluster "$CLUSTER" --service "$svc" --force-new-deployment --no-cli-pager 2>/dev/null && \
    echo "  ✓ $svc" || echo "  - $svc (skipped)"
done

echo "[deploy-endpoint] done"
cat "/tmp/deploy-${SHORT_NAME}-outputs.json" 2>/dev/null || true
