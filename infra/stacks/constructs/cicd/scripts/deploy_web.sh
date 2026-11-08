#!/usr/bin/env bash
set -euo pipefail

cd "$CODEBUILD_SRC_DIR"
ROOT="$(pwd)"

echo "[deploy-web] Installing deps..."
npm install -g aws-cdk@2.1130.0
pip install --quiet uv==0.7.12
uv sync --quiet
source .venv/bin/activate

export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ -z "${REGION:-}" ]; then
  echo "[deploy-web] ERROR: REGION is empty; it is required for __AWS_REGION__ and the static bucket name"
  exit 1
fi
export CDK_DEFAULT_REGION="$REGION"

echo "[deploy-web] building frontend..."
(cd web/frontend && npm ci && npm run build)

echo "[deploy-web] deploying web stack..."
cd "$ROOT"
cdk deploy "${ZIMG_PREFIX}-web" \
  --app "python3 infra/app.py" \
  --require-approval never \
  --exclusively \
  --outputs-file cdk-outputs.json

echo "[deploy-web] injecting Cognito config into index.html..."
WEB_OUTPUTS=$(python3 -c "import json; print(json.dumps(json.load(open('cdk-outputs.json'))['${ZIMG_PREFIX}-web']))")
COGNITO_USER_POOL_ID=$(echo "$WEB_OUTPUTS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('UserPoolId',''))")
COGNITO_APP_CLIENT_ID=$(echo "$WEB_OUTPUTS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('AppClientId',''))")
COGNITO_DOMAIN=$(echo "$WEB_OUTPUTS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('CognitoDomain',''))")
if [ -z "$COGNITO_USER_POOL_ID" ] || [ -z "$COGNITO_APP_CLIENT_ID" ] || [ -z "$COGNITO_DOMAIN" ]; then
  echo "[deploy-web] ERROR: missing Cognito stack output(s) in cdk-outputs.json"
  echo "  UserPoolId='${COGNITO_USER_POOL_ID}' AppClientId='${COGNITO_APP_CLIENT_ID}' CognitoDomain='${COGNITO_DOMAIN}'"
  exit 1
fi
INDEX="web/frontend/dist/index.html"
sed -i.bak \
  -e "s|__COGNITO_USER_POOL_ID__|${COGNITO_USER_POOL_ID}|g" \
  -e "s|__COGNITO_APP_CLIENT_ID__|${COGNITO_APP_CLIENT_ID}|g" \
  -e "s|__COGNITO_DOMAIN__|${COGNITO_DOMAIN}|g" \
  -e "s|__AWS_REGION__|${CDK_DEFAULT_REGION}|g" \
  "$INDEX"
rm -f "${INDEX}.bak"
# Sanity: fail fast if any deploy-time placeholder leaked through. Matches
# every __COGNITO_*__ / __AWS_*__ token (covers all four used by index.html:
# __COGNITO_USER_POOL_ID__, __COGNITO_APP_CLIENT_ID__, __COGNITO_DOMAIN__,
# __AWS_REGION__) without false-positiving on other double-underscore text.
PLACEHOLDER_RE='__(COGNITO|AWS)_[A-Z0-9_]+__'
if grep -Eq "$PLACEHOLDER_RE" "$INDEX"; then
  echo "[deploy-web] ERROR: deploy-time placeholder not replaced in index.html"
  grep -Eo "$PLACEHOLDER_RE" "$INDEX" | sort -u || true
  exit 1
fi

echo "[deploy-web] uploading frontend to S3..."
STATIC_BUCKET="${ZIMG_PREFIX}-web-static-${CDK_DEFAULT_ACCOUNT}-${CDK_DEFAULT_REGION}"
aws s3 sync web/frontend/dist/ "s3://${STATIC_BUCKET}/" --delete

echo "[deploy-web] invalidating CloudFront cache..."
WEB_URL=$(python3 -c "import json; print(json.load(open('cdk-outputs.json'))['${ZIMG_PREFIX}-web']['WebUrl'])" 2>/dev/null || true)
if [ -n "$WEB_URL" ]; then
  DOMAIN=$(echo "$WEB_URL" | sed 's|https://||')
  DIST_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName==\`${DOMAIN}\`].Id|[0]" --output text 2>/dev/null || true)
  if [ -n "$DIST_ID" ] && [ "$DIST_ID" != "None" ]; then
    aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" --query 'Invalidation.Id' --output text
    echo "[deploy-web] cache invalidated: $DIST_ID"
  fi
fi

echo "[deploy-web] done"
