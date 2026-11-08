#!/usr/bin/env bash
set -euo pipefail

# Always run the SFN. The state machine has its own CheckCache → AllCached
# fast-path (lambda head_objects + zip-magic verify) and reaches WarmupCache
# + VerifyNEFFs + PurgeStaleCache afterwards. A script-level shortcut here
# would let an incomplete prior run (e.g. NEFFs uploaded but warmup never
# ran) leave the cache + verify steps unexecuted.

echo "[compile-models] === start $(date) ==="
ACCT=$(aws sts get-caller-identity --query Account --output text)
PREFIX="${ZIMG_PREFIX:-dev-zimg}"
SFN_ARN="arn:aws:states:${AWS_REGION}:${ACCT}:stateMachine:${PREFIX}-compile-sfn"
echo "[compile-models] sfn=$SFN_ARN"

EXEC_NAME="cicd-$(date +%Y%m%d-%H%M%S)"
EXEC_ARN=$(aws stepfunctions start-execution \
  --state-machine-arn "$SFN_ARN" \
  --name "$EXEC_NAME" \
  --input '{}' \
  --query executionArn --output text)
echo "[compile-models] started: $EXEC_ARN"

for i in $(seq 1 180); do
  sleep 60
  STATUS=$(aws stepfunctions describe-execution --execution-arn "$EXEC_ARN" --query status --output text)
  echo "[$i/180] $(date +%H:%M:%S) status=$STATUS"
  case "$STATUS" in
    SUCCEEDED)
      echo "[compile-models] SFN succeeded"
      exit 0
      ;;
    FAILED|TIMED_OUT|ABORTED)
      echo "[compile-models] SFN status=$STATUS"
      aws stepfunctions describe-execution --execution-arn "$EXEC_ARN" --output json
      exit 1
      ;;
  esac
done

echo "[compile-models] timed out waiting for SFN (180 min)"
exit 1
