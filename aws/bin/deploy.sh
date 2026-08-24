#!/usr/bin/env bash
# Build and deploy, aborting on the first failure.
#
# Exists because three separate deploy failures got past a human eye, and each
# one reported success somewhere:
#
#   1. `sam deploy` shipped a stale image. CloudFormation said UPDATE_COMPLETE
#      and every Lambda showed a fresh LastModified. Nothing was wrong except
#      the code.
#   2. A String parameter "false" passed to SAM's boolean `Enabled` coerced to
#      true, so schedules came up ENABLED while describe-stacks displayed
#      `false`.
#   3. `sam build` failed because Docker was not running, left .aws-sam/
#      touched, and the freshness guard then reported "build is current"
#      against artifacts that did not exist.
#
# The common thread is that checking for a bad state after the fact is
# unreliable. Fail fast instead, and verify behaviour rather than status.
set -euo pipefail

ENVNAME="${1:-dev}"
PROFILE="${AWS_PROFILE:-alpaca}"
REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$ENVNAME" in
  dev|prod) ;;
  *) echo "usage: $0 [dev|prod]   (there is no 'default' config-env)"; exit 2 ;;
esac

echo "==> docker"
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop first."; exit 1; }

echo "==> tests"
.venv/bin/python test_suite.py 2>&1 | tail -1

echo "==> sam build"
( cd aws && sam build --profile "$PROFILE" --region "$REGION" ) || {
  echo "BUILD FAILED -- not deploying."; exit 1; }

echo "==> freshness"
./aws/bin/check-build-fresh.sh || exit 1

echo "==> deploy $ENVNAME"
( cd aws && sam deploy --config-env "$ENVNAME" --profile "$PROFILE" \
    --region "$REGION" --no-confirm-changeset )

echo "==> verify"
aws cloudformation describe-stacks --profile "$PROFILE" --region "$REGION" \
  --stack-name "alpaca-alerts-$ENVNAME" \
  --query 'Stacks[0].[StackStatus,LastUpdatedTime]' --output text

want=ENABLED; [ "$ENVNAME" = prod ] && want=DISABLED
got=$(aws events list-rules --profile "$PROFILE" --region "$REGION" \
      --query "length(Rules[?contains(Name,'alpaca-alerts-$ENVNAME') && State=='$want'])" --output text)
tot=$(aws events list-rules --profile "$PROFILE" --region "$REGION" \
      --query "length(Rules[?contains(Name,'alpaca-alerts-$ENVNAME')])" --output text)
echo "rules $want: $got/$tot"
[ "$got" = "$tot" ] || { echo "RULE STATE WRONG -- expected all $want"; exit 1; }

# Smoke-test that the code actually imports and runs. UPDATE_COMPLETE means
# CloudFormation swapped an image, not that the image works: gamma.py was once
# imported by handlers.py but missing from the Dockerfile COPY list, and every
# invocation died on ImportModuleError while the deploy reported success.
echo "==> smoke test"
for fn in refresh scan; do
  out=$(mktemp)
  aws lambda invoke --profile "$PROFILE" --region "$REGION" \
      --function-name "alpaca-$fn-$ENVNAME" --payload '{}' \
      --cli-binary-format raw-in-base64-out "$out" >/dev/null 2>&1 || true
  if grep -q '"errorType"' "$out" 2>/dev/null; then
    echo "  $fn FAILED:"; head -c 300 "$out"; echo; rm -f "$out"; exit 1
  fi
  echo "  $fn ok"
  rm -f "$out"
done

for p in armed live_money kill auto_accept; do
  v=$(aws ssm get-parameter --profile "$PROFILE" --region "$REGION" \
      --name "/alpaca/$ENVNAME/$p" --query Parameter.Value --output text 2>/dev/null || echo "-")
  printf "  %-12s %s\n" "$p" "$v"
done
echo "OK"
