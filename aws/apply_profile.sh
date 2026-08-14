#!/usr/bin/env bash
# Apply a sizing profile to an environment.
#   ./apply_profile.sh dev  measure
#   ./apply_profile.sh prod small-2500
#
# Only touches sizing and strategy shape. Never touches credentials, and never
# touches armed / live_money / kill -- arming is always a separate deliberate
# act, and a profile change must not be able to start trading as a side effect.
#
# No associative arrays: macOS ships bash 3.2, which predates them. The profile
# keys are just the uppercase form of the SSM parameter names, so a lowercase
# translation is all the mapping that was ever needed.
set -euo pipefail
ENV="${1:-}"; PROF="${2:-}"
if [ -z "$ENV" ] || [ -z "$PROF" ]; then
  echo "usage: apply_profile.sh <dev|prod> <profile>"
  echo "profiles:"; ls -1 "$(dirname "$0")/profiles" | sed 's/\.env$//;s/^/  /'; exit 1
fi
F="$(dirname "$0")/profiles/${PROF}.env"
[ -f "$F" ] || { echo "no such profile: $PROF"; ls -1 "$(dirname "$0")/profiles" | sed 's/\.env$//;s/^/  /'; exit 1; }

echo "applying '$PROF' to /alpaca/$ENV"
n=0
while IFS='=' read -r k v || [ -n "$k" ]; do
  case "$k" in ''|\#*) continue;; esac
  k="$(echo "$k" | tr -d '[:space:]')"
  v="$(echo "$v" | tr -d '[:space:]')"
  [ -z "$v" ] && continue
  param="$(echo "$k" | tr '[:upper:]' '[:lower:]')"
  aws ssm put-parameter --name "/alpaca/$ENV/$param" --value "$v" \
    --type String --overwrite --region us-east-1 >/dev/null
  printf "  %-22s %s\n" "$param" "$v"
  n=$((n+1))
done < "$F"
echo "  ($n parameters set)"

echo
echo "unchanged (deliberately):"
for p in armed live_money kill; do
  printf "  %-22s %s\n" "$p" \
    "$(aws ssm get-parameter --name "/alpaca/$ENV/$p" --region us-east-1 --query Parameter.Value --output text 2>/dev/null || echo '?')"
done
