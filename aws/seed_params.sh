#!/usr/bin/env bash
# Seed the SSM parameters a stack needs. Run once per environment.
#   ./seed_params.sh dev
# Secrets are SecureString; the rest are plain so you can read them at a glance.
set -euo pipefail
ENV="${1:?usage: seed_params.sh <dev|prod>}"
P="/alpaca/${ENV}"

put()  { aws ssm put-parameter --name "$P/$1" --value "$2" --type String       --overwrite >/dev/null; echo "  $P/$1 = $2"; }
sec()  { aws ssm put-parameter --name "$P/$1" --value "$2" --type SecureString --overwrite >/dev/null; echo "  $P/$1 = ****"; }

echo "Seeding $P"
sec alpaca_key     "${ALPACA_KEY:?set ALPACA_KEY}"
sec alpaca_secret  "${ALPACA_SECRET:?set ALPACA_SECRET}"
put ntfy_alerts    "${NTFY_ALERTS:?set NTFY_ALERTS}"
sec ntfy_replies   "${NTFY_REPLIES:?set NTFY_REPLIES}"
put symbols        "${SYMBOLS:-SPY,QQQ,IWM}"
put equity         "${EQUITY:-25000}"
put risk_frac      "${RISK_FRAC:-0.02}"
put max_contracts  "${MAX_CONTRACTS:-2}"
put threshold      "${THRESHOLD:-0.15}"
# Both default OFF. Arming is always a separate, deliberate act.
put armed          "off"
put live_money     "off"
put kill           "off"
echo "Done. Stack starts disarmed and on paper."
