#!/bin/sh
# Refuse to deploy an image older than the source it claims to contain.
cd "$(dirname "$0")/../.." || exit 1
B=aws/.aws-sam/build.toml
[ -f "$B" ] || { echo "no build found -- run 'sam build' first"; exit 1; }
STALE=""
for f in *.py aws/*.py; do
  [ -f "$f" ] || continue
  [ "$f" -nt "$B" ] && STALE="$STALE $f"
done
if [ -n "$STALE" ]; then
  echo "STALE BUILD -- these are newer than the last sam build:"
  for f in $STALE; do echo "   $f"; done
  echo "Run 'sam build' before deploying."
  exit 1
fi
echo "build is current"
