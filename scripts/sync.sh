#!/usr/bin/env bash
# Pull latest Substack posts into the site and publish.
# Run from anywhere: bash scripts/sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/sync_substack.py

if [ -n "$(git status --porcelain posts.json)" ]; then
  git add posts.json
  git commit -m "Sync Substack posts"
  git push
  echo "Synced + pushed. Live in ~1 min."
else
  echo "No changes — already up to date."
fi
