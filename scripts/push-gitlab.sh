#!/usr/bin/env bash
set -euo pipefail

# Push to gitlab with docs/*.md included (they are tracked by default).
# Usage: ./scripts/push-gitlab.sh [branch]  (default: master)

BRANCH="${1:-master}"
REMOTE="gitlab"

git push --force --no-verify "$REMOTE" "$BRANCH"

echo "Pushed $BRANCH to $REMOTE (with docs/*.md)"
