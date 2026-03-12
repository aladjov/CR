#!/usr/bin/env bash
set -euo pipefail

# Push to GitHub (origin) WITHOUT docs/*.md files.
# Adds a temporary docs/.gitignore in a worktree so markdown docs stay private.
# Usage: ./scripts/push-github.sh [branch]  (default: master)

BRANCH="${1:-master}"
REMOTE="origin"
WORK=$(mktemp -d)
trap 'git worktree remove --force "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

git worktree add --detach "$WORK" "$BRANCH" --quiet
pushd "$WORK" > /dev/null

# Add docs-level gitignore to exclude *.md from this push
echo '/*.md' > docs/.gitignore
git add docs/.gitignore
git rm --cached docs/*.md 2>/dev/null || true
git commit --amend --no-edit --quiet

git push --force --no-verify "$REMOTE" HEAD:"$BRANCH"

popd > /dev/null
echo "Pushed $BRANCH to $REMOTE (without docs/*.md)"
