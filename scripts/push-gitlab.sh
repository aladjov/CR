#!/usr/bin/env bash
set -euo pipefail

# Push to gitlab with docs/*.md included (they are gitignored for origin).
# Uses a temporary worktree so the working tree is never touched.
# Usage: ./scripts/push-gitlab.sh [branch]  (default: master)

BRANCH="${1:-master}"
REMOTE="gitlab"
WORK=$(mktemp -d)
trap 'git worktree remove --force "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

git worktree add --detach "$WORK" "$BRANCH" --quiet
pushd "$WORK" > /dev/null

# Remove the docs-level gitignore so *.md files are included
rm -f docs/.gitignore
git add docs/.gitignore
git add docs/*.md 2>/dev/null || true
git commit --amend --no-edit --quiet

git push --force-with-lease "$REMOTE" HEAD:"$BRANCH"

popd > /dev/null
echo "Pushed $BRANCH to $REMOTE (with docs/*.md)"
