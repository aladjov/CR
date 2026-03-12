#!/usr/bin/env bash
set -euo pipefail

# Push to GitHub (origin) WITHOUT docs-level *.md files (internal docs).
# docs/wiki/*.md ARE included — they are the public wiki.
# Usage: ./scripts/push-github.sh [branch]  (default: master)

BRANCH="${1:-master}"
REMOTE="origin"
WORK=$(mktemp -d)
trap 'git worktree remove --force "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

git worktree add --detach "$WORK" "$BRANCH" --quiet
pushd "$WORK" > /dev/null

# Remove only docs-level .md files (NOT docs/wiki/*.md or other subdirs).
# IMPORTANT: Do NOT use `git rm --cached docs/*.md` — when no files match the
# shell glob, bash passes the literal "docs/*.md" to git, which interprets it
# as a recursive pathspec and nukes docs/wiki/*.md too.
git ls-files -- 'docs/' | grep '^docs/[^/]*\.md$' | while IFS= read -r f; do
    git rm --cached "$f" --quiet
done

# Add a temporary docs/.gitignore so these .md files stay excluded.
# The leading / anchors to docs/ — does NOT affect docs/wiki/*.md.
echo '/*.md' > docs/.gitignore
git add docs/.gitignore

git commit --amend --no-edit --no-verify --quiet

git push --force "$REMOTE" HEAD:"$BRANCH"

popd > /dev/null

# Update tracking ref so git doesn't report diverged branches.
git update-ref "refs/remotes/$REMOTE/$BRANCH" "refs/heads/$BRANCH"

echo "Pushed $BRANCH to $REMOTE (without docs/*.md, with docs/wiki/*.md)"
