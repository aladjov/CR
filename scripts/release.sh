#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
INIT_PY="$REPO_ROOT/src/customer_retention/__init__.py"
REMOTE="${CR_GIT_REMOTE:-origin}"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

usage() {
    echo "Usage: $0 [--push] <version>"
    echo ""
    echo "  $0 0.99.7a1          Build, publish to PyPI, push commit + tag"
    echo "  $0 --push            Push current branch (no version bump, no release)"
    echo "  $0 v0.99.7a1         Leading 'v' is stripped for file versions"
    exit 1
}

push_only() {
    echo "==> Pushing ${BRANCH} to ${REMOTE}"
    git -C "$REPO_ROOT" push "$REMOTE" "$BRANCH"
    echo "==> Done."
    exit 0
}

[[ $# -lt 1 ]] && usage
[[ "$1" == "--push" ]] && push_only

VERSION="${1#v}"
TAG="v${VERSION}"

echo "==> Releasing ${TAG}  (file version: ${VERSION})"

# --- 1. Strip notebook outputs (keeps package small) --------------------
echo "==> Cleaning notebook outputs"
python "$REPO_ROOT/scripts/notebooks/clean_notebook_outputs.py" \
    --notebooks-dir "$REPO_ROOT/exploration_notebooks"

# --- 2. Update version in pyproject.toml and __init__.py ---------------
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" "$PYPROJECT"
sed -i '' "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" "$INIT_PY"

echo "    pyproject.toml  -> $(grep '^version' "$PYPROJECT")"
echo "    __init__.py     -> $(grep '^__version__' "$INIT_PY")"

# --- 3. Run tests on bumped version before committing ------------------
echo "==> Running tests"
python -m pytest "$REPO_ROOT/tests" -x -q --timeout=120 || {
    echo "!!! Tests failed — reverting version bump"
    git -C "$REPO_ROOT" checkout -- "$PYPROJECT" "$INIT_PY"
    exit 1
}

# --- 4. Commit the version bump ----------------------------------------
git -C "$REPO_ROOT" add "$PYPROJECT" "$INIT_PY" "$REPO_ROOT/exploration_notebooks/"
if git -C "$REPO_ROOT" diff --cached --quiet; then
    echo "    No changes to commit (version already at ${VERSION})"
else
    git -C "$REPO_ROOT" commit -m "Bump version to ${VERSION}"
fi

# --- 5. Tag the commit --------------------------------------------------
if git -C "$REPO_ROOT" rev-parse "$TAG" >/dev/null 2>&1; then
    echo "    Tag ${TAG} already exists — skipping"
else
    git -C "$REPO_ROOT" tag -a "$TAG" -m "Release ${TAG}"
    echo "    Tagged: ${TAG}"
fi

# --- 6. Build -----------------------------------------------------------
echo "==> Building sdist + wheel"
rm -rf "$REPO_ROOT/dist"
uv build "$REPO_ROOT" --out-dir "$REPO_ROOT/dist"

# --- 7. Publish to PyPI -------------------------------------------------
echo "==> Uploading to PyPI"
uvx twine upload "$REPO_ROOT/dist/"*

# --- 8. Push commit + tag -----------------------------------------------
echo "==> Pushing ${BRANCH} and ${TAG} to ${REMOTE}"
git -C "$REPO_ROOT" push "$REMOTE" "$BRANCH"
git -C "$REPO_ROOT" push "$REMOTE" "$TAG"

echo "==> Done. ${TAG} published to PyPI and pushed to ${REMOTE}."
