#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
INIT_PY="$REPO_ROOT/src/customer_retention/__init__.py"

usage() {
    echo "Usage: $0 <version>"
    echo "  e.g.  $0 0.75.1a2"
    echo "        $0 v0.75.1a2   (leading 'v' is stripped for file versions)"
    exit 1
}

[[ $# -ne 1 ]] && usage

# Strip leading 'v' for the version string used in files
VERSION="${1#v}"
TAG="v${VERSION}"

echo "==> Releasing ${TAG}  (file version: ${VERSION})"

# --- 1. Update version in pyproject.toml and __init__.py ---------------
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" "$PYPROJECT"
sed -i '' "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" "$INIT_PY"

echo "    pyproject.toml  -> $(grep '^version' "$PYPROJECT")"
echo "    __init__.py     -> $(grep '^__version__' "$INIT_PY")"

# --- 2. Commit the version bump ----------------------------------------
git -C "$REPO_ROOT" add "$PYPROJECT" "$INIT_PY"
git -C "$REPO_ROOT" commit -m "Bump version to ${VERSION}"

# --- 3. Tag the commit --------------------------------------------------
git -C "$REPO_ROOT" tag -a "$TAG" -m "Release ${TAG}"
echo "    Tagged: ${TAG}"

# --- 4. Build -----------------------------------------------------------
echo "==> Building sdist + wheel"
rm -rf "$REPO_ROOT/dist"
uvx --from build pyproject-build "$REPO_ROOT" --outdir "$REPO_ROOT/dist"

# --- 5. Publish to PyPI -------------------------------------------------
echo "==> Uploading to PyPI"
uvx twine upload "$REPO_ROOT/dist/"*

echo "==> Done. ${TAG} published to PyPI."
