#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
INIT_PY="$REPO_ROOT/src/customer_retention/__init__.py"
REMOTE="${CR_GIT_REMOTE:-origin}"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
EXPECTED_BRANCH=""
ASSUME_YES=0

usage() {
    echo "Usage: $0 [--push] [--branch <name>] [-y|--yes] <version>"
    echo ""
    echo "  $0 0.99.7a1                        Build, publish to PyPI, push commit + tag"
    echo "                                     from whatever branch is checked out."
    echo "  $0 --branch master 0.99.7a1        Abort unless current branch is 'master'."
    echo "  $0 --branch user-extensions/release-a 1.01.3a3"
    echo "                                     Release from a feature branch (e.g.,"
    echo "                                     for a deploy-from-branch workflow)."
    echo "  $0 -y 0.99.7a1                     Skip the pre-release confirmation prompt."
    echo "  $0 --push                          Push current branch (no version bump)."
    echo "  $0 v0.99.7a1                       Leading 'v' is stripped for file versions."
    echo ""
    echo "Env vars:"
    echo "  CR_GIT_REMOTE        Git remote to push to (default: origin)."
    echo "  CR_RELEASE_BRANCH    Same as --branch (flag wins if both set)."
    echo "  CR_ASSUME_YES        Set to 1 to skip the confirmation prompt."
    exit 1
}

push_only() {
    echo "==> Pushing ${BRANCH} to ${REMOTE}"
    git -C "$REPO_ROOT" push "$REMOTE" "$BRANCH"
    echo "==> Done."
    exit 0
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            [[ $# -lt 2 ]] && { echo "error: --branch requires an argument"; exit 1; }
            EXPECTED_BRANCH="$2"
            shift 2
            ;;
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        --push)
            push_only
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

[[ $# -lt 1 ]] && usage

EXPECTED_BRANCH="${EXPECTED_BRANCH:-${CR_RELEASE_BRANCH:-}}"
ASSUME_YES="${ASSUME_YES:-${CR_ASSUME_YES:-0}}"

VERSION="${1#v}"
TAG="v${VERSION}"

if [[ -n "$EXPECTED_BRANCH" && "$EXPECTED_BRANCH" != "$BRANCH" ]]; then
    echo "error: --branch expected '${EXPECTED_BRANCH}' but HEAD is '${BRANCH}'"
    echo "       switch first:  git checkout ${EXPECTED_BRANCH}"
    exit 1
fi

cat <<EOF
==> Release plan
    version:  ${VERSION}
    tag:      ${TAG}
    branch:   ${BRANCH}
    remote:   ${REMOTE}
    pypi:     yes (twine upload dist/*)
EOF

if [[ "$ASSUME_YES" != "1" ]]; then
    read -rp "Proceed? [y/N] " _confirm
    case "$_confirm" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

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
uvx twine upload --verbose "$REPO_ROOT/dist/"*

# --- 8. Push commit + tag -----------------------------------------------
echo "==> Pushing ${BRANCH} and ${TAG} to ${REMOTE}"
git -C "$REPO_ROOT" push "$REMOTE" "$BRANCH"
git -C "$REPO_ROOT" push "$REMOTE" "$TAG"

echo "==> Done. ${TAG} published to PyPI and pushed to ${REMOTE}."
