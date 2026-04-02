#!/usr/bin/env bash
# ============================================================================
# compare_versions.sh — Side-by-side exploration run: old commit vs HEAD
#
# Orchestrates:
#   1. git worktree for the old commit
#   2. Isolated virtualenvs with Jupyter kernels
#   3. Copy notebooks to comparison dir + patch dataset (absolute paths, no backup)
#   4-5. run_exploration.py on copies (CR_EXPERIMENTS_DIR routes artefacts)
#   6. Cell-output capture + drift report
#   7. Cleanup
#
# The default old commit (6209ecd) is the one that generated the retail-churn
# tutorial HTML on Feb 15, 2026 — "Updated retail-churn html tutorial".
#
# Usage:
#   bash scripts/experiments/compare_versions.sh
#   bash scripts/experiments/compare_versions.sh --old-commit abc1234
#   bash scripts/experiments/compare_versions.sh --spark-remote
#   bash scripts/experiments/compare_versions.sh --capture-only
#   bash scripts/experiments/compare_versions.sh --dry-run
#   bash scripts/experiments/compare_versions.sh --cleanup
#   bash scripts/experiments/compare_versions.sh --list-candidates
# ============================================================================

set -euo pipefail

# ---- Defaults ----
OLD_COMMIT="6209ecd"

NEW_COMMIT=""  # empty = HEAD
TIMEOUT=36000
SPARK_REMOTE=""
CAPTURE_ONLY=false
DRY_RUN=false
LIST_CANDIDATES=false
KEEP_WORKTREE=false
CLEANUP=false
DATASET="customer_emails"
RUN_OLD=true
RUN_NEW=true
START_NOTEBOOK=""
OLD_DATABRICKS_RUN=""
NEW_DATABRICKS_RUN=""

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts/experiments"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --old-commit)   OLD_COMMIT="$2"; shift 2 ;;
        --new-commit)   NEW_COMMIT="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --spark-remote) SPARK_REMOTE="--spark-remote"; shift ;;
        --capture-only) CAPTURE_ONLY=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --keep-worktree) KEEP_WORKTREE=true; shift ;;
        --cleanup)      CLEANUP=true; shift ;;
        --old-only)     RUN_OLD=true; RUN_NEW=false; shift ;;
        --new-only)     RUN_OLD=false; RUN_NEW=true; shift ;;
        --start-notebook) START_NOTEBOOK="$2"; shift 2 ;;
        --old-databricks-run) OLD_DATABRICKS_RUN="$2"; shift 2 ;;
        --new-databricks-run) NEW_DATABRICKS_RUN="$2"; shift 2 ;;
        --list-candidates)
            LIST_CANDIDATES=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --old-commit HASH   Baseline commit (default: 6209ecd = tutorial HTML)"
            echo "  --new-commit HASH   Comparison commit (default: HEAD)"
            echo "  --timeout SECS      Per-notebook timeout (default: 36000 = 10h)"
            echo "  --spark-remote      Enable Databricks Connect for new-code run"
            echo "  --capture-only      Skip execution, just re-capture and compare"
            echo "  --dry-run           Show what would be done without executing"
            echo "  --dataset NAME      Dataset to activate (default: customer_emails)"
            echo "  --old-only          Only run the old (baseline) side"
            echo "  --new-only          Only run the new (HEAD) side"
            echo "  --start-notebook NB Resume from this notebook (e.g. 03 or 08_baseline_experiments)"
            echo "  --keep-worktree     Keep the old-code worktree after completion"
            echo "  --cleanup           Remove comparison dir, worktree, and kernels for the given commits"
            echo "  --list-candidates   Show candidate commits near the tutorial date"
            echo "  --old-databricks-run RUN_ID  Use Databricks job run for old side (skip local execution)"
            echo "  --new-databricks-run RUN_ID  Use Databricks job run for new side (skip local execution)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Helper: format a commit with version + date ----
commit_info() {
    # Usage: commit_info <commit-ish>
    # Output: "6209ecd  v0.78.0a6  Feb 15, 2026  Updated retail-churn html tutorial"
    local sha date_str version subject
    sha="$(git -C "$REPO_DIR" rev-parse --short "$1" 2>/dev/null)" || return 1
    date_str="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' "$sha")"
    subject="$(git -C "$REPO_DIR" log -1 --format='%s' "$sha")"
    version="$(git -C "$REPO_DIR" show "$sha":pyproject.toml 2>/dev/null | grep '^version' | head -1 | sed 's/version = "//;s/"//' || echo '?')"
    printf "  %-9s  v%-12s  %-14s  %s\n" "$sha" "$version" "$date_str" "$subject"
}

# ---- List candidate commits ----
if $LIST_CANDIDATES; then
    _HEADER="  COMMIT     VERSION        DATE            MESSAGE"
    _SEP="  ─────────  ────────────   ──────────────  ──────────────────────────────────"

    echo "=== Default baseline (tutorial commit) ==="
    echo ""
    echo "$_HEADER"
    echo "$_SEP"
    commit_info 6209ecd
    echo ""

    echo "=== Current HEAD ==="
    echo ""
    echo "$_HEADER"
    echo "$_SEP"
    commit_info HEAD
    echo ""

    echo "=== All version releases (newest first) ==="
    echo ""
    echo "$_HEADER"
    echo "$_SEP"
    # Show every version-bump commit sorted by date (git log is already date-descending).
    # Insert a marker showing where the tutorial commit sits chronologically.
    _TUTORIAL_EPOCH="$(git -C "$REPO_DIR" log -1 --format='%at' 6209ecd 2>/dev/null || echo '0')"
    _marker_shown=false
    _count=0
    for sha in $(git -C "$REPO_DIR" log --format='%h' --grep="Bump version"); do
        _commit_epoch="$(git -C "$REPO_DIR" log -1 --format='%at' "$sha")"
        if ! $_marker_shown && [ "$_commit_epoch" -le "$_TUTORIAL_EPOCH" ]; then
            echo "  ▸▸▸▸▸▸▸▸▸  tutorial HTML generated here (6209ecd)  ◂◂◂◂◂◂◂◂◂◂◂"
            _marker_shown=true
        fi
        commit_info "$sha"
        _count=$((_count + 1))
    done
    if ! $_marker_shown; then
        echo "  ▸▸▸▸▸▸▸▸▸  tutorial HTML generated here (6209ecd)  ◂◂◂◂◂◂◂◂◂◂◂"
    fi
    echo ""
    echo "  Total: $_count version releases"
    echo ""
    echo "Tip: use --old-commit <COMMIT> to pick any of these as baseline."
    exit 0
fi

# ---- Resolve commits ----
OLD_SHORT="$(echo "$OLD_COMMIT" | cut -c1-7)"
if [ -n "$NEW_COMMIT" ]; then
    NEW_SHORT="$(echo "$NEW_COMMIT" | cut -c1-7)"
    NEW_IS_HEAD=false
else
    NEW_COMMIT="HEAD"
    NEW_SHORT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
    NEW_IS_HEAD=true
fi

if $CLEANUP; then
    NB00="exploration_notebooks/00_start_here.ipynb"
    COMPARISONS_ROOT="$REPO_DIR/experiments/comparisons"

    # Find all comparison dirs matching this old commit (HEAD may have moved since creation)
    _found=()
    if [ -d "$COMPARISONS_ROOT" ]; then
        for d in "$COMPARISONS_ROOT"/${OLD_SHORT}_vs_*/; do
            [ -d "$d" ] && _found+=("$d")
        done
    fi

    if [ ${#_found[@]} -eq 0 ]; then
        echo "Nothing to clean — no comparison dirs matching ${OLD_SHORT}_vs_* found."
        exit 0
    fi

    for COMPARISON_DIR in "${_found[@]}"; do
        COMPARISON_DIR="${COMPARISON_DIR%/}"  # strip trailing slash
        echo "Cleaning up: $(basename "$COMPARISON_DIR")"
        # Remove any worktrees (old side always has one; new side has one when --new-commit is used)
        for wt in "$COMPARISON_DIR"/worktree-*; do
            if [ -d "$wt" ]; then
                echo "  Removing worktree $wt ..."
                git -C "$REPO_DIR" worktree remove "$wt" --force 2>/dev/null || true
            fi
        done
        # Legacy: single "worktree" dir from older runs
        OLD_DIR="$COMPARISON_DIR/worktree"
        if [ -d "$OLD_DIR" ]; then
            echo "  Removing worktree $OLD_DIR ..."
            git -C "$REPO_DIR" worktree remove "$OLD_DIR" --force 2>/dev/null || true
        fi
        echo "  Removing $COMPARISON_DIR ..."
        rm -rf "$COMPARISON_DIR"
    done

    jupyter kernelspec remove cr-baseline -y 2>/dev/null || true
    jupyter kernelspec remove cr-current -y 2>/dev/null || true

    # Prune stale git worktree references left behind by removed dirs
    echo "  Pruning stale git worktrees ..."
    git -C "$REPO_DIR" worktree prune 2>/dev/null || true

    echo "  Done."
    exit 0
fi

# ---- Derived paths ----
NB00="exploration_notebooks/00_start_here.ipynb"

# Everything lives under one comparison folder — easy cleanup with rm -rf
COMPARISON_DIR="$REPO_DIR/experiments/comparisons/${OLD_SHORT}_vs_${NEW_SHORT}"
OLD_DIR="$COMPARISON_DIR/worktree-old"    # git worktree for old commit
NEW_DIR="$COMPARISON_DIR/worktree-new"    # git worktree for new commit (only when --new-commit)
OLD_OUTPUT_DIR="$COMPARISON_DIR/old-$OLD_SHORT"
NEW_OUTPUT_DIR="$COMPARISON_DIR/new-$NEW_SHORT"
REPORT_FILE="$COMPARISON_DIR/drift_report.md"

OLD_RUN_ID="baseline-${OLD_SHORT}"
NEW_RUN_ID="current-${NEW_SHORT}"

# When --new-commit is HEAD, use the repo directly; otherwise use a worktree
if $NEW_IS_HEAD; then
    NEW_WORK_DIR="$REPO_DIR"
else
    NEW_WORK_DIR="$NEW_DIR"
fi

# The old side's actual run_id may differ (older initialize_run ignores CR_RUN_ID)
OLD_ACTUAL_RUN_ID=""

# Resolve human-readable info for both commits
OLD_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' "$OLD_COMMIT" 2>/dev/null || echo '?')"
OLD_VERSION="$(git -C "$REPO_DIR" show "$OLD_COMMIT":pyproject.toml 2>/dev/null | grep '^version' | head -1 | sed 's/version = "//;s/"//' || echo '?')"
NEW_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' "$NEW_COMMIT" 2>/dev/null || echo '?')"
NEW_VERSION="$(git -C "$REPO_DIR" show "$NEW_COMMIT":pyproject.toml 2>/dev/null | grep '^version' | head -1 | sed 's/version = "//;s/"//' || echo '?')"

echo "============================================================"
echo "  Exploration Drift Comparison"
echo ""
if [ -n "$OLD_DATABRICKS_RUN" ]; then
    echo "  Old: Databricks run $OLD_DATABRICKS_RUN"
else
    echo "  Old: $OLD_SHORT  v$OLD_VERSION  ($OLD_DATE)"
fi
if [ -n "$NEW_DATABRICKS_RUN" ]; then
    echo "  New: Databricks run $NEW_DATABRICKS_RUN"
else
    echo "  New: $NEW_SHORT  v$NEW_VERSION  ($NEW_DATE)"
fi
echo ""
echo "  Dataset:   $DATASET"
echo "  Output:    $COMPARISON_DIR/"
echo "============================================================"
echo ""

# ---- Running error log ----
# Write run parameters to the comparison dir log file.
# NOTE: we do NOT use `exec > >(tee ...)` because process substitution pipes
# break papermill's ZMQ kernel communication (hangs at 0% CPU).
# Instead we append to the log explicitly; run_exploration.log from each side
# captures per-notebook error details.
mkdir -p "$COMPARISON_DIR"
LOG_FILE="$COMPARISON_DIR/compare_versions.log"
_log() { echo "$*" | tee -a "$LOG_FILE"; }

_log "compare_versions — $(date '+%Y-%m-%d %H:%M:%S')"
_log ""
_log "  old_commit:   $OLD_COMMIT ($OLD_SHORT, v$OLD_VERSION)"
_log "  new_commit:   HEAD ($NEW_SHORT, v$NEW_VERSION)"
_log "  dataset:      $DATASET"
_log "  timeout:      $TIMEOUT"
_log "  spark_remote: ${SPARK_REMOTE:-no}"
_log "  start_nb:     ${START_NOTEBOOK:-<from beginning>}"
_log ""

# ---- Step 1: Create worktrees ----
_create_worktree() {
    local dir="$1" commit="$2" label="$3"
    if $DRY_RUN; then
        echo "  DRY_RUN: git worktree add $dir $commit --detach"
    elif [ -d "$dir" ]; then
        echo "  $label worktree already exists — reusing"
    else
        git -C "$REPO_DIR" worktree add "$dir" "$commit" --detach
        echo "  Created $label worktree at $dir"
    fi
}

mkdir -p "$COMPARISON_DIR"
if ! $CAPTURE_ONLY; then
    echo "[Step 1] Creating worktrees ..."
    $RUN_OLD && [ -z "$OLD_DATABRICKS_RUN" ] && _create_worktree "$OLD_DIR" "$OLD_COMMIT" "old"
    $RUN_NEW && [ -z "$NEW_DATABRICKS_RUN" ] && ! $NEW_IS_HEAD && _create_worktree "$NEW_DIR" "$NEW_COMMIT" "new"
else
    echo "[Step 1] SKIP — capture-only mode"
fi
echo ""

# ---- Step 2: Virtualenvs + Jupyter kernels ----
echo "[Step 2] Setting up virtualenvs and Jupyter kernels ..."

setup_venv() {
    local dir="$1" name="$2"
    if $DRY_RUN; then
        echo "  DRY_RUN: uv venv + pip install -e '.[dev,ml]' in $dir"
        echo "  DRY_RUN: ipykernel install --name $name"
        return
    fi
    if [ ! -d "$dir/.venv" ]; then
        echo "  Creating venv in $dir ..."
        (cd "$dir" && uv venv .venv)
    fi
    echo "  Installing package in $dir ..."
    # shellcheck disable=SC1091
    (cd "$dir" && source .venv/bin/activate && uv pip install -e ".[dev,ml]" && \
     python -m ipykernel install --user --name "$name" --display-name "CR $name")
}

if ! $CAPTURE_ONLY; then
    $RUN_OLD && setup_venv "$OLD_DIR" "cr-baseline"
    if $RUN_NEW; then
        if $NEW_IS_HEAD && [ -d "$REPO_DIR/.venv" ]; then
            if ! $DRY_RUN; then
                echo "  Using existing venv in $REPO_DIR"
                (cd "$REPO_DIR" && source .venv/bin/activate && \
                 python -m ipykernel install --user --name "cr-current" --display-name "CR current" 2>/dev/null || true)
            fi
        else
            setup_venv "$NEW_WORK_DIR" "cr-current"
        fi
    fi
fi
echo ""

# ---- Step 3: Copy notebooks to commit folders + patch ----
# Running from copies in the comparison dir means:
#   - no backup/restore dance for the repo's NB00
#   - executed notebooks (with outputs) land directly in the comparison dir
#   - everything under experiments/comparisons/ is gitignored already
NB01="exploration_notebooks/01_data_discovery.ipynb"
OLD_NB_DIR="$OLD_OUTPUT_DIR/notebooks"
NEW_NB_DIR="$NEW_OUTPUT_DIR/notebooks"

# Resolve absolute path for dataset fixtures (relative paths break from comparison dir)
_abs_fixture_path() {
    local dir="$1" dataset="$2"
    local p="$dir/tests/fixtures/${dataset}.csv"
    if [ -f "$p" ]; then
        echo "$p"
    else
        echo "$dir/tests/fixtures/${dataset}.csv"
    fi
}

if $RUN_NEW && ! $CAPTURE_ONLY; then
    echo "[Step 3a] Copying + patching new notebooks → $NEW_NB_DIR ..."
    if $DRY_RUN; then
        echo "  DRY_RUN: copy + patch_dataset_config.py --activate $DATASET"
    else
        mkdir -p "$NEW_NB_DIR"
        cp "$NEW_WORK_DIR"/exploration_notebooks/*.ipynb "$NEW_NB_DIR/"
        python "$SCRIPTS_DIR/patch_dataset_config.py" \
            --notebook "$NEW_NB_DIR/00_start_here.ipynb" \
            --activate "$DATASET" \
            --fixture-root "$NEW_WORK_DIR/tests/fixtures"
    fi
else
    echo "[Step 3a] SKIP — not running new side"
fi

if $RUN_OLD && ! $CAPTURE_ONLY; then
    echo "[Step 3b] Copying + patching old notebooks → $OLD_NB_DIR ..."
    if $DRY_RUN; then
        echo "  DRY_RUN: copy + patch_dataset_config.py --activate $DATASET (old)"
    else
        mkdir -p "$OLD_NB_DIR"
        cp "$OLD_DIR"/exploration_notebooks/*.ipynb "$OLD_NB_DIR/"
        # Patch NB00 (datasets = {}) and NB01 (DATA_PATH = "..." in older format)
        _OLD_ALSO_PATCH=""
        if [ -f "$OLD_NB_DIR/01_data_discovery.ipynb" ]; then
            _OLD_ALSO_PATCH="--also-patch $OLD_NB_DIR/01_data_discovery.ipynb"
        fi
        # shellcheck disable=SC2086
        python "$SCRIPTS_DIR/patch_dataset_config.py" \
            --notebook "$OLD_NB_DIR/00_start_here.ipynb" \
            --activate "$DATASET" \
            --fixture-root "$OLD_DIR/tests/fixtures" \
            $_OLD_ALSO_PATCH
    fi
else
    echo "[Step 3b] SKIP — not running old side"
fi
echo ""

# ---------------------------------------------------------------------------
# Build per-side CLI args, handling historical incompatibilities:
#   - --start-notebook: added after 6209ecd; old run_exploration.py rejects it
#   - --run-id:         old initialize_run ignores CR_RUN_ID env, creating its
#                       own namespace; passing --run-id causes a name mismatch
#                       (NB03 can't find project_context.yaml).  Detect by
#                       checking whether the old session.py reads the env var.
# ---------------------------------------------------------------------------
_build_old_extra_args() {
    local args=""
    # Only pass --run-id if old initialize_run actually honours CR_RUN_ID
    if grep -q 'os\.environ\.get.*CR_RUN_ID' \
         "$OLD_DIR/src/customer_retention/analysis/auto_explorer/session.py" 2>/dev/null; then
        args="--run-id $OLD_RUN_ID"
    else
        echo "  (old initialize_run does not honour CR_RUN_ID — omitting --run-id)" >&2
    fi
    # Only pass --start-notebook if the old script accepts it
    if [ -n "$START_NOTEBOOK" ]; then
        if grep -q 'start.notebook' "$OLD_DIR/scripts/notebooks/run_exploration.py" 2>/dev/null; then
            args="$args --start-notebook $START_NOTEBOOK"
        else
            echo "  (old run_exploration.py does not support --start-notebook — skipping)" >&2
        fi
    fi
    echo "$args"
}

# New side always gets the full arg set
_NEW_START_NB_ARG=""
if [ -n "$START_NOTEBOOK" ]; then
    _NEW_START_NB_ARG="--start-notebook $START_NOTEBOOK"
fi

# Helper: discover the actual run_id created by the old side
_discover_old_run_id() {
    local runs_dir="$OLD_OUTPUT_DIR/experiments/runs"
    # 1. sentinel file written by RunNamespace.write_sentinel()
    if [ -f "$runs_dir/.active_run_id" ]; then
        cat "$runs_dir/.active_run_id"
        return
    fi
    # 2. most recently modified run directory
    if [ -d "$runs_dir" ]; then
        # shellcheck disable=SC2012
        ls -1t "$runs_dir" 2>/dev/null | grep -v '^\.' | head -1
        return
    fi
}

# ---- Step 4: Run old version ----
# Notebooks run from the copy in $OLD_NB_DIR.
# CR_EXPERIMENTS_DIR routes RunNamespace artefacts into the comparison dir.
if ! $RUN_OLD; then
    echo "[Step 4] SKIP — new-only mode"
elif [ -n "$OLD_DATABRICKS_RUN" ]; then
    echo "[Step 4] SKIP — using Databricks run $OLD_DATABRICKS_RUN for old side"
elif $CAPTURE_ONLY; then
    echo "[Step 4] SKIP — capture-only mode"
elif $DRY_RUN; then
    echo "[Step 4] Running old version ($OLD_SHORT) ..."
    _OLD_EXTRA_ARGS=$(_build_old_extra_args)
    echo "  DRY_RUN: run_exploration.py --notebooks-dir $OLD_NB_DIR --kernel cr-baseline $_OLD_EXTRA_ARGS"
else
    echo "[Step 4] Running old version ($OLD_SHORT) ..."
    _OLD_EXTRA_ARGS=$(_build_old_extra_args)
    (
        cd "$OLD_DIR"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        export CR_EXPERIMENTS_DIR="$OLD_OUTPUT_DIR/experiments"
        # shellcheck disable=SC2086
        python scripts/notebooks/run_exploration.py \
            --notebooks-dir "$OLD_NB_DIR" \
            --kernel cr-baseline \
            --timeout "$TIMEOUT" \
            $_OLD_EXTRA_ARGS
    ) || _log "  WARNING: Old run had failures (continuing anyway)"

    # Discover the actual run_id (may differ from OLD_RUN_ID for older code)
    OLD_ACTUAL_RUN_ID=$(_discover_old_run_id)
    if [ -z "$OLD_ACTUAL_RUN_ID" ]; then
        OLD_ACTUAL_RUN_ID="$OLD_RUN_ID"
    fi
    _log "  Old run complete.  (run_id: $OLD_ACTUAL_RUN_ID)"
fi
echo ""

# ---- Step 5: Run new version ----
# When --new-commit is a non-HEAD commit, apply the same compatibility
# detection as the old side (it may also lack --start-notebook / --run-id).
_build_new_extra_args() {
    local work_dir="$1" args=""
    if grep -q 'os\.environ\.get.*CR_RUN_ID' \
         "$work_dir/src/customer_retention/analysis/auto_explorer/session.py" 2>/dev/null; then
        args="--run-id $NEW_RUN_ID"
    else
        echo "  (new initialize_run does not honour CR_RUN_ID — omitting --run-id)" >&2
    fi
    if [ -n "$START_NOTEBOOK" ]; then
        if grep -q 'start.notebook' "$work_dir/scripts/notebooks/run_exploration.py" 2>/dev/null; then
            args="$args --start-notebook $START_NOTEBOOK"
        else
            echo "  (new run_exploration.py does not support --start-notebook — skipping)" >&2
        fi
    fi
    echo "$args"
}

if ! $RUN_NEW; then
    echo "[Step 5] SKIP — old-only mode"
elif [ -n "$NEW_DATABRICKS_RUN" ]; then
    echo "[Step 5] SKIP — using Databricks run $NEW_DATABRICKS_RUN for new side"
elif $CAPTURE_ONLY; then
    echo "[Step 5] SKIP — capture-only mode"
elif $DRY_RUN; then
    echo "[Step 5] Running new version ($NEW_SHORT) ..."
    echo "  DRY_RUN: run_exploration.py --notebooks-dir $NEW_NB_DIR --run-id $NEW_RUN_ID --kernel cr-current $SPARK_REMOTE $_NEW_START_NB_ARG"
else
    echo "[Step 5] Running new version ($NEW_SHORT) ..."
    if $NEW_IS_HEAD; then
        _NEW_EXTRA_ARGS="--run-id $NEW_RUN_ID $SPARK_REMOTE $_NEW_START_NB_ARG"
    else
        _NEW_EXTRA_ARGS="$(_build_new_extra_args "$NEW_WORK_DIR") $SPARK_REMOTE"
    fi
    (
        cd "$NEW_WORK_DIR"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        export CR_EXPERIMENTS_DIR="$NEW_OUTPUT_DIR/experiments"
        # shellcheck disable=SC2086
        python scripts/notebooks/run_exploration.py \
            --notebooks-dir "$NEW_NB_DIR" \
            --kernel cr-current \
            --timeout "$TIMEOUT" \
            $_NEW_EXTRA_ARGS
    ) || _log "  WARNING: New run had failures (continuing anyway)"
    _log "  New run complete."
fi
echo ""

# ---- Step 6: Record metadata + collect into comparison dir ----
echo "[Step 6] Recording environment metadata ..."
record_metadata() {
    local output_dir="$1" work_dir="$2" commit_ref="$3"
    if $DRY_RUN; then
        echo "  DRY_RUN: record git sha, version, date, pip freeze → $output_dir"
        return
    fi
    mkdir -p "$output_dir"
    local sha
    sha="$(git -C "$work_dir" rev-parse "$commit_ref")"
    echo "$sha" > "$output_dir/git_sha.txt"
    git -C "$work_dir" log -1 --format='%ad' --date=format:'%b %d, %Y' "$sha" > "$output_dir/commit_date.txt"
    git -C "$work_dir" show "$sha":pyproject.toml 2>/dev/null \
        | grep '^version' | head -1 | sed 's/version = "//;s/"//' > "$output_dir/version.txt"
    (cd "$work_dir" && source .venv/bin/activate && \
     uv pip freeze > "$output_dir/pip_freeze.txt" 2>/dev/null || pip freeze > "$output_dir/pip_freeze.txt" && \
     python --version > "$output_dir/python_version.txt" 2>&1)
    echo "  Recorded: $(cut -c1-7 "$output_dir/git_sha.txt")  v$(cat "$output_dir/version.txt")  $(cat "$output_dir/commit_date.txt")"
}

if ! $CAPTURE_ONLY && ! $DRY_RUN; then
    $RUN_OLD && record_metadata "$OLD_OUTPUT_DIR" "$OLD_DIR" "$OLD_COMMIT"
    $RUN_NEW && record_metadata "$NEW_OUTPUT_DIR" "$NEW_WORK_DIR" "$NEW_COMMIT"
elif $DRY_RUN; then
    $RUN_OLD && record_metadata "$OLD_OUTPUT_DIR" "$OLD_DIR" "$OLD_COMMIT"
    $RUN_NEW && record_metadata "$NEW_OUTPUT_DIR" "$NEW_WORK_DIR" "$NEW_COMMIT"
fi

# Copy run-level YAML artefacts into the comparison dir
copy_run_artefacts() {
    local src_run="$1" dest_dir="$2"
    if [ ! -d "$src_run" ]; then
        echo "  (run dir $src_run not found — skipping artefact copy)"
        return
    fi
    # Copy key YAML files
    for f in project_context.yaml snapshot_grid.yaml exploration_metadata.json; do
        [ -f "$src_run/$f" ] && cp "$src_run/$f" "$dest_dir/"
    done
    # Copy merged findings
    if [ -d "$src_run/merged" ]; then
        mkdir -p "$dest_dir/merged"
        cp "$src_run/merged/"*.yaml "$dest_dir/merged/" 2>/dev/null || true
    fi
    # Copy per-dataset findings
    if [ -d "$src_run/datasets" ]; then
        cp -r "$src_run/datasets" "$dest_dir/datasets" 2>/dev/null || true
    fi
}

if ! $DRY_RUN; then
    # Run artefacts are already in the comparison dir (via CR_EXPERIMENTS_DIR)
    _old_rid="${OLD_ACTUAL_RUN_ID:-$OLD_RUN_ID}"
    OLD_RUN_DIR="$OLD_OUTPUT_DIR/experiments/runs/$_old_rid"
    NEW_RUN_DIR="$NEW_OUTPUT_DIR/experiments/runs/$NEW_RUN_ID"
    $RUN_OLD && copy_run_artefacts "$OLD_RUN_DIR" "$OLD_OUTPUT_DIR"
    $RUN_NEW && copy_run_artefacts "$NEW_RUN_DIR" "$NEW_OUTPUT_DIR"

    # run_exploration.log is written to the notebooks dir (now in comparison dir)
    if $RUN_OLD && [ -f "$OLD_NB_DIR/run_exploration.log" ]; then
        cp "$OLD_NB_DIR/run_exploration.log" "$OLD_OUTPUT_DIR/run_exploration.log"
        _log "  Old run_exploration.log available"
    fi
    if $RUN_NEW && [ -f "$NEW_NB_DIR/run_exploration.log" ]; then
        cp "$NEW_NB_DIR/run_exploration.log" "$NEW_OUTPUT_DIR/run_exploration.log"
        _log "  New run_exploration.log available"
    fi
fi
echo ""

# ---- Step 7: Capture outputs + cell profiles (newest script, both worktrees) ----
echo "[Step 7] Capturing cell outputs and performance profiles ..."

_capture_local() {
    # Usage: _capture_local <notebooks_dir> <output_dir>
    local nb_dir="$1" out_dir="$2"
    python "$SCRIPTS_DIR/capture_notebook_outputs.py" \
        --notebooks-dir "$nb_dir" --output "$out_dir/cell_outputs.md"
    python "$SCRIPTS_DIR/cell_profiling.py" extract \
        --notebooks-dir "$nb_dir" --output "$out_dir/cell_profiles.json"
}

_capture_databricks() {
    # Usage: _capture_databricks <run_id> <output_dir>
    local rid="$1" out_dir="$2"
    mkdir -p "$out_dir"
    python -m customer_retention.integrations.databricks_job_capture \
        --run-id "$rid" \
        --output "$out_dir/cell_outputs.md" \
        --profiles "$out_dir/cell_profiles.json"
}

if $DRY_RUN; then
    $RUN_OLD && echo "  DRY_RUN: capture old → $OLD_OUTPUT_DIR/ (${OLD_DATABRICKS_RUN:+databricks run $OLD_DATABRICKS_RUN}${OLD_DATABRICKS_RUN:-local})"
    $RUN_NEW && echo "  DRY_RUN: capture new → $NEW_OUTPUT_DIR/ (${NEW_DATABRICKS_RUN:+databricks run $NEW_DATABRICKS_RUN}${NEW_DATABRICKS_RUN:-local})"
else
    if $RUN_OLD; then
        if [ -n "$OLD_DATABRICKS_RUN" ]; then
            _capture_databricks "$OLD_DATABRICKS_RUN" "$OLD_OUTPUT_DIR"
        else
            _capture_local "$OLD_NB_DIR" "$OLD_OUTPUT_DIR"
        fi
    fi
    if $RUN_NEW; then
        if [ -n "$NEW_DATABRICKS_RUN" ]; then
            _capture_databricks "$NEW_DATABRICKS_RUN" "$NEW_OUTPUT_DIR"
        else
            _capture_local "$NEW_NB_DIR" "$NEW_OUTPUT_DIR"
        fi
    fi
fi
echo ""

# ---- Step 8: Generate drift report ----
if $RUN_OLD && $RUN_NEW; then
    echo "[Step 8] Generating drift report ..."
    if $DRY_RUN; then
        echo "  DRY_RUN: compare → $REPORT_FILE"
    else
        if [ -f "$OLD_OUTPUT_DIR/cell_outputs.md" ] && [ -f "$NEW_OUTPUT_DIR/cell_outputs.md" ]; then
            python "$SCRIPTS_DIR/compare_exploration_runs.py" \
                --old-manifest "$OLD_OUTPUT_DIR/cell_outputs.md" \
                --new-manifest "$NEW_OUTPUT_DIR/cell_outputs.md" \
                --old-run-dir "$OLD_OUTPUT_DIR" \
                --new-run-dir "$NEW_OUTPUT_DIR" \
                --output "$REPORT_FILE" \
                --format both
            echo "  Report: $REPORT_FILE (+ .html)"
        else
            echo "  SKIP — one or both cell_outputs.md files missing"
        fi
    fi
else
    echo "[Step 8] SKIP — need both sides for drift report (run the other side, then use --capture-only)"
fi
echo ""

# (Steps 9-10 removed: notebooks already live in commit folders, no backup to restore)

# ---- Step 9: Clean up worktrees ----
echo "[Step 9] Cleaning up ..."
if $KEEP_WORKTREE; then
    echo "  Keeping worktrees (--keep-worktree)"
elif ! $DRY_RUN; then
    for _wt in "$OLD_DIR" "$NEW_DIR"; do
        [ -d "$_wt" ] && git -C "$REPO_DIR" worktree remove "$_wt" --force 2>/dev/null && echo "  Removed $_wt"
    done
    jupyter kernelspec remove cr-baseline -y 2>/dev/null || true
    jupyter kernelspec remove cr-current -y 2>/dev/null || true
else
    echo "  DRY_RUN: would remove worktrees + kernels"
fi

echo ""
echo "============================================================"
echo "  DONE"
echo ""
echo "  $COMPARISON_DIR/"
echo "    compare_versions.log"
echo "    drift_report.md    (LLM-optimized)"
echo "    drift_report.html  (human-optimized)"
echo "    old-${OLD_SHORT}/        ← v$OLD_VERSION ($OLD_DATE)"
echo "      notebooks/"
echo "      cell_outputs.md"
echo "      run_exploration.log"
echo "    new-${NEW_SHORT}/        ← v$NEW_VERSION ($NEW_DATE)"
echo "      notebooks/"
echo "      cell_outputs.md"
echo "      run_exploration.log"
if $KEEP_WORKTREE; then
echo "    worktree-old/            ← old code checkout + venv"
! $NEW_IS_HEAD && echo "    worktree-new/            ← new code checkout + venv"
fi
echo ""
echo "  To clean up: bash $0 --old-commit $OLD_COMMIT --cleanup"
echo "============================================================"
