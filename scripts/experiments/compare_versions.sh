#!/usr/bin/env bash
# ============================================================================
# compare_versions.sh — Side-by-side exploration run: old commit vs HEAD
#
# Orchestrates:
#   1. git worktree for the old commit
#   2. Isolated virtualenvs with Jupyter kernels
#   3. Dataset config patching (activate retail dataset in current NB00)
#   4. run_exploration.py on both worktrees
#   5. Cell-output capture (same newest script for both)
#   6. Drift report generation
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
#   bash scripts/experiments/compare_versions.sh --list-candidates
# ============================================================================

set -euo pipefail

# ---- Defaults ----
OLD_COMMIT="6209ecd"
TIMEOUT=900
SPARK_REMOTE=""
CAPTURE_ONLY=false
DRY_RUN=false
LIST_CANDIDATES=false
KEEP_WORKTREE=false
DATASET="customer_retention_retail"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts/experiments"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --old-commit)   OLD_COMMIT="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --spark-remote) SPARK_REMOTE="--spark-remote"; shift ;;
        --capture-only) CAPTURE_ONLY=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --keep-worktree) KEEP_WORKTREE=true; shift ;;
        --list-candidates)
            LIST_CANDIDATES=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --old-commit HASH   Baseline commit (default: 6209ecd = tutorial HTML)"
            echo "  --timeout SECS      Per-notebook timeout (default: 900)"
            echo "  --spark-remote      Enable Databricks Connect for new-code run"
            echo "  --capture-only      Skip execution, just re-capture and compare"
            echo "  --dry-run           Show what would be done without executing"
            echo "  --dataset NAME      Dataset to activate (default: customer_retention_retail)"
            echo "  --keep-worktree     Keep the old-code worktree after completion"
            echo "  --list-candidates   Show candidate commits near the tutorial date"
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

# ---- Derived paths ----
OLD_SHORT="$(echo "$OLD_COMMIT" | cut -c1-7)"
NEW_SHORT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
NB00="exploration_notebooks/00_start_here.ipynb"

# Everything lives under one comparison folder — easy cleanup with rm -rf
COMPARISON_DIR="$REPO_DIR/experiments/comparisons/${OLD_SHORT}_vs_${NEW_SHORT}"
OLD_DIR="$COMPARISON_DIR/worktree"   # git worktree + venv (auto-removed unless --keep-worktree)
OLD_OUTPUT_DIR="$COMPARISON_DIR/$OLD_SHORT"
NEW_OUTPUT_DIR="$COMPARISON_DIR/$NEW_SHORT"
REPORT_FILE="$COMPARISON_DIR/drift_report.md"

# run_exploration.py still writes into each worktree's experiments/runs/;
# we use these as the run-ids so we know where to copy from
OLD_RUN_ID="baseline-${OLD_SHORT}"
NEW_RUN_ID="current-${NEW_SHORT}"

# Resolve human-readable info for both commits
OLD_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' "$OLD_COMMIT" 2>/dev/null || echo '?')"
OLD_VERSION="$(git -C "$REPO_DIR" show "$OLD_COMMIT":pyproject.toml 2>/dev/null | grep '^version' | head -1 | sed 's/version = "//;s/"//' || echo '?')"
NEW_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' HEAD)"
NEW_VERSION="$(grep '^version' "$REPO_DIR/pyproject.toml" | head -1 | sed 's/version = "//;s/"//')"

echo "============================================================"
echo "  Exploration Drift Comparison"
echo ""
echo "  Old: $OLD_SHORT  v$OLD_VERSION  ($OLD_DATE)"
echo "  New: $NEW_SHORT  v$NEW_VERSION  ($NEW_DATE)"
echo ""
echo "  Dataset:   $DATASET"
echo "  Output:    $COMPARISON_DIR/"
echo "============================================================"
echo ""

# ---- Step 1: Create worktree ----
if $CAPTURE_ONLY; then
    echo "[Step 1] SKIP — capture-only mode"
    if [ ! -d "$OLD_OUTPUT_DIR" ]; then
        echo "Error: comparison dir not found at $COMPARISON_DIR — run without --capture-only first" >&2
        exit 1
    fi
else
    echo "[Step 1] Creating git worktree for $OLD_COMMIT ..."
    if $DRY_RUN; then
        echo "  DRY_RUN: mkdir -p $COMPARISON_DIR"
        echo "  DRY_RUN: git worktree add $OLD_DIR $OLD_COMMIT --detach"
    else
        mkdir -p "$COMPARISON_DIR"
        if [ -d "$OLD_DIR" ]; then
            echo "  Worktree already exists — removing first"
            git -C "$REPO_DIR" worktree remove "$OLD_DIR" --force 2>/dev/null || true
        fi
        git -C "$REPO_DIR" worktree add "$OLD_DIR" "$OLD_COMMIT" --detach
        echo "  Created worktree at $OLD_DIR"
    fi
fi
echo ""

# ---- Step 2: Virtualenvs + Jupyter kernels ----
echo "[Step 2] Setting up virtualenvs and Jupyter kernels ..."

setup_venv() {
    local dir="$1" name="$2"
    if $DRY_RUN; then
        echo "  DRY_RUN: uv venv + pip install -e '.[dev]' in $dir"
        echo "  DRY_RUN: ipykernel install --name $name"
        return
    fi
    if [ ! -d "$dir/.venv" ]; then
        echo "  Creating venv in $dir ..."
        (cd "$dir" && uv venv .venv)
    fi
    echo "  Installing package in $dir ..."
    # shellcheck disable=SC1091
    (cd "$dir" && source .venv/bin/activate && uv pip install -e ".[dev]" && \
     python -m ipykernel install --user --name "$name" --display-name "CR $name")
}

if ! $CAPTURE_ONLY; then
    setup_venv "$OLD_DIR" "cr-baseline"
    # For new code: use existing venv or create
    if [ ! -d "$REPO_DIR/.venv" ]; then
        setup_venv "$REPO_DIR" "cr-current"
    else
        echo "  Using existing venv in $REPO_DIR"
        (cd "$REPO_DIR" && source .venv/bin/activate && \
         python -m ipykernel install --user --name "cr-current" --display-name "CR current" 2>/dev/null || true)
    fi
fi
echo ""

# ---- Step 3: Patch dataset in current NB00 ----
echo "[Step 3] Patching dataset in current NB00 → $DATASET ..."
if $DRY_RUN; then
    echo "  DRY_RUN: python patch_dataset_config.py --activate $DATASET"
else
    if ! $CAPTURE_ONLY; then
        python "$SCRIPTS_DIR/patch_dataset_config.py" \
            --notebook "$REPO_DIR/$NB00" \
            --activate "$DATASET"
    fi
fi
echo ""

# ---- Step 4: Run old version ----
echo "[Step 4] Running old version ($OLD_SHORT) ..."
if $CAPTURE_ONLY; then
    echo "  SKIP — capture-only mode"
elif $DRY_RUN; then
    echo "  DRY_RUN: run_exploration.py --run-id $OLD_RUN_ID --kernel cr-baseline"
else
    (
        cd "$OLD_DIR"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        python scripts/notebooks/run_exploration.py \
            --notebooks-dir exploration_notebooks \
            --run-id "$OLD_RUN_ID" \
            --kernel cr-baseline \
            --timeout "$TIMEOUT"
    )
    echo "  Old run complete."
fi
echo ""

# ---- Step 5: Run new version ----
echo "[Step 5] Running new version (HEAD) ..."
if $CAPTURE_ONLY; then
    echo "  SKIP — capture-only mode"
elif $DRY_RUN; then
    echo "  DRY_RUN: run_exploration.py --run-id $NEW_RUN_ID --kernel cr-current $SPARK_REMOTE"
else
    (
        cd "$REPO_DIR"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        python scripts/notebooks/run_exploration.py \
            --notebooks-dir exploration_notebooks \
            --run-id "$NEW_RUN_ID" \
            --kernel cr-current \
            --timeout "$TIMEOUT" \
            $SPARK_REMOTE
    )
    echo "  New run complete."
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
    (cd "$work_dir" && source .venv/bin/activate && pip freeze > "$output_dir/pip_freeze.txt" && \
     python --version > "$output_dir/python_version.txt" 2>&1)
    echo "  Recorded: $(cut -c1-7 "$output_dir/git_sha.txt")  v$(cat "$output_dir/version.txt")  $(cat "$output_dir/commit_date.txt")"
}

if ! $CAPTURE_ONLY && ! $DRY_RUN; then
    record_metadata "$OLD_OUTPUT_DIR" "$REPO_DIR" "$OLD_COMMIT"
    record_metadata "$NEW_OUTPUT_DIR" "$REPO_DIR" "HEAD"
elif $DRY_RUN; then
    record_metadata "$OLD_OUTPUT_DIR" "$REPO_DIR" "$OLD_COMMIT"
    record_metadata "$NEW_OUTPUT_DIR" "$REPO_DIR" "HEAD"
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
    OLD_RUN_DIR="$OLD_DIR/experiments/runs/$OLD_RUN_ID"
    NEW_RUN_DIR="$REPO_DIR/experiments/runs/$NEW_RUN_ID"
    copy_run_artefacts "$OLD_RUN_DIR" "$OLD_OUTPUT_DIR"
    copy_run_artefacts "$NEW_RUN_DIR" "$NEW_OUTPUT_DIR"
fi
echo ""

# ---- Step 7: Capture outputs (newest script, both worktrees) ----
echo "[Step 7] Capturing cell outputs ..."
if $DRY_RUN; then
    echo "  DRY_RUN: capture → $OLD_OUTPUT_DIR/cell_outputs.md"
    echo "  DRY_RUN: capture → $NEW_OUTPUT_DIR/cell_outputs.md"
else
    python "$SCRIPTS_DIR/capture_notebook_outputs.py" \
        --notebooks-dir "$OLD_DIR/exploration_notebooks" \
        --output "$OLD_OUTPUT_DIR/cell_outputs.md"

    python "$SCRIPTS_DIR/capture_notebook_outputs.py" \
        --notebooks-dir "$REPO_DIR/exploration_notebooks" \
        --output "$NEW_OUTPUT_DIR/cell_outputs.md"
fi
echo ""

# ---- Step 8: Generate drift report ----
echo "[Step 8] Generating drift report ..."
if $DRY_RUN; then
    echo "  DRY_RUN: compare → $REPORT_FILE"
else
    python "$SCRIPTS_DIR/compare_exploration_runs.py" \
        --old-manifest "$OLD_OUTPUT_DIR/cell_outputs.md" \
        --new-manifest "$NEW_OUTPUT_DIR/cell_outputs.md" \
        --old-run-dir "$OLD_OUTPUT_DIR" \
        --new-run-dir "$NEW_OUTPUT_DIR" \
        --output "$REPORT_FILE"
    echo "  Report: $REPORT_FILE"
fi
echo ""

# ---- Step 9: Copy executed notebooks into comparison dir ----
echo "[Step 9] Copying executed notebooks ..."
if $DRY_RUN; then
    echo "  DRY_RUN: copy notebooks → $OLD_OUTPUT_DIR/notebooks/"
    echo "  DRY_RUN: copy notebooks → $NEW_OUTPUT_DIR/notebooks/"
else
    mkdir -p "$OLD_OUTPUT_DIR/notebooks" "$NEW_OUTPUT_DIR/notebooks"
    cp "$OLD_DIR"/exploration_notebooks/*.ipynb "$OLD_OUTPUT_DIR/notebooks/" 2>/dev/null || true
    cp "$REPO_DIR"/exploration_notebooks/*.ipynb "$NEW_OUTPUT_DIR/notebooks/" 2>/dev/null || true
    _old_nb_count=$(ls "$OLD_OUTPUT_DIR/notebooks/"*.ipynb 2>/dev/null | wc -l | tr -d ' ')
    _new_nb_count=$(ls "$NEW_OUTPUT_DIR/notebooks/"*.ipynb 2>/dev/null | wc -l | tr -d ' ')
    echo "  Copied ${_old_nb_count} old + ${_new_nb_count} new notebooks"
fi
echo ""

# ---- Step 10: Restore NB00 ----
echo "[Step 10] Restoring NB00 ..."
if ! $DRY_RUN && ! $CAPTURE_ONLY; then
    python "$SCRIPTS_DIR/patch_dataset_config.py" \
        --notebook "$REPO_DIR/$NB00" \
        --restore 2>/dev/null || echo "  (no backup to restore)"
fi
echo ""

# ---- Step 11: Clean up worktree ----
echo "[Step 11] Cleaning up ..."
if $DRY_RUN; then
    if $KEEP_WORKTREE; then
        echo "  DRY_RUN: keeping worktree at $OLD_DIR"
    else
        echo "  DRY_RUN: git worktree remove $OLD_DIR"
        echo "  DRY_RUN: jupyter kernelspec remove cr-baseline"
    fi
elif $KEEP_WORKTREE; then
    echo "  Keeping worktree at $OLD_DIR (--keep-worktree)"
else
    echo "  Removing worktree $OLD_DIR ..."
    git -C "$REPO_DIR" worktree remove "$OLD_DIR" --force 2>/dev/null || true
    jupyter kernelspec remove cr-baseline -y 2>/dev/null || true
    echo "  Cleaned up."
fi

echo ""
echo "============================================================"
echo "  DONE"
echo ""
echo "  $COMPARISON_DIR/"
echo "    drift_report.md"
echo "    ${OLD_SHORT}/"
echo "      notebooks/       ← executed .ipynb from old code (v$OLD_VERSION)"
echo "      cell_outputs.md"
echo "    ${NEW_SHORT}/"
echo "      notebooks/       ← executed .ipynb from new code (v$NEW_VERSION)"
echo "      cell_outputs.md"
if $KEEP_WORKTREE; then
echo "    worktree/          ← old code checkout + venv (--keep-worktree)"
fi
echo ""
if $KEEP_WORKTREE; then
echo "  Worktree kept. To clean up everything:"
echo "    git worktree remove $OLD_DIR && rm -rf $COMPARISON_DIR"
else
echo "  To clean up: rm -rf $COMPARISON_DIR"
fi
echo "============================================================"
