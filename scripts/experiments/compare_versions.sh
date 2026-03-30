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
    echo "=== Candidate baseline commits ==="
    echo ""
    echo "  COMMIT     VERSION        DATE            MESSAGE"
    echo "  ─────────  ────────────   ──────────────  ──────────────────────────────────"
    echo ""
    echo "Tutorial commit (default baseline):"
    commit_info 6209ecd
    echo ""
    echo "Commits touching docs/tutorial/:"
    for sha in $(git -C "$REPO_DIR" log --format='%h' --before="2026-03-01" --after="2026-01-01" -- docs/tutorial/ | head -10); do
        commit_info "$sha"
    done
    echo ""
    echo "Commits before tutorial (potential baselines):"
    for sha in $(git -C "$REPO_DIR" log --format='%h' 6209ecd~5..6209ecd); do
        commit_info "$sha"
    done
    echo ""
    echo "Current HEAD:"
    commit_info HEAD
    exit 0
fi

# ---- Derived paths ----
OLD_SHORT="$(echo "$OLD_COMMIT" | cut -c1-7)"
OLD_DIR="${REPO_DIR}-baseline-${OLD_SHORT}"
OLD_RUN_ID="baseline-${OLD_SHORT}"
NEW_RUN_ID="current-HEAD"
NB00="exploration_notebooks/00_start_here.ipynb"
REPORT_DIR="$REPO_DIR/experiments"

# Resolve human-readable info for both commits
OLD_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' "$OLD_COMMIT" 2>/dev/null || echo '?')"
OLD_VERSION="$(git -C "$REPO_DIR" show "$OLD_COMMIT":pyproject.toml 2>/dev/null | grep '^version' | head -1 | sed 's/version = "//;s/"//' || echo '?')"
NEW_DATE="$(git -C "$REPO_DIR" log -1 --format='%ad' --date=format:'%b %d, %Y' HEAD)"
NEW_VERSION="$(grep '^version' "$REPO_DIR/pyproject.toml" | head -1 | sed 's/version = "//;s/"//')"

echo "============================================================"
echo "  Exploration Drift Comparison"
echo ""
echo "  Old: $OLD_SHORT  v$OLD_VERSION  ($OLD_DATE)"
echo "  New: HEAD     v$NEW_VERSION  ($NEW_DATE)"
echo ""
echo "  Dataset:   $DATASET"
echo "  Repo:      $REPO_DIR"
echo "  Worktree:  $OLD_DIR"
echo "============================================================"
echo ""

# ---- Step 1: Create worktree ----
if $CAPTURE_ONLY; then
    echo "[Step 1] SKIP — capture-only mode (expecting worktree at $OLD_DIR)"
    if [ ! -d "$OLD_DIR" ]; then
        echo "Error: worktree not found at $OLD_DIR — run without --capture-only first" >&2
        exit 1
    fi
else
    echo "[Step 1] Creating git worktree for $OLD_COMMIT ..."
    if $DRY_RUN; then
        echo "  DRY_RUN: git worktree add $OLD_DIR $OLD_COMMIT --detach"
    else
        if [ -d "$OLD_DIR" ]; then
            echo "  Worktree already exists at $OLD_DIR — removing first"
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

# ---- Step 6: Record metadata ----
echo "[Step 6] Recording environment metadata ..."
record_metadata() {
    local run_dir="$1" work_dir="$2"
    if $DRY_RUN; then
        echo "  DRY_RUN: record git sha, version, date, pip freeze in $run_dir"
        return
    fi
    mkdir -p "$run_dir"
    local sha
    sha="$(git -C "$work_dir" rev-parse HEAD)"
    echo "$sha" > "$run_dir/git_sha.txt"
    git -C "$work_dir" log -1 --format='%ad' --date=format:'%b %d, %Y' "$sha" > "$run_dir/commit_date.txt"
    git -C "$work_dir" show "$sha":pyproject.toml 2>/dev/null \
        | grep '^version' | head -1 | sed 's/version = "//;s/"//' > "$run_dir/version.txt"
    (cd "$work_dir" && source .venv/bin/activate && pip freeze > "$run_dir/pip_freeze.txt" && \
     python --version > "$run_dir/python_version.txt" 2>&1)
    echo "  Recorded: $(cat "$run_dir/git_sha.txt" | cut -c1-7)  v$(cat "$run_dir/version.txt")  $(cat "$run_dir/commit_date.txt")"
}

OLD_RUN_DIR="$OLD_DIR/experiments/runs/$OLD_RUN_ID"
NEW_RUN_DIR="$REPO_DIR/experiments/runs/$NEW_RUN_ID"

if ! $CAPTURE_ONLY && ! $DRY_RUN; then
    record_metadata "$OLD_RUN_DIR" "$OLD_DIR"
    record_metadata "$NEW_RUN_DIR" "$REPO_DIR"
fi
echo ""

# ---- Step 7: Capture outputs (newest script, both worktrees) ----
echo "[Step 7] Capturing cell outputs ..."
if $DRY_RUN; then
    echo "  DRY_RUN: capture_notebook_outputs.py for both worktrees"
else
    python "$SCRIPTS_DIR/capture_notebook_outputs.py" \
        --notebooks-dir "$OLD_DIR/exploration_notebooks" \
        --output "$OLD_RUN_DIR/cell_outputs.md"

    python "$SCRIPTS_DIR/capture_notebook_outputs.py" \
        --notebooks-dir "$REPO_DIR/exploration_notebooks" \
        --output "$NEW_RUN_DIR/cell_outputs.md"
fi
echo ""

# ---- Step 8: Generate drift report ----
echo "[Step 8] Generating drift report ..."
REPORT_FILE="$REPORT_DIR/drift_report_${OLD_SHORT}_vs_HEAD.md"
if $DRY_RUN; then
    echo "  DRY_RUN: compare_exploration_runs.py → $REPORT_FILE"
else
    mkdir -p "$REPORT_DIR"
    python "$SCRIPTS_DIR/compare_exploration_runs.py" \
        --old-manifest "$OLD_RUN_DIR/cell_outputs.md" \
        --new-manifest "$NEW_RUN_DIR/cell_outputs.md" \
        --old-run-dir "$OLD_RUN_DIR" \
        --new-run-dir "$NEW_RUN_DIR" \
        --output "$REPORT_FILE"
    echo "  Report: $REPORT_FILE"
fi
echo ""

# ---- Step 9: Restore NB00 and offer cleanup ----
echo "[Step 9] Restoring NB00 ..."
if ! $DRY_RUN && ! $CAPTURE_ONLY; then
    python "$SCRIPTS_DIR/patch_dataset_config.py" \
        --notebook "$REPO_DIR/$NB00" \
        --restore 2>/dev/null || echo "  (no backup to restore)"
fi

echo ""
echo "============================================================"
echo "  DONE"
echo ""
echo "  Drift report: $REPORT_FILE"
echo "  Old manifest: $OLD_RUN_DIR/cell_outputs.md"
echo "  New manifest: $NEW_RUN_DIR/cell_outputs.md"
echo ""
echo "  To clean up the worktree:"
echo "    git worktree remove $OLD_DIR"
echo "    jupyter kernelspec remove cr-baseline -y"
echo "============================================================"
