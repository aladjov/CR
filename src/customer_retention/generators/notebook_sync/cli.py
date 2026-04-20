import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nbformat

from .sync_engine import NotebookSyncEngine
from .sync_report import CellSyncEntry, SyncReport


@dataclass
class PendingWrite:
    user_path: Path
    merged_nb: nbformat.NotebookNode
    report: SyncReport


def sync_notebook(
    repo_path: Path,
    user_path: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> SyncReport:
    repo_nb = nbformat.read(str(repo_path), as_version=4)
    user_nb = nbformat.read(str(user_path), as_version=4)

    engine = NotebookSyncEngine()
    merged, report = engine.sync(repo_nb, user_nb)

    if not dry_run and report.has_changes:
        if backup:
            shutil.copy2(user_path, user_path.with_suffix(".ipynb.bak"))
        nbformat.write(merged, str(user_path))

    return report


def compute_all_syncs(
    repo_dir: Path,
    user_dir: Path,
    notebook: str | None = None,
) -> dict[str, PendingWrite]:
    if notebook:
        names = [notebook]
    else:
        names = sorted(p.name for p in repo_dir.glob("*.ipynb"))

    pending: dict[str, PendingWrite] = {}
    for name in names:
        repo_path = repo_dir / name
        user_path = user_dir / name
        if not repo_path.exists():
            print(f"  {name} — not found in repo dir, skipping")
            continue
        if not user_path.exists():
            print(f"  {name} — not found in user dir, skipping")
            continue
        repo_nb = nbformat.read(str(repo_path), as_version=4)
        user_nb = nbformat.read(str(user_path), as_version=4)
        engine = NotebookSyncEngine()
        merged, report = engine.sync(repo_nb, user_nb)
        pending[name] = PendingWrite(user_path=user_path, merged_nb=merged, report=report)
    return pending


def apply_pending_writes(pending: dict[str, PendingWrite], backup: bool = True) -> None:
    for pw in pending.values():
        if not pw.report.has_changes:
            continue
        if backup:
            shutil.copy2(pw.user_path, pw.user_path.with_suffix(".ipynb.bak"))
        nbformat.write(pw.merged_nb, str(pw.user_path))


def collect_removals(pending: dict[str, PendingWrite]) -> list[tuple[str, CellSyncEntry]]:
    removals: list[tuple[str, CellSyncEntry]] = []
    for name, pw in pending.items():
        for entry in pw.report.removed_entries:
            removals.append((name, entry))
    return removals


def format_removal_summary(removals: list[tuple[str, CellSyncEntry]]) -> str:
    lines = [f"\nWARNING: {len(removals)} cell(s) will be REMOVED:\n"]
    for name, entry in removals:
        lines.append(f"  {name} — cell {entry.cell_id}")
        if entry.source_preview:
            for preview_line in entry.source_preview.split("\n"):
                lines.append(f"    | {preview_line}")
        lines.append("")
    lines.append("Untagged cells are removed during sync.")
    lines.append("To keep a cell, tag it with # @cr:config or # @cr:user_code")
    return "\n".join(lines)


def prompt_removal_confirmation() -> bool:
    response = input("\nProceed with sync? [y/N] ").strip().lower()
    return response == "y"


def sync_directory(
    repo_dir: Path,
    user_dir: Path,
    notebook: str | None = None,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, SyncReport]:
    pending = compute_all_syncs(repo_dir, user_dir, notebook=notebook)

    for name, pw in pending.items():
        label = "would sync" if dry_run else "synced"
        if pw.report.has_changes:
            print(f"  {name} — {label}: {pw.report.format_summary()}")
        else:
            print(f"  {name} — no changes")

    removals = collect_removals(pending)

    if removals and not dry_run and not force:
        print(format_removal_summary(removals))
        if not prompt_removal_confirmation():
            print("\nSync cancelled. No files were modified.")
            return {name: pw.report for name, pw in pending.items()}

    if not dry_run:
        apply_pending_writes(pending, backup=backup)

    return {name: pw.report for name, pw in pending.items()}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Sync exploration and causal notebooks: merge repo updates while "
            "preserving user config and tagged cells."
        ),
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Directory with updated repo notebooks (exploration_notebooks)",
    )
    parser.add_argument(
        "--user-dir",
        default="exploration_notebooks",
        help="User's exploration notebook directory (default: exploration_notebooks)",
    )
    parser.add_argument(
        "--causal-repo-dir",
        default=None,
        help=(
            "Directory with updated causal-track notebooks. Defaults to "
            "'causal_notebooks' under the parent of --repo-dir if that "
            "directory exists, so a single --repo-dir invocation upgrades "
            "both layers in lock-step."
        ),
    )
    parser.add_argument(
        "--causal-user-dir",
        default="causal_notebooks",
        help="User's causal-track notebook directory (default: causal_notebooks)",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Skip causal notebooks even if a causal-repo-dir is detected",
    )
    parser.add_argument(
        "--no-regenerate-causal",
        action="store_true",
        help=(
            "Skip regenerating causal notebooks from their Python source "
            "before syncing. By default ``sync_notebooks`` invokes "
            "``scripts/notebooks/build_causal_notebooks.py`` first so the "
            "repo-side ipynb files are always rebuilt from the current "
            "``build_causal_notebooks.py``. Use this flag when you want to "
            "sync prebuilt notebooks as-is (e.g. in CI against a pinned "
            "artifact)."
        ),
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="Sync a single notebook by filename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak files before overwriting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt when cells are removed",
    )
    args = parser.parse_args(argv)

    repo_dir = Path(args.repo_dir).resolve()
    user_dir = Path(args.user_dir).resolve()

    if not repo_dir.exists():
        print(f"Repo directory not found: {repo_dir}")
        sys.exit(1)
    if not user_dir.exists():
        print(f"User directory not found: {user_dir}")
        sys.exit(1)

    print(f"\n[exploration] {repo_dir.name}/ -> {user_dir}")
    exploration_results = sync_directory(
        repo_dir,
        user_dir,
        notebook=args.notebook,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        force=args.force,
    )

    causal_results: dict[str, SyncReport] = {}
    causal_repo_dir = _resolve_causal_repo_dir(args, repo_dir)
    causal_user_dir = Path(args.causal_user_dir).resolve()
    if causal_repo_dir and causal_user_dir.exists():
        if not args.no_regenerate_causal:
            _regenerate_causal_notebooks(causal_repo_dir, dry_run=args.dry_run)
        print(f"\n[causal]      {causal_repo_dir.name}/ -> {causal_user_dir}")
        causal_results = sync_directory(
            causal_repo_dir,
            causal_user_dir,
            notebook=args.notebook,
            backup=not args.no_backup,
            dry_run=args.dry_run,
            force=args.force,
        )
    elif causal_repo_dir and not causal_user_dir.exists():
        print(f"\n[causal] user dir not found ({causal_user_dir}); skipping")

    total = len(exploration_results) + len(causal_results)
    changed = sum(1 for r in exploration_results.values() if r.has_changes) + sum(
        1 for r in causal_results.values() if r.has_changes
    )
    label = "Would sync" if args.dry_run else "Synced"
    print(f"\n{label} {changed}/{total} notebooks")


def _find_causal_build_script(causal_repo_dir: Path) -> Optional[Path]:
    """Locate ``scripts/notebooks/build_causal_notebooks.py`` next to the repo dir.

    The causal repo dir is conventionally ``{project}/causal_notebooks`` and the
    build script conventionally lives at
    ``{project}/scripts/notebooks/build_causal_notebooks.py``. Look relative
    to the repo dir's parent; return ``None`` if the script is not present so
    regeneration degrades to a warning instead of a hard failure.
    """
    candidate = (
        causal_repo_dir.parent
        / "scripts"
        / "notebooks"
        / "build_causal_notebooks.py"
    )
    return candidate if candidate.exists() else None


def _regenerate_causal_notebooks(causal_repo_dir: Path, dry_run: bool) -> None:
    """Rebuild the causal ipynb artifacts from their Python source before sync.

    Addresses a recurring class of drift: editing ``build_causal_notebooks.py``
    does not automatically refresh the on-disk ipynb files, so syncs have
    silently pushed stale notebooks to Databricks. Running the build step
    first makes the pipeline single-shot.

    Executes the script via ``subprocess.run([sys.executable, script])`` so
    the child inherits the current interpreter and PYTHONPATH without this
    module needing to import script-local symbols. Build output (``wrote
    causal_notebooks/...``) is streamed through so the operator sees exactly
    what was regenerated. ``--dry-run`` skips the build, matching the sync's
    dry-run semantics.
    """
    script = _find_causal_build_script(causal_repo_dir)
    if script is None:
        print(
            f"[causal] regenerate skipped — build script not found next to "
            f"{causal_repo_dir}"
        )
        return
    if dry_run:
        print(f"[causal] would regenerate via {script}")
        return
    print(f"[causal] regenerating from {script.name}")
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        print(f"  {line}")
    for line in result.stderr.splitlines():
        print(f"  {line}", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(
            f"Causal notebook regeneration failed with exit code "
            f"{result.returncode}. Aborting sync so stale artifacts do not "
            "get pushed. Fix the error above or pass --no-regenerate-causal "
            "to bypass."
        )


def _resolve_causal_repo_dir(args, repo_dir: Path) -> Path | None:
    if args.no_causal:
        return None
    if args.causal_repo_dir:
        candidate = Path(args.causal_repo_dir).resolve()
        if not candidate.exists():
            print(f"Causal repo directory not found: {candidate}")
            return None
        return candidate
    sibling = (repo_dir.parent / "causal_notebooks").resolve()
    return sibling if sibling.exists() else None


if __name__ == "__main__":
    main()
