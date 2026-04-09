import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

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
