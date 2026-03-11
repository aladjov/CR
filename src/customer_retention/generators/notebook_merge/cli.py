from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat

from .merge_engine import NotebookMergeEngine
from .merge_report import MergeReport


@dataclass
class PendingMerge:
    name: str
    merged_nb: nbformat.NotebookNode
    report: MergeReport


def _resolve_base_dir() -> Path | None:
    from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
    return ProjectInitializer(project_name="")._get_exploration_source_dir()


def merge_notebook_pair(
    theirs_path: Path, ours_path: Path, output_path: Path,
    base_path: Path | None = None, dry_run: bool = False,
    theirs_label: str = "theirs", ours_label: str = "ours",
) -> MergeReport:
    theirs_nb = nbformat.read(str(theirs_path), as_version=4)
    ours_nb = nbformat.read(str(ours_path), as_version=4)
    if base_path and base_path.exists():
        base_nb = nbformat.read(str(base_path), as_version=4)
    else:
        base_nb = nbformat.v4.new_notebook()

    engine = NotebookMergeEngine(theirs_label, ours_label)
    merged, report = engine.merge(base_nb, theirs_nb, ours_nb)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(merged, str(output_path))
    return report


def compute_all_merges(
    theirs_dir: Path, ours_dir: Path, base_dir: Path,
    notebook: str | None = None,
    theirs_label: str = "theirs", ours_label: str = "ours",
) -> dict[str, PendingMerge]:
    if notebook:
        names = [notebook]
    else:
        theirs_names = {p.name for p in theirs_dir.glob("*.ipynb")}
        ours_names = {p.name for p in ours_dir.glob("*.ipynb")}
        names = sorted(theirs_names | ours_names)

    engine = NotebookMergeEngine(theirs_label, ours_label)
    pending: dict[str, PendingMerge] = {}

    for name in names:
        theirs_path = theirs_dir / name
        ours_path = ours_dir / name
        base_path = base_dir / name

        base_nb = _read_or_empty(base_path)
        theirs_nb = _read_or_empty(theirs_path)
        ours_nb = _read_or_empty(ours_path)

        merged, report = engine.merge(base_nb, theirs_nb, ours_nb)
        pending[name] = PendingMerge(name=name, merged_nb=merged, report=report)
    return pending


def _read_or_empty(path: Path) -> nbformat.NotebookNode:
    if path.exists():
        return nbformat.read(str(path), as_version=4)
    return nbformat.v4.new_notebook()


def apply_pending_merges(pending: dict[str, PendingMerge], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pm in pending.items():
        nbformat.write(pm.merged_nb, str(output_dir / name))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Three-way merge of exploration notebooks",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    dir_parser = sub.add_parser("dir", help="Merge entire notebook directories")
    dir_parser.add_argument("theirs_dir", help="First user's notebook directory")
    dir_parser.add_argument("ours_dir", help="Second user's notebook directory")
    dir_parser.add_argument("-o", "--output", required=True, help="Output directory")
    dir_parser.add_argument("--base-dir", default=None, help="Base notebook directory (default: installed package)")
    dir_parser.add_argument("--notebook", default=None, help="Merge a single notebook by filename")
    dir_parser.add_argument("--theirs-label", default="theirs")
    dir_parser.add_argument("--ours-label", default="ours")
    dir_parser.add_argument("--dry-run", action="store_true")
    dir_parser.add_argument("--fail-on-conflict", action="store_true")

    pair_parser = sub.add_parser("pair", help="Merge a pair of notebook files")
    pair_parser.add_argument("theirs", help="First user's notebook file")
    pair_parser.add_argument("ours", help="Second user's notebook file")
    pair_parser.add_argument("-o", "--output", required=True, help="Output file path")
    pair_parser.add_argument("--base", default=None, help="Base notebook file")
    pair_parser.add_argument("--theirs-label", default="theirs")
    pair_parser.add_argument("--ours-label", default="ours")
    pair_parser.add_argument("--dry-run", action="store_true")
    pair_parser.add_argument("--fail-on-conflict", action="store_true")

    args = parser.parse_args(argv)

    if args.mode == "pair":
        _run_pair(args)
    else:
        _run_dir(args)


def _run_pair(args):
    theirs_path = Path(args.theirs).resolve()
    ours_path = Path(args.ours).resolve()
    output_path = Path(args.output).resolve()
    base_path = Path(args.base).resolve() if args.base else None

    report = merge_notebook_pair(
        theirs_path, ours_path, output_path, base_path,
        dry_run=args.dry_run,
        theirs_label=args.theirs_label, ours_label=args.ours_label,
    )
    _print_report(output_path.name, report, args.dry_run)
    if args.fail_on_conflict and report.has_conflicts:
        sys.exit(1)


def _run_dir(args):
    theirs_dir = Path(args.theirs_dir).resolve()
    ours_dir = Path(args.ours_dir).resolve()
    output_dir = Path(args.output).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else _resolve_base_dir()

    if not base_dir or not base_dir.exists():
        print("Base directory not found. Use --base-dir or install the package.")
        sys.exit(2)

    pending = compute_all_merges(
        theirs_dir, ours_dir, base_dir,
        notebook=args.notebook,
        theirs_label=args.theirs_label, ours_label=args.ours_label,
    )

    has_any_conflict = False
    for name, pm in pending.items():
        _print_report(name, pm.report, args.dry_run)
        if pm.report.has_conflicts:
            has_any_conflict = True

    if not args.dry_run:
        apply_pending_merges(pending, output_dir)

    total = len(pending)
    conflicts = sum(1 for pm in pending.values() if pm.report.has_conflicts)
    label = "Would merge" if args.dry_run else "Merged"
    print(f"\n{label} {total} notebook(s), {conflicts} with conflicts")

    if args.fail_on_conflict and has_any_conflict:
        sys.exit(1)


def _print_report(name: str, report: MergeReport, dry_run: bool):
    label = "would merge" if dry_run else "merged"
    if report.has_conflicts:
        print(f"  {name} — {label}: {report.format_summary()} ** CONFLICTS **")
    elif report.has_changes:
        print(f"  {name} — {label}: {report.format_summary()}")
    else:
        print(f"  {name} — no changes")


if __name__ == "__main__":
    main()
