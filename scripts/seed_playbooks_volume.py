"""Bootstrap the playbooks volume with framework seed YAMLs.

Copies the three framework templates from
``src/customer_retention/stages/causal/seed_yamls/`` into
``{PLAYBOOKS_DIR}/policies/``. Idempotent: existing destination files are
NOT overwritten unless ``--force`` is passed, so re-running the script
after a deployment customizes its policies will never clobber the changes.

Usage:

    # Default — uses get_playbooks_dir() resolved from databricks_init() persisted config
    python scripts/seed_playbooks_volume.py

    # Override the destination explicitly (e.g. for a sibling cluster or local test)
    python scripts/seed_playbooks_volume.py --target-dir /tmp/playbooks_test

    # Force overwrite existing files (use with care)
    python scripts/seed_playbooks_volume.py --force

    # Dry run — show what would be copied without writing anything
    python scripts/seed_playbooks_volume.py --dry-run

The script never touches the per-playbook YAMLs (``low_nps.yaml`` etc.) at
the top level of the playbooks directory — those are private user content
and live outside the framework's bootstrap surface.
"""

from __future__ import annotations

import argparse
import importlib.resources
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

SEED_FILES = (
    "decision_policy.yaml",
    "response_schemas.yaml",
    "vocabularies.yaml",
)


def _resolve_seed_source_dir() -> Optional[Path]:
    """Locate the framework seed YAML directory.

    Tries the source-tree path first (works for editable installs and the
    repo itself), then falls back to importlib.resources for installed
    packages.
    """
    here = Path(__file__).resolve().parent.parent
    candidate = here / "src" / "customer_retention" / "stages" / "causal" / "seed_yamls"
    if candidate.is_dir():
        return candidate
    try:
        resolved = importlib.resources.files("customer_retention.stages.causal") / "seed_yamls"
        resolved_path = Path(str(resolved))
        if resolved_path.is_dir():
            return resolved_path
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass
    return None


def _resolve_target_dir(override: Optional[str]) -> Path:
    """Resolve the destination policies directory.

    Order:
    1. Explicit ``--target-dir`` argument (treated as the policies directory)
    2. ``get_playbooks_dir()`` from the framework config (Databricks volume
       or repo-local fallback) with ``/policies`` appended
    """
    if override:
        return Path(override)
    try:
        from customer_retention.core.config.experiments import get_playbooks_dir

        playbooks_dir = get_playbooks_dir()
        return Path(str(playbooks_dir)) / "policies"
    except ImportError:
        sys.exit(
            "ERROR: customer_retention is not importable. Run from the repo root "
            "with the framework installed (e.g. `pip install -e .`)."
        )


def _copy_one(
    source: Path, destination: Path, *, force: bool, dry_run: bool
) -> str:
    if destination.exists() and not force:
        return f"  SKIP   {destination}  (already exists; use --force to overwrite)"
    if dry_run:
        return f"  WOULD  {source.name} -> {destination}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"  COPIED {source.name} -> {destination}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the playbooks volume's policies/ directory with framework seed YAMLs.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help=(
            "Destination directory for the policies/ files. "
            "Default: <get_playbooks_dir()>/policies."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files at the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing anything.",
    )
    args = parser.parse_args(argv)

    source_dir = _resolve_seed_source_dir()
    if source_dir is None:
        sys.exit(
            "ERROR: framework seed YAML directory not found. Expected at "
            "src/customer_retention/stages/causal/seed_yamls/ relative to the "
            "repo root, or as a package resource of customer_retention.stages.causal."
        )

    target_dir = _resolve_target_dir(args.target_dir)

    print("=" * 60)
    print("Causal-track playbooks volume bootstrap")
    print("=" * 60)
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Force:  {args.force}")
    print(f"  Mode:   {'dry-run' if args.dry_run else 'write'}")
    print()

    missing_sources: List[str] = []
    actions: List[str] = []
    for filename in SEED_FILES:
        source = source_dir / filename
        if not source.exists():
            missing_sources.append(filename)
            continue
        actions.append(_copy_one(source, target_dir / filename, force=args.force, dry_run=args.dry_run))

    if missing_sources:
        print("ERROR: missing seed source files:")
        for name in missing_sources:
            print(f"  - {name}")
        return 1

    for line in actions:
        print(line)

    print()
    print("Done.")
    if not args.dry_run:
        print()
        print("Next steps:")
        print(f"  1. Review {target_dir}/decision_policy.yaml and adjust capacity / holdout / cooldown")
        print(f"  2. Review {target_dir}/vocabularies.yaml and add/remove values per your CS team")
        print("  3. Re-run this script with --force only if you intend to discard your edits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
