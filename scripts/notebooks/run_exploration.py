#!/usr/bin/env python3
"""Run exploration notebooks sequentially with smart skip logic.

Reads findings from the experiments directory to decide which notebooks
to skip:

- Temporal notebooks (01a–01d) are skipped when no event-level datasets exist.
- Text notebooks (01a_a, 02a) are skipped when no TEXT columns are detected.

Cell outputs are preserved after execution so the resulting notebooks
can be used for documentation/HTML export.

Usage:
    python scripts/notebooks/run_exploration.py
    python scripts/notebooks/run_exploration.py --notebooks-dir exploration_notebooks
    python scripts/notebooks/run_exploration.py --dry-run
    python scripts/notebooks/run_exploration.py --timeout 900
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

NOTEBOOKS_ORDER = [
    "00_start_here",
    "01_data_discovery",
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
    "02_column_deep_dive",
    "02a_text_columns_deep_dive",
    "03_quality_assessment",
    "04_relationship_analysis",
    "05_multi_dataset",
    "06_feature_opportunities",
    "07_modeling_readiness",
    "08_baseline_experiments",
    "09_business_alignment",
    "10_spec_generation",
    "11_scoring_validation",
    "12_view_documentation",
]

TEMPORAL_NOTEBOOKS = {
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
}

TEXT_NOTEBOOKS = {
    "01a_a_temporal_text_deep_dive",
    "02a_text_columns_deep_dive",
}

# Notebook after which skip detection should run (findings are available)
_DISCOVERY_NOTEBOOK = "01_data_discovery"

_FINDINGS_EXTENSIONS = ("*.yaml", "*.yml", "*.json")


def _iter_findings_files(findings_dir: Path):
    """Yield findings files matching any supported extension."""
    seen: Set[Path] = set()
    for pattern in _FINDINGS_EXTENSIONS:
        for path in sorted(findings_dir.glob(f"*_findings.{pattern.lstrip('*.')}")):
            if path not in seen:
                seen.add(path)
                yield path


def _detect_skip_set(findings_dir: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Return (notebooks_to_skip, reasons) based on findings."""
    skip: Set[str] = set()
    reasons: Dict[str, str] = {}

    has_event_data = _has_event_level_data(findings_dir)
    has_text_columns = _has_text_columns(findings_dir)

    if not has_event_data:
        for nb in TEMPORAL_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = "no event-level datasets"

    if not has_text_columns:
        for nb in TEXT_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = "no TEXT columns detected"

    return skip, reasons


def _has_event_level_data(findings_dir: Path) -> bool:
    multi_path = findings_dir / "multi_dataset_findings.yaml"
    if multi_path.exists():
        try:
            from customer_retention.analysis.auto_explorer.exploration_manager import (
                MultiDatasetFindings,
            )

            multi = MultiDatasetFindings.load(str(multi_path))
            return bool(multi.event_datasets)
        except Exception:
            pass

    # Fallback: scan individual findings for event_level granularity
    from customer_retention.core.config.column_config import DatasetGranularity

    for path in _iter_findings_files(findings_dir):
        if "multi_dataset" in path.name:
            continue
        try:
            from customer_retention.analysis.auto_explorer.findings import (
                ExplorationFindings,
            )

            findings = ExplorationFindings.load(str(path))
            if (
                findings.time_series_metadata
                and findings.time_series_metadata.granularity
                == DatasetGranularity.EVENT_LEVEL
            ):
                return True
        except Exception:
            continue
    return False


def _has_text_columns(findings_dir: Path) -> bool:
    from customer_retention.core.config.column_config import ColumnType

    for path in _iter_findings_files(findings_dir):
        if "multi_dataset" in path.name:
            continue
        try:
            from customer_retention.analysis.auto_explorer.findings import (
                ExplorationFindings,
            )

            findings = ExplorationFindings.load(str(path))
            if findings.text_processing:
                return True
            if ColumnType.TEXT in findings.column_types.values():
                return True
        except Exception:
            continue
    return False


def _resolve_notebooks(notebooks_dir: Path) -> List[Path]:
    """Return ordered list of existing notebook paths."""
    result = []
    for stem in NOTEBOOKS_ORDER:
        path = notebooks_dir / f"{stem}.ipynb"
        if path.exists():
            result.append(path)
    return result


def _run_notebook(
    notebook_path: Path,
    timeout: int = 600,
    kernel: str = "python3",
) -> Tuple[bool, Optional[str]]:
    """Execute a notebook in-place using papermill, preserving outputs."""
    import papermill as pm

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(notebook_path),
            kernel_name=kernel,
            request_save_on_cell_execute=True,
            cwd=str(notebook_path.parent),
        )
        return True, None
    except pm.PapermillExecutionError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def run_all(
    notebooks_dir: Path,
    findings_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: int = 600,
    kernel: str = "python3",
) -> Dict[str, str]:
    """Run all notebooks, returning {stem: status} for each."""
    if findings_dir is None:
        findings_dir = notebooks_dir.parent / "experiments" / "findings"

    notebooks = _resolve_notebooks(notebooks_dir)
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return {}

    # Skip detection is deferred until after 01_data_discovery runs,
    # because that notebook creates/updates the findings files needed
    # for accurate event-level and text-column detection.
    skip_set: Set[str] = set()
    skip_reasons: Dict[str, str] = {}
    skip_detected = False

    print(f"Findings directory: {findings_dir}")
    print("(skip detection deferred until after data discovery)\n")

    results: Dict[str, str] = {}
    for nb_path in notebooks:
        stem = nb_path.stem

        # After discovery notebook runs (or is skipped), detect skip set
        if not skip_detected and stem != _DISCOVERY_NOTEBOOK and stem != "00_start_here":
            skip_detected = True
            if findings_dir.exists():
                skip_set, skip_reasons = _detect_skip_set(findings_dir)
                if skip_set:
                    print(f"\nSkipping {len(skip_set)} notebooks based on findings:")
                    for nb in sorted(skip_set):
                        print(f"  - {nb}: {skip_reasons[nb]}")
                else:
                    print("\nAll remaining notebooks will run (no skip conditions detected)")
                print()

        if stem in skip_set:
            results[stem] = f"SKIPPED ({skip_reasons[stem]})"
            print(f"  [{stem}] SKIPPED — {skip_reasons[stem]}")
            continue

        if dry_run:
            results[stem] = "DRY_RUN"
            print(f"  [{stem}] would run")
            continue

        print(f"  [{stem}] running ...", end="", flush=True)
        start = time.time()
        ok, error = _run_notebook(nb_path, timeout=timeout, kernel=kernel)
        elapsed = time.time() - start

        if ok:
            results[stem] = f"OK ({elapsed:.0f}s)"
            print(f" OK ({elapsed:.0f}s)")
        else:
            results[stem] = f"FAILED: {error}"
            print(f" FAILED ({elapsed:.0f}s)")
            print(f"    Error: {error[:200]}")

    _print_summary(results)
    return results


def _print_summary(results: Dict[str, str]) -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ok = sum(1 for v in results.values() if v.startswith("OK"))
    skipped = sum(1 for v in results.values() if v.startswith("SKIPPED"))
    failed = sum(1 for v in results.values() if v.startswith("FAILED"))
    dry = sum(1 for v in results.values() if v == "DRY_RUN")
    parts = []
    if ok:
        parts.append(f"{ok} succeeded")
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    if dry:
        parts.append(f"{dry} dry-run")
    print(", ".join(parts))
    if failed:
        print("\nFailed notebooks:")
        for stem, status in results.items():
            if status.startswith("FAILED"):
                print(f"  - {stem}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Run exploration notebooks with smart skip logic",
    )
    parser.add_argument(
        "--notebooks-dir",
        default="exploration_notebooks",
        help="Directory containing notebooks (default: exploration_notebooks)",
    )
    parser.add_argument(
        "--findings-dir",
        default=None,
        help="Findings directory (default: <project>/experiments/findings)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which notebooks would run without executing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per notebook in seconds (default: 600)",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    args = parser.parse_args()
    notebooks_dir = Path(args.notebooks_dir).resolve()
    findings_dir = Path(args.findings_dir).resolve() if args.findings_dir else None
    results = run_all(
        notebooks_dir,
        findings_dir=findings_dir,
        dry_run=args.dry_run,
        timeout=args.timeout,
        kernel=args.kernel,
    )
    if any(v.startswith("FAILED") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
