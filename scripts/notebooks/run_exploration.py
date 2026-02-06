#!/usr/bin/env python3
"""Run exploration notebooks sequentially with smart skip logic.

Reads findings from the experiments directory to decide which notebooks
to skip:

- Temporal notebooks (01a–01d) are skipped when no event-level datasets exist.
- Text notebooks (01a_a, 02a) are skipped when no TEXT columns are detected.

Multi-dataset mode (auto-detected from dataset_context.yaml):
- Runs per-dataset notebooks (01–04) for each dataset (target first)
- Preserves event-level / text-column skip logic per dataset
- Then runs global notebooks (05–12) once

Cell outputs are preserved after execution so the resulting notebooks
can be used for documentation/HTML export.

Usage:
    python scripts/notebooks/run_exploration.py
    python scripts/notebooks/run_exploration.py --notebooks-dir exploration_notebooks
    python scripts/notebooks/run_exploration.py --dry-run
    python scripts/notebooks/run_exploration.py --timeout 900
"""
import argparse
import json
import sys
import time
import traceback
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

SETUP_NOTEBOOKS = {"00_start_here"}

PER_DATASET_STEMS = {
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
}

_DISCOVERY_NOTEBOOK = "01_data_discovery"

_FINDINGS_EXTENSIONS = ("*.yaml", "*.yml", "*.json")


# ---------------------------------------------------------------------------
# Findings helpers
# ---------------------------------------------------------------------------


def _iter_findings_files(findings_dir: Path):
    """Yield findings files matching any supported extension."""
    seen: Set[Path] = set()
    for pattern in _FINDINGS_EXTENSIONS:
        for path in sorted(findings_dir.glob(f"*_findings.{pattern.lstrip('*.')}")):
            if path not in seen:
                seen.add(path)
                yield path


def _detect_skip_set(findings_dir: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Return (notebooks_to_skip, reasons) based on all findings."""
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


def _find_dataset_findings(findings_dir: Path, dataset_name: str) -> Optional[Path]:
    """Find the most recent findings file for a dataset."""
    exact = findings_dir / f"{dataset_name}_findings.yaml"
    if exact.exists():
        return exact
    candidates = sorted(
        findings_dir.glob(f"{dataset_name}*_findings.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _is_event_level_dataset(
    findings_dir: Path,
    dataset_name: str,
    context=None,
    dataset_path: Optional[str] = None,
) -> bool:
    """Determine if a dataset is event-level using multiple signals.

    Priority: findings time_series_metadata > context relationship > direct detection.
    """
    from customer_retention.core.config.column_config import DatasetGranularity

    # Signal 1: findings time_series_metadata
    findings_path = _find_dataset_findings(findings_dir, dataset_name)
    if findings_path:
        try:
            from customer_retention.analysis.auto_explorer.findings import (
                ExplorationFindings,
            )

            findings = ExplorationFindings.load(str(findings_path))
            if findings.time_series_metadata:
                return (
                    findings.time_series_metadata.granularity
                    == DatasetGranularity.EVENT_LEVEL
                )
        except Exception:
            pass

    # Signal 2: context relationship (many_to_one → event-level)
    if context is not None:
        entry = context.datasets.get(dataset_name)
        if entry and getattr(entry, "relationship", None) == "many_to_one":
            return True

    # Signal 3: direct data inspection
    if dataset_path:
        try:
            import pandas as pd

            from customer_retention.stages.profiling import TypeDetector

            raw = (
                pd.read_csv(dataset_path)
                if str(dataset_path).endswith(".csv")
                else pd.read_parquet(dataset_path)
            )
            detector = TypeDetector()
            result = detector.detect_granularity(raw)
            return result.granularity == DatasetGranularity.EVENT_LEVEL
        except Exception:
            pass

    return False


def _has_text_columns_for_dataset(
    findings_dir: Path,
    dataset_name: str,
    dataset_path: Optional[str] = None,
) -> bool:
    """Determine if a dataset has TEXT columns using findings or direct detection."""
    from customer_retention.core.config.column_config import ColumnType

    # Signal 1: findings
    findings_path = _find_dataset_findings(findings_dir, dataset_name)
    if findings_path:
        try:
            from customer_retention.analysis.auto_explorer.findings import (
                ExplorationFindings,
            )

            findings = ExplorationFindings.load(str(findings_path))
            if findings.text_processing:
                return True
            if ColumnType.TEXT in findings.column_types.values():
                return True
            return False
        except Exception:
            pass

    # Signal 2: direct data inspection
    if dataset_path:
        try:
            import pandas as pd

            from customer_retention.stages.profiling import TypeDetector

            raw = (
                pd.read_csv(dataset_path)
                if str(dataset_path).endswith(".csv")
                else pd.read_parquet(dataset_path)
            )
            detector = TypeDetector()
            for col in raw.select_dtypes(include=["object"]).columns:
                result = detector.detect_type(raw[col], col)
                if result.inferred_type == ColumnType.TEXT:
                    return True
        except Exception:
            pass

    return False


def _detect_skip_set_for_dataset(
    findings_dir: Path,
    dataset_name: str,
    context=None,
    dataset_path: Optional[str] = None,
) -> Tuple[Set[str], Dict[str, str]]:
    """Return (notebooks_to_skip, reasons) scoped to a single dataset.

    Uses multiple signals: findings, dataset context, and direct data inspection.
    """
    skip: Set[str] = set()
    reasons: Dict[str, str] = {}

    is_event = _is_event_level_dataset(
        findings_dir, dataset_name, context=context, dataset_path=dataset_path,
    )
    if not is_event:
        for nb in TEMPORAL_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = f"no event-level data ({dataset_name})"

    has_text = _has_text_columns_for_dataset(
        findings_dir, dataset_name, dataset_path=dataset_path,
    )
    if not has_text:
        for nb in TEXT_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = f"no TEXT columns ({dataset_name})"

    return skip, reasons


def _has_event_level_data(findings_dir: Path) -> bool:
    multi_path = findings_dir / "multi_dataset_findings.yaml"
    if multi_path.exists():
        try:
            from customer_retention.analysis.auto_explorer.exploration_manager import (
                MultiDatasetFindings,
            )

            multi = MultiDatasetFindings.load(str(multi_path))
            if multi.event_datasets:
                return True
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Multi-dataset helpers
# ---------------------------------------------------------------------------


def _load_dataset_context(findings_dir: Path):
    """Load DatasetContext from findings, or None if unavailable."""
    context_path = findings_dir / "dataset_context.yaml"
    if not context_path.exists():
        return None
    try:
        from customer_retention.analysis.auto_explorer import DatasetContext

        ctx = DatasetContext.load(context_path)
        if ctx.datasets:
            return ctx
    except Exception:
        pass
    return None


def _ordered_datasets(context) -> List[Tuple[str, str]]:
    """Return (name, path) pairs with the target dataset first."""
    ordered: List[Tuple[str, str]] = []
    for name, entry in context.datasets.items():
        if entry.has_target:
            ordered.insert(0, (name, entry.path))
        else:
            ordered.append((name, entry.path))
    return ordered


def _set_data_path(notebook_path: Path, data_path: str) -> None:
    """Replace the active DATA_PATH assignment in a notebook config cell."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        is_list = isinstance(source, list)
        lines = source if is_list else source.splitlines(True)

        has_active = any(
            ln.lstrip().startswith("DATA_PATH")
            and "=" in ln
            and not ln.lstrip().startswith("#")
            for ln in lines
        )
        if not has_active:
            continue

        new_lines = []
        replaced = False
        for ln in lines:
            stripped = ln.lstrip()
            if (
                not replaced
                and stripped.startswith("DATA_PATH")
                and "=" in stripped
                and not stripped.startswith("#")
            ):
                indent = ln[: len(ln) - len(stripped)]
                new_lines.append(f'{indent}DATA_PATH = "{data_path}"\n')
                replaced = True
            else:
                new_lines.append(ln)

        if replaced:
            cell["source"] = new_lines if is_list else "".join(new_lines)
            notebook_path.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return


# ---------------------------------------------------------------------------
# Notebook resolution and execution
# ---------------------------------------------------------------------------


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
    except pm.PapermillExecutionError:
        return False, traceback.format_exc()
    except Exception:
        return False, traceback.format_exc()


def _execute_one(
    nb_path: Path,
    dataset_name: Optional[str],
    results: Dict[str, str],
    timings: Dict[str, float],
    skip_set: Set[str],
    skip_reasons: Dict[str, str],
    dry_run: bool,
    timeout: int,
    kernel: str,
    error_log: Optional[List[str]] = None,
) -> bool:
    """Execute a single notebook, updating results and timings.

    Returns True on success or skip, False on failure.
    """
    stem = nb_path.stem
    result_key = f"{stem}:{dataset_name}" if dataset_name else stem
    label = f"{stem} | {dataset_name}" if dataset_name else stem

    if stem in skip_set:
        results[result_key] = f"SKIPPED ({skip_reasons[stem]})"
        print(f"  [{label}] SKIPPED — {skip_reasons[stem]}")
        return True

    if dry_run:
        results[result_key] = "DRY_RUN"
        print(f"  [{label}] would run")
        return True

    print(f"  [{label}] running ...", end="", flush=True)
    start = time.time()
    ok, error = _run_notebook(nb_path, timeout=timeout, kernel=kernel)
    elapsed = time.time() - start
    timings[result_key] = elapsed

    if ok:
        results[result_key] = f"OK ({elapsed:.0f}s)"
        print(f"\n  [{label}] OK ({elapsed:.0f}s)")
    else:
        first_line = (error or "unknown error").split("\n")[-2][:200]
        results[result_key] = f"FAILED: {first_line}"
        print(f"\n  [{label}] FAILED ({elapsed:.0f}s)")
        print(f"    Error: {first_line}")
        if error_log is not None:
            error_log.append({
                "notebook": stem,
                "dataset": dataset_name,
                "elapsed": elapsed,
                "traceback": error or "unknown error",
            })

    return ok


# ---------------------------------------------------------------------------
# Flow orchestration
# ---------------------------------------------------------------------------


def _run_single_dataset_flow(
    notebooks: List[Path],
    findings_dir: Path,
    results: Dict[str, str],
    timings: Dict[str, float],
    dry_run: bool,
    timeout: int,
    kernel: str,
    error_log: Optional[List[str]] = None,
) -> None:
    """Original single-dataset sequential execution."""
    skip_set: Set[str] = set()
    skip_reasons: Dict[str, str] = {}
    skip_detected = False

    print(f"Findings directory: {findings_dir}")
    print("(skip detection deferred until after data discovery)\n")

    remaining = [nb for nb in notebooks if nb.stem not in SETUP_NOTEBOOKS]
    for nb_path in remaining:
        stem = nb_path.stem

        if not skip_detected and stem != _DISCOVERY_NOTEBOOK:
            skip_detected = True
            if findings_dir.exists():
                skip_set, skip_reasons = _detect_skip_set(findings_dir)
                if skip_set:
                    print(f"\nSkipping {len(skip_set)} notebooks based on findings:")
                    for nb in sorted(skip_set):
                        print(f"  - {nb}: {skip_reasons[nb]}")
                else:
                    print(
                        "\nAll remaining notebooks will run "
                        "(no skip conditions detected)"
                    )
                print()

        _execute_one(
            nb_path, None, results, timings,
            skip_set, skip_reasons, dry_run, timeout, kernel,
            error_log=error_log,
        )


def _run_multi_dataset_flow(
    notebooks_dir: Path,
    notebooks: List[Path],
    findings_dir: Path,
    context,
    results: Dict[str, str],
    timings: Dict[str, float],
    dry_run: bool,
    timeout: int,
    kernel: str,
    error_log: Optional[List[str]] = None,
) -> None:
    """Multi-dataset execution: per-dataset 01-04, then global 05-12."""
    datasets = _ordered_datasets(context)

    print(f"\nMulti-dataset mode: {len(datasets)} datasets detected")
    for name, path in datasets:
        role = "TARGET" if context.datasets[name].has_target else "source"
        print(f"  {name} ({role}): {path}")

    per_dataset_nbs = [nb for nb in notebooks if nb.stem in PER_DATASET_STEMS]
    global_nbs = [
        nb for nb in notebooks
        if nb.stem not in SETUP_NOTEBOOKS and nb.stem not in PER_DATASET_STEMS
    ]

    for ds_name, ds_path in datasets:
        role = "TARGET" if context.datasets[ds_name].has_target else "source"
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} ({role})")
        print(f"{'=' * 60}")

        if not dry_run:
            for nb_path in per_dataset_nbs:
                _set_data_path(nb_path, ds_path)

        skip_set: Set[str] = set()
        skip_reasons: Dict[str, str] = {}
        skip_detected = False

        for nb_path in per_dataset_nbs:
            stem = nb_path.stem

            if not skip_detected and stem != _DISCOVERY_NOTEBOOK:
                skip_detected = True
                if findings_dir.exists():
                    skip_set, skip_reasons = _detect_skip_set_for_dataset(
                        findings_dir, ds_name,
                        context=context,
                        dataset_path=ds_path,
                    )
                    if skip_set:
                        print(f"\n  Skipping for {ds_name}:")
                        for nb in sorted(skip_set):
                            print(f"    - {nb}: {skip_reasons[nb]}")
                    else:
                        print(
                            f"\n  All per-dataset notebooks will run for {ds_name}"
                        )
                    print()

            _execute_one(
                nb_path, ds_name, results, timings,
                skip_set, skip_reasons, dry_run, timeout, kernel,
                error_log=error_log,
            )

    if global_nbs:
        print(f"\n{'=' * 60}")
        print("Global analysis (all datasets processed)")
        print(f"{'=' * 60}")

        for nb_path in global_nbs:
            _execute_one(
                nb_path, None, results, timings,
                set(), {}, dry_run, timeout, kernel,
                error_log=error_log,
            )


def run_all(
    notebooks_dir: Path,
    findings_dir: Optional[Path] = None,
    docs_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: int = 600,
    kernel: str = "python3",
) -> Dict[str, str]:
    """Run all notebooks, returning {key: status} for each."""
    if findings_dir is None:
        findings_dir = notebooks_dir.parent / "experiments" / "findings"

    notebooks = _resolve_notebooks(notebooks_dir)
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return {}

    total_start = time.time()
    results: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    error_log: List[str] = []

    # Phase 1: Setup (00_start_here)
    setup_nbs = [nb for nb in notebooks if nb.stem in SETUP_NOTEBOOKS]
    for nb_path in setup_nbs:
        _execute_one(
            nb_path, None, results, timings,
            set(), {}, dry_run, timeout, kernel,
            error_log=error_log,
        )

    # Phase 2: Detect multi-dataset from dataset_context.yaml
    context = _load_dataset_context(findings_dir)
    is_multi = context is not None and len(context.datasets) > 1

    if is_multi:
        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run, timeout, kernel,
            error_log=error_log,
        )
    else:
        _run_single_dataset_flow(
            notebooks, findings_dir, results, timings,
            dry_run, timeout, kernel,
            error_log=error_log,
        )

    _print_summary(results, timings, time.time() - total_start)

    if not dry_run:
        if docs_dir is None:
            docs_dir = notebooks_dir.parent / "docs" / "tutorial" / "customer-emails"
        _export_html(notebooks_dir, docs_dir)

    _write_error_log(notebooks_dir, error_log)

    return results


# ---------------------------------------------------------------------------
# Reporting and export
# ---------------------------------------------------------------------------


def _write_error_log(notebooks_dir: Path, error_log: List[Dict]) -> None:
    """Write full exception tracebacks to run_exploration.log, grouped by file and notebook."""
    log_path = notebooks_dir / "run_exploration.log"
    if not error_log:
        if log_path.exists():
            log_path.unlink()
        return

    from collections import OrderedDict

    grouped: OrderedDict[str, List[Dict]] = OrderedDict()
    for entry in error_log:
        key = entry["dataset"] or "(single dataset)"
        grouped.setdefault(key, []).append(entry)

    lines = [
        f"run_exploration — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{len(error_log)} error(s) across {len(grouped)} dataset(s)",
        "",
    ]

    for dataset, entries in grouped.items():
        lines.append("=" * 70)
        lines.append(f"  Dataset: {dataset}")
        lines.append(f"  Errors:  {len(entries)}")
        lines.append("=" * 70)
        lines.append("")

        for entry in entries:
            nb = entry["notebook"]
            elapsed = entry["elapsed"]
            lines.append(f"  --- Notebook: {nb}  ({elapsed:.0f}s) ---")
            lines.append("")
            for tb_line in entry["traceback"].rstrip().splitlines():
                lines.append(f"    {tb_line}")
            lines.append("")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFull error details written to {log_path}")



def _export_html(notebooks_dir: Path, docs_dir: Path) -> None:
    """Export executed notebooks to HTML in the docs directory."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from export_tutorial_html import ExportConfig, TutorialExporter

        config = ExportConfig(output_dir=docs_dir)
        exporter = TutorialExporter(config)

        notebooks = sorted(notebooks_dir.glob("*.ipynb"))
        if not notebooks:
            print(f"\nNo notebooks found in {notebooks_dir} for HTML export")
            return

        print(f"\nExporting {len(notebooks)} notebooks to HTML in {docs_dir} ...")
        exported = []
        for nb_path in notebooks:
            result = exporter.export_notebook(nb_path)
            if result:
                exported.append(result)

        if exported:
            exporter.create_index_page(exported, "Customer Retention Tutorial")
        print(f"HTML export complete: {len(exported)} notebooks exported")
    except Exception as e:
        print(f"\nHTML export failed: {e}")


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m {secs}s"


def _print_summary(
    results: Dict[str, str],
    timings: Dict[str, float],
    total_elapsed: float,
) -> None:
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
        for key, status in results.items():
            if status.startswith("FAILED"):
                print(f"  - {key}: {status}")
    if timings:
        max_name_len = max(len(s) for s in timings)
        print("\nTiming:")
        for key, elapsed in timings.items():
            print(f"  {key:<{max_name_len}}  {_format_duration(elapsed)}")
        print(f"  {'─' * max_name_len}  ────────")
        print(f"  {'Total':<{max_name_len}}  {_format_duration(total_elapsed)}")
    else:
        print(f"\nTotal: {_format_duration(total_elapsed)}")


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
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Output directory for HTML export (default: docs/tutorial/customer-emails)",
    )
    args = parser.parse_args()
    notebooks_dir = Path(args.notebooks_dir).resolve()
    findings_dir = Path(args.findings_dir).resolve() if args.findings_dir else None
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else None
    results = run_all(
        notebooks_dir,
        findings_dir=findings_dir,
        docs_dir=docs_dir,
        dry_run=args.dry_run,
        timeout=args.timeout,
        kernel=args.kernel,
    )
    if any(v.startswith("FAILED") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
