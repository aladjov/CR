#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from customer_retention.analysis.auto_explorer.skip_logic import (
    CRITICAL_GLOBAL,
    NOTEBOOKS_ORDER,
    PER_DATASET_STEMS,
    SETUP_NOTEBOOKS,
)
from customer_retention.analysis.auto_explorer.skip_logic import (
    detect_global_skip_set as _detect_global_skip_set,
)
from customer_retention.analysis.auto_explorer.skip_logic import (
    detect_skip_set_for_dataset as _detect_skip_set_for_dataset,
)
from customer_retention.analysis.auto_explorer.skip_logic import (
    find_dataset_findings as _find_dataset_findings,  # noqa: F401
)
from customer_retention.analysis.auto_explorer.skip_logic import (
    has_text_columns_for_dataset as _has_text_columns_for_dataset,  # noqa: F401
)
from customer_retention.analysis.auto_explorer.skip_logic import (
    is_event_level_dataset as _is_event_level_dataset,  # noqa: F401
)

_DISCOVERY_NOTEBOOK = "01_data_discovery"

_NOTEBOOK_TIMEOUT_MULTIPLIERS = {
    "08_baseline_experiments": 3,
    "10_spec_generation": 5,
}


# ---------------------------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------------------------


def _resolve_namespace(findings_dir: Path, run_id: Optional[str]):
    experiments_dir = findings_dir.parent if findings_dir.name == "findings" else findings_dir
    from customer_retention.analysis.auto_explorer.run_namespace import (
        RunNamespace,
    )

    if run_id:
        return RunNamespace(root=experiments_dir, run_id=run_id)
    return RunNamespace.from_latest(experiments_dir)


# ---------------------------------------------------------------------------
# Findings helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-dataset helpers
# ---------------------------------------------------------------------------


def _load_dataset_context(findings_dir: Path, run_id: Optional[str] = None):
    """Load ProjectContext from run namespace."""
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

    experiments_dir = findings_dir.parent if findings_dir.name == "findings" else findings_dir
    if run_id:
        ns = RunNamespace(root=experiments_dir, run_id=run_id)
    else:
        ns = RunNamespace.from_env_or_latest(root=experiments_dir)
    if ns is None:
        return None
    ns_path = ns.project_context_path
    if ns_path.exists():
        from customer_retention.analysis.auto_explorer import ProjectContext

        ctx = ProjectContext.load(ns_path)
        if ctx.datasets:
            return ctx
    return None


def _ordered_datasets(context) -> List[Tuple[str, str]]:
    """Return (name, path) pairs with the target dataset first."""
    ordered: List[Tuple[str, str]] = []
    for name, entry in context.datasets.items():
        is_target = getattr(entry, "has_target", False) or getattr(entry, "role", None) == "target"
        if is_target:
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
            ln.lstrip().startswith("DATA_PATH") and "=" in ln and not ln.lstrip().startswith("#") for ln in lines
        )
        if not has_active:
            continue

        new_lines = []
        replaced = False
        for ln in lines:
            stripped = ln.lstrip()
            if not replaced and stripped.startswith("DATA_PATH") and "=" in stripped and not stripped.startswith("#"):
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


def _resolve_notebooks(notebooks_dir: Path, start_notebook: Optional[str] = None) -> List[Path]:
    """Return ordered list of existing notebook paths, optionally starting from a given notebook."""
    result = []
    started = start_notebook is None
    for stem in NOTEBOOKS_ORDER:
        if not started:
            if stem == start_notebook or stem.startswith(start_notebook):
                started = True
            else:
                continue
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

    stem = notebook_path.stem
    prev_batch = os.environ.get("CR_BATCH_EXECUTION")
    os.environ["CR_BATCH_EXECUTION"] = "1"
    try:
        pm.execute_notebook(
            str(notebook_path),
            str(notebook_path),
            kernel_name=kernel,
            request_save_on_cell_execute=True,
            cwd=str(notebook_path.parent),
            progress_bar={"desc": f"  Executing ({stem.split('_', 1)[0]})"},
            execution_timeout=timeout,
        )
        return True, None
    except pm.PapermillExecutionError:
        return False, traceback.format_exc()
    except Exception:
        return False, traceback.format_exc()
    finally:
        if prev_batch is None:
            os.environ.pop("CR_BATCH_EXECUTION", None)
        else:
            os.environ["CR_BATCH_EXECUTION"] = prev_batch


_CONNECTION_ERROR_PATTERNS = [
    "SparkConnectGrpcException",
    "UNAVAILABLE",
    "connection",
    "timed out",
    "timeout",
    "BrokenPipeError",
    "ConnectionResetError",
    "EOFError",
]


def _is_connection_error(error_text: str) -> bool:
    lower = error_text.lower()
    return any(pat.lower() in lower for pat in _CONNECTION_ERROR_PATTERNS)


def _restart_spark_session() -> bool:
    try:
        from customer_retention.core.compat.detection import connect_remote_spark

        connect_remote_spark()
        print("    Spark session restarted successfully")
        return True
    except Exception as exc:
        print(f"    Failed to restart Spark session: {exc}")
        return False


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
    error_log: Optional[ProgressiveErrorLog] = None,
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

    effective_timeout = int(timeout * _NOTEBOOK_TIMEOUT_MULTIPLIERS.get(stem, 1))
    spark_remote = os.environ.get("CR_SPARK_REMOTE") == "1"
    max_attempts = 2 if spark_remote else 1

    for attempt in range(1, max_attempts + 1):
        start = time.time()
        ok, error = _run_notebook(nb_path, timeout=effective_timeout, kernel=kernel)
        elapsed = time.time() - start
        timings[result_key] = elapsed

        if ok:
            results[result_key] = f"OK ({elapsed:.0f}s)"
            print(f"  [{label}] OK ({elapsed:.0f}s)")
            if error_log is not None:
                error_log.append_success(stem, dataset_name, elapsed)
            return True

        first_line = (error or "unknown error").split("\n")[-2][:200]

        if attempt < max_attempts and _is_connection_error(error or ""):
            print(f"  [{label}] connection error ({elapsed:.0f}s), restarting Spark...")
            if _restart_spark_session():
                print(f"  [{label}] retrying after Spark restart...")
                continue
            print("    Retry aborted — could not restart Spark session")

        break

    results[result_key] = f"FAILED: {first_line}"
    print(f"  [{label}] FAILED ({elapsed:.0f}s)")
    print(f"    Error: {first_line}")
    if error_log is not None:
        error_log.append_error(
            {
                "notebook": stem,
                "dataset": dataset_name,
                "elapsed": elapsed,
                "traceback": error or "unknown error",
            }
        )

    return False


# ---------------------------------------------------------------------------
# Flow orchestration
# ---------------------------------------------------------------------------


def _run_single_dataset_flow(
    notebooks_dir: Path,
    notebooks: List[Path],
    findings_dir: Path,
    context,
    results: Dict[str, str],
    timings: Dict[str, float],
    dry_run: bool,
    timeout: int,
    kernel: str,
    error_log: Optional[ProgressiveErrorLog] = None,
    namespace=None,
) -> None:
    """Single-dataset sequential execution.

    Patches per-dataset notebooks with the project context dataset path
    so hardcoded DATA_PATH values don't cause a mismatch.
    """
    skip_set: Set[str] = set()
    skip_reasons: Dict[str, str] = {}
    skip_detected = False

    # Patch per-dataset notebooks to match the project context (source of truth)
    if context and context.datasets and not dry_run:
        ds_name = next(iter(context.datasets))
        ds_path = getattr(context.datasets[ds_name], "path", None)
        if ds_path:
            for nb in notebooks:
                if nb.stem in PER_DATASET_STEMS:
                    _set_data_path(nb, ds_path)

    print(f"Findings directory: {findings_dir}")
    print("(skip detection deferred until after data discovery)\n")

    remaining = [nb for nb in notebooks if nb.stem not in SETUP_NOTEBOOKS]
    for nb_path in remaining:
        stem = nb_path.stem

        if not skip_detected and stem != _DISCOVERY_NOTEBOOK:
            skip_detected = True
            dataset_name = None
            dataset_path = None
            if context and context.datasets:
                ds_name = next(iter(context.datasets))
                dataset_name = ds_name
                entry = context.datasets[ds_name]
                dataset_path = getattr(entry, "path", None)

            if dataset_name:
                skip_set, skip_reasons = _detect_skip_set_for_dataset(
                    findings_dir,
                    dataset_name,
                    context=context,
                    dataset_path=dataset_path,
                    namespace=namespace,
                )
            else:
                skip_set, skip_reasons = _detect_skip_set_for_dataset(
                    findings_dir,
                    "",
                    namespace=namespace,
                )

            if skip_set:
                print(f"\nSkipping {len(skip_set)} notebooks based on findings:")
                for nb in sorted(skip_set):
                    print(f"  - {nb}: {skip_reasons[nb]}")
            else:
                print("\nAll remaining notebooks will run (no skip conditions detected)")
            print()

        _execute_one(
            nb_path,
            None,
            results,
            timings,
            skip_set,
            skip_reasons,
            dry_run,
            timeout,
            kernel,
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
    error_log: Optional[ProgressiveErrorLog] = None,
    namespace=None,
) -> None:
    datasets = _ordered_datasets(context)

    print(f"\nMulti-dataset mode: {len(datasets)} datasets detected")
    for name, path in datasets:
        role = (
            "TARGET"
            if getattr(context.datasets[name], "has_target", False)
            or getattr(context.datasets[name], "role", None) == "target"
            else "source"
        )
        print(f"  {name} ({role}): {path}")

    per_dataset_nbs = [nb for nb in notebooks if nb.stem in PER_DATASET_STEMS]
    global_nbs = [nb for nb in notebooks if nb.stem not in SETUP_NOTEBOOKS and nb.stem not in PER_DATASET_STEMS]

    for ds_name, ds_path in datasets:
        os.environ["CR_DATASET_ID"] = ds_name

        role = (
            "TARGET"
            if getattr(context.datasets[ds_name], "has_target", False)
            or getattr(context.datasets[ds_name], "role", None) == "target"
            else "source"
        )
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
                skip_set, skip_reasons = _detect_skip_set_for_dataset(
                    findings_dir,
                    ds_name,
                    context=context,
                    dataset_path=ds_path,
                    namespace=namespace,
                )
                if skip_set:
                    print(f"\n  Skipping for {ds_name}:")
                    for nb in sorted(skip_set):
                        print(f"    - {nb}: {skip_reasons[nb]}")
                else:
                    print(f"\n  All per-dataset notebooks will run for {ds_name}")
                print()

            _execute_one(
                nb_path,
                ds_name,
                results,
                timings,
                skip_set,
                skip_reasons,
                dry_run,
                timeout,
                kernel,
                error_log=error_log,
            )

    os.environ["CR_DATASET_ID"] = datasets[0][0]

    if global_nbs:
        print(f"\n{'=' * 60}")
        print("Global analysis (all datasets processed)")
        print(f"{'=' * 60}")

        global_skip: Set[str] = set()
        global_reasons: Dict[str, str] = {}
        global_skip, global_reasons = _detect_global_skip_set(
            findings_dir,
            context,
            namespace=namespace,
        )
        if global_skip:
            print("\n  Global skip:")
            for nb in sorted(global_skip):
                print(f"    - {nb}: {global_reasons[nb]}")
            print()

        for nb_path in global_nbs:
            ok = _execute_one(
                nb_path,
                None,
                results,
                timings,
                global_skip,
                global_reasons,
                dry_run,
                timeout,
                kernel,
                error_log=error_log,
            )
            if not ok and nb_path.stem in CRITICAL_GLOBAL:
                break


def _detect_run_id_from_context(findings_dir: Path) -> Optional[str]:
    experiments_dir = findings_dir.parent if findings_dir.name == "findings" else findings_dir
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

    ns = RunNamespace.from_env_or_latest(root=experiments_dir)
    if ns:
        return ns.run_id
    return None


def run_all(
    notebooks_dir: Path,
    findings_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: int = 600,
    kernel: str = "python3",
    run_id: Optional[str] = None,
    start_notebook: Optional[str] = None,
) -> Dict[str, str]:
    if findings_dir is None:
        findings_dir = notebooks_dir.parent / "experiments" / "findings"

    notebooks = _resolve_notebooks(notebooks_dir, start_notebook=start_notebook)
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return {}

    total_start = time.time()
    results: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    run_params = {
        "notebooks_dir": str(notebooks_dir),
        "findings_dir": str(findings_dir),
        "timeout": timeout,
        "kernel": kernel,
        "dry_run": dry_run,
        "run_id": run_id,
        "spark_remote": os.environ.get("CR_SPARK_REMOTE") == "1",
        "sample_entities": os.environ.get("CR_SAMPLE_ENTITY_COUNT"),
        "max_grid_dates": os.environ.get("CR_GRID_MAX_DATES"),
    }
    error_log = ProgressiveErrorLog(notebooks_dir, run_params=run_params)

    prev_run_id = os.environ.get("CR_RUN_ID")
    if run_id:
        os.environ["CR_RUN_ID"] = run_id

    # Phase 1: Setup (00_start_here)
    setup_nbs = [nb for nb in notebooks if nb.stem in SETUP_NOTEBOOKS]
    for nb_path in setup_nbs:
        _execute_one(
            nb_path,
            None,
            results,
            timings,
            set(),
            {},
            dry_run,
            timeout,
            kernel,
            error_log=error_log,
        )

    if not run_id:
        run_id = _detect_run_id_from_context(findings_dir)
        if run_id:
            os.environ["CR_RUN_ID"] = run_id

    namespace = _resolve_namespace(findings_dir, run_id)
    if namespace and not dry_run:
        namespace.write_run_pointer()

    context = _load_dataset_context(findings_dir, run_id=run_id)
    is_multi = context is not None and len(context.datasets) > 1

    if is_multi:
        _run_multi_dataset_flow(
            notebooks_dir,
            notebooks,
            findings_dir,
            context,
            results,
            timings,
            dry_run,
            timeout,
            kernel,
            error_log=error_log,
            namespace=namespace,
        )
    else:
        _run_single_dataset_flow(
            notebooks_dir,
            notebooks,
            findings_dir,
            context,
            results,
            timings,
            dry_run,
            timeout,
            kernel,
            error_log=error_log,
            namespace=namespace,
        )

    _print_summary(results, timings, time.time() - total_start)

    error_log.finalize()

    if prev_run_id is None:
        os.environ.pop("CR_RUN_ID", None)
    else:
        os.environ["CR_RUN_ID"] = prev_run_id

    return results


# ---------------------------------------------------------------------------
# Reporting and export
# ---------------------------------------------------------------------------


class ProgressiveErrorLog:
    def __init__(self, notebooks_dir: Path, run_params: Optional[Dict] = None):
        self.log_path = notebooks_dir / "run_exploration.log"
        self.error_count = 0
        header = f"run_exploration — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        if run_params:
            header += self._format_run_params(run_params)
        header += "\n"
        self.log_path.write_text(header, encoding="utf-8")

    @staticmethod
    def _format_run_params(params: Dict) -> str:
        lines: List[str] = []
        _add = lines.append
        _add(f"  notebooks_dir: {params.get('notebooks_dir', '')}")
        _add(f"  findings_dir:  {params.get('findings_dir', '')}")
        _add(f"  timeout:       {params.get('timeout', 600)}")
        _add(f"  kernel:        {params.get('kernel', 'python3')}")
        if params.get("dry_run"):
            _add("  dry_run:       yes")
        if params.get("run_id"):
            _add(f"  run_id:        {params['run_id']}")
        if params.get("spark_remote"):
            _add("  spark_remote:  yes")
        if params.get("sample_entities"):
            _add(f"  sample_entities: {params['sample_entities']}")
        if params.get("max_grid_dates"):
            _add(f"  max_grid_dates:  {params['max_grid_dates']}")
        if params.get("light"):
            _add("  light:         yes (baseline training capped)")
        return "\n" + "\n".join(lines) + "\n"

    def append_error(self, entry: Dict) -> None:
        self.error_count += 1
        nb = entry["notebook"]
        elapsed = entry["elapsed"]
        dataset = entry["dataset"] or "(single dataset)"
        lines = [
            f"  --- Notebook: {nb} | Dataset: {dataset}  ({elapsed:.0f}s) ---",
            "",
        ]
        for tb_line in entry["traceback"].rstrip().splitlines():
            lines.append(f"    {tb_line}")
        lines.append("")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def append_success(self, stem: str, dataset: Optional[str], elapsed: float) -> None:
        label = f"{stem} | {dataset}" if dataset else stem
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"  [{label}] OK ({elapsed:.0f}s)\n")

    def finalize(self) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\nTotal errors: {self.error_count}\n")
        if self.error_count:
            print(f"\nFull error details written to {self.log_path}")
        else:
            print(f"\nAll notebooks succeeded. Log at {self.log_path}")


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
        "--run-id",
        default=None,
        help="Run namespace ID for session-based dataset resolution",
    )
    parser.add_argument(
        "--spark-remote",
        action="store_true",
        default=False,
        help="Execute via Databricks Connect (sets CR_SPARK_REMOTE=1)",
    )
    parser.add_argument(
        "--sample-entities",
        type=int,
        default=None,
        help="Limit to N entity_ids across all datasets (sets CR_SAMPLE_ENTITY_COUNT)",
    )
    parser.add_argument(
        "--max-grid-dates",
        type=int,
        default=None,
        help="Cap snapshot grid to N dates (sets CR_GRID_MAX_DATES)",
    )
    parser.add_argument(
        "--start-notebook",
        default=None,
        help="Start from this notebook stem (e.g. 08_baseline or just 08), skipping earlier ones",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        default=False,
        help="Light mode: cap baseline training data to 200k rows via entity sampling (sets CR_MAX_BASELINE_ROWS)",
    )
    args = parser.parse_args()
    if args.sample_entities:
        os.environ["CR_SAMPLE_ENTITY_COUNT"] = str(args.sample_entities)
    if args.max_grid_dates:
        os.environ["CR_GRID_MAX_DATES"] = str(args.max_grid_dates)
    if args.light:
        os.environ.setdefault("CR_MAX_BASELINE_ROWS", "200000")
    if args.spark_remote:
        os.environ["CR_SPARK_REMOTE"] = "1"
    else:
        _spark_connect_vars = [
            "SPARK_CONNECT_USER_AGENT",
            "DATABRICKS_METADATA_SERVICE_URL",
            "SPARK_REMOTE",
            "DATABRICKS_HOST",
            "DATABRICKS_CLUSTER_ID",
            "DATABRICKS_AUTH_TYPE",
        ]
        for _var in _spark_connect_vars:
            os.environ.pop(_var, None)
    notebooks_dir = Path(args.notebooks_dir).resolve()
    findings_dir = Path(args.findings_dir).resolve() if args.findings_dir else None
    results = run_all(
        notebooks_dir,
        findings_dir=findings_dir,
        dry_run=args.dry_run,
        timeout=args.timeout,
        kernel=args.kernel,
        run_id=args.run_id,
        start_notebook=args.start_notebook,
    )
    if any(v.startswith("FAILED") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
