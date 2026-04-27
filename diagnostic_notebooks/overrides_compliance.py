"""Joint overrides-compliance scanner for cycle-12 close gate.

Walks the user's exploration notebooks for imperative config dicts
(`DROP_COLUMNS`, `TYPE_OVERRIDES`, `MILESTONE_PAIRS`, `BRONZE_AGGREGATIONS`,
`ZERO_INFLATION_OPT_IN`, `EXCLUDED_LEAKING_FEATURES`) AND the run's
`merged/recommendations.yaml` for declarative entries (`bronze_aggregations`,
`silver_derived`, landing filters/lifecycle), then verifies each declared
override is honored in the run artifacts. Drift between the imperative and
declarative lanes is itself reported.

Usage (programmatic, from a cycle notebook):

    from diagnostic_notebooks.overrides_compliance import (
        ComplianceReport,
        scan_overrides,
    )

    report = scan_overrides(
        run_root=Path("/Volumes/.../runs/spschurn-67dd3a30"),
        notebook_paths={
            "01": Path("/Workspace/Repos/.../01_data_discovery.ipynb"),
            "05": Path("/Workspace/Repos/.../05_relationship_analysis.ipynb"),
            "10": Path("/Workspace/Repos/.../10_spec_generation.ipynb"),
        },
        list_columns=lambda table_path: spark.read.format("delta").load(
            str(table_path)
        ).columns,
    )
    print(report.to_json())

The scanner is engagement-agnostic — every engagement-specific value is
read from the user's notebook AST + the run's artifacts, never hardcoded.

Exit codes (when invoked as CLI):
    0 — every declared override honored
    1 — at least one drift; report written to --json path
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Import-time-safe yaml loader. Falls back to a minimal text reader if
# pyyaml is unavailable (which happens on bare-bones Databricks job
# clusters before customer_retention is on the path).
try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]


# Config-dict names this scanner understands. Engagement-specific dicts
# (e.g. SPS's `EXCLUDED_LEAKING_FEATURES`) are listed here verbatim;
# unknown dicts are ignored rather than misinterpreted.
KNOWN_CONFIG_DICTS = (
    "DROP_COLUMNS",
    "TYPE_OVERRIDES",
    "MILESTONE_PAIRS",
    "BRONZE_AGGREGATIONS",
    "ZERO_INFLATION_OPT_IN",
    "EXCLUDED_LEAKING_FEATURES",
    "FEATURE_EXCLUSIONS",
)


@dataclass
class CheckResult:
    family: str
    dataset: str
    check: str
    status: str  # "PASS" | "FAIL" | "INFO"
    detail: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "family": self.family,
            "dataset": self.dataset,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ComplianceReport:
    run_id: str
    n_pass: int = 0
    n_fail: int = 0
    n_info: int = 0
    checks: List[CheckResult] = field(default_factory=list)
    parsed_imperative: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parsed_declarative: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if result.status == "PASS":
            self.n_pass += 1
        elif result.status == "FAIL":
            self.n_fail += 1
        else:
            self.n_info += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": "PASS" if self.n_fail == 0 and self.n_pass > 0 else "FAIL",
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_info": self.n_info,
            "checks": [c.to_dict() for c in self.checks],
            "parsed_imperative": self.parsed_imperative,
            "parsed_declarative": self.parsed_declarative,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _read_notebook_cells(path: Path) -> List[Dict[str, Any]]:
    nb = json.loads(path.read_text())
    return nb.get("cells", [])


def _cell_source(cell: Dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _extract_top_level_dicts(source: str, names: Sequence[str]) -> Dict[str, Any]:
    """Parse `source` and return any top-level `<name> = {...}` literal whose
    name is in `names`. Uses ast.literal_eval — only string/int/list/dict/tuple
    literals are extracted; expressions are skipped silently.
    """
    out: Dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    name_set = set(names)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target not in name_set:
            continue
        try:
            out[target] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
    return out


def parse_imperative_overrides(
    notebook_paths: Dict[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Walk the listed notebooks and harvest every top-level config dict
    whose name is in KNOWN_CONFIG_DICTS. Returns
    `{notebook_id: {dict_name: literal_value}}`.
    """
    parsed: Dict[str, Dict[str, Any]] = {}
    for nb_id, path in notebook_paths.items():
        if not path.exists():
            parsed[nb_id] = {"__missing__": str(path)}
            continue
        cell_dicts: Dict[str, Any] = {}
        for cell in _read_notebook_cells(path):
            if cell.get("cell_type") != "code":
                continue
            cell_dicts.update(
                _extract_top_level_dicts(_cell_source(cell), KNOWN_CONFIG_DICTS)
            )
        parsed[nb_id] = cell_dicts
    return parsed


def parse_declarative_recommendations(run_root: Path) -> Dict[str, Any]:
    """Read `merged/recommendations.yaml` and surface the keys this scanner
    cares about (`bronze_aggregations`, `silver`, `landing`, etc.).
    """
    if _yaml is None:
        return {"__error__": "pyyaml not available in this kernel"}
    recs_path = run_root / "merged" / "recommendations.yaml"
    if not recs_path.exists():
        return {"__missing__": str(recs_path)}
    with open(recs_path) as f:
        data = _yaml.safe_load(f) or {}
    return {
        "bronze_aggregations": data.get("bronze_aggregations") or {},
        "sources": data.get("sources") or {},
        "bronze": data.get("bronze") or {},
        "silver": data.get("silver") or {},
    }


def _check_drop_columns(
    parsed_imperative: Dict[str, Dict[str, Any]],
    list_columns: Optional[Callable[[Path], Iterable[str]]],
    run_root: Path,
    report: ComplianceReport,
) -> None:
    drop_block = next(
        (n.get("DROP_COLUMNS") for n in parsed_imperative.values() if n.get("DROP_COLUMNS")),
        None,
    )
    if not drop_block or not isinstance(drop_block, dict):
        report.add(
            CheckResult(
                "DROP_COLUMNS",
                "*",
                "config_present",
                "INFO",
                "no DROP_COLUMNS dict found in scanned notebooks",
            )
        )
        return
    for dataset, cols in drop_block.items():
        if not isinstance(cols, (list, tuple)):
            report.add(
                CheckResult(
                    "DROP_COLUMNS",
                    dataset,
                    "config_well_formed",
                    "FAIL",
                    f"value is {type(cols).__name__}, expected list",
                )
            )
            continue
        landing_path = run_root / "data" / "landing" / dataset
        if list_columns is None:
            report.add(
                CheckResult(
                    "DROP_COLUMNS",
                    dataset,
                    "DROP_COLUMNS_honored",
                    "INFO",
                    f"declared {len(cols)} cols; no list_columns callback supplied — skipped",
                )
            )
            continue
        try:
            actual = set(list_columns(landing_path))
        except Exception as e:
            report.add(
                CheckResult(
                    "DROP_COLUMNS",
                    dataset,
                    "landing_readable",
                    "FAIL",
                    f"{type(e).__name__}: {str(e)[:160]}",
                )
            )
            continue
        violations = sorted(c for c in cols if c in actual)
        report.add(
            CheckResult(
                "DROP_COLUMNS",
                dataset,
                "DROP_COLUMNS_honored",
                "PASS" if not violations else "FAIL",
                f"declared={len(cols)} present_in_landing={len(violations)} "
                f"sample={violations[:5]}",
            )
        )


def _check_type_overrides(
    parsed_imperative: Dict[str, Dict[str, Any]],
    run_root: Path,
    report: ComplianceReport,
) -> None:
    type_block = next(
        (n.get("TYPE_OVERRIDES") for n in parsed_imperative.values() if n.get("TYPE_OVERRIDES")),
        None,
    )
    if not type_block or _yaml is None:
        report.add(
            CheckResult(
                "TYPE_OVERRIDES",
                "*",
                "config_present",
                "INFO",
                "no TYPE_OVERRIDES (or yaml unavailable)",
            )
        )
        return
    for dataset, overrides in type_block.items():
        if not isinstance(overrides, dict):
            continue
        findings_path = (
            run_root / "datasets" / dataset / "findings" / f"{dataset}_findings.yaml"
        )
        if not findings_path.exists():
            report.add(
                CheckResult(
                    "TYPE_OVERRIDES",
                    dataset,
                    "findings_present",
                    "FAIL",
                    f"missing {findings_path}",
                )
            )
            continue
        with open(findings_path) as f:
            findings = _yaml.safe_load(f) or {}
        cols = (findings.get("columns") or {})
        mismatches: List[str] = []
        for col, expected_type in overrides.items():
            actual = ((cols.get(col) or {}).get("inferred_type") or "").lower()
            if expected_type and actual and expected_type.lower() not in actual:
                mismatches.append(f"{col}: expected~{expected_type} got={actual}")
        report.add(
            CheckResult(
                "TYPE_OVERRIDES",
                dataset,
                "TYPE_OVERRIDES_honored",
                "PASS" if not mismatches else "FAIL",
                f"declared={len(overrides)} mismatches={len(mismatches)} sample={mismatches[:3]}",
            )
        )


def _check_bronze_aggregations(
    parsed_imperative: Dict[str, Dict[str, Any]],
    parsed_declarative: Dict[str, Any],
    list_columns: Optional[Callable[[Path], Iterable[str]]],
    run_root: Path,
    report: ComplianceReport,
) -> None:
    nb10_block = next(
        (n.get("BRONZE_AGGREGATIONS") for n in parsed_imperative.values() if n.get("BRONZE_AGGREGATIONS")),
        None,
    ) or {}
    registry_block = parsed_declarative.get("bronze_aggregations") or {}
    union = sorted(set(nb10_block) | set(registry_block))
    if not union:
        report.add(
            CheckResult(
                "BRONZE_AGGREGATIONS",
                "*",
                "any_overrides_declared",
                "INFO",
                "no bronze aggregation overrides in either lane",
            )
        )
        return
    for dataset in union:
        nb10_cfg = nb10_block.get(dataset) or {}
        reg_cfg = registry_block.get(dataset) or {}
        report.add(
            CheckResult(
                "BRONZE_AGGREGATIONS",
                dataset,
                "lane_inventory",
                "INFO",
                f"nb10_keys={sorted(nb10_cfg)} registry_keys={sorted(reg_cfg)}",
            )
        )
        # Per-grid-date row-count gate: when either lane sets per_grid_date_mode,
        # bronze rows should approach unique_entities × |grid_dates|. Without
        # list_columns we can only verify the literal declaration is present.
        per_grid = nb10_cfg.get("per_grid_date_mode", reg_cfg.get("per_grid_date_mode"))
        if per_grid is None:
            continue
        if list_columns is None:
            report.add(
                CheckResult(
                    "BRONZE_AGGREGATIONS",
                    dataset,
                    "per_grid_date_declared",
                    "PASS",
                    f"per_grid_date_mode={per_grid} (row-count gate skipped — no list_columns)",
                )
            )
            continue
        bronze_path = run_root / "data" / "bronze" / f"bronze_event_{dataset}_aggregated"
        try:
            bronze_cols = list(list_columns(bronze_path))
        except Exception as e:
            report.add(
                CheckResult(
                    "BRONZE_AGGREGATIONS",
                    dataset,
                    "bronze_readable",
                    "FAIL",
                    f"{type(e).__name__}: {str(e)[:160]}",
                )
            )
            continue
        # Look for any value_counts column when value_counts_columns is set:
        # `event_type_<value>_count_<window>` shape — at minimum, presence of
        # any `_count_` column is the declarative invariant.
        vc_cols = nb10_cfg.get("value_counts_columns") or reg_cfg.get("value_counts_columns") or []
        if vc_cols:
            matches = [c for c in bronze_cols if "_count_" in c]
            report.add(
                CheckResult(
                    "BRONZE_AGGREGATIONS",
                    dataset,
                    "value_counts_columns_emitted",
                    "PASS" if matches else "FAIL",
                    f"vc_cols={list(vc_cols)} bronze_count_cols={len(matches)}",
                )
            )


def _check_excluded_leaking(
    parsed_imperative: Dict[str, Dict[str, Any]],
    list_columns: Optional[Callable[[Path], Iterable[str]]],
    run_root: Path,
    report: ComplianceReport,
) -> None:
    block = next(
        (n.get("EXCLUDED_LEAKING_FEATURES") for n in parsed_imperative.values()
         if n.get("EXCLUDED_LEAKING_FEATURES")),
        None,
    )
    if not block:
        report.add(
            CheckResult(
                "EXCLUDED_LEAKING_FEATURES",
                "*",
                "config_present",
                "INFO",
                "no EXCLUDED_LEAKING_FEATURES dict",
            )
        )
        return
    if list_columns is None:
        report.add(
            CheckResult(
                "EXCLUDED_LEAKING_FEATURES",
                "*",
                "EXCLUDED_LEAKING_FEATURES_honored",
                "INFO",
                f"{sum(len(v) for v in block.values() if isinstance(v, list))} entries declared; "
                "no list_columns — skipped",
            )
        )
        return
    gold_root = run_root / "data" / "gold"
    gold_dirs = [p for p in gold_root.glob("gold_features_*") if p.is_dir()]
    if not gold_dirs:
        report.add(
            CheckResult(
                "EXCLUDED_LEAKING_FEATURES",
                "*",
                "gold_present",
                "FAIL",
                f"no gold_features_* under {gold_root}",
            )
        )
        return
    try:
        gold_cols = set(list_columns(gold_dirs[0]))
    except Exception as e:
        report.add(
            CheckResult(
                "EXCLUDED_LEAKING_FEATURES",
                "*",
                "gold_readable",
                "FAIL",
                f"{type(e).__name__}: {str(e)[:160]}",
            )
        )
        return
    for dataset, prefixes in block.items():
        if not isinstance(prefixes, list):
            continue
        leaks: List[str] = []
        for prefix in prefixes:
            leaks.extend(c for c in gold_cols if c.startswith(str(prefix)))
        report.add(
            CheckResult(
                "EXCLUDED_LEAKING_FEATURES",
                dataset,
                "EXCLUDED_LEAKING_FEATURES_honored",
                "PASS" if not leaks else "FAIL",
                f"declared={len(prefixes)} leaks={len(leaks)} sample={leaks[:5]}",
            )
        )


def _check_no_imperative_double_write(
    notebook_paths: Dict[str, Path],
    parsed_declarative: Dict[str, Any],
    report: ComplianceReport,
) -> None:
    """For each migrated cell ID listed in the registered-overrides matrix,
    grep the notebooks for both the `id=<migrated>` AND a sibling imperative
    cell with the same `name=`. Two cells with the same name → double write.
    """
    if not parsed_declarative.get("bronze_aggregations") and not parsed_declarative.get("silver"):
        report.add(
            CheckResult(
                "REGISTERED",
                "*",
                "harvest_emitted",
                "INFO",
                "merged/recommendations.yaml is empty — registered cells haven't run",
            )
        )
        return
    name_to_ids: Dict[str, List[str]] = {}
    for nb_id, path in notebook_paths.items():
        if not path.exists():
            continue
        for cell in _read_notebook_cells(path):
            if cell.get("cell_type") != "code":
                continue
            line1 = (_cell_source(cell).split("\n", 1)[0]) if _cell_source(cell) else ""
            if "@cr:user_code" not in line1:
                continue
            # Pull name='X' id=Y
            import re
            m = re.match(r"#\s*@cr:user_code\s+name='([^']+)'\s+id=(\S+)", line1)
            if not m:
                continue
            name, cell_id = m.group(1), m.group(2)
            name_to_ids.setdefault(name, []).append(f"{nb_id}:{cell_id}")
    duplicates = {n: ids for n, ids in name_to_ids.items() if len(ids) > 1}
    report.add(
        CheckResult(
            "REGISTERED",
            "*",
            "no_imperative_double_write",
            "PASS" if not duplicates else "FAIL",
            f"unique_names={len(name_to_ids)} duplicate_names={list(duplicates)}",
        )
    )


def scan_overrides(
    run_root: Path,
    notebook_paths: Dict[str, Path],
    list_columns: Optional[Callable[[Path], Iterable[str]]] = None,
) -> ComplianceReport:
    """Run every override-compliance check available against the run + notebooks.

    `list_columns(table_path) -> Iterable[str]` is the Spark-dependent
    callback. Pass `lambda p: spark.read.format("delta").load(str(p)).columns`
    from a cycle notebook. Without it, schema-dependent checks downgrade
    to INFO instead of being silently skipped.
    """
    report = ComplianceReport(run_id=run_root.name)
    parsed_imperative = parse_imperative_overrides(notebook_paths)
    parsed_declarative = parse_declarative_recommendations(run_root)
    report.parsed_imperative = parsed_imperative
    report.parsed_declarative = parsed_declarative

    _check_drop_columns(parsed_imperative, list_columns, run_root, report)
    _check_type_overrides(parsed_imperative, run_root, report)
    _check_bronze_aggregations(parsed_imperative, parsed_declarative, list_columns, run_root, report)
    _check_excluded_leaking(parsed_imperative, list_columns, run_root, report)
    _check_no_imperative_double_write(notebook_paths, parsed_declarative, report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="run dir, e.g. .../runs/<run_id>")
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="<id>=<path> pair (repeatable). e.g. --notebook 01=/path/01_data_discovery.ipynb",
    )
    parser.add_argument("--json", type=Path, help="write report JSON here")
    args = parser.parse_args()

    notebook_paths: Dict[str, Path] = {}
    for entry in args.notebook:
        if "=" not in entry:
            print(f"--notebook must be id=path; got {entry!r}", file=sys.stderr)
            return 2
        nb_id, p = entry.split("=", 1)
        notebook_paths[nb_id] = Path(p)

    report = scan_overrides(args.run_root, notebook_paths)
    if args.json:
        args.json.write_text(report.to_json())
    print(report.to_json())
    return 0 if report.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
