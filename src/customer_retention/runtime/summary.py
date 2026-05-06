"""User-code assimilation summary — generic framework mechanism.

Goal: at any checkpoint of an exploration notebook (typically the final
report section), let the operator see at a single glance whether every
registered user_code cell was correctly assimilated by the framework.
"Correctly assimilated" means:

- **Lane 1** — the cell's declarative or imperative registration call
  landed in either ``runtime.registry`` (``@cr.register``) or
  ``RecommendationRegistry`` (``registry.add_*``).
- **Lane 2** — the cell's in-cell side effect actually mutated shared
  exploration state where mutation was expected. For imperative cells
  this is the ``cr.mark_lane2_executed(name)`` flag. For declarative
  landing cells that rebind ``datasets[...]``, it is detected by
  comparing the current ``datasets`` dict against the canonical
  pre-user_code baseline (``namespace.original_datasets``, stashed by
  NB00 cell ``f1eb641c`` immediately after ``dataset_paths``), with
  fallback to the module-level ``_landing_snapshot`` for callers that
  build the summary without a namespace. The verdict is computed
  per-dataset, so multiple Lane-1 declaratives that compose into a
  single rebind (e.g. filter + lifecycle enrichment in one cell) all
  share the same status. For pure config / silver-derived cells, no
  Lane-2 mutation is expected during NB00, so the row is reported as
  "config-only" or "pending until silver stage".

The mechanism carries no engagement-specific knowledge: it does not
hard-code cell IDs, function names, or framework-version labels. Drop
``cr.snapshot_landing_state(datasets)`` near the top of the notebook
that hands user_code its starting state (only needed when no namespace
is available), and call ``cr.print_user_code_summary(datasets, namespace)``
wherever a one-glance audit is useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .registry import RegisteredFunction
from .registry import registry as _runtime_registry

# ----------------------------------------------------------------------------
# Snapshot — taken once before any user_code modifies `datasets[...]`.
# ----------------------------------------------------------------------------

# Module-level singleton; lifecycle is per Python kernel. Key = dataset
# name, value = repr / path / object identity at snapshot time.
_landing_snapshot: Dict[str, str] = {}


def snapshot_landing_state(datasets: Dict[str, Any]) -> Dict[str, str]:
    """Capture the upstream reference of every entry in ``datasets`` so the
    summary can later detect Lane-2 rebinds (cell replaced
    ``datasets[<name>]`` with an enriched / filtered / derived view).

    Returns the captured snapshot (also stored on the runtime registry's
    sidecar). Re-calling overwrites — the operator owns when the
    "before" state is captured. Typically the framework calls this once
    in NB00 right after ``dataset_paths``.
    """
    global _landing_snapshot
    _landing_snapshot = {k: _signature(v) for k, v in datasets.items()}
    return dict(_landing_snapshot)


def get_landing_snapshot() -> Dict[str, str]:
    return dict(_landing_snapshot)


def _signature(value: Any) -> str:
    """Stable string identity for a `datasets[...]` entry. Path strings,
    UC table names, and Spark/pandas DataFrames all map to a string we
    can compare for equality at report time."""
    if value is None:
        return "<none>"
    if isinstance(value, str):
        return value
    # Spark / pyspark.pandas DataFrames carry a stable identity via id().
    return f"<{type(value).__name__}#{id(value)}>"


# ----------------------------------------------------------------------------
# Summary rows
# ----------------------------------------------------------------------------

@dataclass
class AssimilationRow:
    """One line of the user-code assimilation summary.

    ``track`` is the registration mechanism. ``target`` is what the
    cell's Lane-2 mutation should affect (a dataset name, a derived
    column pattern, etc.). ``lane2_status`` is one of ``ok``,
    ``co-applied``, ``missing``, ``config-only``, ``pending``.
    ``co-applied`` (✳️) is a success state used when multiple
    declarative landing registrations target the same dataset and share
    a single composed ``register_temp_view`` rebind (filter + lifecycle
    enrichment in one cell, etc.).
    """
    cell_id: Optional[str]
    name: str
    track: str
    target: str
    lane1_present: bool
    lane2_status: str
    lane2_detail: str
    notebook: Optional[str] = None

    @property
    def emoji(self) -> str:
        if self.lane2_status == "ok":
            return "✅"
        if self.lane2_status == "co-applied":
            return "✳️"
        if self.lane2_status == "missing":
            return "❌"
        if self.lane2_status == "config-only":
            return "·"  # n/a
        if self.lane2_status == "pending":
            return "⏳"
        return "?"


@dataclass
class AssimilationReport:
    rows: List[AssimilationRow] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(
            r.lane2_status in ("ok", "co-applied", "config-only", "pending")
            for r in self.rows
        )

    @property
    def failing(self) -> List[AssimilationRow]:
        return [r for r in self.rows if r.lane2_status == "missing"]


# ----------------------------------------------------------------------------
# Summary builder
# ----------------------------------------------------------------------------

def summarize_user_code(
    datasets: Optional[Dict[str, Any]] = None,
    namespace: Any = None,
    *,
    df_features: Any = None,
) -> AssimilationReport:
    """Inventory every registered user_code cell and assess its Lane-2
    application.

    ``datasets`` and ``namespace`` are optional — when omitted, the
    summary still inspects the registries but Lane-2 evidence for
    declarative landing rebinds is reported as ``pending`` instead of
    ``ok``/``missing``. ``df_features`` (when supplied — typically only
    in NB06+) enables silver-derived Lane-2 verification by checking
    whether the recommended column appears in the feature panel.
    """
    rows: List[AssimilationRow] = []
    rows.extend(_imperative_rows())
    rows.extend(_declarative_rows(datasets, namespace, df_features))
    return AssimilationReport(rows=rows)


def _imperative_rows() -> List[AssimilationRow]:
    out: List[AssimilationRow] = []
    for rf in _runtime_registry.get_registered():
        target = _imperative_target(rf)
        marked = _runtime_registry.is_lane2_executed(rf.name)
        out.append(AssimilationRow(
            cell_id=rf.cell_id,
            name=rf.name,
            track="imperative (@cr.register)",
            target=target,
            lane1_present=True,
            lane2_status="ok" if marked else "missing",
            lane2_detail=(
                "cr.mark_lane2_executed(name) called"
                if marked else
                "cr.mark_lane2_executed(name) NOT called — operator must "
                "add the marker line at the bottom of the user_code cell "
                "after the in-cell side effect runs"
            ),
            notebook=rf.notebook_path.name if rf.notebook_path else None,
        ))
    return out


def _imperative_target(rf: RegisteredFunction) -> str:
    if rf.scope == "dataset" and rf.dataset:
        return rf.dataset
    if rf.scope == "datasets" and rf.datasets:
        return ", ".join(rf.datasets)
    if rf.scope == "wildcard":
        return "*"
    return "?"


def _declarative_rows(
    datasets: Optional[Dict[str, Any]],
    namespace: Any,
    df_features: Any,
) -> List[AssimilationRow]:
    """Inspect the RecommendationRegistry and emit one row per category
    of declarative cell."""
    decl = _load_recommendation_registry(namespace)
    if decl is None:
        return []
    # Compute the per-dataset Lane-2 verdict once. Multiple registrations
    # targeting the same dataset (e.g. filter + lifecycle enrichment in
    # the same cell) compose into a single `register_temp_view` rebind, so
    # they share one verdict — emitting `missing` for the secondary rows
    # would be a phantom failure. ``landing_counts`` lets the row builder
    # tag composed rows with the ``co-applied`` status so the operator
    # can see at a glance that the rebind is shared, not duplicated.
    baseline = _baseline_originals(namespace)
    verdicts = _per_dataset_verdicts(baseline, datasets)
    landing_counts = _landing_registration_counts(decl)
    out: List[AssimilationRow] = []
    out.extend(_landing_filter_rows(decl, datasets, verdicts, landing_counts))
    out.extend(_landing_lifecycle_rows(decl, datasets, verdicts, landing_counts))
    out.extend(_bronze_override_rows(decl))
    out.extend(_silver_derived_rows(decl, df_features))
    return out


def _landing_registration_counts(decl) -> Dict[str, int]:
    """Count how many declarative landing recommendations target each
    dataset (filters + lifecycle enrichments combined). When count > 1,
    those registrations are co-applied via a single ``register_temp_view``
    rebind and should be reported as ``co-applied`` rather than as N
    independent ``ok`` rows.
    """
    counts: Dict[str, int] = {}
    landing = getattr(decl, "landing", None)
    if landing is None:
        return counts
    for rec in (getattr(landing, "filters", None) or []):
        ds = _rec_landing_dataset(rec)
        if ds:
            counts[ds] = counts.get(ds, 0) + 1
    for rec in (getattr(landing, "lifecycle_enrichments", None) or []):
        ds = _rec_landing_dataset(rec)
        if ds:
            counts[ds] = counts.get(ds, 0) + 1
    return counts


def _baseline_originals(namespace: Any) -> Dict[str, str]:
    """Pre-user_code dataset signatures, preferring the canonical
    ``namespace.original_datasets`` (stashed by NB00 cell ``f1eb641c``
    immediately after ``dataset_paths``, before any user_code can mutate
    ``datasets``) over the module-level ``_landing_snapshot``.

    The fallback exists for callers that build the summary without a
    namespace (e.g. unit tests, mid-notebook checkpoints in non-NB00
    notebooks). When both are absent the verdict is ``pending``.
    """
    canonical = getattr(namespace, "original_datasets", None) if namespace is not None else None
    if canonical:
        return {k: _signature(v) for k, v in canonical.items()}
    return dict(_landing_snapshot)


def _per_dataset_verdicts(
    baseline: Dict[str, str],
    datasets: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Return ``{dataset_name: {"status": ..., "before": ..., "after": ...}}``.

    Status is one of ``ok`` / ``missing`` / ``pending``. The verdict is
    keyed on the dataset, not on the individual registration, so that
    composed Lane-1 declaratives sharing one rebind aren't flagged as
    partial.
    """
    out: Dict[str, Dict[str, str]] = {}
    if datasets is None or not baseline:
        return out
    for name, before in baseline.items():
        if name not in datasets:
            out[name] = {"status": "missing", "before": before, "after": "<removed>"}
            continue
        after = _signature(datasets[name])
        status = "ok" if after != before else "missing"
        out[name] = {"status": status, "before": before, "after": after}
    return out


def _load_recommendation_registry(namespace: Any):
    if namespace is None:
        return None
    try:
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
    except ImportError:
        return None
    path = getattr(namespace, "merged_recommendations_path", None)
    if path is None or not path.exists():
        return None
    return RecommendationRegistry.load(path)


def _landing_filter_rows(decl, datasets, verdicts, landing_counts):
    out: List[AssimilationRow] = []
    landing = getattr(decl, "landing", None)
    for rec in (getattr(landing, "filters", None) or []):
        ds = _rec_landing_dataset(rec)
        out.append(_landing_row(rec, ds, "filter", datasets, verdicts, landing_counts))
    return out


def _landing_lifecycle_rows(decl, datasets, verdicts, landing_counts):
    out: List[AssimilationRow] = []
    landing = getattr(decl, "landing", None)
    for rec in (getattr(landing, "lifecycle_enrichments", None) or []):
        ds = _rec_landing_dataset(rec)
        out.append(_landing_row(rec, ds, "lifecycle_enrichment", datasets, verdicts, landing_counts))
    return out


def _landing_row(rec, dataset_name, kind, datasets, verdicts, landing_counts):
    name = _rec_name(rec, default=f"{kind}({dataset_name})")
    verdict = verdicts.get(dataset_name) if dataset_name else None
    composed_count = landing_counts.get(dataset_name, 1) if dataset_name else 1
    if datasets is None or not verdicts:
        status, detail = "pending", "baseline (namespace.original_datasets / snapshot) or live datasets not provided"
    elif verdict is None:
        status, detail = "pending", f"dataset {dataset_name!r} not in baseline"
    elif verdict["status"] == "ok":
        if composed_count > 1:
            status, detail = "co-applied", (
                f"datasets[{dataset_name!r}] composed rebind shared by "
                f"{composed_count} declarative landing rec(s): "
                f"{verdict['before']!r} → {verdict['after']!r}"
            )
        else:
            status, detail = "ok", (
                f"datasets[{dataset_name!r}] rebound: "
                f"{verdict['before']!r} → {verdict['after']!r}"
            )
    elif verdict["after"] == "<removed>":
        status, detail = "missing", f"datasets[{dataset_name!r}] removed since baseline"
    else:
        status, detail = "missing", (
            f"datasets[{dataset_name!r}] still equals the upstream "
            f"baseline ({verdict['before']!r}) — operator's Lane-2 rebind "
            f"to a temp view / enriched frame is missing"
        )
    return AssimilationRow(
        cell_id=_rec_cell_id(rec),
        name=name,
        track=f"declarative landing.{kind}",
        target=dataset_name or "?",
        lane1_present=True,
        lane2_status=status,
        lane2_detail=detail,
        notebook=_rec_notebook(rec),
    )


def _bronze_override_rows(decl):
    out: List[AssimilationRow] = []
    overrides = getattr(decl, "bronze_aggregations", None) or {}
    for ds_name, override in overrides.items():
        out.append(AssimilationRow(
            cell_id=None,
            name=f"bronze_aggregations[{ds_name!r}]",
            track="declarative bronze_aggregations",
            target=ds_name,
            lane1_present=True,
            lane2_status="config-only",
            lane2_detail=(
                f"keys={sorted(override.keys()) if isinstance(override, dict) else '?'} "
                f"— consumed by the bronze codegen at NB10; no datasets[...] "
                f"mutation expected at exploration time"
            ),
            notebook=None,
        ))
    return out


def _silver_derived_rows(decl, df_features):
    out: List[AssimilationRow] = []
    silver = getattr(decl, "silver", None)
    if silver is None:
        return out
    derived = list(getattr(silver, "derived_columns", None) or [])
    for rec in derived:
        col = _rec_silver_column(rec)
        if not col:
            continue
        if df_features is None:
            status = "pending"
            detail = (
                f"silver-derived column {col!r}: not yet checked — "
                f"pass df_features= when calling print_user_code_summary "
                f"from NB06 onwards to verify the column is present"
            )
        else:
            try:
                cols = list(df_features.columns)
            except Exception:
                status, detail = "pending", "df_features.columns not iterable"
            else:
                if col in cols:
                    status, detail = "ok", f"{col!r} present in df_features"
                else:
                    status, detail = "missing", (
                        f"{col!r} NOT in df_features — operator's in-cell "
                        f"derivation either skipped or wrote to a different name"
                    )
        out.append(AssimilationRow(
            cell_id=_rec_cell_id(rec),
            name=_rec_name(rec, default=col or "silver_derived"),
            track="declarative silver.derived",
            target=col or "?",
            lane1_present=True,
            lane2_status=status,
            lane2_detail=detail,
            notebook=_rec_notebook(rec),
        ))
    return out


# ----------------------------------------------------------------------------
# Recommendation introspection helpers — tolerant to dataclass / dict shape.
# ``LayeredRecommendation``'s ``target_column`` field is overloaded: for
# landing recs it carries the dataset name; for silver-derived recs it
# carries the new column name. The track-specific helpers below dispatch
# accordingly so callers don't have to know.
# ----------------------------------------------------------------------------

def _rec_landing_dataset(rec) -> Optional[str]:
    """For landing filter / lifecycle recs, the dataset name lives in
    ``parameters['dataset']`` and is also stored in ``target_column``."""
    params = _get_attr(rec, "parameters")
    if isinstance(params, dict) and params.get("dataset"):
        return str(params["dataset"])
    return _rec_field(rec, "target_column", "dataset")


def _rec_silver_column(rec) -> Optional[str]:
    """For silver-derived recs, ``target_column`` is the new column name."""
    return _rec_field(rec, "target_column", "column", "name")


def _rec_name(rec, default: str = "") -> str:
    val = _rec_field(rec, "id", "name", "rationale_id")
    return val or default


def _rec_cell_id(rec) -> Optional[str]:
    """No declarative call site sets ``cell_id`` today, but if a future
    extension threads it via ``parameters`` we surface it transparently."""
    params = _get_attr(rec, "parameters")
    if isinstance(params, dict) and params.get("cell_id"):
        return str(params["cell_id"])
    return _rec_field(rec, "cell_id", "source_cell_id")


def _rec_notebook(rec) -> Optional[str]:
    return _rec_field(rec, "source_notebook", "notebook")


def _get_attr(rec, name):
    if isinstance(rec, dict):
        return rec.get(name)
    return getattr(rec, name, None)


def _rec_field(rec, *names) -> Optional[str]:
    for n in names:
        v = _get_attr(rec, n)
        if v:
            return str(v)
    return None


# ----------------------------------------------------------------------------
# Rendering — one-glance Markdown table.
# ----------------------------------------------------------------------------

def render_markdown(report: AssimilationReport) -> str:
    if not report.rows:
        return (
            "## User-code assimilation summary\n\n"
            "_No registered user_code cells in this notebook._\n"
        )
    lines = [
        "## User-code assimilation summary",
        "",
        f"**Total registered cells:** {len(report.rows)} — "
        f"{sum(1 for r in report.rows if r.lane2_status == 'ok')} applied, "
        f"{sum(1 for r in report.rows if r.lane2_status == 'co-applied')} co-applied, "
        f"{sum(1 for r in report.rows if r.lane2_status == 'config-only')} config-only, "
        f"{sum(1 for r in report.rows if r.lane2_status == 'pending')} pending, "
        f"**{sum(1 for r in report.rows if r.lane2_status == 'missing')} missing**.",
        "",
        "| Cell ID | Name | Track | Target | Lane 1 | Lane 2 | Detail |",
        "|---------|------|-------|--------|:------:|:------:|--------|",
    ]
    for r in report.rows:
        lines.append(
            f"| `{r.cell_id or '—'}` "
            f"| `{r.name}` "
            f"| {r.track} "
            f"| {r.target} "
            f"| {'✅' if r.lane1_present else '❌'} "
            f"| {r.emoji} "
            f"| {r.lane2_detail} |"
        )
    if report.failing:
        lines.append("")
        lines.append("**Action required:** the rows marked ❌ above did not apply their "
                     "Lane-2 side effect. For imperative cells, add "
                     "`cr.mark_lane2_executed(<fn_name>)` after the in-cell action. "
                     "For declarative landing cells, rebind "
                     "`datasets[<dataset>] = register_temp_view(<enriched_df>, ...)` "
                     "so subsequent notebooks see the post-Lane-2 state.")
    return "\n".join(lines)


def print_user_code_summary(
    datasets: Optional[Dict[str, Any]] = None,
    namespace: Any = None,
    *,
    df_features: Any = None,
) -> AssimilationReport:
    """Build and ``display(...)`` the assimilation summary as Markdown.

    Returns the ``AssimilationReport`` so the caller can `assert
    report.all_ok` if desired.
    """
    report = summarize_user_code(datasets, namespace, df_features=df_features)
    md = render_markdown(report)
    try:
        from IPython.display import Markdown, display  # noqa: PLC0415
        display(Markdown(md))
    except ImportError:
        print(md)
    return report
