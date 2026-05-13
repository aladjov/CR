"""Stage generator for ``c05_snapshot_and_dashboard``.

The final causal-track notebook. Builds the per-scoring-run
``eligibility_snapshot`` table, publishes the dashboard SQL views, and
prints the four-way anchor tuple in force.

Reads ``predictions`` produced by ``c04_batch_inference`` (or the
``s10_batch_inference`` stage of the generated pipeline) — do **not**
trigger scoring here; re-running with the same anchor tuple is an
idempotent MERGE no-op.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import c05_setup_block


class SnapshotAndDashboardStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.SNAPSHOT_AND_DASHBOARD

    @property
    def title(self) -> str:
        return "c05 - Snapshot + Dashboard (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Builds the per-scoring-run `eligibility_snapshot` table, publishes "
            "the dashboard SQL views, and prints the four-way anchor tuple. "
            "Reads predictions from `c04_batch_inference` (or `s10_batch_inference`) "
            "— do not trigger scoring here."
        )

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def _cells(self) -> List[nbformat.NotebookNode]:
        return (
            self.header_cells()
            + c05_setup_block()
            + [
                self.cb.section("5.1 Build Eligibility Snapshot"),
                self.cb.code(_BUILD_SNAPSHOT_CELL),
                self.cb.section("5.2 Compute Per-Slice Top SHAP Drivers"),
                self.cb.code(_COMPUTE_TOP_SHAP_CELL),
                self.cb.section("5.3 Write Run Context (app masthead projection)"),
                self.cb.code(_WRITE_RUN_CONTEXT_CELL),
                self.cb.section("5.3.5 Materialize Provenance + Population Stats UC Tables"),
                self.cb.code(_MATERIALIZE_PROVENANCE_PREREQS_CELL),
                self.cb.section("5.4 Publish Dashboard SQL Views"),
                self.cb.code(_PUBLISH_VIEWS_CELL),
                self.cb.section("5.5 Print Run Summary"),
                self.cb.code(_PRINT_SUMMARY_CELL),
            ]
        )


_BUILD_SNAPSHOT_CELL = '''from customer_retention.stages.causal import SnapshotConfig, build_eligibility_snapshot

snapshot_result = None
if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    print(f"SKIPPED: {ARCHETYPE_CATALOG_FQN} does not exist (run c01..c03 first)")
elif not spark.catalog.tableExists(PREDICTIONS_FQN):
    print(f"SKIPPED: {PREDICTIONS_FQN} not populated (run c04_batch_inference first)")
else:
    snapshot_cfg = SnapshotConfig(
        spark=spark,
        predictions_fqn=PREDICTIONS_FQN,
        archetype_catalog_fqn=ARCHETYPE_CATALOG_FQN,
        eligibility_policy_fqn=ELIGIBILITY_POLICY_FQN,
        decision_policy_fqn=DECISION_POLICY_FQN,
        snapshot_table_fqn=ELIGIBILITY_SNAPSHOT_FQN,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        risk_tier_high=SNAPSHOT_RISK_TIER_HIGH,
        risk_tier_medium=SNAPSHOT_RISK_TIER_MEDIUM,
        capacity_partition_column=SNAPSHOT_CAPACITY_PARTITION_COLUMN or None,
        top_shap_drivers_fqn=TOP_SHAP_DRIVERS_FQN or None,
    )
    snapshot_result = build_eligibility_snapshot(snapshot_cfg)
    print(snapshot_result.summary())
'''


_COMPUTE_TOP_SHAP_CELL = '''from customer_retention.stages.causal import (
    TopDriversConfig,
    compute_and_write_top_shap_drivers,
)

top_drivers_result = None
if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif snapshot_result is None:
    print("SKIPPED: snapshot_result is None — 5.1 did not produce a run")
elif int(SHAP_PER_SLICE_K) <= 0:
    print("SKIPPED: SHAP_PER_SLICE_K == 0 (per-slice SHAP enrichment disabled)")
elif MODEL_URI is None:
    print("SKIPPED: MODEL_URI is None — local runs without an MLflow attribution artifact cannot replay SHAP")
elif not spark.catalog.tableExists(GOLD_FEATURES_FQN):
    print(f"SKIPPED: {GOLD_FEATURES_FQN} does not exist — gold features required for SHAP replay")
else:
    top_drivers_cfg = TopDriversConfig(
        spark=spark,
        snapshot_table_fqn=ELIGIBILITY_SNAPSHOT_FQN,
        gold_features_fqn=GOLD_FEATURES_FQN,
        top_shap_drivers_fqn=TOP_SHAP_DRIVERS_FQN,
        model_name=MODEL_NAME,
        model_version=str(MODEL_VERSION),
        scoring_run_id=snapshot_result.scoring_run_id,
        as_of_date=snapshot_result.as_of_date,
        model_uri=MODEL_URI,
        per_slice_k=int(SHAP_PER_SLICE_K),
        top_drivers_per_row=int(SHAP_TOP_DRIVERS_PER_ROW),
    )
    top_drivers_result = compute_and_write_top_shap_drivers(top_drivers_cfg)
    print(top_drivers_result.summary())
'''


_WRITE_RUN_CONTEXT_CELL = '''from customer_retention.stages.causal import (
    from_project_context,
    write_run_context,
)

RUN_CONTEXT_FQN = f"{CATALOG}.{SCHEMA}.run_context"


def _load_project_context():
    """Best-effort ProjectContext load. Returns None when YAML is unreachable so
    the writer still emits a row with model metadata only."""
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return None
    try:
        _ns = RunNamespace.from_env_or_latest()
    except Exception:
        _ns = None
    if _ns is None:
        return None
    _path = _ns.project_context_path
    if not _path.exists():
        return None
    try:
        return ProjectContext.load(_path)
    except Exception:
        return None


def _resolve_model_type():
    """Try to pull the MLflow flavor off the registered model (e.g. "xgboost")."""
    try:
        import mlflow  # noqa: F401
        if MODEL_URI is None:
            return None
        from mlflow.models import Model as _MlflowModel
        _info = _MlflowModel.load(MODEL_URI)
        _flavors = list((_info.flavors or {}).keys())
        # Prefer the most specific flavor over ``python_function``.
        for preferred in ("xgboost", "lightgbm", "catboost", "pytorch", "tensorflow", "sklearn"):
            if preferred in _flavors:
                return preferred
        _flavors = [f for f in _flavors if f != "python_function"]
        return _flavors[0] if _flavors else None
    except Exception:
        return None


if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif snapshot_result is None:
    print("SKIPPED: snapshot_result is None — 5.1 did not produce a run")
else:
    _ctx = _load_project_context()
    _cfg = from_project_context(
        project_context=_ctx,
        spark=spark,
        table_fqn=RUN_CONTEXT_FQN,
        scoring_run_id=snapshot_result.scoring_run_id,
        as_of_date=snapshot_result.as_of_date,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type=_resolve_model_type(),
    )
    write_run_context(_cfg)
    print(f"Wrote run_context row for scoring_run_id={snapshot_result.scoring_run_id}")
    if _ctx is None:
        print("  (project_context.yaml not reachable — context fields are NULL)")
    else:
        print(f"  horizon_days:       {_cfg.horizon_days}")
        print(f"  primary_objective:  {_cfg.primary_objective}")
        print(f"  temporal_posture:   {_cfg.temporal_posture}")
        print(f"  model_type:         {_cfg.model_type or '(unresolved)'}")
'''


_MATERIALIZE_PROVENANCE_PREREQS_CELL = '''# Materialize the three UC Delta tables that ``publish_dashboard_views``
# (the next cell) reads from: ``feature_meta``, ``column_descriptions``,
# ``feature_population_stats``. Without these tables, the framework's
# ``v_feature_provenance`` view is silently skipped at publish time, which
# breaks the per-row lineage disclosure under both the SHAP and the
# deviation panels of the customer profile.
#
# Why this cell exists despite the framework's own auto-materializer:
# ``publish_dashboard_views`` calls ``_try_materialize_*_from_sidecar()``
# on entry, but those helpers resolve the namespace via
# ``RunNamespace.from_env_or_latest()``. On clusters where the
# experiments directory holds multiple runs (the common case), that
# resolver can pick the wrong run by mtime and read sidecars from a
# namespace that doesn't carry them -- the materialize then silently
# returns ``False`` and ``v_feature_provenance`` is skipped. Pinning the
# namespace explicitly via the ``ENGAGEMENT_*`` constants (the same
# pattern that fixes ``setup_and_resolve_model``) closes that drift.
#
# ``column_descriptions`` is root-scoped (cross-run) by design --
# ``RunNamespace.column_descriptions_dir`` resolves to
# ``root / "column_descriptions"`` (not ``run_dir / ...``). Some
# deployments bootstrap it once under a project-level ``experiments/``
# while runs land under ``experiments_secondary/``; set
# ``ENGAGEMENT_COLUMN_DESCRIPTIONS_ROOT`` in the configuration cell to
# point at the alternate root in that case. Falls back to
# ``ENGAGEMENT_EXPERIMENTS_DIR`` when the override isn't set.
#
# Idempotent: each block checks ``spark.catalog.tableExists`` first and
# skips the write when the UC table is already present. Underlying
# writers use MERGE on composite keys, so even if you remove the
# tableExists guard, re-running would not produce duplicates.
from pathlib import Path
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.stages.causal.feature_meta_writer import (
    FeatureMetaConfig, write_feature_meta,
)
from customer_retention.stages.causal.column_descriptions_writer import (
    ColumnDescriptionsConfig, write_column_descriptions,
)
from customer_retention.stages.causal.population_stats import (
    materialize_population_stats_from_sidecar,
)
from customer_retention.stages.causal.interpretation.sidecars import (
    load_feature_meta_sidecar, load_column_descriptions_sidecar,
)

_ENG_CD_ROOT = globals().get("ENGAGEMENT_COLUMN_DESCRIPTIONS_ROOT") or ENGAGEMENT_EXPERIMENTS_DIR

_ns_run = RunNamespace(root=Path(ENGAGEMENT_EXPERIMENTS_DIR), run_id=ENGAGEMENT_RUN_ID)
_ns_cd  = RunNamespace(root=Path(_ENG_CD_ROOT),                run_id=ENGAGEMENT_RUN_ID)

if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
else:
    _fm_fqn = f"{CATALOG}.{SCHEMA}.feature_meta"
    _cd_fqn = f"{CATALOG}.{SCHEMA}.column_descriptions"
    _ps_fqn = f"{CATALOG}.{SCHEMA}.feature_population_stats"

    # ---- feature_meta -----------------------------------------------------
    if spark.catalog.tableExists(_fm_fqn):
        print(f"[mt4r] feature_meta: UC table already present at {_fm_fqn} -- skipping write.")
    else:
        _meta = load_feature_meta_sidecar(_ns_run) or {}
        if _meta:
            write_feature_meta(FeatureMetaConfig(
                spark=spark, table_fqn=_fm_fqn, run_id=_ns_run.run_id,
                rows=list(_meta.values()),
            ))
            print(f"[mt4r] feature_meta: wrote {len(_meta)} rows from {_ns_run.feature_meta_dir}")
        else:
            print(f"[mt4r] feature_meta: SKIPPED -- no sidecar at {_ns_run.feature_meta_dir}")

    # ---- column_descriptions ---------------------------------------------
    if spark.catalog.tableExists(_cd_fqn):
        print(f"[mt4r] column_descriptions: UC table already present at {_cd_fqn} -- skipping write.")
    else:
        # Engagement-root first (the configured / overridden root), then
        # fall back to the per-run experiments_dir for deployments that
        # bootstrap column_descriptions in the same root as runs.
        _cd_sidecar = load_column_descriptions_sidecar(_ns_cd) or {}
        _cd_source = _ns_cd.column_descriptions_dir
        if not _cd_sidecar and _ENG_CD_ROOT != ENGAGEMENT_EXPERIMENTS_DIR:
            _cd_sidecar = load_column_descriptions_sidecar(_ns_run) or {}
            _cd_source = _ns_run.column_descriptions_dir
        if _cd_sidecar:
            write_column_descriptions(ColumnDescriptionsConfig(
                spark=spark, table_fqn=_cd_fqn, rows=list(_cd_sidecar.values()),
            ))
            print(f"[mt4r] column_descriptions: wrote {len(_cd_sidecar)} rows from {_cd_source}")
        else:
            print(f"[mt4r] column_descriptions: SKIPPED -- no sidecar at {_ns_cd.column_descriptions_dir}")

    # ---- feature_population_stats ----------------------------------------
    if spark.catalog.tableExists(_ps_fqn):
        print(f"[mt4r] feature_population_stats: UC table already present at {_ps_fqn} -- skipping write.")
    else:
        _ps_sidecar = _ns_run.feature_population_stats_dir / "feature_population_stats.json"
        if _ps_sidecar.exists():
            materialize_population_stats_from_sidecar(spark, _ps_sidecar, _ps_fqn)
            print(f"[mt4r] feature_population_stats: materialized from {_ps_sidecar}")
        else:
            print(f"[mt4r] feature_population_stats: SKIPPED -- no sidecar at {_ps_sidecar}")
'''


_PUBLISH_VIEWS_CELL = '''from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_VIEW_NAMES,
    publish_dashboard_views,
)

if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif not spark.catalog.tableExists(ELIGIBILITY_SNAPSHOT_FQN):
    print(f"SKIPPED: {ELIGIBILITY_SNAPSHOT_FQN} not populated yet")
else:
    statements = publish_dashboard_views(spark, CATALOG, SCHEMA)
    print(f"Published {len(statements)} dashboard views:")
    for view_name in DASHBOARD_VIEW_NAMES:
        print(f"  - {CATALOG}.{SCHEMA}.{view_name}")
'''


_PRINT_SUMMARY_CELL = '''if spark is None or not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    print("(no archetype_catalog yet — run c01..c03 first)")
else:
    counts = spark.sql(
        f"SELECT status, COUNT(*) AS n FROM {ARCHETYPE_CATALOG_FQN} GROUP BY status"
    ).collect()
    print("archetype_catalog row counts:")
    for row in counts:
        print(f"  {row['status']}: {row['n']}")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    if snapshot_result is not None:
        print(snapshot_result.summary())
    if top_drivers_result is not None:
        print(top_drivers_result.summary())
'''
