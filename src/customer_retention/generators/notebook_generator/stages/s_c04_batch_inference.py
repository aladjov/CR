"""Stage generator for ``c04_batch_inference``.

Independent scoring refresh notebook on the causal track. Calls
``customer_retention.stages.scoring.batch_inference.run_batch_inference``
against the registered ``@production`` model and writes the
``{catalog}.{schema}.predictions`` + ``inference_audit_log`` Delta tables
that ``c05_snapshot_and_dashboard`` consumes.

Designed to be re-run on a schedule or on demand without touching
archetypes: c02 derives archetypes once per model version; c04 refreshes
predictions per scoring run; c05 joins the two and writes the snapshot.
All heavy work runs inside ``fe.score_batch`` (Databricks Feature
Engineering) — distributed, one Spark job per batch, no driver collect.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import c04_setup_block


class BatchInferenceCausalStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.BATCH_INFERENCE_CAUSAL

    @property
    def title(self) -> str:
        return "c04 - Batch Inference (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Refreshes the `predictions` Delta table by scoring the current "
            "feature-store snapshot against the registered `@production` model. "
            "Independent from c02 (archetypes) and c05 (snapshot + dashboard) so "
            "operators can re-run scoring without recomputing archetypes."
        )

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def _cells(self) -> List[nbformat.NotebookNode]:
        return (
            self.header_cells()
            + c04_setup_block()
            + [
                self.cb.section("4.1 Refresh Predictions"),
                self.cb.code(_REFRESH_PREDICTIONS_CELL),
                self.cb.section("4.2 Print Run Summary"),
                self.cb.code(_PRINT_SUMMARY_CELL),
            ]
        )


_REFRESH_PREDICTIONS_CELL = '''from datetime import datetime, timedelta, timezone

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    run_batch_inference,
)


def _resolve_scope_filter():
    """Return the NB00 scope filter (``ProjectContext.sample_filters``) for the
    target dataset, or ``None`` if project_context is absent / empty. Scoring
    must cover only the entity subset that was in force during exploration
    and training — this keeps the inference population in lock-step with the
    feature-spec gate upstream."""
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return None
    _ns = RunNamespace.from_env_or_latest()
    if _ns is None:
        return None
    _path = _ns.project_context_path
    if not _path.exists():
        return None
    _ctx = ProjectContext.load(_path)
    _filters = getattr(_ctx, "sample_filters", None) or {}
    if not _filters:
        return None
    _target_name = next(
        (name for name, ds in _ctx.datasets.items()
         if getattr(ds, "role", None) == "target"),
        None,
    )
    if _target_name is None and len(_ctx.datasets) == 1:
        _target_name = next(iter(_ctx.datasets))
    if _target_name is None:
        return None
    return _filters.get(_target_name)


def _resolve_already_positive_exclusion():
    """Return ``(target_column, landing_fqn, raw_entity_key)`` so scoring can
    drop entities whose target label has EVER been 1 in landing — applied
    by the framework as a ``left_anti`` join against
    ``landing_<target>.filter(<target_col> = 1)``.

    Entity-level by construction: a row-level ``<target_col> = 0`` clause
    cannot work against a snapshot-panel landing table, because an entity
    with both ``<target_col>=0`` and ``<target_col>=1`` rows would still
    survive the row-level filter (the 0-rows pass; ``distinct(entity_id)``
    re-admits the entity). Anti-join on the "ever positive" entity set is
    the only shape that matches the intent — drop the entity wholesale if
    landing has ANY row at the positive outcome.

    Always-on at scoring time: the trained model has nothing useful to say
    about an already-positive row, and surfacing such rows in the
    CSM-facing dashboard wastes headcount on accounts the business can no
    longer recover. Distinct from ``_resolve_scope_filter`` so the training
    parity contract stays clean — training cohort keeps positives (the
    model needs them to learn the boundary); scoring drops them because
    they aren't actionable.

    Returns ``(None, None, None)`` when project_context is absent or
    cannot resolve a target column / raw entity key — the cell then falls
    back to the unmodified scope filter and logs the omission so the
    operator can see it didn't apply."""
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return None, None, None
    _ns = RunNamespace.from_env_or_latest()
    if _ns is None:
        return None, None, None
    _path = _ns.project_context_path
    if not _path.exists():
        return None, None, None
    _ctx = ProjectContext.load(_path)
    _target_col = getattr(_ctx, "target_column", None)
    if not _target_col:
        return None, None, None
    _target_name = (
        getattr(_ctx, "target_dataset", None)
        or next(
            (n for n, d in _ctx.datasets.items() if getattr(d, "role", None) == "target"),
            None,
        )
    )
    if _target_name is None and len(_ctx.datasets) == 1:
        _target_name = next(iter(_ctx.datasets))
    if _target_name is None:
        return None, None, None
    _target_ds = _ctx.datasets.get(_target_name)
    _raw_key = getattr(_target_ds, "entity_column", None) if _target_ds is not None else None
    if not _raw_key:
        return None, None, None
    _landing_fqn = f"{CATALOG}.{SCHEMA}.landing_{_target_name}"
    return _target_col, _landing_fqn, _raw_key


_scope_filter = _resolve_scope_filter()
_excl_target_col, _excl_landing_fqn, _excl_entity_key = _resolve_already_positive_exclusion()

batch_inference_result = None
_predictions_status = "UNKNOWN"

if spark is None:
    _predictions_status = "SKIPPED: no Spark session (Databricks-only cell)"
    print(_predictions_status)
elif BATCH_INFERENCE_MODE == "never":
    _predictions_status = "SKIPPED: BATCH_INFERENCE_MODE='never'"
    print(_predictions_status)
else:
    _should_run = True
    _stale_reason = "always" if BATCH_INFERENCE_MODE == "always" else None
    if BATCH_INFERENCE_MODE == "auto":
        if not spark.catalog.tableExists(PREDICTIONS_FQN):
            _stale_reason = f"{PREDICTIONS_FQN} does not exist"
        else:
            _latest_ts_row = spark.sql(
                f"SELECT max(inference_point_in_time) AS ts FROM {PREDICTIONS_FQN}"
            ).head()
            _latest_ts = _latest_ts_row["ts"] if _latest_ts_row is not None else None
            if _latest_ts is None:
                _stale_reason = f"{PREDICTIONS_FQN} is empty"
            else:
                if _latest_ts.tzinfo is None:
                    _latest_ts = _latest_ts.replace(tzinfo=timezone.utc)
                _age_hours = (datetime.now(timezone.utc) - _latest_ts).total_seconds() / 3600.0
                if _age_hours > PREDICTIONS_STALE_AFTER_HOURS:
                    _stale_reason = f"latest inference_point_in_time is {_age_hours:.1f}h old (> {PREDICTIONS_STALE_AFTER_HOURS}h)"
                else:
                    _should_run = False
                    _predictions_status = (
                        f"FRESH: latest inference_point_in_time is {_age_hours:.1f}h old "
                        f"(<= {PREDICTIONS_STALE_AFTER_HOURS}h) — skipping"
                    )
                    print(_predictions_status)
    if _should_run:
        print(f"Running batch inference ({_stale_reason})")
        if _scope_filter:
            print(f"Scope filter (from NB00 project_context): {_scope_filter}")
        else:
            print("Scope filter: (none — scoring full entity population)")
        if _excl_target_col and _excl_landing_fqn and _excl_entity_key:
            print(
                f"Already-positive exclusion (entity-level anti-join): "
                f"DROP entity where any row of {_excl_landing_fqn} has "
                f"{_excl_target_col} = 1 (key={_excl_entity_key})"
            )
        else:
            print(
                "Already-positive exclusion: (none — project_context.target_column "
                "or target dataset entity_column not resolvable; entities already "
                "at the positive outcome may be scored)"
            )
        config = BatchInferenceConfig(
            catalog=CATALOG,
            schema=SCHEMA,
            model_uri=MODEL_URI,
            customer_table=GOLD_FEATURES_FQN,
            threshold=SCORING_THRESHOLD,
            risk_tier_high=RISK_TIER_HIGH,
            risk_tier_medium=RISK_TIER_MEDIUM,
            inference_timestamp=datetime.now(timezone.utc),
            filter_expression=_scope_filter,
            exclude_already_positive_target_column=_excl_target_col,
            exclude_already_positive_via_table=_excl_landing_fqn,
            exclude_already_positive_entity_key=_excl_entity_key,
        )
        batch_inference_result = run_batch_inference(config)
        _predictions_status = batch_inference_result.summary()
        print(batch_inference_result.long_summary())
'''


_PRINT_SUMMARY_CELL = '''if batch_inference_result is None:
    print(f"Predictions status: {_predictions_status}")
else:
    print(f"Inference id: {batch_inference_result.inference_id}")
    print(f"Inference timestamp: {batch_inference_result.inference_timestamp}")
    print(f"Scored: {batch_inference_result.total_scored:,}")
    print(f"Predicted churners: {batch_inference_result.predicted_churners:,}")
    print(f"Mean probability: {batch_inference_result.avg_probability:.4f}")
    print(f"Target table: {batch_inference_result.target_table_fqn}")
'''
