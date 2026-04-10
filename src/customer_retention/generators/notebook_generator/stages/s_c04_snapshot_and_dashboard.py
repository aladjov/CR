"""Stage generator for ``c04_snapshot_and_dashboard``.

The final causal-track notebook. Builds the per-scoring-run
``eligibility_snapshot`` table, publishes the dashboard SQL views, and
prints the four-way anchor tuple in force.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import c04_setup_block


class SnapshotAndDashboardStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.SNAPSHOT_AND_DASHBOARD

    @property
    def title(self) -> str:
        return "c04 - Snapshot + Dashboard (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Builds the per-scoring-run `eligibility_snapshot` table, publishes "
            "the dashboard SQL views, and prints the four-way anchor tuple. "
            "Reads predictions from `s10_batch_inference` (do not trigger scoring here)."
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
                self.cb.section("1. Build Eligibility Snapshot"),
                self.cb.code(_BUILD_SNAPSHOT_CELL),
                self.cb.section("2. Publish Dashboard SQL Views"),
                self.cb.code(_PUBLISH_VIEWS_CELL),
                self.cb.section("3. Print Run Summary"),
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
    print(f"SKIPPED: {PREDICTIONS_FQN} not populated (run s10_batch_inference first)")
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
'''
