"""Stage generator for ``c04_snapshot_and_dashboard``.

The final causal-track notebook. Builds the per-scoring-run
``eligibility_snapshot`` table, publishes the dashboard SQL views, and
prints the four-way anchor tuple in force.

Phase 3 placeholder. Cells 1 and 2 are gated on ``RUN_PHASE3``; while
the flag is ``False`` they print a status line and skip, so the run
summary cell at the bottom always executes against any populated
``archetype_catalog``.
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
                self.cb.section("1. Build Eligibility Snapshot (Phase 3 — gated)"),
                self.cb.code(_BUILD_SNAPSHOT_CELL),
                self.cb.section("2. Publish Dashboard SQL Views (Phase 3 — gated)"),
                self.cb.code(_PUBLISH_VIEWS_CELL),
                self.cb.section("3. Print Run Summary"),
                self.cb.code(_PRINT_SUMMARY_CELL),
            ]
        )


_BUILD_SNAPSHOT_CELL = '''if not RUN_PHASE3:
    print("SKIPPED: RUN_PHASE3=False (snapshot_writer.py ships in Phase 3)")
else:
    raise NotImplementedError(
        "snapshot_writer.py ships in Phase 3 — see "
        "docs/causal_track_implementation_plan.md §Phase 3"
    )
'''


_PUBLISH_VIEWS_CELL = '''if not RUN_PHASE3:
    print("SKIPPED: RUN_PHASE3=False (dashboard_views.sql ships in Phase 3)")
else:
    raise NotImplementedError("dashboard_views.sql ships in Phase 3")
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
'''
