"""Stage generator for ``c03_approval_gate``.

The human checkpoint of the causal track. Loads the most recent
``derivation_run_id`` from ``archetype_catalog`` (the rows c02 just
wrote as ``pending_review``), runs the cosine-similarity stability
check against each archetype's prior active row, auto-promotes stable
archetypes, cascades to ``eligibility_policy`` via ``arrays_overlap``,
and prints the manual review queue for anything that didn't auto-promote.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import c03_setup_block


class ApprovalGateStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.APPROVAL_GATE

    @property
    def title(self) -> str:
        return "c03 - Approval Gate (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Auto-promotes stable archetypes (cosine ≥ STABILITY_THRESHOLD) "
            "and cascades the promotion to `eligibility_policy`. Anything "
            "below threshold stays `pending_review` and shows up on the manual "
            "queue printed at the bottom of the notebook."
        )

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def _cells(self) -> List[nbformat.NotebookNode]:
        return (
            self.header_cells()
            + c03_setup_block()
            + [
                self.cb.section("3.1 Resolve Latest Pending Derivation Run"),
                self.cb.code(_RESOLVE_RUN_CELL),
                self.cb.section("3.2 Run Approval Gate"),
                self.cb.code(_APPROVAL_GATE_CELL),
                self.cb.section("3.3 Print Pending Review Queue"),
                self.cb.code(_PRINT_PENDING_CELL),
            ]
        )


_RESOLVE_RUN_CELL = '''from customer_retention.stages.causal import expire_stale_pending

derivation_run_id = None
if spark is not None and spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    row = spark.sql(
        f"SELECT derivation_run_id FROM {ARCHETYPE_CATALOG_FQN} "
        "WHERE status = 'pending_review' AND model_name = ? AND model_version = ? "
        "ORDER BY derivation_run_id DESC LIMIT 1",
        args=[MODEL_NAME, MODEL_VERSION],
    ).head()
    derivation_run_id = row["derivation_run_id"] if row else None

if derivation_run_id is None:
    print("No pending_review derivations found for the current model version.")
else:
    expired = expire_stale_pending(
        spark, ARCHETYPE_CATALOG_FQN, ELIGIBILITY_POLICY_FQN,
        MODEL_NAME, MODEL_VERSION, derivation_run_id,
    )
    if expired:
        print(f"Expired {expired} stale pending_review rows from prior derivation runs")
    print(f"Resolved derivation_run_id: {derivation_run_id}")
'''


_APPROVAL_GATE_CELL = '''from customer_retention.stages.causal import auto_promote_stable

gate_result = None
if derivation_run_id is None:
    print("SKIPPED: nothing pending to approve")
else:
    gate_result = auto_promote_stable(
        spark=spark,
        archetype_table_fqn=ARCHETYPE_CATALOG_FQN,
        policy_table_fqn=ELIGIBILITY_POLICY_FQN,
        derivation_run_id=derivation_run_id,
        threshold=STABILITY_THRESHOLD,
        force=FORCE_APPROVE,
    )
    print(gate_result.summary())
'''


_PRINT_PENDING_CELL = '''from customer_retention.stages.causal import list_pending_review

if derivation_run_id is None:
    print("(no pending queue — derivation_run_id not resolved)")
else:
    pending = list_pending_review(spark, ARCHETYPE_CATALOG_FQN, derivation_run_id)
    if not pending:
        print("All archetypes auto-promoted. Pending review queue is empty.")
    else:
        print(
            "Pending review queue (re-run this notebook with FORCE_APPROVE=True "
            "after manual review):"
        )
        for row in pending:
            print(
                f"  - {row['archetype_id']} v{row['archetype_version']} "
                f"name={row['name']!r} stability={row['stability_vs_prior_version']}"
            )
'''
