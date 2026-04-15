"""Stage generator for ``c01_publish_definitions``.

The first causal-track notebook. Reads ``*.yaml`` from the playbooks
volume and overwrites the definition-layer Delta tables
(``playbook_catalog``, ``playbook_steps``, ``response_schemas``,
``vocabularies``, ``decision_policy``). Lightweight setup — no model URI
lookup, no LLM namer build.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import (
    C01_PIPELINE_SUMMARY_BODY,
    C01_RUN_PIPELINE_BODY,
    C01_RUN_PIPELINE_MD,
    c01_setup_block,
)


class PublishDefinitionsStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.PUBLISH_DEFINITIONS

    @property
    def title(self) -> str:
        return "c01 - Publish Definition Tables (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Reads playbook + policy YAMLs from `PLAYBOOKS_DIR` and overwrites "
            "the definition-layer Delta tables. Always run this notebook before "
            "`c02_archetype_derivation` if YAMLs have changed since the last run."
        )

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def _cells(self) -> List[nbformat.NotebookNode]:
        return (
            self.header_cells()
            + c01_setup_block()
            + [
                self.cb.section("1. Publish Definition Tables (YAML → Delta)"),
                self.cb.code(_PUBLISH_DEFINITIONS_CELL),
                self.cb.markdown(C01_RUN_PIPELINE_MD),
                self.cb.code(C01_RUN_PIPELINE_BODY),
                self.cb.code(C01_PIPELINE_SUMMARY_BODY),
            ]
        )


_PUBLISH_DEFINITIONS_CELL = '''from customer_retention.stages.causal.delta_writer import overwrite_table
from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir
from customer_retention.stages.causal.policy_loader import load_policies_from_dir
from customer_retention.stages.causal.schemas import (
    decision_policy_schema,
    playbook_catalog_schema,
    playbook_steps_schema,
    response_schemas_schema,
    vocabularies_schema,
)

if SKIP_PUBLISH_DEFINITIONS:
    print("SKIPPED: SKIP_PUBLISH_DEFINITIONS=True")
elif spark is None:
    print("SKIPPED: no active Spark session (Databricks-only cell)")
else:
    catalog_rows, step_rows = load_playbooks_from_dir(PLAYBOOKS_DIR)
    policies = load_policies_from_dir(PLAYBOOKS_DIR)

    decision_rows = policies.get("decision_policy", [])
    for row in decision_rows:
        if row.get("risk_tier_high_threshold") is None:
            row["risk_tier_high_threshold"] = RISK_TIER_HIGH_THRESHOLD
        if row.get("risk_tier_medium_threshold") is None:
            row["risk_tier_medium_threshold"] = RISK_TIER_MEDIUM_THRESHOLD

    overwrite_table(spark, catalog_rows, playbook_catalog_schema(), PLAYBOOK_CATALOG_FQN)
    overwrite_table(spark, step_rows, playbook_steps_schema(), PLAYBOOK_STEPS_FQN)
    overwrite_table(spark, decision_rows, decision_policy_schema(), DECISION_POLICY_FQN)
    overwrite_table(spark, policies.get("response_schemas", []), response_schemas_schema(), RESPONSE_SCHEMAS_FQN)
    overwrite_table(spark, policies.get("vocabularies", []), vocabularies_schema(), VOCABULARIES_FQN)
    print(
        f"Published {len(catalog_rows)} playbooks, {len(step_rows)} steps, "
        f"{len(decision_rows)} decision_policy rows"
    )
'''
