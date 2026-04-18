"""Stage generator for ``c02_archetype_derivation``.

The heavy compute notebook of the causal track. Loads the @production
model + the gold features (using the composite-name-qualified table
name resolved in the setup cell), freezes a stratified SHAP background,
runs SHAP via partition-wise pandas_udf, pre-selects the top features by
mean(|SHAP|), runs a silhouette-swept Spark KMeans, fits per-cluster
surrogate trees, and writes the resulting ``archetype_catalog`` /
``eligibility_policy`` rows as ``pending_review``.

Production scoring is **not** in this notebook — it lives in
``s10_batch_inference``. The c04 snapshot notebook reads the predictions
table that s10 writes.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator
from .causal_setup_cell import c02_setup_block


class ArchetypeDerivationStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.ARCHETYPE_DERIVATION

    @property
    def title(self) -> str:
        return "c02 - Archetype Derivation (Causal Track)"

    @property
    def description(self) -> str:
        return (
            "Reads the production model + gold features and derives SHAP-based "
            "archetypes plus eligibility policies. Writes everything as "
            "`pending_review` for the c03 approval gate to consume. Re-running "
            "with no model change is a no-op (skipped via the model_version guard)."
        )

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        return self._cells()

    def _cells(self) -> List[nbformat.NotebookNode]:
        return (
            self.header_cells()
            + c02_setup_block()
            + [
                self.cb.section("2.1 Derive Archetypes + Eligibility Policies"),
                self.cb.code(_DERIVE_ARCHETYPES_CELL),
            ]
        )


_DERIVE_ARCHETYPES_CELL = '''from customer_retention.stages.causal import (
    DerivationConfig,
    build_llm_namer,
    derive_archetypes_and_policies,
)
from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir


def _model_version_already_derived(_spark, table_fqn, model_name, model_version):
    if _spark is None or not _spark.catalog.tableExists(table_fqn):
        return False
    df = _spark.sql(
        f"SELECT 1 FROM {table_fqn} "
        "WHERE model_name = ? AND model_version = ? AND status = 'active' LIMIT 1",
        args=[model_name, model_version],
    )
    return df.limit(1).count() > 0


derivation_result = None
if not is_databricks():
    print("SKIPPED: derivation requires a Spark cluster (Databricks-only cell)")
elif not FORCE_DERIVATION and _model_version_already_derived(
    spark, ARCHETYPE_CATALOG_FQN, MODEL_NAME, MODEL_VERSION
):
    print(f"SKIPPED: active archetypes already exist for {MODEL_NAME} v{MODEL_VERSION}")
else:
    training_df = spark.table(GOLD_FEATURES_FQN)
    feature_columns = [
        c for c in training_df.columns
        if c not in ("account_id", "entity_id", "target", "churn_probability",
                     "event_timestamp", "inference_point_in_time", "model_uri")
    ]
    join_key = "account_id" if "account_id" in training_df.columns else "entity_id"
    catalog_rows, _ = load_playbooks_from_dir(PLAYBOOKS_DIR)
    llm_namer = build_llm_namer(LLM_ENDPOINT_NAME)
    print(f"LLM namer: {llm_namer.model_id}")

    cfg = DerivationConfig(
        spark=spark,
        training_df=training_df,
        raw_feature_df=training_df,
        feature_columns=feature_columns,
        model_uri=MODEL_URI,
        target_column="target",
        join_key=join_key,
        archetype_catalog_fqn=ARCHETYPE_CATALOG_FQN,
        eligibility_policy_fqn=ELIGIBILITY_POLICY_FQN,
        playbooks=catalog_rows,
        gold_feature_names=feature_columns,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        k_range=KMEANS_K_RANGE,
        k_cap=KMEANS_MAX_K,
        feature_cap=KMEANS_FEATURE_CAP,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        llm_namer=llm_namer,
    )
    derivation_result = derive_archetypes_and_policies(cfg)
    print(derivation_result.summary())
'''
