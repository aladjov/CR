"""Batch inference stage for notebook generation.

This stage generates notebooks that perform batch scoring using the feature store
with point-in-time correctness for feature retrieval.
"""

from typing import TYPE_CHECKING, List, Optional

import nbformat

from ..base import NotebookStage
from ..scoring_replay import render_scoring_replay_block
from .base_stage import StageGenerator

if TYPE_CHECKING:
    from customer_retention.runtime.harvest import HarvestResult


class BatchInferenceStage(StageGenerator):
    harvest_result: Optional["HarvestResult"] = None

    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.BATCH_INFERENCE

    def _scoring_replay_cells(self) -> List[nbformat.NotebookNode]:
        if self.harvest_result is None:
            return []
        block = render_scoring_replay_block(self.harvest_result.all_functions())
        if not block:
            return []
        return [
            self.cb.section("3b. User-Extension Replay"),
            self.cb.markdown(
                "User-supplied functions with `replay_at_scoring=True` execute "
                "here so training-time transforms apply identically at scoring."
            ),
            self.cb.code(block),
        ]

    @property
    def title(self) -> str:
        return "10 - Batch Inference with Point-in-Time Features"

    @property
    def description(self) -> str:
        return """Score customers in batch using the production model with point-in-time correct feature retrieval.

**Key Concepts:**
- **Point-in-Time (PIT) Correctness**: Features are retrieved as they existed at inference time
- **Inference Timestamp**: The moment when predictions are made, ensuring no future data leakage
- **Feature Store Integration**: Uses Feast (local) or Databricks Feature Store for consistent feature retrieval

This notebook:
1. Sets the inference timestamp (point-in-time for prediction)
2. Retrieves features from the feature store with PIT correctness
3. Scores customers using the production model
4. Generates a dashboard showing predictions with the inference timestamp
"""

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for local Feast-based batch inference.

        These cells are thin orchestration shells around
        ``customer_retention.stages.scoring.batch_inference.run_batch_inference``.
        All business logic — feature lookup, model load, threshold + risk
        tiers, persistence, audit metadata — lives in the framework module
        and is exercised by direct unit tests. The notebook just configures
        the call and renders the result.
        """
        threshold = self.config.threshold
        name = self.get_dataset_name()
        return self.header_cells() + [
            self.cb.section("1. Setup and Imports"),
            self.cb.code('''from datetime import datetime
from pathlib import Path

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    run_batch_inference,
)

print("Batch inference imports loaded")'''),

            self.cb.section("2. Set Inference Point-in-Time"),
            self.cb.markdown('''**Critical**: The inference timestamp determines what point in time we use to retrieve features.
This ensures we only use data that was available at the time of prediction (no future leakage).

- For **real-time inference**: Use `datetime.now()`
- For **historical backtesting**: Use a specific past timestamp
- For **scheduled batch jobs**: Use the job execution timestamp'''),
            self.cb.code('''# INFERENCE_TIMESTAMP: The point-in-time for this batch prediction.
# Pass None to BatchInferenceConfig to default to datetime.now() at run time.
INFERENCE_TIMESTAMP = datetime.now()

# Alternative: Use a specific historical timestamp for backtesting
# INFERENCE_TIMESTAMP = datetime(2024, 1, 15, 0, 0, 0)

print(f"INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP}")'''),

            self.cb.section("3. Configure the Batch Inference Run"),
            self.cb.markdown('''All scoring parameters live on the `BatchInferenceConfig` dataclass.
Risk tier thresholds and the prediction threshold are explicit so historical runs are reconstructable
from the config that was in force at scoring time.'''),
            self.cb.code(f'''config = BatchInferenceConfig(
    threshold={threshold},
    risk_tier_high=0.6,
    risk_tier_medium=0.3,
    inference_timestamp=INFERENCE_TIMESTAMP,
    model_path=Path("./experiments/data/models/best_model.joblib"),
    feature_store_repo=Path("./experiments/feature_store/feature_repo"),
    output_dir=Path("./experiments/data/predictions"),
    dataset_name={name!r},
    entity_id_columns={self.get_identifier_columns()},
)
print(config)'''),

            *self._scoring_replay_cells(),

            self.cb.section("4. Run Batch Inference"),
            self.cb.markdown('''The framework call handles model loading, feature retrieval with point-in-time
correctness, threshold + risk-tier application, and persistence (timestamped + latest files +
JSON metadata for audit). Returns a `BatchInferenceResult` with all the run statistics.'''),
            self.cb.code('''result = run_batch_inference(config)
print(result.long_summary())'''),

            self.cb.section("5. Inspect Output Files"),
            self.cb.code('''print("Output artifacts written by this run:")
for path in result.output_files:
    print(f"  {path}")'''),

            self.cb.section("6. Summary"),
            self.cb.code('''print("=" * 70)
print("BATCH INFERENCE COMPLETE")
print("=" * 70)
print(result.summary())
print()
print("Next steps:")
print("1. Review high-risk customers for intervention")
print("2. Schedule next batch inference run")
print("3. Monitor model performance over time")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for Databricks Feature Store batch inference.

        Like the local cells, these are thin orchestration shells around
        ``customer_retention.stages.scoring.batch_inference.run_batch_inference``.
        Feature-store lookup, MLflow model load, threshold + risk tiers,
        Delta write, audit log append — all of it lives in the framework
        module. The notebook just configures the call and renders the result.
        """
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        model_name = self.config.mlflow.model_name
        threshold = self.config.threshold

        return self.header_cells() + [
            self.cb.section("1. Setup and Imports"),
            self.cb.code('''from datetime import datetime

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    run_batch_inference,
)

print("Batch inference imports loaded")'''),

            self.cb.section("2. Set Inference Point-in-Time"),
            self.cb.markdown('''**Critical**: The inference timestamp is the point-in-time for feature retrieval.
The Databricks Feature Store uses `timestamp_lookup_key` to ensure PIT correctness.'''),
            self.cb.code('''# INFERENCE_TIMESTAMP: The point-in-time for this batch prediction.
# For production batch jobs, use the job execution timestamp.
INFERENCE_TIMESTAMP = datetime.now()

# Alternative: Use a specific historical timestamp for backtesting
# INFERENCE_TIMESTAMP = datetime(2024, 1, 15, 0, 0, 0)

print(f"INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP}")'''),

            self.cb.section("3. Configure the Batch Inference Run"),
            self.cb.markdown('''All scoring parameters live on the `BatchInferenceConfig` dataclass.
The Databricks path uses `catalog.schema.{model_name}@production` for the model URI and
writes to `{catalog}.{schema}.predictions` plus `{catalog}.{schema}.inference_audit_log`.
Risk tier thresholds are explicit so historical runs are reconstructable.'''),
            self.cb.code(f'''config = BatchInferenceConfig(
    catalog="{catalog}",
    schema="{schema}",
    model_name="{model_name}",
    threshold={threshold},
    risk_tier_high=0.6,
    risk_tier_medium=0.3,
    inference_timestamp=INFERENCE_TIMESTAMP,
    target_table="predictions",
    audit_table="inference_audit_log",
)
print(config)'''),

            *self._scoring_replay_cells(),

            self.cb.section("4. Run Batch Inference"),
            self.cb.markdown('''The framework call resolves the @production model URI, scores via
`fe.score_batch` with point-in-time correctness, applies risk tiers, writes the predictions
table, and appends an audit row. Returns a `BatchInferenceResult` with all run statistics.'''),
            self.cb.code('''result = run_batch_inference(config)
print(result.long_summary())'''),

            self.cb.section("5. Inspect Predictions Table"),
            self.cb.code('''print(f"Predictions table: {result.target_table_fqn}")
display(spark.table(result.target_table_fqn).limit(20))
print()
print("Risk tier breakdown:")
display(spark.table(result.target_table_fqn).groupBy("risk_tier").count().orderBy("risk_tier"))'''),

            self.cb.section("6. Summary"),
            self.cb.code('''print("=" * 70)
print("BATCH INFERENCE COMPLETE")
print("=" * 70)
print(result.summary())
print()
print("The inference_point_in_time column in the predictions table records exactly")
print("when features were retrieved, ensuring full auditability and reproducibility.")
print()
print("Next steps:")
print("1. Review high-risk customers for intervention")
print("2. Set up scheduled inference jobs")
print("3. Monitor prediction drift over time")'''),
        ]
