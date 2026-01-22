from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class IngestionStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.INGESTION

    @property
    def title(self) -> str:
        return "01 - Configuration & Data Ingestion"

    @property
    def description(self) -> str:
        return "Load raw data, configure pipeline context, and save to bronze layer."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        findings_path = self.findings.source_path if self.findings else "./data/customers.csv"
        source_format = getattr(self.findings, "source_format", "csv") if self.findings else "csv"
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.generators.orchestration": ["setup_notebook_context", "PipelineContext"],
                "customer_retention.stages.ingestion": ["DataSourceRegistry"],
                "customer_retention.analysis.auto_explorer": ["ExplorationFindings"],
                "customer_retention.stages.temporal": ["ScenarioDetector", "UnifiedDataPreparer"],
                "datetime": ["datetime"],
                "pathlib": ["Path"],
            }),
            self.cb.section("Configuration"),
            self.cb.code(f'''FINDINGS_PATH = "{findings_path}"
DATA_FORMAT = "{source_format}"
OUTPUT_DIR = Path("./experiments/data")'''),
            self.cb.section("Load Exploration Findings"),
            self.cb.code('''findings = ExplorationFindings.load(FINDINGS_PATH)
print(f"Loaded findings: {findings.row_count} rows, {findings.column_count} columns")
print(f"Target column: {findings.target_column}")'''),
            self.cb.section("Setup Pipeline Context"),
            self.cb.code('''ctx, manager = setup_notebook_context(exploration_findings=findings)
print(f"Pipeline context initialized for: {ctx.config.project_name}")'''),
            self.cb.section("Load Raw Data"),
            self.cb.code('''registry = DataSourceRegistry()
df = registry.load(findings.source_path, format=DATA_FORMAT)
print(f"Loaded {len(df)} rows")
df.head()'''),
            self.cb.section("Detect Timestamp Scenario"),
            self.cb.code('''detector = ScenarioDetector()
scenario, ts_config, discovery_result = detector.detect(df, findings.target_column)
print(f"Detected scenario: {scenario}")
print(f"Strategy: {ts_config.strategy.value}")
print(f"Recommendation: {discovery_result.recommendation}")'''),
            self.cb.section("Prepare Data with Timestamps"),
            self.cb.code('''preparer = UnifiedDataPreparer(OUTPUT_DIR, ts_config)
unified_df = preparer.prepare_from_raw(
    df,
    target_column=findings.target_column,
    entity_column=findings.entity_id_column or "custid"
)
print(f"Prepared {len(unified_df)} rows with timestamps")
print(f"Timestamp columns: feature_timestamp, label_timestamp, label_available_flag")'''),
            self.cb.section("Create Training Snapshot"),
            self.cb.code('''cutoff_date = datetime.now()
snapshot_df, metadata = preparer.create_training_snapshot(unified_df, cutoff_date)
print(f"Created snapshot: {metadata['snapshot_id']}")
print(f"Rows: {metadata['row_count']}")
print(f"Features: {len(metadata['feature_columns'])}")'''),
            self.cb.section("Save Processed Data"),
            self.cb.code('''manager.update(current_df=snapshot_df, current_stage="bronze")
print(f"Pipeline context updated. Use snapshot '{metadata['snapshot_id']}' for training.")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        data_path = self.findings.source_path if self.findings else "/mnt/landing/customers"
        source_format = getattr(self.findings, "source_format", "csv") if self.findings else "csv"
        return self.header_cells() + [
            self.cb.section("Configuration"),
            self.cb.code(f'''CATALOG = "{catalog}"
SCHEMA = "{schema}"
DATA_PATH = "{data_path}"
spark.sql(f"USE CATALOG {{CATALOG}}")
spark.sql(f"USE SCHEMA {{SCHEMA}}")'''),
            self.cb.section("Load Raw Data"),
            self.cb.code(f'''df_raw = (spark.read
    .format("{source_format}")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(DATA_PATH))
print(f"Loaded {{df_raw.count()}} rows")
display(df_raw.limit(10))'''),
            self.cb.section("Save to Bronze Table"),
            self.cb.code('''df_raw.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_customers")
print("Bronze table created")'''),
        ]
