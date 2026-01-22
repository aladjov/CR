from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class ProfilingStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.PROFILING

    @property
    def title(self) -> str:
        return "02 - Data Profiling"

    @property
    def description(self) -> str:
        return "Generate column statistics, type detection, and quality metrics."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.profiling": ["TypeDetector", "ProfilerFactory", "QualityCheckRegistry"],
                "customer_retention.analysis.visualization": ["ChartBuilder"],
                "pandas": ["pd"],
            }),
            self.cb.section("Load Bronze Data"),
            self.cb.code('''df = pd.read_parquet("./experiments/data/bronze/customers.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")'''),
            self.cb.section("Type Detection"),
            self.cb.code('''detector = TypeDetector()
type_results = {col: detector.detect(df[col]) for col in df.columns}
for col, result in type_results.items():
    print(f"{col}: {result.column_type.value} (confidence: {result.confidence:.2f})")'''),
            self.cb.section("Column Profiling"),
            self.cb.code('''factory = ProfilerFactory()
profiles = {}
for col in df.columns:
    profiler = factory.get_profiler(type_results[col].column_type)
    profiles[col] = profiler.profile(df[col])'''),
            self.cb.section("Quality Checks"),
            self.cb.code('''registry = QualityCheckRegistry()
checks = registry.get_all_checks()
results = []
for check in checks:
    for col in df.columns:
        result = check.check(df[col], profiles.get(col))
        if result.passed is False:
            results.append({"column": col, "check": check.name, "severity": result.severity.value, "message": result.message})
quality_df = pd.DataFrame(results)
quality_df'''),
            self.cb.section("Visualize Quality"),
            self.cb.code('''charts = ChartBuilder()
if len(quality_df) > 0:
    fig = charts.quality_heatmap(quality_df)
    fig.show()'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        return self.header_cells() + [
            self.cb.section("Load Bronze Data"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.bronze_customers")
print(f"Loaded {{df.count()}} rows")'''),
            self.cb.section("Basic Statistics"),
            self.cb.code('''summary = df.describe()
display(summary)'''),
            self.cb.section("Column Types and Nulls"),
            self.cb.code('''from pyspark.sql.functions import col, count, when, isnan

null_counts = df.select([
    count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
    for c in df.columns
])
display(null_counts)'''),
            self.cb.section("Distinct Values"),
            self.cb.code('''from pyspark.sql.functions import countDistinct

distinct_counts = df.select([countDistinct(col(c)).alias(c) for c in df.columns])
display(distinct_counts)'''),
            self.cb.section("Save Profiling Results"),
            self.cb.code('''profile_data = {
    "columns": df.columns,
    "dtypes": [str(f.dataType) for f in df.schema.fields],
    "row_count": df.count()
}
import json
dbutils.fs.put("/tmp/profile_results.json", json.dumps(profile_data), overwrite=True)'''),
        ]
