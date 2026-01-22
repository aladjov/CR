from typing import TYPE_CHECKING, Any, Dict, List, Optional

from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation, RecommendationRegistry

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ExplorationFindings


class DatabricksExporter:
    CELL_SEPARATOR = "\n# COMMAND ----------\n"

    def __init__(
        self,
        registry: RecommendationRegistry,
        findings: Optional["ExplorationFindings"] = None,
        catalog: str = "main",
        schema: str = "default"
    ):
        self.registry = registry
        self.findings = findings
        self.catalog = catalog
        self.schema = schema

    def generate_notebook(self) -> str:
        cells = [
            self._header_cell(),
            self._imports_cell(),
            self._config_cell(),
            self.generate_bronze_notebook(),
            self.generate_silver_notebook(),
            self.generate_gold_notebook(),
        ]
        return self.CELL_SEPARATOR.join(cells)

    def generate_source_notebooks(self) -> Dict[str, str]:
        notebooks = {}
        for name, bronze in self.registry.sources.items():
            notebooks[name] = self._generate_source_bronze_notebook(name, bronze)
        return notebooks

    def _generate_source_bronze_notebook(self, name: str, bronze) -> str:
        lines = [
            "# MAGIC %md",
            f"# MAGIC ## Bronze Layer: {name}",
            "",
            "# COMMAND ----------",
            "",
            self._imports_cell(),
            "",
            "# COMMAND ----------",
            "",
            f"# Read from landing zone: {name}",
            f'df_raw = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("{bronze.source_file}")',
            "",
            "# Apply cleaning transformations",
            "df_bronze = df_raw",
        ]
        for rec in bronze.null_handling:
            lines.extend(self._pyspark_null_handling(rec))
        for rec in bronze.outlier_handling:
            lines.extend(self._pyspark_outlier_handling(rec))
        for rec in bronze.type_conversions:
            lines.extend(self._pyspark_type_conversion(rec))
        lines.extend([
            "",
            "# Write to bronze Delta table",
            f'df_bronze.write.format("delta").mode("overwrite").saveAsTable("{self._source_table_path("bronze", name)}")',
            "",
            "display(df_bronze.limit(10))",
        ])
        return "\n".join(lines)

    def generate_silver_merge_notebook(self) -> str:
        lines = [
            "# MAGIC %md",
            "# MAGIC ## Silver Layer: Merge & Aggregations",
            "",
            "# COMMAND ----------",
            "",
            self._imports_cell(),
            "",
            "# COMMAND ----------",
            "",
            "# Read bronze tables",
        ]
        for name in self.registry.source_names:
            lines.append(f'df_{name} = spark.table("{self._source_table_path("bronze", name)}")')
        lines.append("")

        if self.registry.silver and self.registry.silver.joins:
            lines.append("# Merge sources")
            for i, join_rec in enumerate(self.registry.silver.joins):
                params = join_rec.parameters
                left = params["left_source"]
                right = params["right_source"]
                keys = params["join_keys"]
                join_type = params["join_type"]
                if i == 0:
                    lines.append(f'df_merged = df_{left}.join(df_{right}, on={keys}, how="{join_type}")')
                else:
                    if left == "_merged":
                        lines.append(f'df_merged = df_merged.join(df_{right}, on={keys}, how="{join_type}")')
                    else:
                        lines.append(f'df_merged = df_{left}.join(df_{right}, on={keys}, how="{join_type}")')
            lines.append("")
            lines.append("df_silver = df_merged")
        else:
            first_source = self.registry.source_names[0] if self.registry.source_names else "data"
            lines.append(f"df_silver = df_{first_source}")

        if self.registry.silver:
            entity_col = self.registry.silver.entity_column
            for rec in self.registry.silver.aggregations:
                lines.extend(self._pyspark_aggregation(rec, entity_col))

        lines.extend([
            "",
            "# Write to silver Delta table",
            f'df_silver.write.format("delta").mode("overwrite").saveAsTable("{self._table_path("silver")}")',
            "",
            "display(df_silver.limit(10))",
        ])
        return "\n".join(lines)

    def generate_gold_features_notebook(self) -> str:
        lines = [
            "# MAGIC %md",
            "# MAGIC ## Gold Layer: Feature Engineering",
            "",
            "# COMMAND ----------",
            "",
            self._imports_cell(),
            "",
            "# COMMAND ----------",
            "",
            "# Read from silver",
            f'df_silver = spark.table("{self._table_path("silver")}")',
            "",
            "# Apply feature transformations",
            "df_gold = df_silver",
        ]
        if self.registry.gold:
            for rec in self.registry.gold.encoding:
                lines.extend(self._pyspark_encoding(rec))
            for rec in self.registry.gold.scaling:
                lines.extend(self._pyspark_scaling(rec))
            for rec in self.registry.gold.transformations:
                lines.extend(self._pyspark_transformation(rec))
        lines.extend([
            "",
            "# Write to gold Delta table (ML-ready)",
            f'df_gold.write.format("delta").mode("overwrite").saveAsTable("{self._table_path("gold")}")',
            "",
            "display(df_gold.limit(10))",
        ])
        return "\n".join(lines)

    def export_notebook_structure(self) -> Dict[str, Any]:
        structure = {
            "bronze": {},
            "silver": self.generate_silver_merge_notebook(),
            "gold": self.generate_gold_features_notebook(),
        }
        for name, code in self.generate_source_notebooks().items():
            structure["bronze"][name] = code
        return structure

    def generate_bronze_notebook(self) -> str:
        lines = [
            "# MAGIC %md",
            "# MAGIC ## Bronze Layer: Data Cleaning",
            "",
            "# COMMAND ----------",
            "",
            "# Read from landing zone",
            f'df_raw = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("{self._landing_path()}")',
            "",
            "# Apply cleaning transformations",
            "df_bronze = df_raw",
        ]
        if self.registry.bronze:
            for rec in self.registry.bronze.null_handling:
                lines.extend(self._pyspark_null_handling(rec))
            for rec in self.registry.bronze.outlier_handling:
                lines.extend(self._pyspark_outlier_handling(rec))
            for rec in self.registry.bronze.type_conversions:
                lines.extend(self._pyspark_type_conversion(rec))
        lines.extend([
            "",
            "# Write to bronze Delta table",
            f'df_bronze.write.format("delta").mode("overwrite").saveAsTable("{self._table_path("bronze")}")',
            "",
            "display(df_bronze.limit(10))",
        ])
        return "\n".join(lines)

    def generate_silver_notebook(self) -> str:
        lines = [
            "# MAGIC %md",
            "# MAGIC ## Silver Layer: Joins & Aggregations",
            "",
            "# COMMAND ----------",
            "",
            "# Read from bronze",
            f'df_bronze = spark.table("{self._table_path("bronze")}")',
            "",
            "# Apply aggregations",
            "df_silver = df_bronze",
        ]
        if self.registry.silver:
            entity_col = self.registry.silver.entity_column
            for rec in self.registry.silver.aggregations:
                lines.extend(self._pyspark_aggregation(rec, entity_col))
        lines.extend([
            "",
            "# Write to silver Delta table",
            f'df_silver.write.format("delta").mode("overwrite").saveAsTable("{self._table_path("silver")}")',
            "",
            "display(df_silver.limit(10))",
        ])
        return "\n".join(lines)

    def generate_gold_notebook(self) -> str:
        lines = [
            "# MAGIC %md",
            "# MAGIC ## Gold Layer: Feature Engineering",
            "",
            "# COMMAND ----------",
            "",
            "# Read from silver",
            f'df_silver = spark.table("{self._table_path("silver")}")',
            "",
            "# Apply feature transformations",
            "df_gold = df_silver",
        ]
        if self.registry.gold:
            for rec in self.registry.gold.encoding:
                lines.extend(self._pyspark_encoding(rec))
            for rec in self.registry.gold.scaling:
                lines.extend(self._pyspark_scaling(rec))
            for rec in self.registry.gold.transformations:
                lines.extend(self._pyspark_transformation(rec))
        lines.extend([
            "",
            "# Write to gold Delta table (ML-ready)",
            f'df_gold.write.format("delta").mode("overwrite").saveAsTable("{self._table_path("gold")}")',
            "",
            "display(df_gold.limit(10))",
        ])
        return "\n".join(lines)

    def to_notebook_cells(self) -> List[Dict[str, str]]:
        return [
            {"content": self._header_cell(), "type": "markdown"},
            {"content": self._imports_cell(), "type": "code"},
            {"content": self._config_cell(), "type": "code"},
            {"content": self.generate_bronze_notebook(), "type": "code"},
            {"content": self.generate_silver_notebook(), "type": "code"},
            {"content": self.generate_gold_notebook(), "type": "code"},
        ]

    def _header_cell(self) -> str:
        source = self.findings.source_path if self.findings else "data"
        return f"""# MAGIC %md
# MAGIC # Data Pipeline: {source}
# MAGIC
# MAGIC Auto-generated pipeline using medallion architecture.
# MAGIC
# MAGIC | Layer | Description |
# MAGIC |-------|-------------|
# MAGIC | Bronze | Cleaned raw data |
# MAGIC | Silver | Aggregated/joined data |
# MAGIC | Gold | ML-ready features |"""

    def _imports_cell(self) -> str:
        return """from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline"""

    def _config_cell(self) -> str:
        return f"""# Configuration
CATALOG = "{self.catalog}"
SCHEMA = "{self.schema}"
LANDING_PATH = "{self._landing_path()}"

# Set catalog context
spark.sql(f"USE CATALOG {{CATALOG}}")
spark.sql(f"USE SCHEMA {{SCHEMA}}")"""

    def _landing_path(self) -> str:
        if self.findings:
            return self.findings.source_path
        if self.registry.bronze:
            return self.registry.bronze.source_file
        return "/mnt/landing/data"

    def _table_path(self, layer: str) -> str:
        return f"{self.catalog}.{self.schema}.{layer}_customers"

    def _source_table_path(self, layer: str, source_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{layer}_{source_name}"

    def _pyspark_null_handling(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        strategy = rec.parameters.get("strategy", "median")
        lines = ["", f"# {rec.rationale}"]
        if strategy == "median":
            lines.extend([
                f"median_val = df_bronze.approxQuantile('{col}', [0.5], 0.01)[0]",
                f"df_bronze = df_bronze.na.fill({{'{col}': median_val}})",
            ])
        elif strategy == "mean":
            lines.extend([
                f"mean_val = df_bronze.agg(F.mean('{col}')).collect()[0][0]",
                f"df_bronze = df_bronze.na.fill({{'{col}': mean_val}})",
            ])
        elif strategy == "mode":
            lines.extend([
                f"mode_val = df_bronze.groupBy('{col}').count().orderBy(F.desc('count')).first()[0]",
                f"df_bronze = df_bronze.na.fill({{'{col}': mode_val}})",
            ])
        else:
            lines.append(f"df_bronze = df_bronze.na.fill({{'{col}': 0}})")
        return lines

    def _pyspark_outlier_handling(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        factor = rec.parameters.get("factor", 1.5)
        return [
            "",
            f"# {rec.rationale}",
            f"quantiles = df_bronze.approxQuantile('{col}', [0.25, 0.75], 0.01)",
            "q1, q3 = quantiles[0], quantiles[1]",
            "iqr = q3 - q1",
            f"lower_bound = q1 - {factor} * iqr",
            f"upper_bound = q3 + {factor} * iqr",
            f"df_bronze = df_bronze.withColumn('{col}', F.when(F.col('{col}') < lower_bound, lower_bound)",
            f"                                          .when(F.col('{col}') > upper_bound, upper_bound)",
            f"                                          .otherwise(F.col('{col}')))",
        ]

    def _pyspark_type_conversion(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        target_type = rec.parameters.get("target_type", "string")
        pyspark_type = {"datetime": "timestamp", "int": "integer", "float": "double"}.get(target_type, target_type)
        return [
            "",
            f"# {rec.rationale}",
            f"df_bronze = df_bronze.withColumn('{col}', F.col('{col}').cast('{pyspark_type}'))",
        ]

    def _pyspark_aggregation(self, rec: LayeredRecommendation, entity_col: str) -> List[str]:
        col = rec.target_column
        agg = rec.parameters.get("aggregation", "sum")
        feature_name = f"{col}_{agg}"
        window = f"Window.partitionBy('{entity_col}')"
        agg_func = {"sum": "F.sum", "mean": "F.mean", "avg": "F.avg", "count": "F.count", "max": "F.max", "min": "F.min"}.get(agg, "F.sum")
        return [
            "",
            f"# {rec.rationale}",
            f"window_spec = {window}",
            f"df_silver = df_silver.withColumn('{feature_name}', {agg_func}('{col}').over(window_spec))",
        ]

    def _pyspark_encoding(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        method = rec.parameters.get("method", "one_hot")
        if method == "one_hot":
            return [
                "",
                f"# {rec.rationale}",
                f"indexer_{col} = StringIndexer(inputCol='{col}', outputCol='{col}_idx', handleInvalid='keep')",
                f"encoder_{col} = OneHotEncoder(inputCol='{col}_idx', outputCol='{col}_onehot')",
                f"pipeline_{col} = Pipeline(stages=[indexer_{col}, encoder_{col}])",
                f"df_gold = pipeline_{col}.fit(df_gold).transform(df_gold)",
                f"df_gold = df_gold.drop('{col}', '{col}_idx')",
            ]
        return ["", f"# {rec.rationale} - {method} encoding (implement as needed)"]

    def _pyspark_scaling(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        return [
            "",
            f"# {rec.rationale}",
            f"assembler_{col} = VectorAssembler(inputCols=['{col}'], outputCol='{col}_vec')",
            f"scaler_{col} = StandardScaler(inputCol='{col}_vec', outputCol='{col}_scaled', withMean=True, withStd=True)",
            f"df_gold = assembler_{col}.transform(df_gold)",
            f"df_gold = scaler_{col}.fit(df_gold).transform(df_gold)",
            f"df_gold = df_gold.drop('{col}', '{col}_vec')",
        ]

    def _pyspark_transformation(self, rec: LayeredRecommendation) -> List[str]:
        col = rec.target_column
        method = rec.parameters.get("method", "log")
        if method == "log":
            return [
                "",
                f"# {rec.rationale}",
                f"df_gold = df_gold.withColumn('{col}', F.log1p(F.col('{col}')))",
            ]
        elif method == "sqrt":
            return [
                "",
                f"# {rec.rationale}",
                f"df_gold = df_gold.withColumn('{col}', F.sqrt(F.col('{col}')))",
            ]
        return []
