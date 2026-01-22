from typing import TYPE_CHECKING, List, Optional

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    LayeredRecommendation,
    RecommendationRegistry,
)

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ExplorationFindings


class PipelineDocGenerator:
    def __init__(self, registry: RecommendationRegistry, findings: Optional["ExplorationFindings"] = None):
        self.registry = registry
        self.findings = findings

    def generate(self) -> str:
        sections = [
            self._generate_header(),
            self._generate_assistant_prompt(),
            self._generate_data_overview(),
            self._generate_column_schema(),
            self._generate_sources_spec(),
            self._generate_bronze_spec(),
            self._generate_silver_spec(),
            self._generate_gold_spec(),
            self._generate_notebook_structure(),
            self._generate_delta_structure(),
        ]
        return "\n\n".join(s for s in sections if s)

    def _generate_header(self) -> str:
        return "# Pipeline Specification\n\nGenerated pipeline specification for LLM context."

    def _generate_assistant_prompt(self) -> str:
        lines = [
            "## Assistant Instructions",
            "",
            "Generate a standalone PySpark data pipeline following the medallion architecture.",
            "The code should:",
            "- Use native PySpark DataFrame operations (no external framework dependencies)",
            "- Read from landing zone and write to Delta tables",
            "- Implement Bronze (cleaning), Silver (aggregations), and Gold (feature engineering) layers",
            "- Use `pyspark.sql.functions` for transformations",
            "- Use `pyspark.ml.feature` for encoding and scaling",
        ]
        return "\n".join(lines)

    def _generate_data_overview(self) -> str:
        lines = ["## Data Overview", ""]
        if self.findings:
            lines.append(f"- **Source**: {self.findings.source_path}")
            lines.append(f"- **Format**: {self.findings.source_format}")
            lines.append(f"- **Rows**: {self.findings.row_count:,}")
            lines.append(f"- **Columns**: {self.findings.column_count}")
            if self.findings.target_column:
                lines.append(f"- **Target**: `{self.findings.target_column}` (variable to predict)")
            if self.findings.identifier_columns:
                lines.append(f"- **Identifiers**: {', '.join(f'`{c}`' for c in self.findings.identifier_columns)}")
        else:
            lines.append("No data overview available.")
        return "\n".join(lines)

    def _generate_column_schema(self) -> str:
        lines = ["## Column Schema", ""]
        if not self.findings or not self.findings.columns:
            lines.append("No column schema available.")
            return "\n".join(lines)

        lines.append("| Column | Type | Null % | Statistics |")
        lines.append("|--------|------|--------|------------|")

        for name, col in self.findings.columns.items():
            col_type = col.inferred_type.name if col.inferred_type else "UNKNOWN"
            null_pct = col.universal_metrics.get("null_percentage", 0) if col.universal_metrics else 0
            stats = self._format_column_stats(col)
            lines.append(f"| `{name}` | {col_type} | {null_pct:.1f}% | {stats} |")

        return "\n".join(lines)

    def _format_column_stats(self, col) -> str:
        parts = []
        if col.universal_metrics:
            if "distinct_count" in col.universal_metrics:
                parts.append(f"distinct={col.universal_metrics['distinct_count']}")
            if "null_count" in col.universal_metrics:
                parts.append(f"nulls={col.universal_metrics['null_count']}")
        if col.type_metrics:
            if "mean" in col.type_metrics:
                parts.append(f"mean={col.type_metrics['mean']}")
            if "std" in col.type_metrics:
                parts.append(f"std={col.type_metrics['std']}")
            if "skewness" in col.type_metrics:
                parts.append(f"skew={col.type_metrics['skewness']}")
        return ", ".join(parts) if parts else "-"

    def _generate_sources_spec(self) -> str:
        if not self.registry.sources:
            return ""

        lines = [
            "## Data Sources",
            "",
            "Multiple source files to process. Bronze notebooks can run in parallel.",
            "",
            "| Source | File Path | Recommendations |",
            "|--------|-----------|-----------------|",
        ]

        for name, bronze in self.registry.sources.items():
            rec_count = len(bronze.all_recommendations)
            lines.append(f"| `{name}` | {bronze.source_file} | {rec_count} |")

        lines.extend([
            "",
            "### Per-Source Bronze Transformations",
            "",
        ])

        for name, bronze in self.registry.sources.items():
            lines.append(f"#### {name}")
            lines.append(f"- **Source file**: {bronze.source_file}")
            if bronze.null_handling:
                lines.append(f"- **Null handling**: {', '.join(r.target_column for r in bronze.null_handling)}")
            if bronze.outlier_handling:
                lines.append(f"- **Outlier handling**: {', '.join(r.target_column for r in bronze.outlier_handling)}")
            if bronze.type_conversions:
                lines.append(f"- **Type conversions**: {', '.join(r.target_column for r in bronze.type_conversions)}")
            lines.append("")

        return "\n".join(lines)

    def _generate_bronze_spec(self) -> str:
        lines = ["## Bronze Layer: Data Cleaning", ""]
        if not self.registry.bronze or not self.registry.bronze.all_recommendations:
            if not self.registry.sources:
                lines.append("No bronze recommendations.")
            else:
                lines.append("See Per-Source Bronze Transformations above.")
            return "\n".join(lines)

        lines.extend([
            "Implementation: Use `df.na.fill()` for imputation, `df.withColumn()` with `F.when()` for capping.",
            "",
        ])

        if self.registry.bronze.null_handling:
            lines.append("### Null Handling")
            lines.append("Use `df.na.fill()` or compute median/mean then fillna.")
            for rec in self.registry.bronze.null_handling:
                lines.extend(self._format_recommendation(rec))

        if self.registry.bronze.outlier_handling:
            lines.append("### Outlier Handling")
            lines.append("Use IQR method: compute Q1, Q3, then clip with `F.when()`.")
            for rec in self.registry.bronze.outlier_handling:
                lines.extend(self._format_recommendation(rec))

        if self.registry.bronze.type_conversions:
            lines.append("### Type Conversions")
            lines.append("Use `df.withColumn(col, F.col(col).cast(type))`.")
            for rec in self.registry.bronze.type_conversions:
                lines.extend(self._format_recommendation(rec))

        if self.registry.bronze.filtering:
            lines.append("### Filtering")
            lines.append("Use `df.drop(columns)` or `df.filter()`.")
            for rec in self.registry.bronze.filtering:
                lines.extend(self._format_recommendation(rec))

        return "\n".join(lines)

    def _generate_silver_spec(self) -> str:
        lines = ["## Silver Layer: Joins & Aggregations", ""]
        if not self.registry.silver or not self.registry.silver.all_recommendations:
            lines.append("No silver recommendations.")
            return "\n".join(lines)

        lines.append(f"- **Entity Column**: `{self.registry.silver.entity_column}`")
        if self.registry.silver.time_column:
            lines.append(f"- **Time Column**: `{self.registry.silver.time_column}`")
        lines.extend([
            "",
            "Implementation: Use `Window.partitionBy()` for aggregations, `df.join()` for joins.",
            "",
        ])

        if self.registry.silver.joins:
            lines.append("### Joins")
            lines.append("Merge sources using `df.join()`. Execute after all bronze notebooks complete.")
            for rec in self.registry.silver.joins:
                lines.extend(self._format_recommendation(rec))

        if self.registry.silver.aggregations:
            lines.append("### Aggregations")
            lines.append("Use window functions: `F.sum().over(Window.partitionBy(entity))`.")
            for rec in self.registry.silver.aggregations:
                lines.extend(self._format_recommendation(rec))

        if self.registry.silver.derived_columns:
            lines.append("### Derived Columns")
            for rec in self.registry.silver.derived_columns:
                lines.extend(self._format_recommendation(rec))

        return "\n".join(lines)

    def _generate_gold_spec(self) -> str:
        lines = ["## Gold Layer: Feature Engineering", ""]
        if not self.registry.gold or not self.registry.gold.all_recommendations:
            lines.append("No gold recommendations.")
            return "\n".join(lines)

        lines.append(f"- **Target Column**: `{self.registry.gold.target_column}`")
        lines.extend([
            "",
            "Implementation: Use `StringIndexer` + `OneHotEncoder` for categoricals, `StandardScaler` for numerics.",
            "",
        ])

        if self.registry.gold.encoding:
            lines.append("### Encoding")
            lines.append("For one_hot: `StringIndexer(inputCol) -> OneHotEncoder(inputCol_idx)`.")
            for rec in self.registry.gold.encoding:
                lines.extend(self._format_recommendation(rec))

        if self.registry.gold.scaling:
            lines.append("### Scaling")
            lines.append("Use `VectorAssembler` then `StandardScaler` from pyspark.ml.feature.")
            for rec in self.registry.gold.scaling:
                lines.extend(self._format_recommendation(rec))

        if self.registry.gold.transformations:
            lines.append("### Transformations")
            lines.append("Use `F.log1p()` for log, `F.sqrt()` for sqrt transformations.")
            for rec in self.registry.gold.transformations:
                lines.extend(self._format_recommendation(rec))

        if self.registry.gold.feature_selection:
            lines.append("### Feature Selection")
            for rec in self.registry.gold.feature_selection:
                lines.extend(self._format_recommendation(rec))

        return "\n".join(lines)

    def _generate_notebook_structure(self) -> str:
        if not self.registry.sources:
            return ""

        source_names = list(self.registry.sources.keys())
        lines = [
            "## Notebook Structure",
            "",
            "Organize notebooks by layer. Bronze notebooks are independent and can run in parallel.",
            "",
            "```",
            "pipeline/",
            "├── bronze/                    # Parallel execution",
        ]

        for name in source_names:
            lines.append(f"│   └── bronze_{name}.py")

        lines.extend([
            "├── silver/                    # After bronze completes",
            "│   └── silver_merge.py        # Joins all bronze outputs",
            "└── gold/                      # After silver completes",
            "    └── gold_features.py       # Final ML-ready features",
            "```",
            "",
            "### Execution Order",
            "",
            f"1. **Bronze** (parallel): Run all bronze notebooks independently ({', '.join(source_names)})",
            "2. **Silver** (sequential): After all bronze complete, run silver merge notebook",
            "3. **Gold** (sequential): After silver completes, run gold features notebook",
        ])
        return "\n".join(lines)

    def _generate_delta_structure(self) -> str:
        lines = [
            "## Delta Table Structure",
            "",
            "Write each layer to Delta tables using:",
            "```",
            'df.write.format("delta").mode("overwrite").saveAsTable("catalog.schema.layer_table")',
            "```",
            "",
            "Suggested table names:",
        ]

        if self.registry.sources:
            for name in self.registry.sources.keys():
                lines.append(f"- Bronze ({name}): `{{catalog}}.{{schema}}.bronze_{name}`")
        else:
            lines.append("- Bronze: `{catalog}.{schema}.bronze_customers`")

        lines.extend([
            "- Silver: `{catalog}.{schema}.silver_customers`",
            "- Gold: `{catalog}.{schema}.gold_customers`",
        ])
        return "\n".join(lines)

    def _format_recommendation(self, rec: LayeredRecommendation) -> List[str]:
        lines = [
            f"#### {rec.target_column}",
            f"- **Action**: {rec.action}",
            f"- **Parameters**: {rec.parameters}",
            f"- **Rationale**: {rec.rationale}",
            f"- **Source**: {rec.source_notebook}",
            "",
        ]
        return lines
