from typing import Optional
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.core.config.column_config import ColumnType


class LLMContextBuilder:
    def __init__(self,
                 include_databricks: bool = False,
                 include_framework_docs: bool = True,
                 max_sample_values: int = 10):
        self.include_databricks = include_databricks
        self.include_framework_docs = include_framework_docs
        self.max_sample_values = max_sample_values

    def build_exploration_context(self, findings: ExplorationFindings) -> str:
        lines = [
            "# Data Exploration Context",
            "",
            "## Dataset Overview",
            f"- **Source:** {findings.source_path}",
            f"- **Format:** {findings.source_format}",
            f"- **Rows:** {findings.row_count:,}",
            f"- **Columns:** {findings.column_count}",
            f"- **Overall Quality Score:** {findings.overall_quality_score:.1f}/100",
            ""
        ]
        if findings.target_column:
            lines.extend([
                "## Target Information",
                f"- **Target Column:** {findings.target_column}",
                f"- **Target Type:** {findings.target_type}",
                ""
            ])
        lines.extend([
            "## Column Details",
            "",
            "| Column | Type | Confidence | Nulls | Notes |",
            "|--------|------|------------|-------|-------|"
        ])
        for name, col in findings.columns.items():
            null_pct = col.universal_metrics.get("null_percentage", 0)
            notes = "; ".join(col.evidence[:2]) if col.evidence else ""
            lines.append(
                f"| {name} | {col.inferred_type.value} | {col.confidence:.0%} | {null_pct:.1f}% | {notes[:50]} |"
            )
        lines.append("")
        lines.extend(self._build_column_details(findings))
        if findings.critical_issues:
            lines.extend([
                "## Critical Issues",
                ""
            ])
            for issue in findings.critical_issues:
                lines.append(f"- {issue}")
            lines.append("")
        if findings.warnings:
            lines.extend([
                "## Warnings",
                ""
            ])
            for warning in findings.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        return "\n".join(lines)

    def _build_column_details(self, findings: ExplorationFindings) -> list:
        lines = ["## Detailed Column Information", ""]
        for name, col in findings.columns.items():
            lines.append(f"### {name}")
            lines.append(f"- **Type:** {col.inferred_type.value}")
            lines.append(f"- **Confidence:** {col.confidence:.0%}")
            if col.universal_metrics:
                metrics = col.universal_metrics
                lines.append(f"- **Null Count:** {metrics.get('null_count', 0)}")
                lines.append(f"- **Distinct Count:** {metrics.get('distinct_count', 'N/A')}")
            if col.type_metrics:
                metrics = col.type_metrics
                if "mean" in metrics:
                    lines.append(f"- **Mean:** {metrics['mean']:.2f}")
                if "std" in metrics:
                    lines.append(f"- **Std:** {metrics['std']:.2f}")
                if "min_value" in metrics:
                    lines.append(f"- **Range:** {metrics['min_value']} to {metrics.get('max_value', 'N/A')}")
                if "top_categories" in metrics:
                    top = metrics["top_categories"][:3]
                    lines.append(f"- **Top Categories:** {top}")
            lines.append("")
        return lines

    def build_configuration_context(self, findings: ExplorationFindings, user_goal: str) -> str:
        lines = [
            "# Pipeline Configuration Context",
            "",
            f"## User Goal",
            f"{user_goal}",
            "",
            self.build_exploration_context(findings),
            "",
            "## Recommendations Summary",
            ""
        ]
        lines.append("### Suggested Transformations")
        for name, col in findings.columns.items():
            if col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
                lines.append(f"- **{name}:** Apply standard scaling")
            elif col.inferred_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
                lines.append(f"- **{name}:** Apply encoding (one-hot or target)")
            elif col.inferred_type == ColumnType.DATETIME:
                lines.append(f"- **{name}:** Extract temporal features")
        lines.append("")
        return "\n".join(lines)

    def build_databricks_context(self, findings: ExplorationFindings) -> str:
        lines = [
            "# Databricks Integration Context",
            "",
            "## Available Databricks Features",
            "",
            "### Delta Lake",
            "- ACID transactions for data reliability",
            "- Schema enforcement and evolution",
            "- Time travel for data versioning",
            "",
            "### Delta Live Tables (DLT)",
            "- Declarative pipeline definitions",
            "- Automatic dependency management",
            "- Built-in expectations for quality",
            "",
            "### Unity Catalog",
            "- Centralized data governance",
            "- Fine-grained access control",
            "- Data lineage tracking",
            "",
            "### Feature Store",
            "- Centralized feature repository",
            "- Point-in-time feature lookups",
            "- Online/offline feature serving",
            "",
            "### Spark Considerations",
            f"- Dataset has {findings.row_count:,} rows",
        ]
        if findings.row_count > 1_000_000:
            lines.append("- Consider partitioning for large dataset")
        lines.extend([
            "- Use DataFrame API for transformations",
            "- Leverage Spark ML for scalable modeling",
            ""
        ])
        return "\n".join(lines)

    def build_framework_docs_context(self) -> str:
        lines = [
            "# Customer Retention Framework Documentation",
            "",
            "## ColumnType Reference",
            "",
            "Available column types in the framework:",
            ""
        ]
        for col_type in ColumnType:
            lines.append(f"- **{col_type.name}:** {col_type.value}")
        lines.extend([
            "",
            "## Key Modules",
            "",
            "### Profiling",
            "- TypeDetector: Automatic type inference",
            "- ColumnProfiler: Statistical profiling per type",
            "- QualityChecks: Data quality validation",
            "",
            "### Transformation",
            "- NumericTransformer: Scaling, log transforms, binning",
            "- CategoricalEncoder: One-hot, target, ordinal encoding",
            "- DatetimeTransformer: Temporal feature extraction",
            "",
            "### Modeling",
            "- BaselineTrainer: Quick baseline models",
            "- CrossValidator: Robust cross-validation",
            "- HyperparameterTuner: Automated tuning",
            "",
            "### Validation",
            "- DataQualityGate: Data quality checks",
            "- LeakageGate: Feature leakage detection",
            "- ModelValidityGate: Model performance validation",
            ""
        ])
        return "\n".join(lines)

    def build_full_context(self, findings: ExplorationFindings, user_goal: str = "") -> str:
        sections = [
            self.build_exploration_context(findings),
            "---",
        ]
        if user_goal:
            sections.append(f"## User Goal\n{user_goal}\n")
            sections.append("---")
        if self.include_framework_docs:
            sections.append(self.build_framework_docs_context())
            sections.append("---")
        if self.include_databricks:
            sections.append(self.build_databricks_context(findings))
        return "\n\n".join(sections)
