class PromptTemplates:
    INFER_COLUMN_TYPES = """Given the following column information, infer the semantic type for each column.

Available column types:
- IDENTIFIER: Unique keys, IDs, codes
- TARGET: The prediction target (binary or multiclass)
- BINARY: Two-value columns (yes/no, true/false, 0/1)
- NUMERIC_CONTINUOUS: Continuous numeric values (amounts, measurements)
- NUMERIC_DISCRETE: Discrete numeric values (counts, ratings)
- CATEGORICAL_NOMINAL: Categories without order (colors, types)
- CATEGORICAL_ORDINAL: Categories with order (ratings, levels)
- CATEGORICAL_CYCLICAL: Cyclical categories (days, months)
- DATETIME: Date/time values
- TEXT: Free-form text

For each column, provide:
1. Inferred type
2. Confidence (0-100%)
3. Evidence supporting your inference

{context}

Please analyze each column and provide your type inference."""

    SUGGEST_TARGET_COLUMN = """Based on the data exploration findings below, suggest the most appropriate target column for a machine learning model.

Consider:
- Column names that suggest outcomes (churn, target, label, outcome, class)
- Binary or low-cardinality categorical columns
- Columns that seem to represent what we want to predict

{context}

Provide:
1. Recommended target column
2. Confidence level
3. Rationale for your choice
4. Alternative candidates (if any)"""

    RECOMMEND_FEATURES = """Based on the data exploration findings, recommend feature engineering opportunities.

Consider:
- Datetime columns: temporal features (year, month, day, day of week, days since)
- Numeric columns: binning, scaling, log transforms for skewed data
- Categorical columns: encoding strategies, interaction features
- Cross-column features: ratios, differences, combinations

{context}

For each recommendation, provide:
1. Source column(s)
2. Proposed feature name
3. Feature type and computation
4. Priority (high/medium/low)
5. Implementation hint"""

    GENERATE_PIPELINE_CONFIG = """Generate a production pipeline configuration based on the exploration findings.

The configuration should include:
- Data source specifications
- Schema definitions
- Bronze layer transforms (raw data ingestion)
- Silver layer transforms (cleaning and standardization)
- Gold layer transforms (feature engineering)
- Model configuration
- Quality gates

{context}

User Goal: {user_goal}

Generate a complete pipeline specification in YAML format."""

    EXPLAIN_QUALITY_ISSUES = """Explain the data quality issues found in the exploration and provide remediation recommendations.

For each issue:
1. Describe the problem in business terms
2. Explain the potential impact on model performance
3. Recommend specific remediation steps
4. Prioritize by severity

{context}

Provide a clear, actionable quality improvement plan."""

    GENERATE_DLT_CODE = """Generate Databricks Delta Live Tables (DLT) code based on the pipeline specification.

Requirements:
- Use @dlt.table decorators
- Include expectations for quality checks
- Follow medallion architecture (bronze/silver/gold)
- Include proper schema definitions

{context}

Generate production-ready DLT Python code."""

    @classmethod
    def format_prompt(cls, template: str, **kwargs) -> str:
        return template.format(**kwargs)
