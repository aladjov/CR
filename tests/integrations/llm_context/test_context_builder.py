import pytest

from customer_retention.core.config.column_config import ColumnType
from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.integrations.llm_context.context_builder import LLMContextBuilder
from customer_retention.integrations.llm_context.prompts import PromptTemplates


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
            universal_metrics={"null_count": 0, "distinct_count": 1000}
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric with many values"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0},
            type_metrics={"mean": 45.2, "std": 15.3, "min_value": 18, "max_value": 85}
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["String categorical"],
            universal_metrics={"null_count": 0, "distinct_count": 3},
            type_metrics={"top_categories": [("Monthly", 500), ("Annual", 300), ("Two-Year", 200)]}
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"],
            universal_metrics={"null_count": 0, "distinct_count": 2}
        )
    }
    return ExplorationFindings(
        source_path="data/customers.csv",
        source_format="csv",
        row_count=1000,
        column_count=4,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"],
        overall_quality_score=85.0,
        critical_issues=["5% missing in age column"],
        warnings=["Consider encoding contract_type"]
    )


class TestLLMContextBuilderInit:
    def test_default_init(self):
        builder = LLMContextBuilder()
        assert builder is not None
        assert builder.include_framework_docs

    def test_custom_init(self):
        builder = LLMContextBuilder(
            include_databricks=True,
            include_framework_docs=False,
            max_sample_values=5
        )
        assert builder.include_databricks
        assert not builder.include_framework_docs
        assert builder.max_sample_values == 5


class TestBuildExplorationContext:
    def test_returns_string(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert isinstance(context, str)
        assert len(context) > 100

    def test_includes_column_names(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert "customer_id" in context
        assert "age" in context
        assert "contract_type" in context
        assert "churned" in context

    def test_includes_column_types(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert "identifier" in context.lower()
        assert "numeric" in context.lower()
        assert "categorical" in context.lower()

    def test_includes_quality_info(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert "quality" in context.lower() or "missing" in context.lower()

    def test_includes_target_info(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert "target" in context.lower()
        assert "churned" in context

    def test_includes_row_count(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        assert "1000" in context or "1,000" in context


class TestBuildConfigurationContext:
    def test_returns_string(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_configuration_context(
            sample_findings,
            user_goal="Predict customer churn"
        )

        assert isinstance(context, str)
        assert len(context) > 100

    def test_includes_user_goal(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_configuration_context(
            sample_findings,
            user_goal="Predict customer churn"
        )

        assert "churn" in context.lower()

    def test_includes_recommendations(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_configuration_context(
            sample_findings,
            user_goal="Build ML pipeline"
        )

        assert "recommend" in context.lower() or "suggest" in context.lower() or len(context) > 200


class TestBuildDatabricksContext:
    def test_returns_string(self, sample_findings):
        builder = LLMContextBuilder(include_databricks=True)
        context = builder.build_databricks_context(sample_findings)

        assert isinstance(context, str)
        assert len(context) > 50

    def test_includes_databricks_specific_content(self, sample_findings):
        builder = LLMContextBuilder(include_databricks=True)
        context = builder.build_databricks_context(sample_findings)

        databricks_terms = ["databricks", "spark", "delta", "dlt", "unity", "catalog", "feature"]
        assert any(term in context.lower() for term in databricks_terms)


class TestBuildFrameworkDocsContext:
    def test_returns_string(self):
        builder = LLMContextBuilder(include_framework_docs=True)
        context = builder.build_framework_docs_context()

        assert isinstance(context, str)
        assert len(context) > 100

    def test_includes_column_types(self):
        builder = LLMContextBuilder(include_framework_docs=True)
        context = builder.build_framework_docs_context()

        assert "ColumnType" in context or "column" in context.lower()


class TestBuildFullContext:
    def test_combines_all_contexts(self, sample_findings):
        builder = LLMContextBuilder(
            include_databricks=True,
            include_framework_docs=True
        )
        context = builder.build_full_context(
            sample_findings,
            user_goal="Predict churn"
        )

        assert isinstance(context, str)
        assert len(context) > 500

    def test_has_clear_sections(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_full_context(
            sample_findings,
            user_goal="Predict churn"
        )

        assert "##" in context or "---" in context or "\n\n" in context


class TestPromptTemplates:
    def test_infer_column_types_template(self):
        template = PromptTemplates.INFER_COLUMN_TYPES
        assert isinstance(template, str)
        assert len(template) > 50
        assert "column" in template.lower()

    def test_suggest_target_template(self):
        template = PromptTemplates.SUGGEST_TARGET_COLUMN
        assert isinstance(template, str)
        assert "target" in template.lower()

    def test_recommend_features_template(self):
        template = PromptTemplates.RECOMMEND_FEATURES
        assert isinstance(template, str)
        assert "feature" in template.lower()

    def test_generate_pipeline_config_template(self):
        template = PromptTemplates.GENERATE_PIPELINE_CONFIG
        assert isinstance(template, str)
        assert "pipeline" in template.lower() or "config" in template.lower()

    def test_explain_quality_issues_template(self):
        template = PromptTemplates.EXPLAIN_QUALITY_ISSUES
        assert isinstance(template, str)
        assert "quality" in template.lower() or "issue" in template.lower()

    def test_format_prompt_with_context(self, sample_findings):
        builder = LLMContextBuilder()
        context = builder.build_exploration_context(sample_findings)

        formatted = PromptTemplates.format_prompt(
            PromptTemplates.SUGGEST_TARGET_COLUMN,
            context=context
        )

        assert isinstance(formatted, str)
        assert "customer_id" in formatted or "context" in formatted.lower()

    def test_all_templates_are_strings(self):
        templates = [
            PromptTemplates.INFER_COLUMN_TYPES,
            PromptTemplates.SUGGEST_TARGET_COLUMN,
            PromptTemplates.RECOMMEND_FEATURES,
            PromptTemplates.GENERATE_PIPELINE_CONFIG,
            PromptTemplates.EXPLAIN_QUALITY_ISSUES,
        ]
        for template in templates:
            assert isinstance(template, str)
            assert len(template) > 20
