from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
from customer_retention.core.config.column_config import ColumnType


def _make_findings(source_path="data.csv", target="churned"):
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id", inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95, evidence=["Unique"], universal_metrics={"null_count": 0},
        ),
        "age": ColumnFinding(
            name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=["Numeric"],
            universal_metrics={"null_count": 0}, type_metrics={"skewness": 0.5},
        ),
        "churned": ColumnFinding(
            name="churned", inferred_type=ColumnType.BINARY,
            confidence=0.9, evidence=["Binary"],
            universal_metrics={"null_count": 0, "distinct_count": 2},
        ),
    }
    return ExplorationFindings(
        source_path=source_path, source_format="csv",
        row_count=100, column_count=3, columns=columns,
        target_column=target, identifier_columns=["customer_id"],
    )


def _make_registry():
    reg = RecommendationRegistry()
    reg.init_bronze("data.csv")
    reg.init_silver("customer_id")
    reg.init_gold("churned")
    return reg


def _make_df():
    return pd.DataFrame({"customer_id": [1, 2, 3], "age": [25, 30, 35], "churned": [0, 1, 0]})


@pytest.fixture
def namespace_dir(tmp_path):
    run_dir = tmp_path / "runs" / "test-run"
    datasets_dir = run_dir / "datasets" / "mydata" / "findings"
    datasets_dir.mkdir(parents=True)
    merged_dir = run_dir / "merged"
    merged_dir.mkdir(parents=True)
    landing = run_dir / "datasets" / "mydata" / "landing" / "mydata"
    landing.mkdir(parents=True)
    return run_dir


class TestAnalysisContextConstruction:
    def test_creates_from_findings_path(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        df = _make_df()
        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=df):
                ctx = AnalysisContext("test_notebook.ipynb")

        assert ctx.findings.row_count == 100
        assert ctx.target == "churned"
        assert len(ctx.df) == 3

    def test_creates_with_namespace(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        namespace = MagicMock()
        namespace.project_context_path = tmp_path / "nonexistent.yaml"
        df = _make_df()

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, "mydata")):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=df):
                ctx = AnalysisContext("test_notebook.ipynb")

        assert ctx.namespace is namespace
        assert ctx.dataset_name == "mydata"

    def test_prefer_merged_kwarg_passed(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "merged_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)) as mock_lnf:
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                AnalysisContext("test.ipynb", prefer_merged=True)

        assert mock_lnf.call_args.kwargs.get("prefer_merged") is True


class TestAnalysisContextRegistryLoading:
    def test_loads_existing_registry(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        registry = _make_registry()
        registry.add_bronze_null("age", "median", "test", "nb04")
        recs_path = str(tmp_path / "test_recommendations.yaml")
        registry.save(recs_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert len(ctx.registry.all_recommendations) == 1

    def test_creates_fresh_registry_when_none_exists(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.registry is not None
        assert ctx.registry.bronze is not None
        assert ctx.registry.gold is not None

    def test_namespace_merged_recommendations_loaded(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "merged_findings.yaml")
        findings.save(findings_path)

        registry = _make_registry()
        registry.add_gold_encoding("age", "standard", "test", "nb04")
        merged_recs_path = tmp_path / "recommendations.yaml"
        registry.save(str(merged_recs_path))

        namespace = MagicMock()
        namespace.merged_recommendations_path = merged_recs_path
        namespace.project_context_path = tmp_path / "nonexistent.yaml"

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb", prefer_merged=True)

        assert len(ctx.registry.all_recommendations) >= 1


class TestAnalysisContextDataLoading:
    def test_loads_from_namespace_merged(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "merged_findings.yaml")
        findings.save(findings_path)

        namespace = MagicMock()
        namespace.silver_merged_path = tmp_path / "silver_merged"
        namespace.silver_merged_path.mkdir()
        namespace.merged_recommendations_path = tmp_path / "nonexistent.yaml"
        namespace.project_context_path = tmp_path / "nonexistent.yaml"

        df = _make_df()
        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.get_delta") as mock_delta:
                mock_delta.return_value.read.return_value = df
                ctx = AnalysisContext("test.ipynb", prefer_merged=True)

        assert len(ctx.df) == 3
        assert ctx.data_source == "silver_merged"

    def test_loads_from_active_dataset(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        namespace = MagicMock()
        namespace.project_context_path = tmp_path / "nonexistent.yaml"
        namespace.merged_recommendations_path = tmp_path / "nonexistent.yaml"

        df = _make_df()
        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, "mydata")):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.load_active_dataset",
                       return_value=df):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.data_source == "mydata"

    def test_loads_aggregated_delta(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings(source_path=str(tmp_path / "agg_table"))
        findings_path = str(tmp_path / "test_aggregated_findings.yaml")
        findings.save(findings_path)

        agg_dir = tmp_path / "agg_table"
        agg_dir.mkdir()

        namespace = MagicMock()
        namespace.project_context_path = tmp_path / "nonexistent.yaml"
        namespace.merged_recommendations_path = tmp_path / "nonexistent.yaml"

        df = _make_df()
        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, "mydata")):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.get_delta") as mock_delta:
                mock_delta.return_value.read.return_value = df
                ctx = AnalysisContext("test.ipynb")

        assert "aggregated" in ctx.data_source


class TestAnalysisContextSave:
    def test_save_persists_findings_and_registry(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        ctx.registry.add_bronze_null("age", "median", "missing", "test")
        ctx.findings.metadata["custom_key"] = "custom_value"
        ctx.save()

        reloaded = ExplorationFindings.load(findings_path)
        assert reloaded.metadata["custom_key"] == "custom_value"

        recs_path = findings_path.replace("_findings.yaml", "_recommendations.yaml")
        assert Path(recs_path).exists()

    def test_context_manager_saves_on_exit(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                with AnalysisContext("test.ipynb") as ctx:
                    ctx.registry.add_bronze_null("age", "median", "test", "nb")

        recs_path = findings_path.replace("_findings.yaml", "_recommendations.yaml")
        assert Path(recs_path).exists()

    def test_context_manager_does_not_save_on_exception(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        recs_path = findings_path.replace("_findings.yaml", "_recommendations.yaml")
        with pytest.raises(RuntimeError):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                       return_value=(findings_path, None, None)):
                with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                           return_value=_make_df()):
                    with AnalysisContext("test.ipynb") as ctx:
                        ctx.registry.add_bronze_null("age", "median", "test", "nb")
                        raise RuntimeError("boom")

        assert not Path(recs_path).exists()


class TestAnalysisContextRunStep:
    def test_run_calls_analyze(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        step = MagicMock()
        step.analyze = MagicMock()

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        ctx.run(step)
        step.analyze.assert_called_once_with(ctx)

    def test_run_multiple_steps(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        call_order = []

        class StepA:
            name = "step_a"
            def analyze(self, ctx):
                call_order.append("a")
                ctx.registry.add_bronze_null("age", "median", "from A", "step_a")

        class StepB:
            name = "step_b"
            def analyze(self, ctx):
                call_order.append("b")
                ctx.registry.add_gold_encoding("age", "standard", "from B", "step_b")

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        ctx.run(StepA())
        ctx.run(StepB())

        assert call_order == ["a", "b"]
        assert len(ctx.registry.all_recommendations) == 2


class TestAnalysisContextBuilder:
    def test_builder_returns_recommendation_builder(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext
        from customer_retention.analysis.auto_explorer.recommendation_builder import RecommendationBuilder

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        builder = ctx.builder
        assert isinstance(builder, RecommendationBuilder)
        assert builder.registry is ctx.registry

    def test_builder_is_cached(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.builder is ctx.builder


class TestAnalysisContextRecommendationsPath:
    def test_standard_findings_path(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "mydata_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.recommendations_path == str(tmp_path / "mydata_recommendations.yaml")

    def test_merged_namespace_uses_merged_recommendations_path(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "merged_findings.yaml")
        findings.save(findings_path)

        namespace = MagicMock()
        namespace.merged_recommendations_path = tmp_path / "merged_recs.yaml"
        namespace.project_context_path = tmp_path / "nonexistent.yaml"

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb", prefer_merged=True)

        assert ctx.recommendations_path == str(tmp_path / "merged_recs.yaml")


class TestAnalysisContextTargetResolution:
    def test_resolves_from_findings(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings(target="churned")
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.target == "churned"

    def test_resolves_from_project_context(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings(target=None)
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        namespace = MagicMock()
        namespace.project_context_path = tmp_path / "project_context.yaml"
        namespace.merged_recommendations_path = tmp_path / "nonexistent.yaml"

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, namespace, "mydata")):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                with patch("customer_retention.analysis.auto_explorer.analysis_context.resolve_target_column",
                           return_value="churned"):
                    ctx = AnalysisContext("test.ipynb")

        assert ctx.target == "churned"


class TestAnalysisContextEdgeCases:
    def test_no_target_column_still_works(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings(target=None)
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=_make_df()):
                ctx = AnalysisContext("test.ipynb")

        assert ctx.target is None
        assert ctx.registry.gold is not None

    def test_empty_dataframe(self, tmp_path):
        from customer_retention.analysis.auto_explorer.analysis_context import AnalysisContext

        findings = _make_findings()
        findings_path = str(tmp_path / "test_findings.yaml")
        findings.save(findings_path)

        with patch("customer_retention.analysis.auto_explorer.analysis_context.load_notebook_findings",
                   return_value=(findings_path, None, None)):
            with patch("customer_retention.analysis.auto_explorer.analysis_context.AnalysisContext._load_dataframe",
                       return_value=pd.DataFrame()):
                ctx = AnalysisContext("test.ipynb")

        assert len(ctx.df) == 0
