import pytest
import pandas as pd
import numpy as np

from customer_retention.analysis.recommendations.base import (
    Platform,
    RecommendationResult,
    BaseRecommendation,
    CleaningRecommendation,
    TransformRecommendation,
)
from customer_retention.analysis.recommendations.pipeline import RecommendationPipeline


class AddOneRecommendation(TransformRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "add_one"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["fitted"] = True

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col] + 1
        return RecommendationResult(data=df, columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return f"df[{self.columns}] = df[{self.columns}] + 1"

    def _generate_databricks_code(self) -> str:
        return f"df = df.withColumn('{self.columns[0]}', df['{self.columns[0]}'] + 1)"


class MultiplyTwoRecommendation(TransformRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "multiply_two"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["multiplier"] = 2

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col] * 2
        return RecommendationResult(data=df, columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return f"df[{self.columns}] = df[{self.columns}] * 2"

    def _generate_databricks_code(self) -> str:
        return f"df = df.withColumn('{self.columns[0]}', df['{self.columns[0]}'] * 2)"


class FillNullsRecommendation(CleaningRecommendation):
    @property
    def recommendation_type(self) -> str:
        return "fill_nulls"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["fill_value"] = 0

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return RecommendationResult(data=df, columns_affected=self.columns, rows_before=len(df), rows_after=len(df))

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return f"df[{self.columns}] = df[{self.columns}].fillna(0)"

    def _generate_databricks_code(self) -> str:
        return f"df = df.fillna({{'{self.columns[0]}': 0}})"


class TestRecommendationPipelineInit:
    def test_empty_pipeline(self):
        pipeline = RecommendationPipeline()
        assert pipeline.recommendations == []
        assert pipeline._is_fitted is False

    def test_pipeline_with_recommendations(self):
        recs = [
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="multiply")
        ]
        pipeline = RecommendationPipeline(recommendations=recs)
        assert len(pipeline.recommendations) == 2


class TestRecommendationPipelineAdd:
    def test_add_returns_self(self):
        pipeline = RecommendationPipeline()
        result = pipeline.add(AddOneRecommendation(columns=["a"], rationale="test"))
        assert result is pipeline

    def test_add_appends_recommendation(self):
        pipeline = RecommendationPipeline()
        pipeline.add(AddOneRecommendation(columns=["a"], rationale="add"))
        pipeline.add(MultiplyTwoRecommendation(columns=["a"], rationale="mult"))
        assert len(pipeline.recommendations) == 2

    def test_fluent_chaining(self):
        pipeline = (
            RecommendationPipeline()
            .add(AddOneRecommendation(columns=["a"], rationale="add"))
            .add(MultiplyTwoRecommendation(columns=["a"], rationale="mult"))
        )
        assert len(pipeline.recommendations) == 2


class TestRecommendationPipelineFit:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_fit_returns_self(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        result = pipeline.fit(sample_df)
        assert result is pipeline

    def test_fit_sets_is_fitted(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        assert pipeline._is_fitted is False
        pipeline.fit(sample_df)
        assert pipeline._is_fitted is True

    def test_fit_fits_all_recommendations(self, sample_df):
        rec1 = AddOneRecommendation(columns=["a"], rationale="add")
        rec2 = MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        pipeline = RecommendationPipeline([rec1, rec2])
        pipeline.fit(sample_df)
        assert rec1._is_fitted is True
        assert rec2._is_fitted is True


class TestRecommendationPipelineTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_transform_applies_all_recommendations_in_order(self, sample_df):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ])
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df)
        expected = (sample_df["a"] + 1) * 2
        assert list(result["a"]) == list(expected)

    def test_transform_with_platform(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df, Platform.DATABRICKS)
        assert list(result["a"]) == [2, 3, 4]

    def test_transform_preserves_other_columns(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df)
        assert list(result["b"]) == [4, 5, 6]


class TestRecommendationPipelineFitTransform:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3]})

    def test_fit_transform_combines_fit_and_transform(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        result = pipeline.fit_transform(sample_df)
        assert pipeline._is_fitted is True
        assert list(result["a"]) == [2, 3, 4]


class TestRecommendationPipelineGenerateCode:
    def test_generate_code_local(self):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ])
        code = pipeline.generate_code(Platform.LOCAL)
        assert "+ 1" in code
        assert "* 2" in code

    def test_generate_code_databricks(self):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ])
        code = pipeline.generate_code(Platform.DATABRICKS)
        assert "withColumn" in code

    def test_generate_code_empty_pipeline(self):
        pipeline = RecommendationPipeline()
        code = pipeline.generate_code()
        assert code == ""


class TestRecommendationPipelineChaining:
    @pytest.fixture
    def df_with_nulls(self):
        return pd.DataFrame({"a": [1.0, None, 3.0], "b": [4, 5, 6]})

    def test_cleaning_then_transform(self, df_with_nulls):
        pipeline = RecommendationPipeline([
            FillNullsRecommendation(columns=["a"], rationale="fill"),
            AddOneRecommendation(columns=["a"], rationale="add")
        ])
        result = pipeline.fit_transform(df_with_nulls)
        assert list(result["a"]) == [2.0, 1.0, 4.0]

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add a"),
            MultiplyTwoRecommendation(columns=["b"], rationale="mult b")
        ])
        result = pipeline.fit_transform(df)
        assert list(result["a"]) == [2, 3]
        assert list(result["b"]) == [20, 40]


class TestRecommendationPipelineToDict:
    def test_to_dict(self):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["b"], rationale="mult")
        ])
        d = pipeline.to_dict()
        assert "recommendations" in d
        assert len(d["recommendations"]) == 2
        assert d["is_fitted"] is False

    def test_to_dict_after_fit(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(df)
        d = pipeline.to_dict()
        assert d["is_fitted"] is True


class TestRecommendationPipelineLen:
    def test_len_empty(self):
        assert len(RecommendationPipeline()) == 0

    def test_len_with_recommendations(self):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ])
        assert len(pipeline) == 2


class TestRecommendationPipelineIteration:
    def test_iteration(self):
        recs = [
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ]
        pipeline = RecommendationPipeline(recs)
        collected = list(pipeline)
        assert collected == recs


class MockMLflowAdapter:
    def __init__(self):
        self.logged_params = {}
        self.logged_metrics = {}
        self.log_params_call_count = 0
        self.log_metrics_call_count = 0

    def log_params(self, params):
        self.logged_params.update(params)
        self.log_params_call_count += 1

    def log_metrics(self, metrics):
        self.logged_metrics.update(metrics)
        self.log_metrics_call_count += 1


class TestRecommendationPipelineMLflowIntegration:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_transform_with_mlflow_adapter(self, sample_df):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add")
        ])
        pipeline.fit(sample_df)
        adapter = MockMLflowAdapter()
        result = pipeline.transform(sample_df, mlflow_adapter=adapter)
        assert adapter.log_params_call_count >= 1
        assert list(result["a"]) == [2, 3, 4]

    def test_transform_passes_mlflow_to_each_recommendation(self, sample_df):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["a"], rationale="mult")
        ])
        pipeline.fit(sample_df)
        adapter = MockMLflowAdapter()
        pipeline.transform(sample_df, mlflow_adapter=adapter)
        assert adapter.log_params_call_count == 2

    def test_transform_without_mlflow_works(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        result = pipeline.fit_transform(sample_df)
        assert list(result["a"]) == [2, 3, 4]

    def test_fit_transform_with_mlflow_adapter(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        adapter = MockMLflowAdapter()
        result = pipeline.fit_transform(sample_df, mlflow_adapter=adapter)
        assert adapter.log_params_call_count >= 1
        assert list(result["a"]) == [2, 3, 4]

    def test_mlflow_receives_fit_params(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        adapter = MockMLflowAdapter()
        pipeline.transform(sample_df, mlflow_adapter=adapter)
        assert "fitted" in adapter.logged_params

    def test_mlflow_with_platform_databricks(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        adapter = MockMLflowAdapter()
        result = pipeline.transform(sample_df, Platform.DATABRICKS, mlflow_adapter=adapter)
        assert adapter.log_params_call_count >= 1


class TestRecommendationPipelineFromFindings:
    def test_from_findings_returns_pipeline(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings, ColumnFinding
        from customer_retention.core.config.column_config import ColumnType
        findings = ExplorationFindings(
            source_path="test.csv", source_format="csv", row_count=100, column_count=2,
            columns={
                "age": ColumnFinding(
                    name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.95, evidence=["numeric"],
                    cleaning_recommendations=["impute_median"],
                    transformation_recommendations=["standard_scale"]
                ),
            },
            target_column=None
        )
        pipeline = RecommendationPipeline.from_findings(findings)
        assert isinstance(pipeline, RecommendationPipeline)
        assert len(pipeline.recommendations) == 2

    def test_from_findings_empty_findings(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        findings = ExplorationFindings(
            source_path="test.csv", source_format="csv", row_count=0, column_count=0,
            columns={}, target_column=None
        )
        pipeline = RecommendationPipeline.from_findings(findings)
        assert len(pipeline.recommendations) == 0


class TestRecommendationPipelineFeatureCatalog:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_to_feature_catalog_returns_catalog(self, sample_df):
        from customer_retention.stages.features.feature_definitions import FeatureCatalog
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        catalog = pipeline.to_feature_catalog()
        assert isinstance(catalog, FeatureCatalog)

    def test_to_feature_catalog_contains_features(self, sample_df):
        pipeline = RecommendationPipeline([
            AddOneRecommendation(columns=["a"], rationale="add"),
            MultiplyTwoRecommendation(columns=["b"], rationale="mult")
        ])
        pipeline.fit(sample_df)
        catalog = pipeline.to_feature_catalog()
        assert len(catalog) == 2

    def test_to_feature_catalog_empty_pipeline(self):
        pipeline = RecommendationPipeline()
        catalog = pipeline.to_feature_catalog()
        assert len(catalog) == 0

    def test_to_feature_catalog_feature_names(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        catalog = pipeline.to_feature_catalog()
        names = catalog.list_names()
        assert len(names) == 1
        assert "a_add_one" in names[0]

    def test_to_feature_catalog_feature_has_source_columns(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        catalog = pipeline.to_feature_catalog()
        feature = catalog.get("a_add_one")
        assert feature.source_columns == ["a"]

    def test_to_feature_catalog_feature_has_derivation(self, sample_df):
        pipeline = RecommendationPipeline([AddOneRecommendation(columns=["a"], rationale="add")])
        pipeline.fit(sample_df)
        catalog = pipeline.to_feature_catalog()
        feature = catalog.get("a_add_one")
        assert feature.derivation is not None
        assert len(feature.derivation) > 0
