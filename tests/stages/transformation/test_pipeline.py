import pytest
import pandas as pd
import numpy as np
from customer_retention.core.config import ColumnType
from customer_retention.stages.transformation import (
    TransformationPipeline, TransformationManifest, PipelineResult
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "custid": ["C001", "C002", "C003", "C004", "C005"],
        "age": [25.0, 30.0, None, 45.0, 35.0],
        "income": [50000.0, 75000.0, 60000.0, 100000.0, 80000.0],
        "city": ["NYC", "LA", "NYC", "SF", "LA"],
        "is_active": [1, 0, 1, 1, 0],
        "created": pd.to_datetime(["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-20", "2024-05-05"]),
        "target": [1, 0, 1, 1, 0]
    })


@pytest.fixture
def column_types():
    return {
        "custid": ColumnType.IDENTIFIER,
        "age": ColumnType.NUMERIC_CONTINUOUS,
        "income": ColumnType.NUMERIC_CONTINUOUS,
        "city": ColumnType.CATEGORICAL_NOMINAL,
        "is_active": ColumnType.BINARY,
        "created": ColumnType.DATETIME,
        "target": ColumnType.TARGET
    }


class TestPipelineExecution:
    def test_pipeline_executes_without_error(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.df is not None
        assert len(result.df) == 5

    def test_pipeline_handles_missing_values(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.df.isna().sum().sum() == 0

    def test_pipeline_drops_identifier_column(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert "custid" not in result.df.columns

    def test_pipeline_preserves_target(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert "target" in result.df.columns
        pd.testing.assert_series_equal(
            result.df["target"], sample_dataframe["target"], check_names=False
        )


class TestPipelineExecutionOrder:
    def test_execution_order_correct(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        expected_order = [
            "drop_columns", "handle_missing", "treat_outliers",
            "transform_datetime", "transform_numeric",
            "encode_categorical", "standardize_binary", "validate"
        ]
        assert result.manifest.execution_order == expected_order


class TestTransformationManifest:
    def test_manifest_captures_missing_handling(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.manifest.missing_value_handling is not None
        assert "age" in result.manifest.missing_value_handling

    def test_manifest_captures_column_mapping(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.manifest.column_mapping is not None
        assert len(result.manifest.column_mapping) > 0

    def test_manifest_has_metadata(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.manifest.timestamp is not None
        assert result.manifest.input_rows == 5
        assert result.manifest.output_rows == 5


class TestFitTransformSeparation:
    def test_fit_then_transform_on_new_data(self, sample_dataframe, column_types):
        train = sample_dataframe.iloc[:3]
        test = sample_dataframe.iloc[3:]

        pipeline = TransformationPipeline(column_types=column_types)
        pipeline.fit(train)
        result = pipeline.transform(test)

        assert result.df is not None
        assert len(result.df) == 2

    def test_transform_without_fit_raises_error(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)

        with pytest.raises(ValueError, match="not fitted"):
            pipeline.transform(sample_dataframe)


class TestPipelineConfiguration:
    def test_auto_from_profile(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(
            column_types=column_types, auto_from_profile=True
        )
        result = pipeline.fit_transform(sample_dataframe)

        assert result.df is not None

    def test_custom_column_configs(self, sample_dataframe, column_types):
        column_configs = {
            "age": {"missing_strategy": "mean"}
        }
        pipeline = TransformationPipeline(
            column_types=column_types, column_configs=column_configs
        )
        result = pipeline.fit_transform(sample_dataframe)

        assert result.df is not None


class TestQualityGate:
    def test_validation_passes_on_clean_data(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        assert result.validation_passed is True

    def test_no_nulls_after_transform(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        non_target_cols = [c for c in result.df.columns if c != "target"]
        assert result.df[non_target_cols].isna().sum().sum() == 0

    def test_no_infinities_after_transform(self, sample_dataframe, column_types):
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(sample_dataframe)

        numeric_cols = result.df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result.df[numeric_cols].values).any()


class TestEdgeCases:
    def test_handles_all_null_column(self):
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0],
            "col2": [None, None, None],
            "target": [1, 0, 1]
        })
        column_types = {
            "col1": ColumnType.NUMERIC_CONTINUOUS,
            "col2": ColumnType.NUMERIC_CONTINUOUS,
            "target": ColumnType.TARGET
        }
        pipeline = TransformationPipeline(column_types=column_types)
        result = pipeline.fit_transform(df)

        assert "col2" not in result.df.columns or result.df["col2"].isna().sum() == 0

    def test_handles_constant_column(self):
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0],
            "col2": [5.0, 5.0, 5.0],
            "target": [1, 0, 1]
        })
        column_types = {
            "col1": ColumnType.NUMERIC_CONTINUOUS,
            "col2": ColumnType.NUMERIC_CONTINUOUS,
            "target": ColumnType.TARGET
        }
        pipeline = TransformationPipeline(
            column_types=column_types, drop_constant_columns=True
        )
        result = pipeline.fit_transform(df)

        assert result.df is not None
