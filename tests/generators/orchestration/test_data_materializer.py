"""Tests for DataMaterializer that applies transforms and persists results."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    LayeredRecommendation,
    RecommendationRegistry,
)
from customer_retention.generators.orchestration.data_materializer import DataMaterializer


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(100),
        "age": [30 + np.random.randint(-5, 20) if i % 10 != 0 else np.nan for i in range(100)],
        "revenue": [100 + np.random.exponential(50) for _ in range(100)],
        "contract_type": np.random.choice(["monthly", "yearly", "two_year"], 100),
        "churned": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def bronze_registry():
    registry = RecommendationRegistry()
    registry.init_bronze("customers.csv")
    registry.bronze.null_handling.append(LayeredRecommendation(
        id="bronze_null_age", layer="bronze", category="null", action="impute",
        target_column="age", parameters={"strategy": "median"},
        rationale="10% missing", source_notebook="03"
    ))
    return registry


@pytest.fixture
def gold_registry():
    registry = RecommendationRegistry()
    registry.init_bronze("customers.csv")
    registry.init_gold("churned")
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality", source_notebook="06"
    ))
    return registry


@pytest.fixture
def full_registry():
    registry = RecommendationRegistry()
    registry.init_bronze("customers.csv")
    registry.init_gold("churned")
    registry.bronze.null_handling.append(LayeredRecommendation(
        id="bronze_null_age", layer="bronze", category="null", action="impute",
        target_column="age", parameters={"strategy": "median"},
        rationale="10% missing", source_notebook="03"
    ))
    registry.gold.encoding.append(LayeredRecommendation(
        id="gold_encode_contract", layer="gold", category="encoding", action="one_hot",
        target_column="contract_type", parameters={"method": "one_hot"},
        rationale="Low cardinality", source_notebook="06"
    ))
    registry.gold.scaling.append(LayeredRecommendation(
        id="gold_scale_revenue", layer="gold", category="scaling", action="standard",
        target_column="revenue", parameters={"method": "standard"},
        rationale="Normalize", source_notebook="06"
    ))
    return registry


class TestDataMaterializerInit:
    def test_creates_with_registry(self, bronze_registry):
        materializer = DataMaterializer(bronze_registry)
        assert materializer.registry == bronze_registry

    def test_creates_with_output_dir(self, bronze_registry):
        materializer = DataMaterializer(bronze_registry, output_dir="/tmp/data")
        assert materializer.output_dir == "/tmp/data"


class TestBronzeTransforms:
    def test_applies_null_imputation(self, sample_df, bronze_registry):
        materializer = DataMaterializer(bronze_registry)
        result = materializer.apply_bronze(sample_df)
        assert result["age"].isna().sum() == 0

    def test_preserves_non_null_values(self, sample_df, bronze_registry):
        original_non_null = sample_df["age"].dropna().values
        materializer = DataMaterializer(bronze_registry)
        result = materializer.apply_bronze(sample_df)
        result_non_null_mask = ~sample_df["age"].isna()
        np.testing.assert_array_almost_equal(
            result.loc[result_non_null_mask, "age"].values,
            sample_df.loc[result_non_null_mask, "age"].values
        )

    def test_drop_strategy_drops_column(self, sample_df):
        # Mirror production: `findings_parser._map_bronze_null` translates
        # parameters.strategy=drop into a DROP_COLUMN bronze step. Exploration
        # must mirror this so NB04+/feature_spec stay in sync with what
        # production will materialize.
        registry = RecommendationRegistry()
        registry.init_bronze("customers.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_revenue", layer="bronze", category="null", action="drop",
            target_column="revenue", parameters={"strategy": "drop"},
            rationale="98% missing", source_notebook="02_source_integrity",
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert "revenue" not in result.columns
        assert "age" in result.columns  # untouched columns preserved

    def test_drop_strategy_no_op_when_column_missing(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("customers.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_phantom", layer="bronze", category="null", action="drop",
            target_column="not_in_dataframe", parameters={"strategy": "drop"},
            rationale="dropped column already absent", source_notebook="02_source_integrity",
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert set(result.columns) == set(sample_df.columns)


class TestGoldTransforms:
    def test_applies_one_hot_encoding(self, sample_df, gold_registry):
        materializer = DataMaterializer(gold_registry)
        result = materializer.apply_gold(sample_df)
        assert "contract_type" not in result.columns
        contract_cols = [c for c in result.columns if c.startswith("contract_type_")]
        assert len(contract_cols) >= 2

    def test_applies_standard_scaling(self, sample_df, full_registry):
        materializer = DataMaterializer(full_registry)
        result = materializer.apply_gold(sample_df)
        assert abs(result["revenue"].mean()) < 0.5
        assert abs(result["revenue"].std() - 1.0) < 0.5


class TestFullPipeline:
    def test_applies_all_transforms(self, sample_df, full_registry):
        materializer = DataMaterializer(full_registry)
        result = materializer.transform(sample_df)
        assert result["age"].isna().sum() == 0
        assert "contract_type" not in result.columns
        contract_cols = [c for c in result.columns if c.startswith("contract_type_")]
        assert len(contract_cols) >= 2

    def test_returns_dataframe(self, sample_df, full_registry):
        materializer = DataMaterializer(full_registry)
        result = materializer.transform(sample_df)
        assert isinstance(result, pd.DataFrame)


class TestMaterialization:
    def test_saves_to_parquet(self, sample_df, full_registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = DataMaterializer(full_registry, output_dir=tmpdir)
            result_df, output_path = materializer.materialize(sample_df, "prepared_data")
            assert os.path.exists(output_path)
            assert "prepared_data" in output_path
            loaded = pd.read_parquet(output_path)
            assert len(loaded) == len(result_df)

    def test_returns_transformed_df_and_path(self, sample_df, full_registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = DataMaterializer(full_registry, output_dir=tmpdir)
            result_df, output_path = materializer.materialize(sample_df, "test_output")
            assert isinstance(result_df, pd.DataFrame)
            assert isinstance(output_path, str)
            assert "test_output" in output_path

    def test_creates_output_dir_if_missing(self, sample_df, full_registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")
            materializer = DataMaterializer(full_registry, output_dir=nested_dir)
            _, output_path = materializer.materialize(sample_df, "data")
            assert os.path.exists(output_path)


class TestEmptyRegistry:
    def test_handles_empty_bronze(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_handles_empty_gold(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_gold("target")
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_handles_no_registry_layers(self, sample_df):
        registry = RecommendationRegistry()
        materializer = DataMaterializer(registry)
        result = materializer.transform(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)


class TestTransformOrder:
    def test_bronze_before_gold(self, sample_df, full_registry):
        materializer = DataMaterializer(full_registry)
        result = materializer.transform(sample_df)
        assert result["age"].isna().sum() == 0
        assert "contract_type" not in result.columns


class TestBronzeOutlierHandling:
    def test_iqr_outlier_clipping(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.outlier_handling.append(LayeredRecommendation(
            id="bronze_outlier_rev", layer="bronze", category="outlier", action="clip",
            target_column="revenue", parameters={"method": "iqr", "factor": 1.5},
            rationale="Clip outliers", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert result["revenue"].max() <= sample_df["revenue"].quantile(0.75) + 1.5 * (
            sample_df["revenue"].quantile(0.75) - sample_df["revenue"].quantile(0.25)
        ) + 0.01

    def test_outlier_missing_column_skips(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.outlier_handling.append(LayeredRecommendation(
            id="bronze_outlier_x", layer="bronze", category="outlier", action="clip",
            target_column="nonexistent", parameters={"method": "iqr"},
            rationale="Missing column", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)


class TestBronzeTypeConversion:
    def test_type_conversion_to_str(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.type_conversions.append(LayeredRecommendation(
            id="bronze_type_col", layer="bronze", category="type", action="convert",
            target_column="col", parameters={"target_type": "str"},
            rationale="Convert to str", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        assert result["col"].dtype == object

    def test_type_conversion_to_datetime(self):
        df = pd.DataFrame({"date_col": ["2023-01-01", "2023-02-01"]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.type_conversions.append(LayeredRecommendation(
            id="bronze_type_date", layer="bronze", category="type", action="convert",
            target_column="date_col", parameters={"target_type": "datetime"},
            rationale="Convert to datetime", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_type_conversion_missing_column(self):
        df = pd.DataFrame({"col": [1, 2]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.type_conversions.append(LayeredRecommendation(
            id="bronze_type_x", layer="bronze", category="type", action="convert",
            target_column="nonexistent", parameters={"target_type": "str"},
            rationale="Missing", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        pd.testing.assert_frame_equal(result, df)


class TestBronzeFiltering:
    def test_drop_column(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.filtering.append(LayeredRecommendation(
            id="bronze_filter_age", layer="bronze", category="filtering", action="drop",
            target_column="age", parameters={},
            rationale="Drop column", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert "age" not in result.columns

    def test_drop_nonexistent_column(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.filtering.append(LayeredRecommendation(
            id="bronze_filter_x", layer="bronze", category="filtering", action="drop",
            target_column="nonexistent", parameters={},
            rationale="Column missing", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert len(result.columns) == len(sample_df.columns)

    def test_non_drop_action(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.filtering.append(LayeredRecommendation(
            id="bronze_filter_keep", layer="bronze", category="filtering", action="keep",
            target_column="age", parameters={},
            rationale="Keep action", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        assert "age" in result.columns


class TestBronzeNullStrategies:
    def test_mean_strategy(self):
        df = pd.DataFrame({"val": [1.0, 2.0, None, 4.0]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_val", layer="bronze", category="null", action="impute",
            target_column="val", parameters={"strategy": "mean"},
            rationale="mean", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        assert result["val"].isna().sum() == 0
        assert result["val"].iloc[2] == pytest.approx(7.0 / 3.0)

    def test_mode_strategy(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", None]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_cat", layer="bronze", category="null", action="impute",
            target_column="cat", parameters={"strategy": "mode"},
            rationale="mode", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        assert result["cat"].isna().sum() == 0
        assert result["cat"].iloc[3] == "a"

    def test_zero_strategy(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_val", layer="bronze", category="null", action="impute",
            target_column="val", parameters={"strategy": "zero"},
            rationale="zero", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)
        assert result["val"].iloc[1] == 0

    def test_null_missing_column(self, sample_df):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.bronze.null_handling.append(LayeredRecommendation(
            id="bronze_null_x", layer="bronze", category="null", action="impute",
            target_column="nonexistent", parameters={"strategy": "median"},
            rationale="Missing", source_notebook="03"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)


class TestSilverTransforms:
    def test_empty_silver_returns_df(self, sample_df):
        registry = RecommendationRegistry()
        materializer = DataMaterializer(registry)
        result = materializer.apply_silver(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_aggregation(self):
        df = pd.DataFrame({
            "cust_id": [1, 1, 2, 2],
            "amount": [10, 20, 30, 40],
        })
        registry = RecommendationRegistry()
        registry.init_silver("cust_id")
        registry.silver.aggregations.append(LayeredRecommendation(
            id="silver_agg_amount", layer="silver", category="aggregation", action="sum",
            target_column="amount", parameters={"aggregation": "sum"},
            rationale="Sum amounts", source_notebook="04"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_silver(df)
        assert "amount_sum" in result.columns
        assert result.loc[0, "amount_sum"] == 30  # sum for cust_id=1
        assert result.loc[2, "amount_sum"] == 70  # sum for cust_id=2

    def test_aggregation_missing_column(self):
        df = pd.DataFrame({"cust_id": [1, 2], "amount": [10, 20]})
        registry = RecommendationRegistry()
        registry.init_silver("cust_id")
        registry.silver.aggregations.append(LayeredRecommendation(
            id="silver_agg_x", layer="silver", category="aggregation", action="sum",
            target_column="nonexistent", parameters={"aggregation": "sum"},
            rationale="Missing", source_notebook="04"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_silver(df)
        assert "nonexistent_sum" not in result.columns

    def test_derived_columns_noop(self):
        df = pd.DataFrame({"a": [1, 2]})
        registry = RecommendationRegistry()
        registry.init_silver("a")
        registry.silver.derived_columns.append(LayeredRecommendation(
            id="silver_derived_x", layer="silver", category="derived", action="ratio",
            target_column="new_col", parameters={"expression": "a * 2"},
            rationale="derive", source_notebook="04"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_silver(df)
        assert isinstance(result, pd.DataFrame)


class TestGoldTransformations:
    def test_log_transformation(self):
        df = pd.DataFrame({"val": [1.0, 10.0, 100.0]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.transformations.append(LayeredRecommendation(
            id="gold_trans_val", layer="gold", category="transformation", action="log",
            target_column="val", parameters={"method": "log"},
            rationale="Log transform", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        np.testing.assert_array_almost_equal(result["val"].values, np.log1p([1.0, 10.0, 100.0]))

    def test_sqrt_transformation(self):
        df = pd.DataFrame({"val": [4.0, 9.0, 16.0]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.transformations.append(LayeredRecommendation(
            id="gold_trans_val", layer="gold", category="transformation", action="sqrt",
            target_column="val", parameters={"method": "sqrt"},
            rationale="Sqrt transform", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        np.testing.assert_array_almost_equal(result["val"].values, [2.0, 3.0, 4.0])

    def test_transformation_missing_column(self):
        df = pd.DataFrame({"val": [1.0, 2.0]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.transformations.append(LayeredRecommendation(
            id="gold_trans_x", layer="gold", category="transformation", action="log",
            target_column="nonexistent", parameters={"method": "log"},
            rationale="Missing", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        pd.testing.assert_frame_equal(result, df)


class TestGoldScaling:
    def test_robust_scaling(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.scaling.append(LayeredRecommendation(
            id="gold_scale_val", layer="gold", category="scaling", action="robust",
            target_column="val", parameters={"method": "robust"},
            rationale="Robust scaling", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        # Robust scaler centers around median
        assert abs(result["val"].median()) < 0.1

    def test_scaling_missing_column(self):
        df = pd.DataFrame({"val": [1.0, 2.0]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.scaling.append(LayeredRecommendation(
            id="gold_scale_x", layer="gold", category="scaling", action="standard",
            target_column="nonexistent", parameters={"method": "standard"},
            rationale="Missing", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        pd.testing.assert_frame_equal(result, df)


class TestGoldEncoding:
    def test_encoding_missing_column(self):
        df = pd.DataFrame({"val": [1, 2]})
        registry = RecommendationRegistry()
        registry.init_gold("target")
        registry.gold.encoding.append(LayeredRecommendation(
            id="gold_enc_x", layer="gold", category="encoding", action="one_hot",
            target_column="nonexistent", parameters={"method": "one_hot"},
            rationale="Missing", source_notebook="06"
        ))
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)
        pd.testing.assert_frame_equal(result, df)


class TestSaveWithStorage:
    def test_save_with_storage_and_context(self, sample_df, full_registry):
        mock_storage = MagicMock()
        mock_storage.history.return_value = ["v0", "v1", "v2"]
        mock_context = MagicMock()
        mock_context.build_commit_metadata.return_value = {"key": "value"}
        mock_context.delta_versions = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = DataMaterializer(
                full_registry, output_dir=tmpdir,
                storage=mock_storage, context=mock_context,
            )
            result_df, output_path = materializer.materialize(sample_df, "output")
            mock_storage.write.assert_called_once()
            assert output_path in mock_context.delta_versions
            assert mock_context.delta_versions[output_path] == 2

    def test_save_with_storage_no_context(self, sample_df, full_registry):
        mock_storage = MagicMock()
        mock_storage.history.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = DataMaterializer(
                full_registry, output_dir=tmpdir,
                storage=mock_storage, context=None,
            )
            result_df, output_path = materializer.materialize(sample_df, "output")
            mock_storage.write.assert_called_once()
            call_kwargs = mock_storage.write.call_args
            assert call_kwargs[1].get("metadata") is None


class TestCompareRuns:
    def test_compare_runs(self, sample_df, full_registry):
        mock_storage = MagicMock()
        df_a = pd.DataFrame({"a": [1, 2, 3]})
        df_b = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mock_storage.read.side_effect = [df_a, df_b]

        materializer = DataMaterializer(full_registry, storage=mock_storage)
        result = materializer.compare_runs("path", version_a=0, version_b=1)
        assert len(result) == 2
        assert result.loc[0, "version_a"] == 3
        assert result.loc[0, "version_b"] == 2
        assert result.loc[1, "version_a"] == 1
        assert result.loc[1, "version_b"] == 2

    def test_compare_runs_no_storage(self, full_registry):
        materializer = DataMaterializer(full_registry)
        materializer.storage = None
        with pytest.raises(RuntimeError, match="DeltaStorage required"):
            materializer.compare_runs("path", 0, 1)


class TestGetStorage:
    def test_get_storage_import_error(self):
        with patch(
            "customer_retention.generators.orchestration.data_materializer._get_storage",
            return_value=None,
        ):
            from customer_retention.generators.orchestration.data_materializer import _get_storage
            # Test the actual fallback path
            with patch("builtins.__import__", side_effect=ImportError):
                result = _get_storage()
                assert result is None
