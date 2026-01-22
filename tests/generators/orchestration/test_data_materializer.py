"""Tests for DataMaterializer that applies transforms and persists results."""
import os
import tempfile

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
            assert output_path.endswith(".parquet")
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
