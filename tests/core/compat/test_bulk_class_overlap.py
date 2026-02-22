import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_class_overlap
from customer_retention.core.utils.leakage import calculate_class_overlap


class TestBulkClassOverlapMatchesPerColumn:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        target = np.array([0] * 100 + [1] * 100)
        return pd.DataFrame({
            "normal": np.random.randn(n),
            "spread": np.concatenate([np.random.uniform(0, 50, 100), np.random.uniform(30, 80, 100)]),
            "separated": np.concatenate([np.random.uniform(0, 10, 100), np.random.uniform(50, 60, 100)]),
        }), pd.Series(target)

    def test_matches_per_column_results(self, sample_df):
        df, target = sample_df
        columns = ["normal", "spread", "separated"]
        bulk = bulk_class_overlap(df, columns, target)
        for col in columns:
            expected = calculate_class_overlap(df[col], target)
            assert abs(bulk[col] - expected) < 1e-10, f"{col}: {bulk[col]} != {expected}"

    def test_returns_all_requested_columns(self, sample_df):
        df, target = sample_df
        columns = ["normal", "spread", "separated"]
        bulk = bulk_class_overlap(df, columns, target)
        assert set(bulk.keys()) == set(columns)


class TestBulkClassOverlapEdgeCases:
    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        assert bulk_class_overlap(df, [], pd.Series([0, 1])) == {}

    def test_single_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["x"], target)
        assert "x" in result
        expected = calculate_class_overlap(df["x"], target)
        assert abs(result["x"] - expected) < 1e-10

    def test_perfect_separation_returns_zero(self):
        df = pd.DataFrame({"sep": [0.0, 0.0, 100.0, 100.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["sep"], target)
        assert result["sep"] == 0.0

    def test_full_overlap_high_percentage(self):
        df = pd.DataFrame({"same": [1.0, 2.0, 1.0, 2.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["same"], target)
        assert result["same"] == 100.0

    def test_constant_column_returns_100(self):
        df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["const"], target)
        assert result["const"] == 100.0

    def test_all_nan_one_class_returns_100(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, 1.0, 2.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["x"], target)
        assert result["x"] == 100.0

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        target = pd.Series([0, 1])
        result = bulk_class_overlap(df, ["a", "nonexistent"], target)
        assert "a" in result
        assert "nonexistent" not in result

    def test_multiple_columns_mixed(self):
        df = pd.DataFrame({
            "separated": [0.0, 0.0, 100.0, 100.0],
            "overlapping": [1.0, 2.0, 1.5, 2.5],
            "constant": [7.0, 7.0, 7.0, 7.0],
        })
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["separated", "overlapping", "constant"], target)
        assert result["separated"] == 0.0
        assert result["constant"] == 100.0
        assert 0 < result["overlapping"] <= 100

    def test_nan_in_both_classes(self):
        df = pd.DataFrame({"x": [np.nan, 1.0, np.nan, 2.0]})
        target = pd.Series([0, 0, 1, 1])
        result = bulk_class_overlap(df, ["x"], target)
        expected = calculate_class_overlap(df["x"], target)
        assert abs(result["x"] - expected) < 1e-10
