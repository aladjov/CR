import pandas as pd
import pytest

from customer_retention.stages.profiling.stats_helpers import calculate_group_retention_stats


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "group_col": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "target": [1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    })


class TestCalculateGroupRetentionStats:
    def test_returns_dataframe(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5)
        assert list(result.columns) == ["group", "retained_count", "count", "retention_rate", "lift"]

    def test_correct_counts(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5)
        a_row = result[result["group"] == "A"].iloc[0]
        assert a_row["count"] == 3
        assert a_row["retained_count"] == 2

    def test_correct_retention_rate(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5)
        b_row = result[result["group"] == "B"].iloc[0]
        assert pytest.approx(b_row["retention_rate"], abs=1e-6) == 1 / 3

    def test_lift_calculation(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5)
        a_row = result[result["group"] == "A"].iloc[0]
        expected_lift = (2 / 3) / 0.5
        assert pytest.approx(a_row["lift"], abs=1e-6) == expected_lift

    def test_lift_zero_overall_rate(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.0)
        assert (result["lift"] == 0).all()

    def test_min_samples_filters(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5, min_samples=4)
        assert len(result) == 1
        assert result.iloc[0]["group"] == "C"

    def test_min_samples_zero_keeps_all(self, sample_df):
        result = calculate_group_retention_stats(sample_df, "group_col", "target", 0.5, min_samples=0)
        assert len(result) == 3

    def test_sort_ascending(self, sample_df):
        result = calculate_group_retention_stats(
            sample_df, "group_col", "target", 0.5, sort_by="group", ascending=True,
        )
        assert list(result["group"]) == ["A", "B", "C"]

    def test_sort_descending(self, sample_df):
        result = calculate_group_retention_stats(
            sample_df, "group_col", "target", 0.5, sort_by="retention_rate", ascending=False,
        )
        assert result.iloc[0]["retention_rate"] >= result.iloc[-1]["retention_rate"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"group_col": pd.Series(dtype=str), "target": pd.Series(dtype=float)})
        result = calculate_group_retention_stats(df, "group_col", "target", 0.5)
        assert len(result) == 0

    def test_single_group(self):
        df = pd.DataFrame({"g": ["X", "X", "X"], "t": [1, 0, 1]})
        result = calculate_group_retention_stats(df, "g", "t", 0.5)
        assert len(result) == 1
        assert result.iloc[0]["group"] == "X"
        assert result.iloc[0]["count"] == 3
