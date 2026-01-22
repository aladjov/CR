import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalyzer, CutoffAnalysis


class TestCutoffAnalyzer:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        np.random.seed(42)
        return pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(len(dates))],
            "feature_timestamp": dates,
            "label_timestamp": dates + pd.Timedelta(days=180),
            "value": np.random.randn(len(dates)),
        })

    @pytest.fixture
    def analyzer(self):
        return CutoffAnalyzer()


class TestAnalyzeDistribution(TestCutoffAnalyzer):
    def test_returns_cutoff_analysis(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert isinstance(result, CutoffAnalysis)
        assert result.timestamp_column == "feature_timestamp"
        assert result.total_rows == len(sample_df)

    def test_bins_cover_date_range(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert len(result.bins) > 0
        assert result.bins[0] >= sample_df["feature_timestamp"].min()
        assert result.bins[-1] <= sample_df["feature_timestamp"].max()

    def test_cumulative_percentages_increase(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        for i in range(1, len(result.train_percentages)):
            assert result.train_percentages[i] >= result.train_percentages[i - 1]

    def test_train_plus_score_equals_100(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        for train_pct, score_pct in zip(result.train_percentages, result.score_percentages):
            assert abs(train_pct + score_pct - 100.0) < 0.1

    def test_custom_bin_count(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp", n_bins=10)

        assert len(result.bins) == 10

    def test_row_counts_per_bin(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert len(result.bin_counts) == len(result.bins)
        assert sum(result.bin_counts) == len(sample_df)


class TestSuggestCutoff(TestCutoffAnalyzer):
    def test_suggest_cutoff_default_split(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        suggested = result.suggest_cutoff()  # Default is now 0.9

        assert isinstance(suggested, datetime)
        train_pct = result.get_train_percentage(suggested)
        assert 85 <= train_pct <= 95

    def test_suggest_cutoff_70_30_split(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        suggested = result.suggest_cutoff(train_ratio=0.7)

        train_pct = result.get_train_percentage(suggested)
        assert 65 <= train_pct <= 75

    def test_get_train_percentage_for_specific_date(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        mid_date = datetime(2024, 6, 1)

        train_pct = result.get_train_percentage(mid_date)
        assert 0 < train_pct < 100


class TestGetSplitAtDate(TestCutoffAnalyzer):
    def test_get_split_returns_counts(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.get_split_at_date(cutoff)

        assert "train_count" in split
        assert "score_count" in split
        assert "train_pct" in split
        assert "score_pct" in split
        assert split["train_count"] + split["score_count"] == result.total_rows


class TestToDataFrame(TestCutoffAnalyzer):
    def test_to_dataframe_structure(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        df = result.to_dataframe()

        assert "date" in df.columns
        assert "bin_count" in df.columns
        assert "train_pct" in df.columns
        assert "score_pct" in df.columns
        assert "cumulative_count" in df.columns

    def test_to_dataframe_row_count(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp", n_bins=20)
        df = result.to_dataframe()

        assert len(df) == 20


class TestPercentageMilestones(TestCutoffAnalyzer):
    def test_milestones_at_5_percent_intervals(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        milestones = result.get_percentage_milestones(step=5)

        assert len(milestones) > 0
        for m in milestones:
            assert "date" in m
            assert "train_pct" in m
            assert "score_pct" in m

    def test_milestones_at_10_percent_intervals(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        milestones = result.get_percentage_milestones(step=10)

        pcts = [m["train_pct"] for m in milestones]
        assert all(p >= 10 for p in pcts)

    def test_milestones_dates_are_ordered(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        milestones = result.get_percentage_milestones(step=5)

        dates = [m["date"] for m in milestones]
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i - 1]


class TestEdgeCases(TestCutoffAnalyzer):
    def test_handles_single_date(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01"] * 10),
            "value": range(10),
        })

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")
        assert result.total_rows == 10

    def test_handles_missing_timestamps(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01", None, "2024-06-01", None, "2024-12-01"]),
            "value": range(5),
        })

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")
        assert result.total_rows == 3

    def test_auto_detect_timestamp_column(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df)

        assert result.timestamp_column in ["feature_timestamp", "label_timestamp"]
