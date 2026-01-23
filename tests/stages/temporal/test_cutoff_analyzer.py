from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalysis, CutoffAnalyzer, SplitResult


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


class TestCoverageReporting(TestCutoffAnalyzer):
    def test_reports_source_and_covered_rows(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert result.source_rows == len(sample_df)
        assert result.covered_rows == len(sample_df)
        assert result.coverage_ratio == 1.0

    def test_coverage_ratio_with_nulls(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(
                ["2024-01-01", None, "2024-06-01", None, "2024-12-01"]
            ),
            "value": range(5),
        })

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")

        assert result.source_rows == 5
        assert result.covered_rows == 3
        assert abs(result.coverage_ratio - 0.6) < 0.01

    def test_total_rows_equals_covered_rows(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        # Backward compat: total_rows still reflects non-null rows used in analysis
        assert result.total_rows == result.covered_rows


class TestAnalyzeWithDerivedSeries(TestCutoffAnalyzer):
    def test_analyze_with_timestamp_series(self, analyzer, sample_df):
        series = sample_df["feature_timestamp"].copy()
        series.name = "my_series"

        result = analyzer.analyze(sample_df, timestamp_series=series)

        assert result.timestamp_column == "my_series"
        assert result.total_rows == len(sample_df)

    def test_timestamp_series_overrides_column(self, analyzer, sample_df):
        series = sample_df["label_timestamp"].copy()
        series.name = "override_ts"

        result = analyzer.analyze(
            sample_df, timestamp_column="feature_timestamp", timestamp_series=series
        )

        assert result.timestamp_column == "override_ts"
        assert result.date_range[0] == series.min()

    def test_timestamp_series_with_nulls_reports_coverage(self, analyzer):
        df = pd.DataFrame({"value": range(10)})
        series = pd.Series(
            pd.to_datetime(["2024-01-01", None, "2024-03-01", None, "2024-05-01"]
                           + [None] * 5)
        )
        series.name = "sparse_ts"

        result = analyzer.analyze(df, timestamp_series=series)

        assert result.source_rows == 10
        assert result.covered_rows == 3
        assert abs(result.coverage_ratio - 0.3) < 0.01

    def test_low_coverage_warning_threshold(self, analyzer):
        df = pd.DataFrame({"value": range(10)})
        series = pd.Series(
            pd.to_datetime(["2024-01-01", None, "2024-03-01", None, None]
                           + [None] * 5)
        )
        series.name = "very_sparse"

        with pytest.warns(UserWarning, match="Low timestamp coverage"):
            analyzer.analyze(df, timestamp_series=series)

    def test_backward_compat_without_series(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert result.timestamp_column == "feature_timestamp"
        assert result.total_rows == len(sample_df)
        assert result.source_rows == len(sample_df)
        assert result.covered_rows == len(sample_df)


class TestResolvedTimestampSeriesStored(TestCutoffAnalyzer):
    def test_resolved_series_not_none_after_analyze(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert result.resolved_timestamp_series is not None

    def test_resolved_series_index_matches_input(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert result.resolved_timestamp_series.index.equals(sample_df.index)

    def test_resolved_series_contains_nan_for_unparseable(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": ["2024-01-01", "not_a_date", "2024-06-01", None, "2024-12-01"],
            "value": range(5),
        })

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")

        assert result.resolved_timestamp_series.isna().sum() >= 2  # "not_a_date" and None

    def test_existing_fields_unchanged(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")

        assert result.timestamp_column == "feature_timestamp"
        assert result.total_rows == len(sample_df)
        assert result.source_rows == len(sample_df)
        assert result.covered_rows == len(sample_df)
        assert len(result.bins) > 0
        assert len(result.bin_counts) > 0


class TestSplitAtCutoff(TestCutoffAnalyzer):
    @pytest.fixture
    def df_with_nulls(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-03-01", None,
            "2024-05-01", "2024-06-01", None, "2024-08-01",
            "2024-09-01", "2024-10-01",
        ])
        return pd.DataFrame({
            "feature_timestamp": dates,
            "value": range(10),
        })

    def test_split_returns_split_result(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)

        assert isinstance(split, SplitResult)

    def test_train_rows_have_ts_lte_cutoff(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)
        train_ts = result.resolved_timestamp_series.loc[split.train_df.index]

        assert (train_ts <= cutoff).all()

    def test_score_rows_have_ts_gt_cutoff(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)
        score_ts = result.resolved_timestamp_series.loc[split.score_df.index]

        assert (score_ts > cutoff).all()

    def test_train_score_unresolvable_equals_original(self, analyzer, df_with_nulls):
        result = analyzer.analyze(df_with_nulls, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)

        assert split.train_count + split.score_count + split.unresolvable_count == split.original_count

    def test_unresolvable_contains_null_rows(self, analyzer, df_with_nulls):
        result = analyzer.analyze(df_with_nulls, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)

        assert split.unresolvable_count == 2  # Two None timestamps

    def test_raises_on_index_corruption(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        # Corrupt the stored series by changing its index
        result.resolved_timestamp_series = result.resolved_timestamp_series.iloc[:5]

        with pytest.raises(Exception):
            result.split_at_cutoff(datetime(2024, 6, 1))

    def test_preserves_original_index(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.date_range("2024-01-01", periods=5, freq="ME"),
            "value": range(5),
        }, index=[10, 20, 30, 40, 50])

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")
        split = result.split_at_cutoff(datetime(2024, 3, 15))

        all_indices = sorted(
            split.train_df.index.tolist() +
            split.score_df.index.tolist() +
            split.unresolvable_df.index.tolist()
        )
        assert all_indices == [10, 20, 30, 40, 50]

    def test_all_train_when_cutoff_after_max(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2025, 12, 31)

        split = result.split_at_cutoff(cutoff)

        assert split.train_count == len(sample_df)
        assert split.score_count == 0

    def test_all_score_when_cutoff_before_min(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2022, 1, 1)

        split = result.split_at_cutoff(cutoff)

        assert split.score_count == len(sample_df)
        assert split.train_count == 0

    def test_raises_value_error_when_no_resolved_series(self):
        analysis = CutoffAnalysis(
            timestamp_column="ts", total_rows=10, bins=[], bin_counts=[],
            train_percentages=[], score_percentages=[],
            date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
        )

        with pytest.raises(ValueError, match="No resolved timestamp series"):
            analysis.split_at_cutoff(datetime(2024, 6, 1))

    def test_timestamp_source_matches_analysis(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)

        assert split.timestamp_source == "feature_timestamp"

    def test_works_with_coalesced_series(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01", None, "2024-03-01", None, "2024-05-01"]),
            "label_timestamp": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]),
            "value": range(5),
        })
        coalesced = df["feature_timestamp"].combine_first(df["label_timestamp"])
        coalesced.name = "last_action_date"

        result = analyzer.analyze(df, timestamp_series=coalesced)
        split = result.split_at_cutoff(datetime(2024, 3, 20))

        assert split.train_count + split.score_count == 5
        assert split.unresolvable_count == 0


class TestSplitResultValidation(TestCutoffAnalyzer):
    def test_counts_match_dataframes(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        split = result.split_at_cutoff(datetime(2024, 6, 1))

        assert split.train_count == len(split.train_df)
        assert split.score_count == len(split.score_df)
        assert split.unresolvable_count == len(split.unresolvable_df)

    def test_cutoff_date_stored(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        cutoff = datetime(2024, 6, 1)

        split = result.split_at_cutoff(cutoff)

        assert split.cutoff_date == cutoff

    def test_timestamp_source_stored(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        split = result.split_at_cutoff(datetime(2024, 6, 1))

        assert split.timestamp_source == "feature_timestamp"


class TestSplitEdgeCases(TestCutoffAnalyzer):
    def test_raises_value_error_when_no_source_df(self):
        ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
        analysis = CutoffAnalysis(
            timestamp_column="ts", total_rows=2, bins=[], bin_counts=[],
            train_percentages=[], score_percentages=[],
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 1)),
            resolved_timestamp_series=ts,
        )

        with pytest.raises(ValueError, match="No source DataFrame"):
            analysis.split_at_cutoff(datetime(2024, 3, 1))

    def test_all_nat_series_gives_empty_analysis(self, analyzer):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime([None, None, None]),
            "value": [1, 2, 3],
        })

        result = analyzer.analyze(df, timestamp_column="feature_timestamp")

        assert result.total_rows == 0
        assert result.resolved_timestamp_series is not None

    def test_auto_detect_nonstandard_datetime_column(self, analyzer):
        df = pd.DataFrame({
            "created_at": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
            "value": [1, 2, 3],
        })

        result = analyzer.analyze(df)

        assert result.timestamp_column == "created_at"

    def test_get_train_percentage_after_all_bins(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        # Cutoff well after all bins
        pct = result.get_train_percentage(datetime(2030, 1, 1))

        assert pct == result.train_percentages[-1]

    def test_suggest_cutoff_with_ratio_1(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, timestamp_column="feature_timestamp")
        # With ratio=1.0, only last bin at 100% satisfies
        suggested = result.suggest_cutoff(train_ratio=1.0)

        assert isinstance(suggested, datetime)
