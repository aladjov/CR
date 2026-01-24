import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.time_series_profiler import (
    ActivitySegmentResult,
    classify_activity_segments,
)


@pytest.fixture
def varied_lifecycles():
    np.random.seed(42)
    n = 200
    counts = np.random.choice([1, 3, 10, 15, 20, 50, 100], size=n, p=[0.05, 0.15, 0.2, 0.2, 0.2, 0.15, 0.05])
    return pd.DataFrame({
        "entity": [f"E{i}" for i in range(n)],
        "first_event": pd.Timestamp("2020-01-01"),
        "last_event": pd.Timestamp("2023-01-01"),
        "duration_days": np.random.randint(10, 1000, size=n),
        "event_count": counts,
    })


@pytest.fixture
def single_event_only():
    return pd.DataFrame({
        "entity": ["A", "B", "C"],
        "first_event": pd.Timestamp("2020-01-01"),
        "last_event": pd.Timestamp("2020-01-01"),
        "duration_days": [0, 0, 0],
        "event_count": [1, 1, 1],
    })


class TestClassifyActivitySegments:
    def test_returns_activity_segment_result(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        assert isinstance(result, ActivitySegmentResult)

    def test_lifecycles_has_activity_segment_column(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        assert "activity_segment" in result.lifecycles.columns

    def test_all_entities_classified(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        assert result.lifecycles["activity_segment"].notna().all()
        assert len(result.lifecycles) == len(varied_lifecycles)

    def test_segment_names(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        expected = {"One-time", "Low Activity", "Medium Activity", "High Activity"}
        assert set(result.lifecycles["activity_segment"].unique()).issubset(expected)

    def test_one_time_entities_have_count_1(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        one_time = result.lifecycles[result.lifecycles["activity_segment"] == "One-time"]
        assert (one_time["event_count"] == 1).all()

    def test_high_activity_above_q75(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        q75 = varied_lifecycles["event_count"].quantile(0.75)
        high = result.lifecycles[result.lifecycles["activity_segment"] == "High Activity"]
        assert (high["event_count"] > q75).all()

    def test_thresholds_match_quartiles(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        assert result.q25_threshold == varied_lifecycles["event_count"].quantile(0.25)
        assert result.q75_threshold == varied_lifecycles["event_count"].quantile(0.75)

    def test_all_one_time_entities(self, single_event_only):
        result = classify_activity_segments(single_event_only)
        assert (result.lifecycles["activity_segment"] == "One-time").all()


class TestActivitySegmentRecommendations:
    def test_recommendations_is_dataframe(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        assert isinstance(result.recommendations, pd.DataFrame)

    def test_recommendations_has_all_present_segments(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        present = set(result.lifecycles["activity_segment"].unique())
        assert set(result.recommendations["Segment"]) == present

    def test_recommendations_ordered_by_share_descending(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        shares = result.recommendations["Share"].str.rstrip("%").astype(float)
        assert (shares == shares.sort_values(ascending=False).values).all()

    def test_recommendations_has_expected_columns(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        for col in ["Segment", "Entities", "Share", "Avg Events", "Feature Approach", "Modeling Implication"]:
            assert col in result.recommendations.columns, f"Missing column: {col}"

    def test_recommendations_share_sums_to_100(self, varied_lifecycles):
        result = classify_activity_segments(varied_lifecycles)
        total = result.recommendations["Share"].str.rstrip("%").astype(float).sum()
        assert abs(total - 100.0) < 0.5
