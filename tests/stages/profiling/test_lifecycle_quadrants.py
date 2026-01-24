from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.time_series_profiler import (
    LifecycleQuadrantResult,
    classify_lifecycle_quadrants,
)


@pytest.fixture
def balanced_lifecycles():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "entity": [f"E{i}" for i in range(n)],
        "first_event": pd.Timestamp("2020-01-01"),
        "last_event": [pd.Timestamp("2020-01-01") + timedelta(days=np.random.randint(1, 3000)) for _ in range(n)],
        "duration_days": np.random.randint(1, 3000, size=n),
        "event_count": np.random.randint(1, 100, size=n),
    })


@pytest.fixture
def extreme_lifecycles():
    return pd.DataFrame({
        "entity": ["long_intense", "long_sparse", "short_intense", "short_sparse"],
        "first_event": pd.Timestamp("2020-01-01"),
        "last_event": pd.Timestamp("2023-01-01"),
        "duration_days": [2000, 2000, 10, 10],
        "event_count": [500, 5, 50, 2],
    })


class TestClassifyLifecycleQuadrants:
    def test_returns_lifecycle_quadrant_result(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        assert isinstance(result, LifecycleQuadrantResult)

    def test_lifecycles_has_quadrant_column(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        assert "lifecycle_quadrant" in result.lifecycles.columns
        assert "intensity" in result.lifecycles.columns

    def test_all_entities_classified(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        assert result.lifecycles["lifecycle_quadrant"].notna().all()
        assert len(result.lifecycles) == len(balanced_lifecycles)

    def test_four_quadrant_names(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        expected = {"Steady & Loyal", "Occasional & Loyal", "Intense & Brief", "One-shot"}
        assert set(result.lifecycles["lifecycle_quadrant"].unique()).issubset(expected)

    def test_thresholds_are_medians(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        assert result.tenure_threshold == balanced_lifecycles["duration_days"].median()

    def test_extreme_entities_classified_correctly(self, extreme_lifecycles):
        result = classify_lifecycle_quadrants(extreme_lifecycles)
        lc = result.lifecycles.set_index("entity")
        assert lc.loc["long_intense", "lifecycle_quadrant"] == "Steady & Loyal"
        assert lc.loc["long_sparse", "lifecycle_quadrant"] == "Occasional & Loyal"
        assert lc.loc["short_intense", "lifecycle_quadrant"] == "Intense & Brief"
        assert lc.loc["short_sparse", "lifecycle_quadrant"] == "One-shot"

    def test_zero_duration_uses_clip(self):
        lc = pd.DataFrame({
            "entity": ["A", "B"],
            "first_event": pd.Timestamp("2020-01-01"),
            "last_event": pd.Timestamp("2020-01-01"),
            "duration_days": [0, 100],
            "event_count": [5, 5],
        })
        result = classify_lifecycle_quadrants(lc)
        assert (result.lifecycles["intensity"] >= 0).all()


class TestRecommendationsTable:
    def test_recommendations_is_dataframe(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        assert isinstance(result.recommendations, pd.DataFrame)

    def test_recommendations_has_all_present_quadrants(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        present = result.lifecycles["lifecycle_quadrant"].unique()
        assert set(result.recommendations["Quadrant"]) == set(present)

    def test_recommendations_ordered_by_share_descending(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        shares = result.recommendations["Share"].str.rstrip("%").astype(float)
        assert (shares == shares.sort_values(ascending=False).values).all()

    def test_recommendations_has_expected_columns(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        for col in ["Quadrant", "Entities", "Share", "Windows", "Feature Strategy", "Risk"]:
            assert col in result.recommendations.columns, f"Missing column: {col}"

    def test_recommendations_share_sums_to_100(self, balanced_lifecycles):
        result = classify_lifecycle_quadrants(balanced_lifecycles)
        total = result.recommendations["Share"].str.rstrip("%").astype(float).sum()
        assert abs(total - 100.0) < 0.5

    def test_extreme_data_all_quadrants_present(self, extreme_lifecycles):
        result = classify_lifecycle_quadrants(extreme_lifecycles)
        assert len(result.recommendations) == 4
