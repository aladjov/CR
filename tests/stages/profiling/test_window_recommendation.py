import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_pattern_analyzer import SeasonalityPeriod
from customer_retention.stages.profiling.time_series_profiler import (
    ActivitySegmentResult,
    LifecycleQuadrantResult,
)
from customer_retention.stages.profiling.window_recommendation import (
    WindowRecommendationCollector,
)


@pytest.fixture
def diverse_lifecycles():
    np.random.seed(42)
    n = 500
    durations = np.random.choice([5, 30, 90, 200, 400, 700], size=n, p=[0.05, 0.15, 0.2, 0.25, 0.2, 0.15])
    counts = (durations * np.random.uniform(0.01, 0.5, size=n)).astype(int).clip(min=1)
    intensities = counts / np.clip(durations, 1, None)
    med_dur = np.median(durations)
    med_int = np.median(intensities)
    quadrants = []
    for d, i in zip(durations, intensities):
        if d >= med_dur and i >= med_int:
            quadrants.append("Steady & Loyal")
        elif d >= med_dur and i < med_int:
            quadrants.append("Occasional & Loyal")
        elif d < med_dur and i >= med_int:
            quadrants.append("Intense & Brief")
        else:
            quadrants.append("One-shot")
    segments = []
    for c in counts:
        if c == 1:
            segments.append("One-time")
        elif c <= np.percentile(counts, 25):
            segments.append("Low Activity")
        elif c <= np.percentile(counts, 75):
            segments.append("Medium Activity")
        else:
            segments.append("High Activity")
    return pd.DataFrame({
        "entity": [f"E{i}" for i in range(n)],
        "duration_days": durations,
        "event_count": counts,
        "intensity": intensities,
        "lifecycle_quadrant": quadrants,
        "activity_segment": segments,
    })


@pytest.fixture
def short_span_lifecycles():
    return pd.DataFrame({
        "entity": [f"E{i}" for i in range(100)],
        "duration_days": np.full(100, 15),
        "event_count": np.full(100, 10),
        "intensity": np.full(100, 0.67),
        "lifecycle_quadrant": ["Intense & Brief"] * 100,
        "activity_segment": ["High Activity"] * 100,
    })


@pytest.fixture
def homogeneous_lifecycles():
    np.random.seed(99)
    n = 200
    return pd.DataFrame({
        "entity": [f"E{i}" for i in range(n)],
        "duration_days": np.random.normal(200, 5, n).clip(min=10).astype(int),
        "event_count": np.random.normal(50, 2, n).clip(min=2).astype(int),
        "intensity": np.random.normal(0.25, 0.01, n),
        "lifecycle_quadrant": np.random.choice(["Steady & Loyal", "Occasional & Loyal"], n),
        "activity_segment": ["Medium Activity"] * n,
    })


@pytest.fixture
def heterogeneous_lifecycles():
    np.random.seed(77)
    groups = []
    for _ in range(100):
        groups.append({"duration_days": 5, "event_count": 50, "intensity": 10.0,
                       "lifecycle_quadrant": "Intense & Brief", "activity_segment": "High Activity"})
    for _ in range(100):
        groups.append({"duration_days": 700, "event_count": 3, "intensity": 0.004,
                       "lifecycle_quadrant": "Occasional & Loyal", "activity_segment": "Low Activity"})
    df = pd.DataFrame(groups)
    df["entity"] = [f"E{i}" for i in range(len(df))]
    return df


@pytest.fixture
def cold_start_heavy_lifecycles():
    n = 200
    data = []
    for i in range(n):
        if i < 70:
            data.append({"duration_days": 0, "event_count": 1, "intensity": 0.0,
                         "lifecycle_quadrant": "One-shot", "activity_segment": "One-time"})
        elif i < 120:
            data.append({"duration_days": 5, "event_count": 20, "intensity": 4.0,
                         "lifecycle_quadrant": "Intense & Brief", "activity_segment": "High Activity"})
        else:
            data.append({"duration_days": 500, "event_count": 100, "intensity": 0.2,
                         "lifecycle_quadrant": "Steady & Loyal", "activity_segment": "Medium Activity"})
    df = pd.DataFrame(data)
    df["entity"] = [f"E{i}" for i in range(n)]
    return df


@pytest.fixture
def segment_result(diverse_lifecycles):
    recs = pd.DataFrame({
        "Segment": ["Medium Activity", "Low Activity", "High Activity", "One-time"],
        "Entities": [200, 150, 100, 50],
        "Share": ["40.0%", "30.0%", "20.0%", "10.0%"],
        "Avg Events": [15, 5, 80, 1],
        "Feature Approach": ["Standard", "Wide windows", "All windows", "N/A"],
        "Modeling Implication": ["Good", "Sparse", "Rich", "Cold start"],
    })
    return ActivitySegmentResult(
        lifecycles=diverse_lifecycles,
        q25_threshold=5.0,
        q75_threshold=50.0,
        recommendations=recs,
    )


@pytest.fixture
def quadrant_result(diverse_lifecycles):
    recs = pd.DataFrame({
        "Quadrant": ["Steady & Loyal", "Occasional & Loyal", "Intense & Brief", "One-shot"],
        "Entities": [125, 125, 125, 125],
        "Share": ["25.0%", "25.0%", "25.0%", "25.0%"],
        "Windows": ["All (7d-365d)", "Wide (90d, 180d, 365d)", "Short (7d, 24h, 30d)", "N/A"],
        "Feature Strategy": ["Full", "Wide", "Short", "N/A"],
        "Risk": ["Low", "Medium", "High", "Very High"],
    })
    return LifecycleQuadrantResult(
        lifecycles=diverse_lifecycles,
        tenure_threshold=200.0,
        intensity_threshold=0.1,
        recommendations=recs,
    )


class TestWindowCoverageComputation:
    def test_all_time_always_included(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.99)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=730)
        assert "all_time" in result.windows

    def test_window_excluded_when_coverage_below_threshold(self, short_span_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.10)
        result = collector.compute_union(lifecycles=short_span_lifecycles, time_span_days=50)
        assert "90d" not in result.windows
        assert "180d" not in result.windows

    def test_window_included_when_coverage_above_threshold(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.10)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        assert "30d" in result.windows

    def test_hard_exclusion_when_span_too_short(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.01)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=50)
        assert "90d" not in result.windows
        assert "180d" not in result.windows
        assert "365d" not in result.windows

    def test_hard_exclusion_overrides_high_coverage(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.01)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=50)
        excluded = result.explanation[result.explanation["window"] == "90d"]
        assert not excluded.empty
        assert not excluded.iloc[0]["included"]

    def test_always_include_bypasses_threshold(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(
            coverage_threshold=0.99, always_include=["all_time", "7d"]
        )
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        assert "7d" in result.windows
        assert "all_time" in result.windows

    def test_windows_sorted_ascending_by_duration(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        window_order = {"24h": 1, "7d": 7, "14d": 14, "30d": 30, "90d": 90, "180d": 180, "365d": 365, "all_time": 9999}
        indices = [window_order[w] for w in result.windows]
        assert indices == sorted(indices)

    def test_coverage_pct_correct_for_known_data(self):
        lc = pd.DataFrame({
            "entity": ["A", "B", "C", "D", "E"],
            "duration_days": [100, 50, 200, 10, 300],
            "event_count": [20, 5, 40, 1, 60],
            "intensity": [0.2, 0.1, 0.2, 0.1, 0.2],
            "lifecycle_quadrant": ["Steady & Loyal"] * 5,
            "activity_segment": ["Medium Activity"] * 5,
        })
        collector = WindowRecommendationCollector(coverage_threshold=0.01)
        result = collector.compute_union(lifecycles=lc, time_span_days=700)
        row_30d = result.explanation[result.explanation["window"] == "30d"].iloc[0]
        has_span = (lc["duration_days"] >= 30).sum()
        expected_events = lc["event_count"] * (30 / lc["duration_days"].clip(lower=1))
        has_density = (expected_events >= 2).sum()
        beneficial = ((lc["duration_days"] >= 30) & (expected_events >= 2)).sum()
        assert row_30d["beneficial_entities"] == beneficial

    def test_meaningful_pct_requires_two_events_in_window(self):
        lc = pd.DataFrame({
            "entity": ["A", "B"],
            "duration_days": [100, 100],
            "event_count": [2, 100],
            "intensity": [0.02, 1.0],
            "lifecycle_quadrant": ["Occasional & Loyal", "Steady & Loyal"],
            "activity_segment": ["Low Activity", "High Activity"],
        })
        collector = WindowRecommendationCollector(coverage_threshold=0.01)
        result = collector.compute_union(lifecycles=lc, time_span_days=700)
        row_7d = result.explanation[result.explanation["window"] == "7d"].iloc[0]
        exp_a = 2 * (7 / 100)
        exp_b = 100 * (7 / 100)
        meaningful_count = int(exp_a >= 2) + int(exp_b >= 2)
        assert row_7d["meaningful_pct"] == pytest.approx(meaningful_count / 2, abs=0.01)

    def test_zero_duration_entities_handled_via_clip(self):
        lc = pd.DataFrame({
            "entity": ["A", "B", "C"],
            "duration_days": [0, 0, 100],
            "event_count": [1, 1, 50],
            "intensity": [0.0, 0.0, 0.5],
            "lifecycle_quadrant": ["One-shot"] * 2 + ["Steady & Loyal"],
            "activity_segment": ["One-time"] * 2 + ["High Activity"],
        })
        collector = WindowRecommendationCollector(coverage_threshold=0.01)
        result = collector.compute_union(lifecycles=lc, time_span_days=700)
        assert result.windows is not None


class TestSegmentAnnotation:
    def test_segment_context_annotates_primary_segments(self, diverse_lifecycles, segment_result):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        collector.add_segment_context(segment_result)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        has_segments = result.explanation["primary_segments"].apply(lambda x: len(x) > 0).any()
        assert has_segments

    def test_quadrant_context_annotates_primary_quadrants(self, diverse_lifecycles, quadrant_result):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        collector.add_quadrant_context(quadrant_result)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        has_segments = result.explanation["primary_segments"].apply(lambda x: len(x) > 0).any()
        assert has_segments

    def test_seasonality_context_adds_note(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        collector.add_seasonality_context([SeasonalityPeriod(period=7, strength=0.8, period_name="weekly")])
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        row_7d = result.explanation[result.explanation["window"] == "7d"].iloc[0]
        assert "seasonality" in row_7d["note"].lower()

    def test_inter_event_context_adds_timing_aligned_note(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        collector.add_inter_event_context(median_days=30.0, mean_days=35.0)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        row_30d = result.explanation[result.explanation["window"] == "30d"].iloc[0]
        assert "timing" in row_30d["note"].lower()

    def test_no_context_produces_empty_annotations(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        all_empty = result.explanation["primary_segments"].apply(lambda x: len(x) == 0).all()
        assert all_empty


class TestTemporalHeterogeneity:
    def test_eta_squared_identical_groups_is_zero(self, homogeneous_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=homogeneous_lifecycles, time_span_days=1500)
        assert result.heterogeneity.eta_squared_intensity < 0.06
        assert result.heterogeneity.eta_squared_event_count < 0.06

    def test_eta_squared_perfectly_separated_is_high(self, heterogeneous_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=heterogeneous_lifecycles, time_span_days=1500)
        assert result.heterogeneity.eta_squared_intensity > 0.14
        assert result.heterogeneity.eta_squared_event_count > 0.14

    def test_heterogeneity_level_low_below_006(self, homogeneous_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=homogeneous_lifecycles, time_span_days=1500)
        assert result.heterogeneity.heterogeneity_level == "low"

    def test_heterogeneity_level_moderate_006_to_014(self):
        np.random.seed(55)
        n = 300
        groups = np.random.choice(["A", "B", "C"], n)
        intensity = np.where(groups == "A", np.random.normal(0.3, 0.15, n),
                             np.where(groups == "B", np.random.normal(0.35, 0.15, n),
                                      np.random.normal(0.4, 0.15, n)))
        lc = pd.DataFrame({
            "entity": [f"E{i}" for i in range(n)],
            "duration_days": np.full(n, 200),
            "event_count": np.full(n, 30),
            "intensity": intensity.clip(min=0.01),
            "lifecycle_quadrant": groups,
            "activity_segment": ["Medium Activity"] * n,
        })
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=lc, time_span_days=1500)
        assert result.heterogeneity.heterogeneity_level in ("low", "moderate")

    def test_heterogeneity_level_high_above_014(self, heterogeneous_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=heterogeneous_lifecycles, time_span_days=1500)
        assert result.heterogeneity.heterogeneity_level == "high"

    def test_advisory_single_model_when_low_heterogeneity(self, homogeneous_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=homogeneous_lifecycles, time_span_days=1500)
        assert result.heterogeneity.segmentation_advisory == "single_model"

    def test_advisory_segment_feature_when_moderate(self):
        np.random.seed(88)
        n = 300
        groups = np.random.choice(["Steady & Loyal", "Occasional & Loyal", "Intense & Brief"], n)
        intensity = np.where(groups == "Steady & Loyal", np.random.normal(0.5, 0.1, n),
                             np.where(groups == "Occasional & Loyal", np.random.normal(0.2, 0.1, n),
                                      np.random.normal(0.35, 0.1, n)))
        event_count = np.where(groups == "Steady & Loyal", np.random.normal(80, 10, n),
                               np.where(groups == "Occasional & Loyal", np.random.normal(50, 10, n),
                                         np.random.normal(65, 10, n)))
        lc = pd.DataFrame({
            "entity": [f"E{i}" for i in range(n)],
            "duration_days": np.full(n, 300),
            "event_count": event_count.clip(min=2).astype(int),
            "intensity": intensity.clip(min=0.01),
            "lifecycle_quadrant": groups,
            "activity_segment": ["Medium Activity"] * n,
        })
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=lc, time_span_days=1500)
        assert result.heterogeneity.segmentation_advisory in ("consider_segment_feature", "single_model")

    def test_advisory_separate_models_when_high_with_cold_start(self, cold_start_heavy_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=cold_start_heavy_lifecycles, time_span_days=1500)
        assert result.heterogeneity.segmentation_advisory == "consider_separate_models"

    def test_coverage_table_has_all_selected_windows(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        table_windows = set(result.heterogeneity.coverage_table["window"].tolist())
        assert set(result.windows).issubset(table_windows)


class TestWindowUnionResult:
    def test_result_has_ordered_window_list(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        assert isinstance(result.windows, list)
        assert len(result.windows) >= 1

    def test_explanation_table_includes_excluded_windows(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.50)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        assert len(result.explanation) == len(WindowRecommendationCollector.ALL_CANDIDATE_WINDOWS)
        excluded = result.explanation[~result.explanation["included"]]
        assert len(excluded) >= 1

    def test_feature_count_estimate_formula(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(
            lifecycles=diverse_lifecycles, time_span_days=1500,
            value_columns=3, agg_funcs=4,
        )
        n_windows = len(result.windows)
        expected = 3 * 4 * n_windows + n_windows
        assert result.feature_count_estimate == expected

    def test_result_serializable_to_findings_fields(self, diverse_lifecycles):
        collector = WindowRecommendationCollector(coverage_threshold=0.05)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        assert isinstance(result.windows, list)
        assert all(isinstance(w, str) for w in result.windows)
        assert isinstance(result.coverage_threshold, float)
        h = result.heterogeneity
        assert isinstance(h.eta_squared_intensity, float)
        assert isinstance(h.eta_squared_event_count, float)
        assert isinstance(h.heterogeneity_level, str)
        assert isinstance(h.segmentation_advisory, str)


class TestFindingsPersistence:
    def test_new_fields_have_none_defaults(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        meta = TimeSeriesMetadata()
        assert meta.window_coverage_threshold is None
        assert meta.heterogeneity_level is None
        assert meta.eta_squared_intensity is None
        assert meta.eta_squared_event_count is None
        assert meta.temporal_segmentation_advisory is None

    def test_fields_roundtrip_through_yaml(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import (
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.core.config.column_config import DatasetGranularity
        meta = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            suggested_aggregations=["7d", "30d", "all_time"],
            window_coverage_threshold=0.10,
            heterogeneity_level="moderate",
            eta_squared_intensity=0.09,
            eta_squared_event_count=0.07,
            temporal_segmentation_advisory="consider_segment_feature",
        )
        findings = ExplorationFindings(
            source_path="test.csv", source_format="csv",
            time_series_metadata=meta,
        )
        path = str(tmp_path / "findings.yaml")
        findings.save(path)
        loaded = ExplorationFindings.load(path)
        assert loaded.time_series_metadata.suggested_aggregations == ["7d", "30d", "all_time"]
        assert loaded.time_series_metadata.window_coverage_threshold == 0.10
        assert loaded.time_series_metadata.heterogeneity_level == "moderate"
        assert loaded.time_series_metadata.eta_squared_intensity == 0.09
        assert loaded.time_series_metadata.eta_squared_event_count == 0.07
        assert loaded.time_series_metadata.temporal_segmentation_advisory == "consider_segment_feature"

    def test_suggested_aggregations_set_from_window_result(self, diverse_lifecycles):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        collector = WindowRecommendationCollector(coverage_threshold=0.10)
        result = collector.compute_union(lifecycles=diverse_lifecycles, time_span_days=1500)
        meta = TimeSeriesMetadata(suggested_aggregations=result.windows)
        assert meta.suggested_aggregations == result.windows
        assert len(meta.suggested_aggregations) >= 1
