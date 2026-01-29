import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_coverage import (
    EntityWindowCoverage,
    FeatureAvailabilityResult,
    analyze_feature_availability,
    analyze_temporal_coverage,
    derive_drift_implications,
)


@pytest.fixture
def steady_events():
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    np.random.seed(42)
    entities = np.random.choice([f"E{i}" for i in range(100)], size=len(dates) * 3)
    times = np.random.choice(dates, size=len(entities))
    return pd.DataFrame({"entity": entities, "event_date": times})


@pytest.fixture
def gapped_events():
    before = pd.date_range("2020-01-01", "2020-06-30", freq="D")
    after = pd.date_range("2020-10-01", "2021-06-30", freq="D")
    np.random.seed(42)
    entities_before = np.random.choice([f"E{i}" for i in range(50)], size=len(before) * 2)
    times_before = np.random.choice(before, size=len(entities_before))
    entities_after = np.random.choice([f"E{i}" for i in range(50)], size=len(after) * 2)
    times_after = np.random.choice(after, size=len(entities_after))
    return pd.DataFrame({
        "entity": np.concatenate([entities_before, entities_after]),
        "event_date": np.concatenate([times_before, times_after]),
    })


@pytest.fixture
def growing_events():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    rows = []
    for d in dates:
        day_idx = (d - dates[0]).days
        n_events = max(1, int(1 + day_idx * 0.05))
        for _ in range(n_events):
            rows.append({"entity": f"E{np.random.randint(0, 200)}", "event_date": d})
    return pd.DataFrame(rows)


@pytest.fixture
def declining_events():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    rows = []
    total_days = len(dates)
    for i, d in enumerate(dates):
        n_events = max(1, int(30 - i * 0.04))
        for _ in range(n_events):
            rows.append({"entity": f"E{np.random.randint(0, 200)}", "event_date": d})
    return pd.DataFrame(rows)


@pytest.fixture
def recent_only_events():
    np.random.seed(42)
    dates = pd.date_range("2023-11-01", "2023-12-31", freq="D")
    entities = np.random.choice([f"E{i}" for i in range(50)], size=len(dates) * 5)
    times = np.random.choice(dates, size=len(entities))
    return pd.DataFrame({"entity": entities, "event_date": times})


class TestGapDetection:
    def test_no_gaps_in_steady_data(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        major_gaps = [g for g in result.gaps if g.severity == "major"]
        assert len(major_gaps) == 0

    def test_detects_major_gap(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        major_gaps = [g for g in result.gaps if g.severity == "major"]
        assert len(major_gaps) >= 1
        gap = major_gaps[0]
        assert gap.duration_days >= 30

    def test_gap_has_start_end_dates(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        for gap in result.gaps:
            assert gap.start < gap.end
            assert gap.duration_days == pytest.approx((gap.end - gap.start).days, abs=1)

    def test_gap_severity_classification(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        for gap in result.gaps:
            if gap.duration_days < 7:
                assert gap.severity == "minor"
            elif gap.duration_days < 30:
                assert gap.severity == "moderate"
            else:
                assert gap.severity == "major"

    def test_single_day_data_no_crash(self):
        df = pd.DataFrame({
            "entity": ["A", "B", "C"],
            "event_date": pd.Timestamp("2023-01-15"),
        })
        result = analyze_temporal_coverage(df, "entity", "event_date")
        assert result.time_span_days == 0
        assert len(result.gaps) == 0


class TestEntityWindowCoverage:
    def test_all_time_coverage_is_100_pct(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        all_time = next(c for c in result.entity_window_coverage if c.window == "all_time")
        assert all_time.coverage_pct == 1.0

    def test_short_window_lower_coverage_than_long(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        coverages = {c.window: c.coverage_pct for c in result.entity_window_coverage}
        if "7d" in coverages and "365d" in coverages:
            assert coverages["7d"] <= coverages["365d"]

    def test_coverage_pct_between_0_and_1(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        for c in result.entity_window_coverage:
            assert 0.0 <= c.coverage_pct <= 1.0

    def test_active_entities_count_correct(self, recent_only_events):
        result = analyze_temporal_coverage(recent_only_events, "entity", "event_date")
        all_time = next(c for c in result.entity_window_coverage if c.window == "all_time")
        assert all_time.active_entities == recent_only_events["entity"].nunique()

    def test_custom_windows(self, steady_events):
        result = analyze_temporal_coverage(
            steady_events, "entity", "event_date",
            candidate_windows=["30d", "90d", "all_time"],
        )
        windows = [c.window for c in result.entity_window_coverage]
        assert windows == ["30d", "90d", "all_time"]

    def test_reference_date_affects_coverage(self, steady_events):
        early_ref = pd.Timestamp("2020-03-01")
        result = analyze_temporal_coverage(
            steady_events, "entity", "event_date", reference_date=early_ref,
        )
        cov_365 = next(c for c in result.entity_window_coverage if c.window == "365d")
        assert cov_365.coverage_pct < 1.0


class TestVolumeTrend:
    def test_growing_trend(self, growing_events):
        result = analyze_temporal_coverage(growing_events, "entity", "event_date")
        assert result.volume_trend == "growing"
        assert result.volume_change_pct > 0

    def test_declining_trend(self, declining_events):
        result = analyze_temporal_coverage(declining_events, "entity", "event_date")
        assert result.volume_trend == "declining"
        assert result.volume_change_pct < 0

    def test_stable_trend(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert result.volume_trend in ("stable", "growing")

    def test_volume_change_pct_is_float(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert isinstance(result.volume_change_pct, float)


class TestRecommendations:
    def test_gap_produces_recommendation(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        assert any("gap" in r.lower() for r in result.recommendations)

    def test_declining_produces_recommendation(self, declining_events):
        result = analyze_temporal_coverage(declining_events, "entity", "event_date")
        assert any("declin" in r.lower() for r in result.recommendations)

    def test_short_span_produces_recommendation(self, recent_only_events):
        result = analyze_temporal_coverage(recent_only_events, "entity", "event_date")
        assert any("short" in r.lower() or "limited" in r.lower() for r in result.recommendations)

    def test_steady_data_minimal_recommendations(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        warning_recs = [r for r in result.recommendations if "warning" in r.lower() or "gap" in r.lower()]
        assert len(warning_recs) == 0


class TestTemporalCoverageResult:
    def test_result_has_time_span(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert result.time_span_days > 0

    def test_result_has_first_last_event(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert result.first_event < result.last_event

    def test_result_gaps_is_list(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert isinstance(result.gaps, list)

    def test_result_entity_window_coverage_is_list(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert isinstance(result.entity_window_coverage, list)
        assert all(isinstance(c, EntityWindowCoverage) for c in result.entity_window_coverage)

    def test_events_over_time_dataframe(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert isinstance(result.events_over_time, pd.Series)
        assert len(result.events_over_time) > 0

    def test_new_entities_over_time_dataframe(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        assert isinstance(result.new_entities_over_time, pd.Series)
        assert len(result.new_entities_over_time) > 0


class TestDriftImplication:
    def test_steady_data_low_risk(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.risk_level == "low"
        assert drift.volume_drift_risk == "none"
        assert drift.regime_count == 1

    def test_gapped_data_has_regime_boundaries(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.regime_count >= 2
        assert len(drift.regime_boundaries) >= 1
        for boundary in drift.regime_boundaries:
            assert isinstance(boundary, pd.Timestamp)

    def test_gapped_data_recommends_training_start(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.recommended_training_start is not None
        assert drift.recommended_training_start > result.first_event

    def test_growing_volume_drift_risk(self, growing_events):
        result = analyze_temporal_coverage(growing_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.volume_drift_risk == "growing"

    def test_declining_volume_drift_risk(self, declining_events):
        result = analyze_temporal_coverage(declining_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.volume_drift_risk == "declining"
        assert drift.risk_level in ("moderate", "high")

    def test_population_stability_between_0_and_1(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert 0.0 <= drift.population_stability <= 1.0

    def test_population_stability_low_for_uneven_influx(self):
        np.random.seed(42)
        early = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        late = pd.date_range("2020-10-01", "2021-06-30", freq="D")
        entities_early = [f"E{i}" for i in range(200)] * 2
        times_early = np.random.choice(early, size=len(entities_early))
        entities_late = [f"L{i}" for i in range(200)] * 2
        times_late = np.random.choice(late, size=len(entities_late))
        df = pd.DataFrame({
            "entity": np.concatenate([entities_early, entities_late]),
            "event_date": np.concatenate([times_early, times_late]),
        })
        result = analyze_temporal_coverage(df, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.population_stability < 0.7

    def test_short_span_moderate_risk(self, recent_only_events):
        result = analyze_temporal_coverage(recent_only_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.risk_level in ("moderate", "high")

    def test_rationale_populated(self, gapped_events):
        result = analyze_temporal_coverage(gapped_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert len(drift.rationale) > 0
        assert all(isinstance(r, str) for r in drift.rationale)

    def test_single_day_data_no_crash(self):
        df = pd.DataFrame({
            "entity": ["A", "B", "C"],
            "event_date": pd.Timestamp("2023-01-15"),
        })
        result = analyze_temporal_coverage(df, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.risk_level in ("moderate", "high")
        assert drift.regime_count == 1

    def test_regime_count_one_without_major_gaps(self, steady_events):
        result = analyze_temporal_coverage(steady_events, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.regime_count == 1
        assert drift.recommended_training_start is None

    def test_coverage_decay_increases_risk(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        rows = []
        for d in dates:
            day_idx = (d - dates[0]).days
            n_entities = max(1, int(50 - day_idx * 0.05))
            for j in range(3):
                rows.append({"entity": f"E{np.random.randint(0, n_entities)}", "event_date": d})
        df = pd.DataFrame(rows)
        result = analyze_temporal_coverage(df, "entity", "event_date")
        drift = derive_drift_implications(result)
        assert drift.population_stability < 0.8


class TestFeatureAvailability:
    @pytest.fixture
    def data_with_new_tracking(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "event_date": dates,
            "always_present": np.random.rand(len(dates)),
            "new_feature": [None]*180 + list(np.random.rand(len(dates)-180)),  # starts mid-year
        })
        return df

    @pytest.fixture
    def data_with_retired_tracking(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "event_date": dates,
            "always_present": np.random.rand(len(dates)),
            "retired_feature": list(np.random.rand(180)) + [None]*(len(dates)-180),  # stops mid-year
        })
        return df

    @pytest.fixture
    def data_with_partial_window(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "event_date": dates,
            "always_present": np.random.rand(len(dates)),
            "partial_feature": [None]*60 + list(np.random.rand(180)) + [None]*(len(dates)-240),
        })
        return df

    def test_detects_new_tracking(self, data_with_new_tracking):
        result = analyze_feature_availability(data_with_new_tracking, "event_date")
        assert isinstance(result, FeatureAvailabilityResult)
        assert "new_feature" in result.new_tracking
        assert "always_present" not in result.new_tracking

    def test_detects_retired_tracking(self, data_with_retired_tracking):
        result = analyze_feature_availability(data_with_retired_tracking, "event_date")
        assert "retired_feature" in result.retired_tracking
        assert "always_present" not in result.retired_tracking

    def test_detects_partial_window(self, data_with_partial_window):
        result = analyze_feature_availability(data_with_partial_window, "event_date")
        assert "partial_feature" in result.partial_window

    def test_full_coverage_marked_correctly(self, data_with_new_tracking):
        result = analyze_feature_availability(data_with_new_tracking, "event_date")
        always_feat = next(f for f in result.features if f.column == "always_present")
        assert always_feat.availability_type == "full"

    def test_feature_availability_has_dates(self, data_with_new_tracking):
        result = analyze_feature_availability(data_with_new_tracking, "event_date")
        new_feat = next(f for f in result.features if f.column == "new_feature")
        assert new_feat.first_valid_date is not None
        assert new_feat.last_valid_date is not None
        assert new_feat.days_from_start > 0

    def test_generates_recommendations(self, data_with_new_tracking):
        result = analyze_feature_availability(data_with_new_tracking, "event_date")
        assert len(result.recommendations) > 0
        new_recs = [r for r in result.recommendations if r.get("column") == "new_feature"]
        assert len(new_recs) >= 1
        assert new_recs[0]["issue"] == "new_tracking"

    def test_train_test_split_warning(self, data_with_new_tracking):
        result = analyze_feature_availability(data_with_new_tracking, "event_date")
        general_recs = [r for r in result.recommendations if r.get("column") == "_general_"]
        assert len(general_recs) >= 1
        assert "availability" in general_recs[0]["reason"].lower()
        assert any("split" in opt.lower() for opt in general_recs[0]["options"])

    def test_exclude_columns_respected(self, data_with_new_tracking):
        result = analyze_feature_availability(
            data_with_new_tracking, "event_date", exclude_columns=["always_present"]
        )
        columns_analyzed = [f.column for f in result.features]
        assert "always_present" not in columns_analyzed

    def test_empty_column_detected(self):
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        df = pd.DataFrame({
            "event_date": dates,
            "empty_col": [None] * len(dates),
            "full_col": np.random.rand(len(dates)),
        })
        result = analyze_feature_availability(df, "event_date")
        empty_feat = next(f for f in result.features if f.column == "empty_col")
        assert empty_feat.availability_type == "empty"
        assert empty_feat.coverage_pct == 0

    def test_threshold_customization(self, data_with_new_tracking):
        result_strict = analyze_feature_availability(
            data_with_new_tracking, "event_date", late_start_threshold_pct=5.0
        )
        result_lenient = analyze_feature_availability(
            data_with_new_tracking, "event_date", late_start_threshold_pct=60.0
        )
        assert len(result_strict.new_tracking) >= len(result_lenient.new_tracking)
