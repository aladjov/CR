from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
from customer_retention.core.config.column_config import DatasetGranularity


class TestPopulateFromCoverage:
    def _make_metadata(self):
        return TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id", time_column="event_date",
        )

    def test_populates_windows(self):
        meta = self._make_metadata()
        meta.populate_from_coverage(
            windows=["180d", "365d", "all_time"], coverage_threshold=0.10,
        )
        assert meta.suggested_aggregations == ["180d", "365d", "all_time"]
        assert meta.window_coverage_threshold == 0.10

    def test_populates_heterogeneity(self):
        meta = self._make_metadata()
        meta.populate_from_heterogeneity(
            heterogeneity_level="high",
            eta_squared_intensity=0.18,
            eta_squared_event_count=0.22,
            segmentation_advisory="consider_segment_feature",
        )
        assert meta.heterogeneity_level == "high"
        assert meta.eta_squared_intensity == 0.18
        assert meta.eta_squared_event_count == 0.22
        assert meta.temporal_segmentation_advisory == "consider_segment_feature"

    def test_populates_segmentation_recommendation_from_advisory(self):
        meta = self._make_metadata()
        meta.populate_from_heterogeneity(
            heterogeneity_level="high",
            eta_squared_intensity=0.18,
            eta_squared_event_count=0.22,
            segmentation_advisory="consider_segment_feature",
        )
        assert meta.temporal_segmentation_recommendation == "include_lifecycle_quadrant"

    def test_single_model_advisory_skips_segmentation(self):
        meta = self._make_metadata()
        meta.populate_from_heterogeneity(
            heterogeneity_level="low",
            eta_squared_intensity=0.02,
            eta_squared_event_count=0.03,
            segmentation_advisory="single_model",
        )
        assert meta.temporal_segmentation_recommendation is None

    def test_populates_drift(self):
        meta = self._make_metadata()
        meta.populate_from_drift(
            risk_level="moderate", volume_drift_risk="growing",
            population_stability=0.65, regime_count=2,
            recommended_training_start="2020-06-01",
        )
        assert meta.drift_risk_level == "moderate"
        assert meta.volume_drift_risk == "growing"
        assert meta.population_stability == 0.65
        assert meta.regime_count == 2
        assert meta.recommended_training_start == "2020-06-01"

    def test_populates_drift_with_none_training_start(self):
        meta = self._make_metadata()
        meta.populate_from_drift(
            risk_level="low", volume_drift_risk="none",
            population_stability=0.85, regime_count=1,
            recommended_training_start=None,
        )
        assert meta.recommended_training_start is None

    def test_serialization_round_trip_with_populated_fields(self):
        meta = self._make_metadata()
        meta.populate_from_coverage(
            windows=["180d", "all_time"], coverage_threshold=0.10,
        )
        meta.populate_from_heterogeneity(
            heterogeneity_level="moderate",
            eta_squared_intensity=0.10,
            eta_squared_event_count=0.12,
            segmentation_advisory="consider_segment_feature",
        )
        meta.populate_from_drift(
            risk_level="low", volume_drift_risk="none",
            population_stability=0.80, regime_count=1,
            recommended_training_start=None,
        )
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        findings = ExplorationFindings(
            source_path="test.csv", source_format="csv",
            time_series_metadata=meta,
        )
        yaml_str = findings.to_yaml()
        loaded = ExplorationFindings.from_yaml(yaml_str)
        ts = loaded.time_series_metadata
        assert ts.suggested_aggregations == ["180d", "all_time"]
        assert ts.window_coverage_threshold == 0.10
        assert ts.heterogeneity_level == "moderate"
        assert ts.drift_risk_level == "low"
        assert ts.temporal_segmentation_recommendation == "include_lifecycle_quadrant"
