from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from customer_retention.stages.profiling.pattern_analysis_config import (
    AggregationFeatureConfig,
    FindingsValidationResult,
    create_momentum_ratio_features,
    create_recency_bucket_feature,
    deduplicate_events,
    get_duplicate_event_count,
    validate_temporal_findings,
)


@dataclass
class MockTimeSeriesMetadata:
    entity_column: str = "customer_id"
    time_column: str = "event_date"
    suggested_aggregations: Optional[List[str]] = None
    temporal_segmentation_recommendation: Optional[str] = None


@dataclass
class MockFindings:
    time_series_metadata: Optional[MockTimeSeriesMetadata] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    target_column: Optional[str] = None


class TestFindingsValidationResult:
    def test_valid_result(self):
        result = FindingsValidationResult(valid=True)
        assert result.valid
        assert result.missing_sections == []
        assert result.warnings == []

    def test_invalid_result_with_missing(self):
        result = FindingsValidationResult(
            valid=False, missing_sections=["time_series_metadata (run 01a first)"]
        )
        assert not result.valid
        assert len(result.missing_sections) == 1

    def test_valid_result_with_warnings(self):
        result = FindingsValidationResult(
            valid=True, warnings=["No trend analysis found in 01c"]
        )
        assert result.valid
        assert len(result.warnings) == 1

    def test_print_summary_invalid(self, capsys):
        result = FindingsValidationResult(
            valid=False,
            missing_sections=["time_series_metadata (run 01a first)"],
            warnings=["No trend analysis found in 01c"],
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "MISSING REQUIRED ANALYSIS" in captured.out
        assert "time_series_metadata" in captured.out
        assert "Warning" in captured.out

    def test_print_summary_valid_with_warnings(self, capsys):
        result = FindingsValidationResult(
            valid=True, warnings=["No trend analysis found in 01c"]
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "MISSING" not in captured.out
        assert "Warning" in captured.out


class TestValidateTemporalFindings:
    def test_missing_time_series_metadata(self):
        findings = MockFindings(time_series_metadata=None)
        result = validate_temporal_findings(findings)
        assert not result.valid
        assert any("time_series_metadata" in m for m in result.missing_sections)

    def test_missing_temporal_patterns(self):
        findings = MockFindings(
            time_series_metadata=MockTimeSeriesMetadata(
                suggested_aggregations=["7d", "30d"]
            ),
            metadata={},
        )
        result = validate_temporal_findings(findings)
        assert not result.valid
        assert any("temporal_patterns" in m for m in result.missing_sections)

    def test_valid_findings(self):
        findings = MockFindings(
            time_series_metadata=MockTimeSeriesMetadata(
                suggested_aggregations=["7d", "30d"]
            ),
            metadata={
                "temporal_patterns": {
                    "trend": {"direction": "increasing"},
                    "recency": {"median_days": 30},
                    "momentum": {"_divergent_columns": []},
                }
            },
        )
        result = validate_temporal_findings(findings)
        assert result.valid
        assert len(result.missing_sections) == 0

    def test_warning_no_aggregation_windows(self):
        findings = MockFindings(
            time_series_metadata=MockTimeSeriesMetadata(suggested_aggregations=None),
            metadata={
                "temporal_patterns": {
                    "trend": {},
                    "recency": {},
                    "momentum": {},
                }
            },
        )
        result = validate_temporal_findings(findings)
        assert result.valid  # Still valid, just with warnings
        assert any("aggregation windows" in w for w in result.warnings)

    def test_warning_missing_sections_in_patterns(self):
        findings = MockFindings(
            time_series_metadata=MockTimeSeriesMetadata(
                suggested_aggregations=["7d", "30d"]
            ),
            metadata={
                "temporal_patterns": {
                    "trend": {},
                    # Missing recency and momentum
                }
            },
        )
        result = validate_temporal_findings(findings)
        assert result.valid  # Still valid, just with warnings
        assert any("recency" in w for w in result.warnings)
        assert any("momentum" in w for w in result.warnings)

    def test_none_metadata(self):
        findings = MockFindings(
            time_series_metadata=MockTimeSeriesMetadata(
                suggested_aggregations=["7d"]
            ),
            metadata=None,
        )
        result = validate_temporal_findings(findings)
        assert not result.valid
        assert any("temporal_patterns" in m for m in result.missing_sections)


class TestAggregationFeatureConfig:
    def test_default_values(self):
        config = AggregationFeatureConfig()
        assert config.trend_features == []
        assert config.seasonality_features == []
        assert config.cohort_features == []
        assert config.recency_features == []
        assert config.categorical_features == []
        assert config.velocity_features == []
        assert config.momentum_features == []
        assert config.lag_features == []
        assert config.sparkline_features == []
        assert config.priority_features == []
        assert config.scaling_recommendations == []
        assert config.divergent_columns == []
        assert config.feature_flags == {}

    def test_from_findings_empty_patterns(self):
        findings = MockFindings(metadata={})
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.trend_features == []
        assert config.feature_flags == {}

    def test_from_findings_with_trend_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add_trend_feature", "features": ["trend_slope", "trend_intercept"]}
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "trend_slope" in config.trend_features
        assert "trend_intercept" in config.trend_features

    def test_from_findings_with_seasonality_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "seasonality": {
                        "recommendations": [
                            {"action": "add_cyclical_feature", "features": ["day_of_week_sin", "day_of_week_cos"]}
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "day_of_week_sin" in config.seasonality_features
        assert "day_of_week_cos" in config.seasonality_features

    def test_from_findings_with_recency_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "recency": {
                        "recommendations": [
                            {"action": "add_recency_features", "features": ["days_since_last_event", "log_recency"]}
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "days_since_last_event" in config.recency_features
        assert "log_recency" in config.recency_features

    def test_from_findings_cohort_skip_action(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "cohort": {
                        "recommendations": [
                            {"action": "skip_cohort_features", "features": ["cohort_month"], "reason": "Not enough data"},
                            {"action": "add_cohort_feature", "features": ["cohort_quarter"]}
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "cohort_month" not in config.cohort_features
        assert "cohort_quarter" in config.cohort_features

    def test_from_findings_with_categorical_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "categorical": {
                        "recommendations": [
                            {"action": "create_risk_indicator", "features": ["channel_is_high_risk"]}
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "channel_is_high_risk" in config.categorical_features

    def test_from_findings_with_velocity_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "velocity": {
                        "recommendations": [
                            {
                                "action": "add_velocity_feature",
                                "source_column": "amount",
                                "features": ["amount_velocity_7d"],
                                "priority": 1,
                                "effect_size": 0.85,
                            }
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "amount_velocity_7d" in config.velocity_features

    def test_from_findings_with_momentum_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "momentum": {
                        "recommendations": [
                            {
                                "action": "add_momentum_feature",
                                "source_column": "frequency",
                                "features": ["frequency_momentum_7_30"],
                                "priority": 2,
                                "effect_size": 0.55,
                            }
                        ],
                        "_divergent_columns": ["amount"],
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "frequency_momentum_7_30" in config.momentum_features
        assert "amount" in config.divergent_columns

    def test_from_findings_with_lag_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "lag": {
                        "recommendations": [
                            {
                                "action": "add_lag_feature",
                                "source_column": "amount",
                                "features": ["amount_lag_7d"],
                                "priority": 1,
                            }
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "amount_lag_7d" in config.lag_features

    def test_from_findings_with_divergent_columns(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "momentum": {
                        "_divergent_columns": ["amount", "frequency"]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "amount" in config.divergent_columns
        assert "frequency" in config.divergent_columns

    def test_from_findings_with_feature_flags(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "feature_flags": {
                        "include_recency": True,
                        "include_tenure": False,
                        "include_lifecycle_quadrant": True,
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.feature_flags["include_recency"] is True
        assert config.feature_flags["include_tenure"] is False
        assert config.feature_flags["include_lifecycle_quadrant"] is True

    def test_get_all_features(self):
        config = AggregationFeatureConfig(
            trend_features=["trend_slope"],
            seasonality_features=["day_of_week"],
            cohort_features=["cohort_month"],
            recency_features=["recency_days"],
            categorical_features=["channel_risk"],
            velocity_features=["amount_velocity_7d"],
            momentum_features=["freq_momentum_7_30"],
            lag_features=["amount_lag_7d"],
        )
        all_features = config.get_all_features()
        assert len(all_features) == 8
        assert "trend_slope" in all_features
        assert "day_of_week" in all_features
        assert "cohort_month" in all_features
        assert "recency_days" in all_features
        assert "channel_risk" in all_features
        assert "amount_velocity_7d" in all_features
        assert "freq_momentum_7_30" in all_features
        assert "amount_lag_7d" in all_features

    def test_get_all_features_empty(self):
        config = AggregationFeatureConfig()
        all_features = config.get_all_features()
        assert all_features == []

    def test_print_summary(self, capsys):
        config = AggregationFeatureConfig(
            trend_features=["trend_slope"],
            recency_features=["recency_days"],
            divergent_columns=["amount"],
            feature_flags={"include_recency": True},
        )
        config.print_summary()
        captured = capsys.readouterr()
        assert "AGGREGATION FEATURE CONFIG" in captured.out
        assert "trend_slope" in captured.out
        assert "recency_days" in captured.out
        assert "amount" in captured.out
        assert "include_recency" in captured.out

    def test_from_findings_multiple_recommendations_same_section(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add_trend", "features": ["slope"]},
                            {"action": "add_acceleration", "features": ["acceleration"]},
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "slope" in config.trend_features
        assert "acceleration" in config.trend_features

    def test_from_findings_none_metadata(self):
        findings = MockFindings(metadata=None)
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.trend_features == []
        assert config.feature_flags == {}

    def test_from_findings_recommendation_without_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "investigate_trend", "reason": "Unusual pattern"}
                            # No "features" key
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.trend_features == []

    def test_from_findings_with_sparkline_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "sparkline": {
                        "recommendations": [
                            {"action": "add_trend_feature", "feature": "amount",
                             "features": ["amount_add_trend_feature"], "priority": "high"},
                            {"action": "robust_scale", "feature": "variance_col",
                             "features": ["variance_col_robust_scale"], "priority": "medium"},
                        ]
                    }
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "amount_add_trend_feature" in config.sparkline_features
        assert len(config.scaling_recommendations) == 1

    def test_from_findings_with_priority_features(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "effect_size": {
                        "recommendations": [
                            {"action": "prioritize_feature", "feature": "opened",
                             "effect_size": 0.99, "priority": "high"},
                        ]
                    },
                    "predictive_power": {
                        "recommendations": [
                            {"action": "include_feature", "feature": "clicked",
                             "iv": 0.383, "ks": 0.334, "priority": "high"},
                        ]
                    },
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "opened" in config.priority_features
        assert "clicked" in config.priority_features

    def test_get_all_features_deduplication(self):
        config = AggregationFeatureConfig(
            trend_features=["amount_trend"],
            sparkline_features=["amount_trend"],  # Duplicate
            velocity_features=["amount_velocity"],
        )
        all_features = config.get_all_features()
        assert all_features.count("amount_trend") == 1
        assert len(all_features) == 2

    def test_get_priority_features(self):
        config = AggregationFeatureConfig(priority_features=["feat1", "feat2"])
        assert config.get_priority_features() == ["feat1", "feat2"]

    def test_from_findings_with_text_pca_columns(self):
        findings = MockFindings(
            metadata={"temporal_patterns": {}},
        )
        findings.text_processing = {
            "description": type("TP", (), {
                "component_columns": ["desc_pc0", "desc_pc1", "desc_pc2"]
            })(),
        }
        config = AggregationFeatureConfig.from_findings(findings)
        assert "desc_pc0" in config.text_pca_columns
        assert "desc_pc1" in config.text_pca_columns
        assert "desc_pc2" in config.text_pca_columns

    def test_from_findings_empty_text_processing(self):
        findings = MockFindings(metadata={"temporal_patterns": {}})
        findings.text_processing = {}
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.text_pca_columns == []

    def test_from_findings_no_text_processing_attribute(self):
        findings = MockFindings(metadata={"temporal_patterns": {}})
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.text_pca_columns == []

    def test_get_all_features_includes_text_pca(self):
        config = AggregationFeatureConfig(
            trend_features=["slope"],
            text_pca_columns=["pc0", "pc1"],
        )
        all_features = config.get_all_features()
        assert "pc0" in all_features
        assert "pc1" in all_features

    def test_format_recommendation_summary_all_sections(self):
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {"recommendations": [{"action": "add", "features": ["f1"]}]},
                    "seasonality": {"recommendations": [{"action": "add", "features": ["f2", "f3"]}]},
                    "recency": {"recommendations": [{"action": "add", "features": ["f4"]}]},
                    "cohort": {"recommendations": [{"action": "skip_cohort_features", "features": ["f5"]}]},
                    "sparkline": {"recommendations": [{"action": "trend", "features": ["f6"], "feature": "col1"}]},
                    "effect_size": {"recommendations": [{"action": "prioritize_feature", "feature": "f7", "priority": "high"}]},
                    "predictive_power": {"recommendations": [{"action": "include_feature", "feature": "f8", "iv": 0.3, "ks": 0.4, "priority": "high"}]},
                    "velocity": {"recommendations": []},
                    "momentum": {"recommendations": [{"action": "add", "features": ["f9"]}]},
                    "lag": {"recommendations": []},
                }
            }
        )
        config = AggregationFeatureConfig.from_findings(findings)
        summary = config.format_recommendation_summary()
        assert "trend" in summary
        assert "seasonality" in summary
        assert "recency" in summary
        assert "cohort" in summary
        assert "sparkline" in summary
        assert "effect_size" in summary
        assert "predictive_power" in summary

    def test_format_recommendation_summary_empty(self):
        config = AggregationFeatureConfig()
        summary = config.format_recommendation_summary()
        assert "RECOMMENDATION" in summary

    def test_get_duplicate_event_count_with_duplicates(self):

        findings = MockFindings(
            metadata={
                "temporal_quality": {
                    "issues": {"duplicate_events": 371}
                }
            }
        )
        assert get_duplicate_event_count(findings) == 371

    def test_get_duplicate_event_count_no_quality(self):

        findings = MockFindings(metadata={})
        assert get_duplicate_event_count(findings) == 0

    def test_get_duplicate_event_count_none_metadata(self):

        findings = MockFindings(metadata=None)
        assert get_duplicate_event_count(findings) == 0


class TestDeduplicateEvents:
    def test_removes_duplicates(self):

        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B", "B"],
            "event_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "value": [1, 2, 3, 4, 5],
        })
        result, removed = deduplicate_events(df, "entity", "event_date", duplicate_count=2)
        assert removed == 2
        assert len(result) == 3

    def test_no_op_when_zero_duplicates(self):

        df = pd.DataFrame({
            "entity": ["A", "B"], "event_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        })
        result, removed = deduplicate_events(df, "entity", "event_date", duplicate_count=0)
        assert removed == 0
        assert len(result) == 2

    def test_keeps_first_occurrence(self):

        df = pd.DataFrame({
            "entity": ["A", "A"],
            "event_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "value": [10, 20],
        })
        result, removed = deduplicate_events(df, "entity", "event_date", duplicate_count=1)
        assert result.iloc[0]["value"] == 10


class TestCreateRecencyBucketFeature:
    def test_creates_bucket_column(self):

        df = pd.DataFrame({"days_since_last_event": [3, 15, 60, 120, 250]})
        result = create_recency_bucket_feature(df)
        assert "recency_bucket" in result.columns
        assert len(result) == 5

    def test_correct_bucket_assignment(self):

        df = pd.DataFrame({"days_since_last_event": [5, 20, 60, 150, 300]})
        result = create_recency_bucket_feature(df)
        buckets = result["recency_bucket"].tolist()
        assert str(buckets[0]) == "0-7d"
        assert str(buckets[1]) == "8-30d"
        assert str(buckets[2]) == "31-90d"
        assert str(buckets[3]) == "91-180d"
        assert str(buckets[4]) == ">180d"

    def test_custom_recency_column(self):

        df = pd.DataFrame({"my_recency": [5, 20]})
        result = create_recency_bucket_feature(df, recency_column="my_recency")
        assert "recency_bucket" in result.columns

    def test_no_recency_column_returns_unchanged(self):

        df = pd.DataFrame({"other_col": [1, 2]})
        result = create_recency_bucket_feature(df)
        assert "recency_bucket" not in result.columns

    def test_recency_bucket_dtype_is_object(self):
        df = pd.DataFrame({"days_since_last_event": [5, 20, 60]})
        result = create_recency_bucket_feature(df)
        assert result["recency_bucket"].dtype == "object"


class TestCreateMomentumRatioFeatures:
    def test_creates_ratio_feature(self):

        df = pd.DataFrame({
            "entity": ["A", "B", "C"],
            "opened_mean_30d": [10.0, 5.0, 8.0],
            "opened_mean_90d": [8.0, 6.0, 4.0],
        })
        recs = [{"source_column": "opened", "params": {"short_window": 30, "long_window": 90}}]
        result = create_momentum_ratio_features(df, recs)
        assert "opened_momentum_30_90" in result.columns
        assert result.loc[0, "opened_momentum_30_90"] == 10.0 / 8.0

    def test_handles_zero_long_window(self):

        df = pd.DataFrame({
            "entity": ["A"],
            "opened_mean_30d": [10.0],
            "opened_mean_90d": [0.0],
        })
        recs = [{"source_column": "opened", "params": {"short_window": 30, "long_window": 90}}]
        result = create_momentum_ratio_features(df, recs)
        assert "opened_momentum_30_90" in result.columns

    def test_skips_missing_columns(self):

        df = pd.DataFrame({"entity": ["A"], "other_col": [1.0]})
        recs = [{"source_column": "opened", "params": {"short_window": 30, "long_window": 90}}]
        result = create_momentum_ratio_features(df, recs)
        assert "opened_momentum_30_90" not in result.columns

    def test_empty_recs(self):

        df = pd.DataFrame({"entity": ["A"], "val": [1.0]})
        result = create_momentum_ratio_features(df, [])
        assert len(result.columns) == 2

    def test_malformed_rec_missing_params(self):

        df = pd.DataFrame({"entity": ["A"], "opened_mean_30d": [10.0]})
        recs = [{"source_column": "opened"}]
        result = create_momentum_ratio_features(df, recs)
        assert len(result.columns) == 2

    def test_malformed_rec_missing_source(self):

        df = pd.DataFrame({"entity": ["A"], "opened_mean_30d": [10.0]})
        recs = [{"params": {"short_window": 30, "long_window": 90}}]
        result = create_momentum_ratio_features(df, recs)
        assert len(result.columns) == 2


class TestEdgeCases:
    def test_text_pca_with_none_component_columns(self):
        findings = MockFindings(metadata={"temporal_patterns": {}})
        findings.text_processing = {
            "description": type("TP", (), {"component_columns": None})(),
        }
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.text_pca_columns == []

    def test_text_pca_with_missing_component_columns(self):
        findings = MockFindings(metadata={"temporal_patterns": {}})
        findings.text_processing = {
            "description": type("TP", (), {})(),
        }
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.text_pca_columns == []

    def test_recency_bucket_with_nan_values(self):

        df = pd.DataFrame({"days_since_last_event": [5, float("nan"), 60]})
        result = create_recency_bucket_feature(df)
        assert "recency_bucket" in result.columns
        assert pd.isna(result.iloc[1]["recency_bucket"])

    def test_momentum_ratio_nan_in_short_col(self):

        df = pd.DataFrame({
            "entity": ["A"],
            "opened_mean_30d": [float("nan")],
            "opened_mean_90d": [5.0],
        })
        recs = [{"source_column": "opened", "params": {"short_window": 30, "long_window": 90}}]
        result = create_momentum_ratio_features(df, recs)
        assert "opened_momentum_30_90" in result.columns

    def test_dedup_on_empty_dataframe(self):

        df = pd.DataFrame({"entity": pd.Series([], dtype=str), "event_date": pd.Series([], dtype="datetime64[ns]")})
        result, removed = deduplicate_events(df, "entity", "event_date", duplicate_count=10)
        assert removed == 0
        assert len(result) == 0

    def test_duplicate_event_count_with_nested_none(self):

        findings = MockFindings(metadata={"temporal_quality": {"issues": None}})
        assert get_duplicate_event_count(findings) == 0

    def test_format_summary_includes_text_pca_label(self):
        config = AggregationFeatureConfig(text_pca_columns=["pc0"])
        summary = config.format_summary()
        assert "Text PCA" in summary
