from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from customer_retention.integrations.iteration.recommendation_tracker import (
    RecommendationTracker,
    RecommendationType,
)


@dataclass
class MockFindings:
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class TestAddFromTemporalFindings:
    def test_empty_findings(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(metadata={})
        tracked = tracker.add_from_temporal_findings(findings)
        assert tracked == []
        assert len(tracker.recommendations) == 0

    def test_none_metadata(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(metadata=None)
        tracked = tracker.add_from_temporal_findings(findings)
        assert tracked == []

    def test_trend_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {
                                "action": "add_trend_features",
                                "features": ["trend_slope", "trend_intercept"],
                                "priority": "high",
                                "reason": "Strong upward trend detected",
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        assert all(t.recommendation_type == RecommendationType.FEATURE for t in tracked)
        assert all(t.source_column == "trend" for t in tracked)
        feature_names = [t.action for t in tracked]
        assert "trend_slope" in feature_names
        assert "trend_intercept" in feature_names
        # Check description includes priority
        assert any("[high]" in t.description for t in tracked)

    def test_seasonality_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "seasonality": {
                        "recommendations": [
                            {
                                "action": "add_cyclical_feature",
                                "features": ["day_of_week_sin", "day_of_week_cos"],
                                "priority": "medium",
                                "reason": "Weekly pattern detected",
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        assert all(t.source_column == "seasonality" for t in tracked)

    def test_recency_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "recency": {
                        "recommendations": [
                            {
                                "action": "add_recency_features",
                                "features": ["days_since_last_event", "log_recency"],
                                "priority": "high",
                                "reason": "Strong recency effect (d=0.8)",
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        assert all(t.source_column == "recency" for t in tracked)

    def test_categorical_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "categorical": {
                        "recommendations": [
                            {
                                "action": "create_risk_indicator",
                                "features": ["channel_is_high_risk", "region_is_high_risk"],
                                "priority": "medium",
                                "reason": "High churn rate categories detected",
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        assert all(t.source_column == "categorical" for t in tracked)

    def test_cohort_skip_action_excluded(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "cohort": {
                        "recommendations": [
                            {
                                "action": "skip_cohort_features",
                                "features": ["cohort_month"],
                                "reason": "Insufficient data for cohort analysis",
                            },
                            {
                                "action": "add_cohort_feature",
                                "features": ["cohort_quarter"],
                                "priority": "low",
                                "reason": "Quarterly cohorts show variation",
                            },
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].action == "cohort_quarter"

    def test_velocity_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "velocity": {
                        "recommendations": [
                            {
                                "action": "add_velocity_feature",
                                "source_column": "amount",
                                "description": "Add 7d velocity for amount (d=0.85)",
                                "features": ["amount_velocity_7d"],
                                "priority": 1,  # Integer priority
                                "effect_size": 0.85,
                                "params": {"window_days": 7},
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].source_column == "amount"
        assert tracked[0].action == "amount_velocity_7d"
        assert "[high]" in tracked[0].description
        assert "d=0.85" in tracked[0].description

    def test_momentum_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "momentum": {
                        "recommendations": [
                            {
                                "action": "add_momentum_feature",
                                "source_column": "frequency",
                                "description": "Add 7d/30d momentum for frequency",
                                "features": ["frequency_momentum_7_30"],
                                "priority": 2,  # Integer priority
                                "effect_size": 0.55,
                                "params": {"window_pair": "7_30"},
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].source_column == "frequency"
        assert tracked[0].action == "frequency_momentum_7_30"
        assert "[medium]" in tracked[0].description  # priority=2 -> medium

    def test_lag_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "lag": {
                        "recommendations": [
                            {
                                "action": "add_lag_feature",
                                "source_column": "amount",
                                "description": "Add 7d lag for amount (r=0.75)",
                                "features": ["amount_lag_7d"],
                                "priority": 1,
                                "params": {"lag_days": 7},
                            }
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].source_column == "amount"
        assert tracked[0].action == "amount_lag_7d"
        assert "[high]" in tracked[0].description

    def test_mixed_standard_and_temporal_sections(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add_trend", "features": ["slope"], "priority": "high", "reason": "r"}
                        ]
                    },
                    "velocity": {
                        "recommendations": [
                            {
                                "action": "add_velocity",
                                "source_column": "amount",
                                "features": ["velocity_7d"],
                                "priority": 1,
                                "description": "d",
                            }
                        ]
                    },
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        sources = {t.source_column for t in tracked}
        assert "trend" in sources
        assert "amount" in sources

    def test_multiple_sections(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add", "features": ["trend_feat"], "priority": "high", "reason": "r1"}
                        ]
                    },
                    "recency": {
                        "recommendations": [
                            {"action": "add", "features": ["recency_feat"], "priority": "medium", "reason": "r2"}
                        ]
                    },
                    "cohort": {
                        "recommendations": [
                            {"action": "add", "features": ["cohort_feat"], "priority": "low", "reason": "r3"}
                        ]
                    },
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 3
        sources = {t.source_column for t in tracked}
        assert sources == {"trend", "recency", "cohort"}

    def test_recommendations_without_features_skipped(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "investigate", "reason": "Unusual pattern"},
                            {"action": "add", "features": ["valid_feature"], "priority": "high", "reason": "r"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].action == "valid_feature"

    def test_empty_features_list_skipped(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add", "features": [], "reason": "Empty"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 0

    def test_default_priority_and_reason(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"features": ["feat1"]}  # No action, priority, or reason
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert "[medium]" in tracked[0].description
        assert "From trend analysis" in tracked[0].description

    def test_recommendations_added_to_tracker(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add", "features": ["feat1", "feat2"], "priority": "high", "reason": "r"}
                        ]
                    }
                }
            }
        )
        tracker.add_from_temporal_findings(findings)
        assert len(tracker.recommendations) == 2
        # Verify we can retrieve by ID
        rec = tracker.get("feature_trend_feat1")
        assert rec is not None
        assert rec.action == "feat1"

    def test_multiple_recommendations_same_section(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add_slope", "features": ["slope"], "priority": "high", "reason": "r1"},
                            {"action": "add_accel", "features": ["acceleration"], "priority": "medium", "reason": "r2"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 2
        actions = {t.action for t in tracked}
        assert "slope" in actions
        assert "acceleration" in actions

    def test_section_without_recommendations_key(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "direction": "increasing",
                        "strength": 0.8,
                        # No "recommendations" key
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 0

    def test_sparkline_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "sparkline": {
                        "recommendations": [
                            {"action": "add_trend_feature", "feature": "amount",
                             "reason": "Opposite trends detected", "priority": "high",
                             "features": ["amount_trend"]},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].source_column == "sparkline"
        assert "[high]" in tracked[0].description

    def test_effect_size_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "effect_size": {
                        "recommendations": [
                            {"action": "prioritize_feature", "feature": "opened",
                             "effect_size": 0.99, "priority": "high",
                             "reason": "Cohen's d=0.99 shows large effect"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].action == "opened"
        assert "prioritize" in tracked[0].description

    def test_effect_size_drop_recommendations_skipped(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "effect_size": {
                        "recommendations": [
                            {"action": "consider_dropping", "feature": "weak_feat",
                             "effect_size": 0.05, "priority": "low"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 0

    def test_predictive_power_recommendations(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "predictive_power": {
                        "recommendations": [
                            {"action": "include_feature", "feature": "clicked",
                             "iv": 0.383, "ks": 0.334, "priority": "high"},
                        ]
                    }
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        assert len(tracked) == 1
        assert tracked[0].action == "clicked"
        assert "IV=0.383" in tracked[0].description
        assert "KS=0.334" in tracked[0].description

    def test_deduplication_same_feature_multiple_sections(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add", "features": ["amount"], "priority": "high", "reason": "r"}
                        ]
                    },
                    # Same feature from different section - different source_column means different ID
                    "velocity": {
                        "recommendations": [
                            {"action": "add", "source_column": "amount",
                             "features": ["amount_velocity"], "priority": 1, "description": "d"}
                        ]
                    },
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        # Different features (amount vs amount_velocity), so both should be tracked
        assert len(tracked) == 2

    def test_deduplication_exact_same_feature(self, tmp_path):
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        findings = MockFindings(
            metadata={
                "temporal_patterns": {
                    "trend": {
                        "recommendations": [
                            {"action": "add", "features": ["amount"], "priority": "high", "reason": "r1"},
                            {"action": "add", "features": ["amount"], "priority": "medium", "reason": "r2"},
                        ]
                    },
                }
            }
        )
        tracked = tracker.add_from_temporal_findings(findings)
        # Same feature from same section - should only appear once
        assert len(tracked) == 1
