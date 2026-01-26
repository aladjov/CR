import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling import (
    AggregationMethod,
    TargetColumnDetector,
    TargetLevel,
    TargetLevelAnalyzer,
)


@pytest.fixture
def entity_level_data():
    """Data where target is consistent within entities."""
    np.random.seed(42)
    entities = [f"E{i}" for i in range(100)]
    data = []
    for entity in entities:
        n_events = np.random.randint(3, 10)
        target = np.random.choice([0, 1])  # Same for all events of entity
        for _ in range(n_events):
            data.append({
                "entity_id": entity,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 365)),
                "target": target,
                "value": np.random.randn()
            })
    return pd.DataFrame(data)


@pytest.fixture
def event_level_data():
    """Data where target varies within entities."""
    np.random.seed(42)
    entities = [f"E{i}" for i in range(100)]
    data = []
    for entity in entities:
        n_events = np.random.randint(5, 15)
        for _ in range(n_events):
            data.append({
                "entity_id": entity,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 365)),
                "target": np.random.choice([0, 1]),  # Varies within entity
                "value": np.random.randn()
            })
    return pd.DataFrame(data)


class TestTargetLevel:
    def test_enum_values(self):
        assert TargetLevel.ENTITY_LEVEL.value == "entity_level"
        assert TargetLevel.EVENT_LEVEL.value == "event_level"
        assert TargetLevel.UNKNOWN.value == "unknown"
        assert TargetLevel.MISSING.value == "missing"


class TestAggregationMethod:
    def test_enum_values(self):
        assert AggregationMethod.MAX.value == "max"
        assert AggregationMethod.MEAN.value == "mean"
        assert AggregationMethod.SUM.value == "sum"
        assert AggregationMethod.LAST.value == "last"
        assert AggregationMethod.FIRST.value == "first"


class TestTargetLevelAnalyzer:
    def test_detects_entity_level(self, entity_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(entity_level_data, "target", "entity_id")

        assert result.level == TargetLevel.ENTITY_LEVEL
        assert result.suggested_aggregation is None
        assert result.variation_pct < 5

    def test_detects_event_level(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(event_level_data, "target", "entity_id")

        assert result.level == TargetLevel.EVENT_LEVEL
        assert result.suggested_aggregation is not None
        assert result.variation_pct > 5

    def test_missing_column(self, entity_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(entity_level_data, "nonexistent", "entity_id")

        assert result.level == TargetLevel.MISSING

    def test_none_columns(self, entity_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(entity_level_data, None, "entity_id")

        assert result.level == TargetLevel.UNKNOWN


class TestAggregation:
    def test_max_aggregation(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.MAX
        )

        assert "target_entity" in df_result.columns
        assert result.entity_target_column == "target_entity"
        assert result.aggregation_used == AggregationMethod.MAX

    def test_mean_aggregation(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.MEAN
        )

        assert "target_entity" in df_result.columns
        # Mean of binary values should be between 0 and 1
        assert df_result["target_entity"].between(0, 1).all()

    def test_sum_aggregation(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.SUM
        )

        assert "target_entity" in df_result.columns
        # Sum should be >= 0
        assert (df_result["target_entity"] >= 0).all()

    def test_last_aggregation(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.LAST
        )

        assert "target_entity" in df_result.columns
        # Should be 0 or 1
        assert df_result["target_entity"].isin([0, 1]).all()

    def test_entity_level_no_aggregation(self, entity_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            entity_level_data, "target", "entity_id", "timestamp", AggregationMethod.MAX
        )

        # No new column created - target already at entity level
        assert result.entity_target_column == "target"
        assert "target_entity" not in df_result.columns


class TestEntityDistribution:
    def test_entity_distribution_computed(self, entity_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(entity_level_data, "target", "entity_id")

        assert result.entity_distribution is not None
        assert 0 in result.entity_distribution.value_counts or 1 in result.entity_distribution.value_counts

    def test_event_distribution_computed(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(event_level_data, "target", "entity_id")

        assert result.event_distribution is not None
        assert result.event_distribution.total == len(event_level_data)


class TestTargetColumnDetector:
    def test_override(self, entity_level_data):
        class MockFindings:
            columns = {}

        detector = TargetColumnDetector()
        col, method = detector.detect(MockFindings(), entity_level_data, override="target")

        assert col == "target"
        assert method == "override"

    def test_defer(self, entity_level_data):
        class MockFindings:
            columns = {}

        detector = TargetColumnDetector()
        col, method = detector.detect(MockFindings(), entity_level_data, override="DEFER_TO_MULTI_DATASET")

        assert col is None
        assert method == "deferred"

    def test_not_found(self, entity_level_data):
        class MockFindings:
            columns = {}

        detector = TargetColumnDetector()
        col, method = detector.detect(MockFindings(), entity_level_data, override=None)

        assert col is None
        assert method == "not-found"

    def test_auto_detected(self, entity_level_data):
        from customer_retention.core.config.column_config import ColumnType

        class MockColumnInfo:
            def __init__(self, inferred_type):
                self.inferred_type = inferred_type

        class MockFindings:
            columns = {"target": MockColumnInfo(ColumnType.TARGET)}

        detector = TargetColumnDetector()
        col, method = detector.detect(MockFindings(), entity_level_data, override=None)

        assert col == "target"
        assert method == "auto-detected"

    def test_binary_candidate(self, entity_level_data):
        from customer_retention.core.config.column_config import ColumnType

        class MockColumnInfo:
            def __init__(self, inferred_type):
                self.inferred_type = inferred_type

        class MockFindings:
            columns = {"churn_flag": MockColumnInfo(ColumnType.BINARY)}

        detector = TargetColumnDetector()
        col, method = detector.detect(MockFindings(), entity_level_data, override=None)

        assert col == "churn_flag"
        assert method == "binary-candidate"

    def test_print_detection_deferred(self, capsys):
        detector = TargetColumnDetector()
        detector.print_detection(None, "deferred")
        captured = capsys.readouterr()
        assert "deferred" in captured.out.lower()

    def test_print_detection_override(self, capsys):
        detector = TargetColumnDetector()
        detector.print_detection("target", "override")
        captured = capsys.readouterr()
        assert "override" in captured.out.lower()

    def test_print_detection_auto(self, capsys):
        detector = TargetColumnDetector()
        detector.print_detection("target", "auto-detected")
        captured = capsys.readouterr()
        assert "Auto-detected" in captured.out

    def test_print_detection_binary_candidate(self, capsys):
        detector = TargetColumnDetector()
        detector.print_detection("churn", "binary-candidate", ["other_col"])
        captured = capsys.readouterr()
        assert "binary candidate" in captured.out.lower()

    def test_print_detection_not_found(self, capsys):
        detector = TargetColumnDetector()
        detector.print_detection(None, "not-found")
        captured = capsys.readouterr()
        assert "No target" in captured.out


class TestTargetDistribution:
    def test_as_percentages(self):
        from customer_retention.stages.profiling.target_level_analyzer import TargetDistribution

        dist = TargetDistribution(value_counts={0: 80, 1: 20}, total=100)
        pcts = dist.as_percentages

        assert pcts[0] == 80.0
        assert pcts[1] == 20.0

    def test_get_label(self):
        from customer_retention.stages.profiling.target_level_analyzer import TargetDistribution

        dist = TargetDistribution(value_counts={0: 80, 1: 20}, total=100)

        assert dist.get_label(1) == "Churned"
        assert dist.get_label(0) == "Retained"
        assert dist.get_label(2) == "2"


class TestAggregationEdgeCases:
    def test_first_aggregation(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.FIRST
        )

        assert "target_entity" in df_result.columns
        assert df_result["target_entity"].isin([0, 1]).all()

    def test_first_aggregation_no_time(self, event_level_data):
        from customer_retention.stages.profiling import AggregationMethod

        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", None, AggregationMethod.FIRST
        )

        assert "target_entity" in df_result.columns

    def test_last_aggregation_no_time(self, event_level_data):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", None, AggregationMethod.LAST
        )

        assert "target_entity" in df_result.columns
        assert "Warning" in " ".join(result.messages)


class TestTargetLevelAnalyzerPrintAnalysis:
    def test_print_analysis_entity_level(self, entity_level_data, capsys):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(entity_level_data, "target", "entity_id")
        analyzer.print_analysis(result)

        captured = capsys.readouterr()
        assert "TARGET LEVEL ANALYSIS" in captured.out
        assert "ENTITY_LEVEL" in captured.out

    def test_print_analysis_event_level(self, event_level_data, capsys):
        analyzer = TargetLevelAnalyzer()
        result = analyzer.detect_level(event_level_data, "target", "entity_id")
        analyzer.print_analysis(result)

        captured = capsys.readouterr()
        assert "TARGET LEVEL ANALYSIS" in captured.out
        assert "EVENT-LEVEL" in captured.out

    def test_print_analysis_with_aggregation(self, event_level_data, capsys):
        analyzer = TargetLevelAnalyzer()
        df_result, result = analyzer.aggregate_to_entity(
            event_level_data, "target", "entity_id", "timestamp", AggregationMethod.MAX
        )
        analyzer.print_analysis(result)

        captured = capsys.readouterr()
        assert "Aggregation applied" in captured.out


class TestSuggestAggregation:
    def test_suggest_max_for_non_binary(self):
        analyzer = TargetLevelAnalyzer()
        result = analyzer._suggest_aggregation({0: 50, 1: 30, 2: 20}, is_binary=False)
        assert result == AggregationMethod.MAX

    def test_suggest_max_for_skewed_binary(self):
        analyzer = TargetLevelAnalyzer()
        result = analyzer._suggest_aggregation({0: 90, 1: 10}, is_binary=True)
        assert result == AggregationMethod.MAX

    def test_suggest_max_for_balanced_binary(self):
        analyzer = TargetLevelAnalyzer()
        result = analyzer._suggest_aggregation({0: 50, 1: 50}, is_binary=True)
        assert result == AggregationMethod.MAX
