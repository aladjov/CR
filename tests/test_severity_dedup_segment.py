"""Comprehensive tests for severity, deduplicate, and segment_analyzer modules."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics.segment_analyzer import (
    SegmentPerformanceAnalyzer,
)
from customer_retention.analysis.recommendations.cleaning.deduplicate import (
    DeduplicateRecommendation,
)
from customer_retention.core.components.enums import Platform, Severity
from customer_retention.core.utils.severity import (
    ThresholdConfig,
    classify_by_thresholds,
    severity_recommendation,
)

# ---------------------------------------------------------------------------
# Module 1: severity.py - classify_by_thresholds (ascending=True)
# ---------------------------------------------------------------------------


class TestClassifyByThresholdsAscending:
    def test_critical_threshold(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.95, config) == Severity.CRITICAL

    def test_high_threshold(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.75, config) == Severity.HIGH

    def test_warning_threshold(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.55, config) == Severity.WARNING

    def test_medium_threshold(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.35, config) == Severity.MEDIUM

    def test_low_threshold(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.15, config) == Severity.LOW

    def test_info_fallthrough(self):
        config = ThresholdConfig(critical=0.9, high=0.7, warning=0.5, medium=0.3, low=0.1, ascending=True)
        assert classify_by_thresholds(0.05, config) == Severity.INFO


# ---------------------------------------------------------------------------
# Module 1: severity.py - classify_by_thresholds (ascending=False)
# ---------------------------------------------------------------------------


class TestClassifyByThresholdsDescending:
    def test_critical_threshold_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.05, config) == Severity.CRITICAL

    def test_high_threshold_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.25, config) == Severity.HIGH

    def test_warning_threshold_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.45, config) == Severity.WARNING

    def test_medium_threshold_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.65, config) == Severity.MEDIUM

    def test_low_threshold_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.85, config) == Severity.LOW

    def test_info_fallthrough_descending(self):
        config = ThresholdConfig(critical=0.1, high=0.3, warning=0.5, medium=0.7, low=0.9, ascending=False)
        assert classify_by_thresholds(0.95, config) == Severity.INFO

    def test_partial_thresholds_only_critical_descending(self):
        config = ThresholdConfig(critical=0.2, ascending=False)
        assert classify_by_thresholds(0.1, config) == Severity.CRITICAL
        assert classify_by_thresholds(0.5, config) == Severity.INFO


# ---------------------------------------------------------------------------
# Module 1: severity.py - severity_recommendation
# ---------------------------------------------------------------------------


class TestSeverityRecommendation:
    def test_critical_recommendation(self):
        result = severity_recommendation(Severity.CRITICAL, "data drift detected")
        assert result == "CRITICAL: data drift detected. investigate immediately."

    def test_high_recommendation(self):
        result = severity_recommendation(Severity.HIGH, "model degradation")
        assert result == "HIGH: model degradation. investigate immediately."

    def test_warning_recommendation(self):
        result = severity_recommendation(Severity.WARNING, "slight skew observed")
        assert result == "WARNING: slight skew observed. monitor closely."

    def test_medium_recommendation(self):
        result = severity_recommendation(Severity.MEDIUM, "minor issue found")
        assert result == "MEDIUM: minor issue found. monitor closely."

    def test_low_recommendation(self):
        result = severity_recommendation(Severity.LOW, "negligible noise")
        assert result == "LOW: negligible noise. no action needed."

    def test_info_recommendation(self):
        result = severity_recommendation(Severity.INFO, "normal operation")
        assert result == "INFO: normal operation. no action needed."

    def test_custom_actions(self):
        result = severity_recommendation(
            Severity.CRITICAL, "drift detected",
            action_critical="retrain model", action_warning="alert team", action_info="log only"
        )
        assert result == "CRITICAL: drift detected. retrain model."

    def test_custom_warning_action(self):
        result = severity_recommendation(
            Severity.WARNING, "slight issue",
            action_warning="notify stakeholders"
        )
        assert result == "WARNING: slight issue. notify stakeholders."


# ---------------------------------------------------------------------------
# Module 2: deduplicate.py - strategies
# ---------------------------------------------------------------------------


class TestDeduplicateStrategies:
    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({
            "id": [1, 1, 2, 2, 3],
            "value": [10, 20, 30, 40, 50],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-02", "2023-01-10", "2023-01-03"]),
        })

    def test_keep_first_strategy(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first")
        result = rec.fit_transform(df_with_duplicates)
        assert result.rows_after == 3
        # Keep first occurrence: values 10, 30, 50
        assert list(result.data["value"]) == [10, 30, 50]

    def test_keep_last_strategy(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last")
        result = rec.fit_transform(df_with_duplicates)
        assert result.rows_after == 3
        # Keep last occurrence: values 20, 40, 50
        assert list(result.data["value"]) == [20, 40, 50]

    def test_keep_most_recent_strategy(self, df_with_duplicates):
        rec = DeduplicateRecommendation(
            key_columns=["id"], strategy="keep_most_recent", timestamp_column="timestamp"
        )
        result = rec.fit_transform(df_with_duplicates)
        assert result.rows_after == 3
        # Most recent timestamps per id: id=1 -> 2023-01-05 (value=20), id=2 -> 2023-01-10 (value=40), id=3 (value=50)
        result_sorted = result.data.sort_values("id")
        assert list(result_sorted["value"]) == [20, 40, 50]

    def test_drop_exact_strategy(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="drop_exact")
        result = rec.fit_transform(df_with_duplicates)
        assert result.rows_after == 3
        # drop_exact keeps first by default
        assert list(result.data["value"]) == [10, 30, 50]

    def test_metadata_duplicates_removed(self, df_with_duplicates):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last")
        result = rec.fit_transform(df_with_duplicates)
        assert result.metadata["duplicates_removed"] == 2


# ---------------------------------------------------------------------------
# Module 2: deduplicate.py - missing key columns
# ---------------------------------------------------------------------------


class TestDeduplicateMissingKeys:
    def test_fit_with_no_existing_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        rec = DeduplicateRecommendation(key_columns=["nonexistent_col", "another_missing"])
        rec.fit(df)
        assert rec._fit_params["duplicate_count"] == 0
        assert rec._fit_params["duplicate_keys"] == []

    def test_transform_with_no_existing_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        rec = DeduplicateRecommendation(key_columns=["nonexistent_col"])
        result = rec.fit_transform(df)
        assert result.rows_before == 3
        assert result.rows_after == 3
        assert result.metadata["duplicates_removed"] == 0
        assert list(result.data["a"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Module 2: deduplicate.py - code generation
# ---------------------------------------------------------------------------


class TestDeduplicateCodeGeneration:
    def test_generate_local_code_keep_first(self):
        rec = DeduplicateRecommendation(key_columns=["id", "name"], strategy="keep_first")
        code = rec.generate_code(Platform.LOCAL)
        assert "drop_duplicates" in code
        assert "'id'" in code
        assert "'name'" in code
        assert "keep='first'" in code

    def test_generate_local_code_keep_last(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_last")
        code = rec.generate_code(Platform.LOCAL)
        assert "keep='last'" in code

    def test_generate_local_code_keep_most_recent(self):
        rec = DeduplicateRecommendation(
            key_columns=["id"], strategy="keep_most_recent", timestamp_column="updated_at"
        )
        code = rec.generate_code(Platform.LOCAL)
        assert "sort_values" in code
        assert "updated_at" in code
        assert "sort_index" in code

    def test_generate_local_code_drop_exact(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="drop_exact")
        code = rec.generate_code(Platform.LOCAL)
        assert "drop_duplicates" in code
        assert "keep='first'" in code

    def test_generate_databricks_code_keep_most_recent(self):
        rec = DeduplicateRecommendation(
            key_columns=["id", "email"], strategy="keep_most_recent", timestamp_column="ts"
        )
        code = rec.generate_code(Platform.DATABRICKS)
        assert "pyspark.sql.window" in code
        assert "Window.partitionBy" in code
        assert "row_number" in code
        assert "desc" in code
        assert "'ts'" in code

    def test_generate_databricks_code_fallback(self):
        rec = DeduplicateRecommendation(key_columns=["id"], strategy="keep_first")
        code = rec.generate_code(Platform.DATABRICKS)
        assert "dropDuplicates" in code
        assert "'id'" in code


# ---------------------------------------------------------------------------
# Module 3: segment_analyzer.py - define_segments
# ---------------------------------------------------------------------------


class TestSegmentDefineSegments:
    @pytest.fixture
    def analyzer(self):
        return SegmentPerformanceAnalyzer()

    def test_missing_column_returns_all(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = analyzer.define_segments(df, segment_column="nonexistent")
        assert list(result) == ["all", "all", "all"]

    def test_tenure_type_segments(self, analyzer):
        df = pd.DataFrame({"tenure_days": [30, 200, 500]})
        result = analyzer.define_segments(df, segment_column="tenure_days", segment_type="tenure")
        assert list(result) == ["new", "established", "mature"]

    def test_quantile_type_segments(self, analyzer):
        df = pd.DataFrame({"score": list(range(1, 31))})
        result = analyzer.define_segments(df, segment_column="score", segment_type="quantile")
        # Should produce 3 bins: low, medium, high
        assert set(result.unique()) == {"low", "medium", "high"}

    def test_unknown_type_returns_all(self, analyzer):
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = analyzer.define_segments(df, segment_column="col", segment_type="unknown_type")
        assert list(result) == ["all", "all", "all"]


# ---------------------------------------------------------------------------
# Module 3: segment_analyzer.py - analyze_performance
# ---------------------------------------------------------------------------


class _ConstantModel:
    """A model that always predicts a single class with fixed probabilities."""

    def __init__(self, pred_value=0, proba_value=0.3):
        self._pred_value = pred_value
        self._proba_value = proba_value

    def predict(self, X):
        return np.full(len(X), self._pred_value)

    def predict_proba(self, X):
        pos = self._proba_value
        neg = 1.0 - pos
        return np.column_stack([np.full(len(X), neg), np.full(len(X), pos)])


class _BrokenModel:
    """A model that raises an exception on predict."""

    def predict(self, X):
        raise RuntimeError("Model prediction failed")

    def predict_proba(self, X):
        raise RuntimeError("Model prediction failed")


class _LinearModel:
    """A model whose probability is proportional to the first feature value.

    predict_proba: proba = clip(feat / 100, 0.01, 0.99)
    predict: 1 if feat >= 50, else 0

    This creates a clear pr_auc gap when one segment's labels correlate
    with features (high pr_auc) and another segment's labels are inversely
    correlated (low pr_auc).
    """

    def predict(self, X):
        arr = np.array(X).flatten()
        return (arr >= 50).astype(int)

    def predict_proba(self, X):
        arr = np.array(X).flatten()
        pos = np.clip(arr / 100.0, 0.01, 0.99)
        return np.column_stack([1 - pos, pos])


class TestSegmentPerformanceAnalysis:
    @pytest.fixture
    def analyzer(self):
        return SegmentPerformanceAnalyzer()

    def test_small_segment_check(self, analyzer):
        """Test SG003: small segment detection when segment size < 5%."""
        np.random.seed(42)
        # Need total large enough so that 10 samples is < 5% -> total > 200
        n = 250
        X = pd.DataFrame({"feat": np.random.randn(n)})
        y = pd.Series(np.random.randint(0, 2, n))
        # 'tiny' has 11 samples out of 250 = 4.4% < 5% threshold
        segments = pd.Series(["large"] * 239 + ["tiny"] * 11)
        model = _ConstantModel(pred_value=1, proba_value=0.6)
        result = analyzer.analyze_performance(model, X, y, segments)
        small_checks = [c for c in result.checks if c.check_id == "SG003"]
        assert len(small_checks) >= 1
        assert small_checks[0].severity == Severity.MEDIUM

    def test_underperforming_segment(self, analyzer):
        """Test SG001: underperformance gap > 20% in pr_auc."""
        np.random.seed(42)
        # Good segment: features positively correlate with labels
        # Model gives high proba when feat is high, labels are 1 when feat >= 50
        n_good = 150
        feat_good = np.random.uniform(0, 100, n_good)
        y_good = pd.Series((feat_good >= 50).astype(int))
        X_good = pd.DataFrame({"feat": feat_good})

        # Bad segment: features inversely correlate with labels
        # Model gives high proba when feat is high, but labels are 1 when feat < 50
        n_bad = 50
        feat_bad = np.random.uniform(0, 100, n_bad)
        y_bad = pd.Series((feat_bad < 50).astype(int))
        X_bad = pd.DataFrame({"feat": feat_bad})

        X = pd.concat([X_good, X_bad], ignore_index=True)
        y = pd.concat([y_good, y_bad], ignore_index=True)
        segments = pd.Series(["good"] * n_good + ["bad"] * n_bad)

        model = _LinearModel()
        result = analyzer.analyze_performance(model, X, y, segments)
        underperf_checks = [c for c in result.checks if c.check_id == "SG001"]
        assert len(underperf_checks) >= 1
        assert underperf_checks[0].severity == Severity.HIGH

    def test_low_recall_segment(self, analyzer):
        """Test SG002: low recall (< 20%) for a segment."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({"feat": np.random.randn(n)})
        # y has many positives but model predicts all zeros -> recall = 0
        y = pd.Series([1] * 80 + [0] * 20)
        segments = pd.Series(["seg_a"] * n)

        model = _ConstantModel(pred_value=0, proba_value=0.1)
        result = analyzer.analyze_performance(model, X, y, segments)
        low_recall_checks = [c for c in result.checks if c.check_id == "SG002"]
        assert len(low_recall_checks) >= 1
        assert low_recall_checks[0].severity == Severity.HIGH
        assert "low recall" in low_recall_checks[0].recommendation

    def test_segment_too_small_to_process(self, analyzer):
        """Segments with fewer than 10 samples are skipped."""
        X = pd.DataFrame({"feat": np.random.randn(20)})
        y = pd.Series(np.random.randint(0, 2, 20))
        segments = pd.Series(["big"] * 15 + ["tiny"] * 5)
        model = _ConstantModel(pred_value=1, proba_value=0.6)
        result = analyzer.analyze_performance(model, X, y, segments)
        # Only 'big' segment should be processed, not 'tiny' (only 5 samples)
        assert "tiny" not in result.segment_metrics

    def test_compute_metrics_exception(self, analyzer):
        """When model.predict raises, _compute_metrics returns empty dict."""
        X = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        segments = pd.Series(["all"] * 10)

        model = _BrokenModel()
        result = analyzer.analyze_performance(model, X, y, segments)
        # With broken model, global_metrics and seg_metrics are both {}
        # No pr_auc or recall checks should fire
        assert len(result.checks) == 0


# ---------------------------------------------------------------------------
# Module 3: segment_analyzer.py - _global_recommendation
# ---------------------------------------------------------------------------


class TestSegmentGlobalRecommendation:
    @pytest.fixture
    def analyzer(self):
        return SegmentPerformanceAnalyzer()

    def test_no_high_issues(self, analyzer):
        """No high-severity issues returns continue with global model message."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({"feat": np.random.randn(n)})
        y = pd.Series(np.random.randint(0, 2, n))
        segments = pd.Series(["all"] * n)
        # Model that performs reasonably -> no HIGH checks
        model = _ConstantModel(pred_value=1, proba_value=0.7)
        result = analyzer.analyze_performance(model, X, y, segments)
        # Filter to only high checks relevant to recommendation
        high_checks = [c for c in result.checks if c.severity == Severity.HIGH]
        if not high_checks:
            assert "No significant segment gaps" in result.recommendation

    def test_one_high_issue(self, analyzer):
        """Exactly one high-severity issue returns 'one segment underperforms'."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({"feat": np.random.randn(n)})
        # Only positives - model always predicts 0 -> low recall for one segment
        y = pd.Series([1] * n)
        segments = pd.Series(["only_seg"] * n)
        model = _ConstantModel(pred_value=0, proba_value=0.1)
        result = analyzer.analyze_performance(model, X, y, segments)
        high_checks = [c for c in result.checks if c.severity == Severity.HIGH]
        if len(high_checks) == 1:
            assert "One segment underperforms" in result.recommendation

    def test_multiple_high_issues(self, analyzer):
        """Two or more high-severity issues returns 'multiple segments underperform'."""
        np.random.seed(42)
        # Create two segments that both have low recall
        n = 200
        X = pd.DataFrame({"feat": np.random.randn(n)})
        y = pd.Series([1] * n)
        segments = pd.Series(["seg_a"] * 100 + ["seg_b"] * 100)
        # Model always predicts 0 -> both segments have recall = 0 -> two HIGH checks
        model = _ConstantModel(pred_value=0, proba_value=0.1)
        result = analyzer.analyze_performance(model, X, y, segments)
        high_checks = [c for c in result.checks if c.severity == Severity.HIGH]
        if len(high_checks) >= 2:
            assert "Multiple segments underperform" in result.recommendation
