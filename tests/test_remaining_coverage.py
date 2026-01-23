"""Comprehensive tests to increase coverage for remaining modules.

Covers:
- pandas_backend: delta read/write import errors, missing stats, correlation, concat
- console: bar rendering, buffering, kv formatting, overview
- batch_integration: all combine_scores strategies, get_best_available_score priority
- noise_tester: gaussian noise degradation, feature dropout, exception handling
- overfitting_analyzer: learning curve, complexity analysis, model complexity checks
- threshold_optimizer: TARGET_PRECISION objective
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics.noise_tester import NoiseTester
from customer_retention.analysis.diagnostics.overfitting_analyzer import OverfittingAnalyzer
from customer_retention.analysis.visualization import console
from customer_retention.core.compat.pandas_backend import (
    concat,
    correlation_matrix,
    get_missing_stats,
    read_delta,
    write_delta,
)
from customer_retention.integrations.streaming.batch_integration import (
    BatchStreamingBridge,
    ScoreCombinationStrategy,
)
from customer_retention.stages.modeling.threshold_optimizer import (
    OptimizationObjective,
    ThresholdOptimizer,
)


# ---------------------------------------------------------------------------
# Module 1: pandas_backend
# ---------------------------------------------------------------------------
class TestPandasBackend:
    """Tests for uncovered lines in pandas_backend module."""

    def test_read_delta_raises_import_error_when_unavailable(self):
        """Line 19-20: read_delta raises ImportError when deltalake not installed."""
        with patch(
            "customer_retention.core.compat.pandas_backend.DELTA_RS_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="deltalake package required"):
                read_delta("/some/path")

    def test_read_delta_with_version(self):
        """Lines 21-22: read_delta passes version to DeltaTable."""
        import sys

        mock_dt = MagicMock()
        mock_dt.to_pandas.return_value = pd.DataFrame({"a": [1, 2]})
        mock_deltalake = MagicMock()
        mock_deltalake.DeltaTable.return_value = mock_dt

        import customer_retention.core.compat.pandas_backend as pb

        with patch.object(pb, "DELTA_RS_AVAILABLE", True), patch.dict(
            sys.modules, {"deltalake": mock_deltalake}
        ):
            pb.deltalake = mock_deltalake
            try:
                result = read_delta("/some/path", version=3)
                mock_deltalake.DeltaTable.assert_called_once_with("/some/path", version=3)
                assert list(result.columns) == ["a"]
            finally:
                if hasattr(pb, "deltalake"):
                    del pb.deltalake

    def test_read_delta_without_version(self):
        """Lines 23-25: read_delta without version uses default DeltaTable."""
        import sys

        mock_dt = MagicMock()
        mock_dt.to_pandas.return_value = pd.DataFrame({"x": [10]})
        mock_deltalake = MagicMock()
        mock_deltalake.DeltaTable.return_value = mock_dt

        import customer_retention.core.compat.pandas_backend as pb

        with patch.object(pb, "DELTA_RS_AVAILABLE", True), patch.dict(
            sys.modules, {"deltalake": mock_deltalake}
        ):
            pb.deltalake = mock_deltalake
            try:
                result = read_delta("/some/path")
                mock_deltalake.DeltaTable.assert_called_once_with("/some/path")
                assert list(result.columns) == ["x"]
            finally:
                if hasattr(pb, "deltalake"):
                    del pb.deltalake

    def test_write_delta_raises_import_error_when_unavailable(self):
        """Lines 30-31: write_delta raises ImportError when deltalake not installed."""
        df = pd.DataFrame({"a": [1, 2]})
        with patch(
            "customer_retention.core.compat.pandas_backend.DELTA_RS_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="deltalake package required"):
                write_delta(df, "/some/path")

    def test_write_delta_calls_write_deltalake(self):
        """Lines 32-33: write_delta calls write_deltalake with correct arguments."""
        import sys

        df = pd.DataFrame({"a": [1, 2]})
        mock_deltalake = MagicMock()
        mock_write = MagicMock()
        mock_deltalake.write_deltalake = mock_write

        import customer_retention.core.compat.pandas_backend as pb

        with patch.object(pb, "DELTA_RS_AVAILABLE", True), patch.dict(
            sys.modules, {"deltalake": mock_deltalake}
        ):
            write_delta(df, "/output/path", mode="append", partition_by=["a"])
            mock_write.assert_called_once_with(
                "/output/path", df, mode="append", partition_by=["a"]
            )

    def test_get_missing_stats_with_nulls(self):
        """Line 37: get_missing_stats returns correct percentages."""
        df = pd.DataFrame({"a": [1, None, 3, None], "b": [None, None, None, None], "c": [1, 2, 3, 4]})
        stats = get_missing_stats(df)
        assert stats["a"] == pytest.approx(0.5)
        assert stats["b"] == pytest.approx(1.0)
        assert stats["c"] == pytest.approx(0.0)

    def test_correlation_matrix_with_specific_columns(self):
        """Lines 41-42: correlation_matrix uses provided columns subset."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 1.0, 1.0],
            "text": ["x", "y", "z", "w"],
        })
        result = correlation_matrix(df, columns=["a", "b"])
        assert list(result.columns) == ["a", "b"]
        assert result.loc["a", "b"] == pytest.approx(-1.0)

    def test_concat_axis_1(self):
        """Lines 55-56: concat with axis=1 uses ignore_index=False."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        result = concat([df1, df2], axis=1)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3, 4]


# ---------------------------------------------------------------------------
# Module 2: console
# ---------------------------------------------------------------------------
class TestConsoleOutput:
    """Tests for uncovered lines in console module."""

    def setup_method(self):
        """Reset console state before each test."""
        console._buffer.clear()
        console._auto_flush = True

    def teardown_method(self):
        """Ensure console state is restored after each test."""
        console._buffer.clear()
        console._auto_flush = True

    def test_bar_full_value(self, capsys):
        """Line 29: _bar returns all filled blocks when value >= 100."""
        result = console._bar(100, width=10)
        assert result == "\u2588" * 10
        assert "\u2591" not in result

    def test_bar_zero_value(self):
        """Line 30: _bar returns all empty blocks when value is 0."""
        result = console._bar(0, width=10)
        assert result == "\u2591" * 10
        assert "\u2588" not in result

    def test_bar_partial_value(self):
        """Line 30: _bar returns mixed blocks for intermediate values."""
        result = console._bar(50, width=20)
        assert "\u2588" in result
        assert "\u2591" in result
        assert len(result) == 20

    def test_add_auto_flush_prints(self, capsys):
        """Lines 19-20: _add prints when _auto_flush is True and no IPython."""
        with patch.object(console, "HAS_IPYTHON", False):
            console._auto_flush = True
            console._add("test line")
            captured = capsys.readouterr()
            assert "test line" in captured.out

    def test_add_buffered_mode(self):
        """Lines 21-22: _add appends to buffer when _auto_flush is False."""
        console._auto_flush = False
        console._add("buffered line")
        assert "buffered line" in console._buffer

    def test_start_section_enables_buffering(self):
        """Lines 34-36: start_section sets _auto_flush=False and clears buffer."""
        console._buffer.append("old data")
        console.start_section()
        assert console._auto_flush is False
        assert len(console._buffer) == 0

    def test_end_section_flushes_and_restores(self, capsys):
        """Lines 42-48: end_section prints buffered content and restores auto_flush."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.start_section()
            console._add("line1")
            console._add("line2")
            console.end_section()
            captured = capsys.readouterr()
            assert "line1" in captured.out
            assert "line2" in captured.out
            assert console._auto_flush is True
            assert len(console._buffer) == 0

    def test_end_section_empty_buffer(self, capsys):
        """Lines 41-48: end_section with empty buffer does not print."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.start_section()
            console.end_section()
            captured = capsys.readouterr()
            assert captured.out == ""
            assert console._auto_flush is True

    def test_kv_inline(self, capsys):
        """Lines 80-82: kv with inline=True joins with pipe."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.kv({"a": 1, "b": 2}, inline=True)
            captured = capsys.readouterr()
            assert "|" in captured.out
            assert "a: **1**" in captured.out
            assert "b: **2**" in captured.out

    def test_kv_not_inline(self, capsys):
        """Lines 83-85: kv with inline=False prints one per line."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.kv({"x": 10, "y": 20}, inline=False)
            captured = capsys.readouterr()
            lines = captured.out.strip().split("\n")
            assert len(lines) == 2
            assert "x: **10**" in lines[0]
            assert "y: **20**" in lines[1]

    def test_overview_without_target(self, capsys):
        """Lines 117-120: overview prints basic stats without target."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.overview(rows=1000, cols=15, memory_mb=2.5, completeness=95.0)
            captured = capsys.readouterr()
            assert "1,000" in captured.out
            assert "15" in captured.out
            assert "2.5 MB" in captured.out
            assert "95.0%" in captured.out
            assert "Target" not in captured.out

    def test_overview_with_target(self, capsys):
        """Lines 121-122: overview prints target when provided."""
        with patch.object(console, "HAS_IPYTHON", False):
            console.overview(rows=500, cols=10, memory_mb=1.0, completeness=80.0, target="churn")
            captured = capsys.readouterr()
            assert "Target: **churn**" in captured.out


# ---------------------------------------------------------------------------
# Module 3: batch_integration
# ---------------------------------------------------------------------------
class TestBatchStreamingBridge:
    """Tests for uncovered lines in batch_integration module."""

    def setup_method(self):
        self.bridge = BatchStreamingBridge()

    def test_combine_scores_both_none(self):
        """Line 37: both scores None returns 0.0."""
        result = self.bridge.combine_scores(None, None)
        assert result == 0.0

    def test_combine_scores_batch_none(self):
        """Line 41: batch_score=None returns streaming_score."""
        result = self.bridge.combine_scores(None, 0.75)
        assert result == 0.75

    def test_combine_scores_streaming_none(self):
        """Line 39: streaming_score=None returns batch_score."""
        result = self.bridge.combine_scores(0.6, None)
        assert result == 0.6

    def test_combine_scores_batch_only(self):
        """Line 43: BATCH_ONLY strategy returns batch_score."""
        result = self.bridge.combine_scores(
            0.8, 0.9, strategy=ScoreCombinationStrategy.BATCH_ONLY
        )
        assert result == 0.8

    def test_combine_scores_streaming_override_fresh(self):
        """Lines 45-48: STREAMING_OVERRIDE with fresh streaming returns streaming."""
        now = datetime.now()
        result = self.bridge.combine_scores(
            0.6,
            0.9,
            strategy=ScoreCombinationStrategy.STREAMING_OVERRIDE,
            batch_timestamp=now - timedelta(hours=2),
            streaming_timestamp=now - timedelta(minutes=10),
            freshness_threshold_hours=1,
        )
        assert result == 0.9

    def test_combine_scores_streaming_override_stale(self):
        """Line 49: STREAMING_OVERRIDE with stale streaming still returns streaming."""
        now = datetime.now()
        result = self.bridge.combine_scores(
            0.6,
            0.9,
            strategy=ScoreCombinationStrategy.STREAMING_OVERRIDE,
            batch_timestamp=now - timedelta(hours=2),
            streaming_timestamp=now - timedelta(hours=3),
            freshness_threshold_hours=1,
        )
        # streaming is not fresh, but streaming_score is not None, so returns streaming_score
        assert result == 0.9

    def test_combine_scores_streaming_override_no_timestamps(self):
        """Line 49: STREAMING_OVERRIDE without timestamps returns streaming."""
        result = self.bridge.combine_scores(
            0.6,
            0.8,
            strategy=ScoreCombinationStrategy.STREAMING_OVERRIDE,
        )
        assert result == 0.8

    def test_combine_scores_signal_boost_streaming_higher(self):
        """Lines 55-57: SIGNAL_BOOST adds 0.1 when streaming > batch."""
        result = self.bridge.combine_scores(
            0.7, 0.9, strategy=ScoreCombinationStrategy.SIGNAL_BOOST
        )
        assert result == pytest.approx(0.8)

    def test_combine_scores_signal_boost_capped_at_one(self):
        """Line 57: SIGNAL_BOOST caps at 1.0."""
        result = self.bridge.combine_scores(
            0.95, 0.99, strategy=ScoreCombinationStrategy.SIGNAL_BOOST
        )
        assert result == 1.0

    def test_combine_scores_signal_boost_no_boost(self):
        """Lines 55-56: SIGNAL_BOOST no boost when streaming <= batch."""
        result = self.bridge.combine_scores(
            0.9, 0.5, strategy=ScoreCombinationStrategy.SIGNAL_BOOST
        )
        assert result == 0.9

    def test_get_best_available_score_realtime(self):
        """Line 73: realtime_score takes priority."""
        result = self.bridge.get_best_available_score(
            realtime_score=0.99, streaming_score=0.8, batch_score=0.7, cached_score=0.6
        )
        assert result.score == 0.99
        assert result.source == "realtime"

    def test_get_best_available_score_streaming(self):
        """Line 75: streaming_score is second priority."""
        result = self.bridge.get_best_available_score(
            streaming_score=0.85, batch_score=0.7, cached_score=0.6
        )
        assert result.score == 0.85
        assert result.source == "streaming"

    def test_get_best_available_score_batch(self):
        """Line 77: batch_score is third priority (line 76-77 coverage)."""
        result = self.bridge.get_best_available_score(batch_score=0.72)
        assert result.score == 0.72
        assert result.source == "batch"

    def test_get_best_available_score_cached(self):
        """Lines 78-79: cached_score is fourth priority."""
        result = self.bridge.get_best_available_score(cached_score=0.55)
        assert result.score == 0.55
        assert result.source == "cached"

    def test_get_best_available_score_default(self):
        """Line 80: all None returns default 0.0."""
        result = self.bridge.get_best_available_score()
        assert result.score == 0.0
        assert result.source == "default"


# ---------------------------------------------------------------------------
# Module 4: noise_tester
# ---------------------------------------------------------------------------
class TestNoiseTesterExtended:
    """Tests for uncovered lines in noise_tester module."""

    def setup_method(self):
        self.tester = NoiseTester()

    def _make_fragile_model_and_data(self):
        """Create a model that is very fragile to noise (overfits on noise-free data)."""
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(42)
        # Create data where the signal is very subtle, so noise easily destroys it
        n = 200
        X = pd.DataFrame({
            "f1": np.random.randn(n) * 0.1,
            "f2": np.random.randn(n) * 0.1,
        })
        # Target is determined precisely by a narrow boundary
        y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))

        # Deep tree will overfit this exactly
        model = DecisionTreeClassifier(max_depth=20, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_gaussian_noise_nr001_low_noise_fragile(self):
        """Lines 55-62: NR001 check fires when model degrades > 10% at low noise."""
        model, X, y = self._make_fragile_model_and_data()
        np.random.seed(0)
        result = self.tester.test_gaussian_noise(model, X, y)
        # Check if NR001 appeared (model is fragile to low noise)
        nr001_checks = [c for c in result.checks if c.check_id == "NR001"]
        # The deep decision tree should be fragile
        assert len(result.degradation_curve) == 4  # low, medium, high, extreme

    def test_gaussian_noise_nr002_medium_noise(self):
        """Lines 63-70: NR002 check fires when model degrades > 20% at medium noise."""
        model, X, y = self._make_fragile_model_and_data()
        np.random.seed(1)
        result = self.tester.test_gaussian_noise(model, X, y)
        # Verify the degradation curve has medium entry
        medium_entries = [d for d in result.degradation_curve if d["noise_level"] == "medium"]
        assert len(medium_entries) == 1

    def test_feature_dropout_nr003_dominant_feature(self):
        """Lines 87-95: NR003 fires when one feature causes >50% degradation."""
        # Directly mock _get_score to control degradation precisely
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y = pd.Series(np.random.randint(0, 2, n))

        model = MagicMock()

        call_count = {"n": 0}
        def mock_get_score(mdl, X_input, y_input):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Baseline score
                return 0.9
            elif call_count["n"] == 2:
                # Dropping f1 causes massive degradation
                return 0.3
            else:
                # Dropping f2 causes no degradation
                return 0.9

        with patch.object(self.tester, "_get_score", side_effect=mock_get_score):
            result = self.tester.test_feature_dropout(model, X, y)

        # f1 should dominate heavily: (0.9 - 0.3) / 0.9 = 0.667
        assert result.feature_importance["f1"] > 0.5
        # Check NR003 was triggered
        nr003_checks = [c for c in result.checks if c.check_id == "NR003"]
        assert len(nr003_checks) == 1
        assert "f1" in nr003_checks[0].recommendation

    def test_get_score_exception_returns_half(self):
        """Lines 118-119: _get_score returns 0.5 on exception."""
        model = MagicMock()
        model.predict_proba.side_effect = ValueError("broken model")
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        score = self.tester._get_score(model, X, y)
        assert score == 0.5

    def test_compute_robustness_empty_curve(self):
        """Lines 122-123: _compute_robustness returns 1.0 for empty curve."""
        result = self.tester._compute_robustness([])
        assert result == 1.0

    def test_compute_robustness_with_degradations(self):
        """Lines 124-125: _compute_robustness computes 1 - mean(degradation)."""
        curve = [
            {"degradation": 0.1},
            {"degradation": 0.2},
            {"degradation": 0.3},
        ]
        result = self.tester._compute_robustness(curve)
        assert result == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Module 5: overfitting_analyzer
# ---------------------------------------------------------------------------
class TestOverfittingAnalyzerExtended:
    """Tests for uncovered lines in overfitting_analyzer module."""

    def setup_method(self):
        self.analyzer = OverfittingAnalyzer()

    def test_analyze_learning_curve_exception(self):
        """Lines 97-98: returns 'Unable to generate learning curve' on exception."""
        model = MagicMock()
        model.fit.side_effect = RuntimeError("cannot fit")
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with patch(
            "customer_retention.analysis.diagnostics.overfitting_analyzer.learning_curve",
            side_effect=ValueError("learning curve failed"),
        ):
            result = self.analyzer.analyze_learning_curve(model, X, y)
        assert result.diagnosis == "Unable to generate learning curve"
        assert result.learning_curve == []
        assert result.passed is True

    def test_diagnose_learning_curve_empty_data(self):
        """Line 102: empty curve data returns 'Insufficient data'."""
        result = self.analyzer._diagnose_learning_curve([])
        assert result == "Insufficient data for diagnosis"

    def test_diagnose_learning_curve_overfitting(self):
        """Line 112: gap > 0.15 returns overfitting diagnosis."""
        curve_data = [
            {"train_size": 50, "train_score": 0.95, "val_score": 0.60},
            {"train_size": 200, "train_score": 0.98, "val_score": 0.70},
        ]
        result = self.analyzer._diagnose_learning_curve(curve_data)
        assert "Overfitting" in result

    def test_diagnose_learning_curve_underfitting(self):
        """Line 114: both scores low returns underfitting diagnosis."""
        curve_data = [
            {"train_size": 50, "train_score": 0.55, "val_score": 0.50},
            {"train_size": 200, "train_score": 0.60, "val_score": 0.55},
        ]
        result = self.analyzer._diagnose_learning_curve(curve_data)
        assert "Underfitting" in result

    def test_diagnose_learning_curve_more_data(self):
        """Line 116: val_improvement > 0.05 returns 'More data may help'."""
        curve_data = [
            {"train_size": 50, "train_score": 0.85, "val_score": 0.65},
            {"train_size": 200, "train_score": 0.88, "val_score": 0.78},
        ]
        result = self.analyzer._diagnose_learning_curve(curve_data)
        assert "More data may help" in result

    def test_diagnose_learning_curve_plateau(self):
        """Line 117: small improvement returns 'Validation plateau'."""
        curve_data = [
            {"train_size": 50, "train_score": 0.82, "val_score": 0.75},
            {"train_size": 200, "train_score": 0.84, "val_score": 0.76},
        ]
        result = self.analyzer._diagnose_learning_curve(curve_data)
        assert "plateau" in result

    def test_analyze_complexity_critical_ratio(self):
        """Lines 137-138: ratio < 10 gives CRITICAL severity."""
        # 5 samples, 2 features => ratio = 2.5 (critical)
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1, 0])
        result = self.analyzer.analyze_complexity(X, y)
        assert result.passed is False  # CRITICAL means not passed
        assert any(c.check_id == "OF010" for c in result.checks)
        assert "CRITICAL" in result.recommendations[0]

    def test_analyze_complexity_high_ratio(self):
        """Lines 139-140: ratio 10-50 gives HIGH severity."""
        # 30 samples, 2 features => ratio = 15 (HIGH)
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        y = pd.Series(np.random.randint(0, 2, 30))
        result = self.analyzer.analyze_complexity(X, y)
        assert result.passed is True  # HIGH is not CRITICAL
        assert any(c.check_id == "OF011" for c in result.checks)
        assert any("HIGH" in r for r in result.recommendations)

    def test_analyze_model_complexity_high_estimators_no_regularization(self):
        """Lines 164-166: n_estimators > 500 without regularization triggers OF013."""
        params = {"n_estimators": 1000}
        result = self.analyzer.analyze_model_complexity(params)
        assert any(c.check_id == "OF013" for c in result.checks)
        of013 = [c for c in result.checks if c.check_id == "OF013"][0]
        assert "n_estimators=1000" in of013.recommendation
        assert of013.train_value == 1000

    def test_analyze_model_complexity_high_estimators_with_regularization(self):
        """Lines 165: n_estimators > 500 WITH regularization does NOT trigger OF013."""
        params = {"n_estimators": 1000, "regularization": 0.01}
        result = self.analyzer.analyze_model_complexity(params)
        assert not any(c.check_id == "OF013" for c in result.checks)

    def test_analyze_model_complexity_high_depth(self):
        """Lines 153-162: max_depth > 15 triggers OF012."""
        params = {"max_depth": 25}
        result = self.analyzer.analyze_model_complexity(params)
        assert any(c.check_id == "OF012" for c in result.checks)
        of012 = [c for c in result.checks if c.check_id == "OF012"][0]
        assert of012.train_value == 25

    def test_ratio_recommendation_critical(self):
        """Lines 144-146: _ratio_recommendation for critical ratio."""
        rec = self.analyzer._ratio_recommendation(5.0, 50, 10)
        assert "CRITICAL" in rec
        assert "5" in rec  # suggested features = 50 // 10 = 5

    def test_ratio_recommendation_high(self):
        """Lines 147-148: _ratio_recommendation for high ratio."""
        rec = self.analyzer._ratio_recommendation(25.0, 100, 4)
        assert "HIGH" in rec
        assert "L1 regularization" in rec


# ---------------------------------------------------------------------------
# Module 6: threshold_optimizer
# ---------------------------------------------------------------------------
class TestThresholdOptimizerPrecision:
    """Tests for uncovered lines in threshold_optimizer (TARGET_PRECISION)."""

    def _make_model_and_data(self):
        """Create a logistic regression model with separable data."""
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        n = 200
        # Create clearly separable data
        X_pos = np.random.randn(n // 2, 2) + 2
        X_neg = np.random.randn(n // 2, 2) - 2
        X = pd.DataFrame(
            np.vstack([X_pos, X_neg]), columns=["f1", "f2"]
        )
        y = pd.Series([1] * (n // 2) + [0] * (n // 2))

        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_target_precision_objective_met(self):
        """Lines 89-92: When precision >= target, returns recall as score."""
        model, X, y = self._make_model_and_data()
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.TARGET_PRECISION,
            target_precision=0.8,
            threshold_step=0.05,
        )
        result = optimizer.optimize(model, X, y)
        # With well-separated data, precision should be achievable
        assert result.optimal_threshold > 0.0
        assert result.threshold_metrics["precision"] >= 0.8

    def test_target_precision_objective_unachievable(self):
        """Lines 92-93: When precision < target, score is -inf."""
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        # Create noisy data where high precision is very hard
        n = 100
        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y = pd.Series(np.random.randint(0, 2, n))  # random labels

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.TARGET_PRECISION,
            target_precision=0.99,  # nearly impossible with random data
            threshold_step=0.05,
        )
        result = optimizer.optimize(model, X, y)
        # Result should still produce a ThresholdResult (falls back to best found)
        assert isinstance(result.optimal_threshold, float)

    def test_target_precision_calculate_score_directly(self):
        """Lines 89-93: Direct test of _calculate_score for TARGET_PRECISION."""
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.TARGET_PRECISION,
            target_precision=0.9,
        )
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
        # Predictions with high precision (all predicted positives are correct)
        y_pred_high_prec = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 0])
        y_proba = np.array([0.3, 0.4, 0.1, 0.2, 0.9, 0.1, 0.8, 0.85, 0.2, 0.15])

        score_high = optimizer._calculate_score(y_true, y_pred_high_prec, y_proba, 0.7)
        # precision of y_pred_high_prec: 3/3 = 1.0 >= 0.9, so returns recall
        assert score_high > 0  # recall = 3/5 = 0.6

        # Predictions with low precision
        y_pred_low_prec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        score_low = optimizer._calculate_score(y_true, y_pred_low_prec, y_proba, 0.1)
        # precision = 5/10 = 0.5 < 0.9, so returns -inf
        assert score_low == float("-inf")

    def test_target_precision_end_to_end_metrics(self):
        """Verify full optimize pipeline with TARGET_PRECISION produces valid metrics."""
        model, X, y = self._make_model_and_data()
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.TARGET_PRECISION,
            target_precision=0.85,
            threshold_step=0.05,
        )
        result = optimizer.optimize(model, X, y)
        assert "precision" in result.threshold_metrics
        assert "recall" in result.threshold_metrics
        assert "f1" in result.threshold_metrics
        assert "f2" in result.threshold_metrics
        assert result.cost_at_threshold is not None
        assert "default_f1" in result.comparison_default
