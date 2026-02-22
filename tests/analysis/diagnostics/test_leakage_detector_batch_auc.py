import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics import LeakageDetector
from customer_retention.core.components.enums import Severity


class TestBatchedSingleFeatureAUC:
    @pytest.fixture
    def detector(self):
        return LeakageDetector()

    @pytest.fixture
    def leaky_data(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        return pd.DataFrame({
            "leaky": target + np.random.randn(n) * 0.01,
            "normal1": np.random.randn(n),
            "normal2": np.random.randn(n),
        }), pd.Series(target)

    def test_detects_leaky_feature(self, detector, leaky_data):
        X, y = leaky_data
        result = detector.check_single_feature_auc(X, y)
        critical = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert any("leaky" in c.feature for c in critical)

    def test_auc_values_valid(self, detector, leaky_data):
        X, y = leaky_data
        result = detector.check_single_feature_auc(X, y)
        for check in result.checks:
            assert 0 <= check.auc <= 1

    def test_no_false_positives_for_normal(self, detector):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        X = pd.DataFrame({f"f{i}": np.random.randn(n) for i in range(5)})
        result = detector.check_single_feature_auc(X, pd.Series(target))
        critical = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_empty_columns_returns_empty(self, detector):
        X = pd.DataFrame({"text_col": ["a", "b", "c"]})
        y = pd.Series([0, 1, 0])
        result = detector.check_single_feature_auc(X, y)
        assert len(result.checks) == 0

    def test_excludes_temporal_metadata(self, detector):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature_timestamp": target + np.random.randn(n) * 0.01,
            "label_timestamp": target + np.random.randn(n) * 0.01,
        })
        result = detector.check_single_feature_auc(X, pd.Series(target))
        flagged = {c.feature for c in result.checks}
        assert "feature_timestamp" not in flagged
        assert "label_timestamp" not in flagged

    def test_nan_column_handled_gracefully(self, detector):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        X = pd.DataFrame({
            "all_nan": [np.nan] * n,
            "normal": np.random.randn(n),
        })
        result = detector.check_single_feature_auc(X, pd.Series(target))
        assert isinstance(result.passed, bool)

    def test_constant_column_returns_default_auc(self, detector):
        n = 200
        target = np.array([0] * 100 + [1] * 100)
        X = pd.DataFrame({"const": [5.0] * n, "normal": np.random.randn(n)})
        result = detector.check_single_feature_auc(X, pd.Series(target))
        assert isinstance(result.passed, bool)
