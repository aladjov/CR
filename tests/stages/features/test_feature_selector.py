import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import FeatureSelector, SelectionMethod


class TestVarianceSelection:
    @pytest.fixture
    def df_with_low_variance(self):
        np.random.seed(42)
        return pd.DataFrame({
            "constant": [1.0] * 100,
            "near_constant": [1.0] * 99 + [2.0],
            "normal_var": np.random.randn(100),
            "high_var": np.random.randn(100) * 10,
            "target": np.random.choice([0, 1], 100)
        })

    def test_removes_constant_features(self, df_with_low_variance):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(df_with_low_variance)

        assert "constant" not in result.selected_features
        assert "normal_var" in result.selected_features
        assert "high_var" in result.selected_features

    def test_variance_threshold_configurable(self, df_with_low_variance):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.5,
            target_column="target"
        )
        result = selector.fit_transform(df_with_low_variance)

        assert "constant" not in result.selected_features
        assert "near_constant" not in result.selected_features


class TestCorrelationSelection:
    @pytest.fixture
    def df_with_correlation(self):
        np.random.seed(42)
        base = np.random.randn(100)
        return pd.DataFrame({
            "feature1": base,
            "feature2": base + np.random.randn(100) * 0.01,  # highly correlated
            "feature3": base + np.random.randn(100) * 0.1,   # correlated
            "feature4": np.random.randn(100),  # independent
            "target": np.random.choice([0, 1], 100)
        })

    def test_removes_highly_correlated_features(self, df_with_correlation):
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION,
            correlation_threshold=0.95,
            target_column="target"
        )
        result = selector.fit_transform(df_with_correlation)

        # Either feature1 or feature2 should be dropped
        correlated_pair = {"feature1", "feature2"}
        selected_set = set(result.selected_features)
        dropped_correlated = correlated_pair - selected_set

        assert len(dropped_correlated) >= 1

    def test_correlation_threshold_configurable(self, df_with_correlation):
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION,
            correlation_threshold=0.80,
            target_column="target"
        )
        result = selector.fit_transform(df_with_correlation)

        # With lower threshold, more features should be dropped
        assert len(result.dropped_features) >= 1


class TestPreserveFeatures:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "constant": [1.0] * 100,
            "important": np.random.randn(100),
            "normal": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

    def test_preserves_specified_features(self, sample_df):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target",
            preserve_features=["constant"]
        )
        result = selector.fit_transform(sample_df)

        # constant should be preserved even though it has zero variance
        assert "constant" in result.selected_features


class TestSelectionResult:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "constant": [1.0] * 100,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

    def test_result_contains_selected_features(self, sample_df):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(sample_df)

        assert hasattr(result, "selected_features")
        assert isinstance(result.selected_features, list)

    def test_result_contains_dropped_features(self, sample_df):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(sample_df)

        assert hasattr(result, "dropped_features")
        assert "constant" in result.dropped_features

    def test_result_contains_drop_reasons(self, sample_df):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(sample_df)

        assert hasattr(result, "drop_reasons")
        assert "constant" in result.drop_reasons

    def test_result_contains_dataframe(self, sample_df):
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(sample_df)

        assert hasattr(result, "df")
        assert "constant" not in result.df.columns
        assert "feature1" in result.df.columns


class TestFitTransformSeparation:
    def test_fit_then_transform(self):
        np.random.seed(42)
        train = pd.DataFrame({
            "constant": [1.0] * 100,
            "feature1": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })
        test = pd.DataFrame({
            "constant": [1.0] * 50,
            "feature1": np.random.randn(50),
            "target": np.random.choice([0, 1], 50)
        })

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        selector.fit(train)
        result = selector.transform(test)

        assert "constant" not in result.df.columns
        assert "feature1" in result.df.columns


class TestMaxFeatures:
    def test_max_features_limits_output(self):
        np.random.seed(42)
        df = pd.DataFrame({
            f"feature{i}": np.random.randn(100) * (i + 1)
            for i in range(10)
        })
        df["target"] = np.random.choice([0, 1], 100)

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.0,
            target_column="target",
            max_features=5
        )
        result = selector.fit_transform(df)

        assert len(result.selected_features) <= 5


class TestSelectionMethod:
    def test_all_methods_exist(self):
        expected_methods = [
            "VARIANCE", "CORRELATION", "MUTUAL_INFO",
            "IMPORTANCE", "RECURSIVE", "L1_SELECTION"
        ]
        for method in expected_methods:
            assert hasattr(SelectionMethod, method)


class TestEdgeCases:
    def test_handles_single_feature(self):
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(df)

        assert "feature1" in result.selected_features

    def test_handles_all_constant_features(self):
        df = pd.DataFrame({
            "const1": [1.0] * 100,
            "const2": [2.0] * 100,
            "target": np.random.choice([0, 1], 100)
        })

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        result = selector.fit_transform(df)

        assert len(result.selected_features) == 0

    def test_handles_null_values(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })
        df.loc[0, "feature1"] = np.nan

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target"
        )
        # Should not raise error
        result = selector.fit_transform(df)
        assert result.df is not None


class TestCombinedSelection:
    def test_variance_then_correlation(self):
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            "constant": [1.0] * 100,
            "feature1": base,
            "feature2": base + np.random.randn(100) * 0.01,
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            correlation_threshold=0.95,
            target_column="target",
            apply_correlation_filter=True
        )
        result = selector.fit_transform(df)

        # constant should be dropped (variance)
        assert "constant" not in result.selected_features
        # One of feature1/feature2 should be dropped (correlation)
        assert len(set(result.selected_features) & {"feature1", "feature2"}) <= 1
