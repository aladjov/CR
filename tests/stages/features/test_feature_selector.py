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


class TestBatchedCorrelationSelection:
    def test_correlation_selection_uses_batched_corr(self, monkeypatch):
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            "feature1": base,
            "feature2": base + np.random.randn(100) * 0.01,
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })

        calls = []
        original_batched = None

        import customer_retention.stages.features.feature_selector as fs_mod
        original_batched = fs_mod.batched_corr_matrix

        def tracking_batched(*args, **kwargs):
            calls.append(1)
            return original_batched(*args, **kwargs)

        monkeypatch.setattr(fs_mod, "batched_corr_matrix", tracking_batched)

        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION,
            correlation_threshold=0.95,
            target_column="target",
        )
        selector.fit_transform(df)
        assert len(calls) >= 1

    def test_correlation_selection_drops_same_features(self):
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            "feature1": base,
            "feature2": base + np.random.randn(100) * 0.01,
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION,
            correlation_threshold=0.95,
            target_column="target",
        )
        result = selector.fit_transform(df)
        correlated_pair = {"feature1", "feature2"}
        assert len(correlated_pair - set(result.selected_features)) >= 1


class TestBatchedVarianceSelection:
    def test_variance_selection_batched(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "constant": [1.0] * 100,
            "low_var": [1.0] * 99 + [1.001],
            "normal": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="target",
        )
        result = selector.fit_transform(df)
        assert "constant" not in result.selected_features
        assert "low_var" not in result.selected_features
        assert "normal" in result.selected_features


class TestL1Selection:
    @pytest.fixture
    def df_with_target_signal(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        return pd.DataFrame({
            "relevant1": target * 2.0 + np.random.randn(n) * 0.1,
            "relevant2": target * 1.5 + np.random.randn(n) * 0.3,
            "noise1": np.random.randn(n),
            "noise2": np.random.randn(n),
            "noise3": np.random.randn(n),
            "target": target,
        })

    def test_l1_drops_irrelevant_features(self, df_with_target_signal):
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df_with_target_signal)
        assert len(result.dropped_features) > 0
        relevant_kept = {"relevant1", "relevant2"} & set(result.selected_features)
        assert len(relevant_kept) >= 1

    def test_l1_keeps_strongest_signal(self, df_with_target_signal):
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df_with_target_signal)
        assert "relevant1" in result.selected_features

    def test_l1_populates_importance_scores(self, df_with_target_signal):
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df_with_target_signal)
        assert result.importance_scores is not None
        assert isinstance(result.importance_scores, dict)
        assert "relevant1" in result.importance_scores
        assert result.importance_scores["relevant1"] > 0

    def test_l1_preserves_preserved_features(self, df_with_target_signal):
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
            preserve_features=["noise1"],
        )
        result = selector.fit_transform(df_with_target_signal)
        assert "noise1" in result.selected_features

    def test_l1_handles_multiclass(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1, 2], n)
        df = pd.DataFrame({
            "signal": target * 3.0 + np.random.randn(n) * 0.1,
            "noise": np.random.randn(n),
            "target": target,
        })
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df)
        assert "signal" in result.selected_features
        assert result.importance_scores is not None
        assert result.importance_scores["signal"] > result.importance_scores.get("noise", 0)

    def test_l1_requires_target_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        selector = FeatureSelector(method=SelectionMethod.L1_SELECTION)
        with pytest.raises(ValueError, match="target_column"):
            selector.fit(df)

    def test_l1_handles_nan_in_features(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "signal": target * 2.0 + np.random.randn(n) * 0.1,
            "with_nan": np.random.randn(n),
            "target": target,
        })
        df.loc[0:5, "with_nan"] = np.nan
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df)
        assert result.df is not None

    def test_l1_raises_on_insufficient_rows(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        with pytest.raises(ValueError, match="at least 10 rows"):
            selector.fit(df)

    def test_l1_drop_reasons_mention_l1(self, df_with_target_signal):
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df_with_target_signal)
        for reason in result.drop_reasons.values():
            assert "l1" in reason.lower() or "zero" in reason.lower()


class TestRunSelectionPipeline:
    @pytest.fixture
    def df_mixed(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        base = np.random.randn(n)
        return pd.DataFrame({
            "constant": [1.0] * n,
            "signal": target * 2.0 + np.random.randn(n) * 0.1,
            "corr1": base,
            "corr2": base + np.random.randn(n) * 0.01,
            "noise1": np.random.randn(n),
            "noise2": np.random.randn(n),
            "target": target,
        })

    def test_variance_and_correlation_only(self, df_mixed):
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        result = run_selection_pipeline(
            df_mixed, target_column="target",
            variance_threshold=0.01, correlation_threshold=0.95,
            l1_enabled=False,
        )
        assert "constant" in result.dropped_features
        corr_pair = {"corr1", "corr2"}
        assert len(corr_pair - set(result.selected_features)) >= 1
        assert result.method_used == SelectionMethod.CORRELATION

    def test_full_pipeline_with_l1(self, df_mixed):
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        result = run_selection_pipeline(
            df_mixed, target_column="target",
            variance_threshold=0.01, correlation_threshold=0.95,
            l1_enabled=True,
        )
        assert "constant" in result.dropped_features
        assert "signal" in result.selected_features
        assert result.method_used == SelectionMethod.L1_SELECTION

    def test_merges_drop_reasons_across_stages(self, df_mixed):
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        result = run_selection_pipeline(
            df_mixed, target_column="target",
            variance_threshold=0.01, correlation_threshold=0.95,
            l1_enabled=True,
        )
        reasons = set(result.drop_reasons.values())
        has_variance = any("variance" in r.lower() for r in reasons)
        assert has_variance
        assert len(result.drop_reasons) == len(result.dropped_features)

    def test_max_features_respected(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            f"f{i}": target * (i + 1) + np.random.randn(n) * 0.5
            for i in range(20)
        })
        df["target"] = target
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        result = run_selection_pipeline(
            df, target_column="target",
            l1_enabled=True, max_features=5,
        )
        assert len(result.selected_features) <= 5

    def test_preserves_target_column_in_output(self, df_mixed):
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        result = run_selection_pipeline(
            df_mixed, target_column="target",
            l1_enabled=True,
        )
        assert "target" in result.df.columns
        assert "target" not in result.dropped_features


class TestDistributedL1Selection:
    def test_spark_l1_returns_feature_selection_result(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_l1_selection
        feature_cols = ["f1", "f2", "f3"]
        mock_spark_df = MagicMock()
        mock_model = MagicMock()
        mock_model.coefficients.toArray.return_value = np.array([0.5, 0.0, 0.3])
        mock_lr_class = MagicMock(return_value=MagicMock(fit=MagicMock(return_value=mock_model)))
        mock_assembler = MagicMock()
        mock_assembler_inst = MagicMock()
        mock_assembler_inst.transform.return_value = mock_spark_df
        mock_assembler.return_value = mock_assembler_inst
        mock_scaler_class = MagicMock()
        mock_scaler_model = MagicMock()
        mock_scaler_model.transform.return_value = mock_spark_df
        mock_scaler_class.return_value = MagicMock(fit=MagicMock(return_value=mock_scaler_model))
        with patch("customer_retention.stages.features.feature_selector._import_spark_ml") as mock_imports:
            mock_imports.return_value = (mock_lr_class, mock_assembler, mock_scaler_class, MagicMock())
            dropped, reasons, scores = _spark_l1_selection(mock_spark_df, "target", feature_cols)
        assert "f2" in dropped
        assert "f1" not in dropped
        assert "f3" not in dropped
        assert reasons["f2"] == "L1 zero coefficient"
        assert scores["f1"] == 0.5

    def test_spark_l1_all_nonzero_drops_nothing(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_l1_selection
        feature_cols = ["a", "b"]
        mock_spark_df = MagicMock()
        mock_model = MagicMock()
        mock_model.coefficients.toArray.return_value = np.array([1.0, 0.5])
        mock_lr_class = MagicMock(return_value=MagicMock(fit=MagicMock(return_value=mock_model)))
        mock_assembler = MagicMock()
        mock_assembler.return_value = MagicMock(transform=MagicMock(return_value=mock_spark_df))
        mock_scaler_class = MagicMock()
        mock_scaler_class.return_value = MagicMock(fit=MagicMock(return_value=MagicMock(transform=MagicMock(return_value=mock_spark_df))))
        with patch("customer_retention.stages.features.feature_selector._import_spark_ml") as mock_imports:
            mock_imports.return_value = (mock_lr_class, mock_assembler, mock_scaler_class, MagicMock())
            dropped, reasons, scores = _spark_l1_selection(mock_spark_df, "target", feature_cols)
        assert dropped == []
        assert reasons == {}


    def test_spark_l1_all_zero_drops_nothing(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_l1_selection
        feature_cols = ["a", "b", "c"]
        mock_spark_df = MagicMock()
        mock_model = MagicMock()
        mock_model.coefficients.toArray.return_value = np.array([0.0, 0.0, 0.0])
        mock_lr_class = MagicMock(return_value=MagicMock(fit=MagicMock(return_value=mock_model)))
        mock_assembler = MagicMock()
        mock_assembler.return_value = MagicMock(transform=MagicMock(return_value=mock_spark_df))
        mock_scaler_class = MagicMock()
        mock_scaler_class.return_value = MagicMock(fit=MagicMock(return_value=MagicMock(transform=MagicMock(return_value=mock_spark_df))))
        with patch("customer_retention.stages.features.feature_selector._import_spark_ml") as mock_imports:
            mock_imports.return_value = (mock_lr_class, mock_assembler, mock_scaler_class, MagicMock())
            dropped, reasons, scores = _spark_l1_selection(mock_spark_df, "target", feature_cols)
        assert dropped == []
        assert reasons == {}
        assert len(scores) == 3

    def test_l1_all_zero_coefficients_keeps_all_features(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "noise1": np.random.randn(n) * 0.001,
            "noise2": np.random.randn(n) * 0.001,
            "target": np.random.choice([0, 1], n),
        })
        selector = FeatureSelector(
            method=SelectionMethod.L1_SELECTION,
            target_column="target",
        )
        result = selector.fit_transform(df)
        if all(v == 0.0 for v in (result.importance_scores or {}).values()):
            assert len(result.selected_features) > 0


class TestPrecomputedCorrMatrix:
    @pytest.fixture
    def df_correlated(self):
        np.random.seed(42)
        base = np.random.randn(200)
        return pd.DataFrame({
            "f1": base,
            "f2": base + np.random.randn(200) * 0.01,
            "f3": np.random.randn(200),
            "target": np.random.choice([0, 1], 200),
        })

    def test_selector_uses_precomputed_corr(self, df_correlated):
        precomputed = df_correlated[["f1", "f2", "f3"]].corr()
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=0.95,
            target_column="target", precomputed_corr_matrix=precomputed,
        )
        result = selector.fit_transform(df_correlated)
        assert len({"f1", "f2"} - set(result.selected_features)) >= 1

    def test_pipeline_uses_precomputed_corr(self, df_correlated):
        from customer_retention.stages.features.feature_selector import run_selection_pipeline
        precomputed = df_correlated[["f1", "f2", "f3"]].corr()
        result = run_selection_pipeline(
            df_correlated, target_column="target",
            correlation_threshold=0.95, l1_enabled=False,
            precomputed_corr_matrix=precomputed,
        )
        assert len({"f1", "f2"} - set(result.selected_features)) >= 1

    def test_precomputed_corr_skips_recomputation(self, df_correlated):
        precomputed = df_correlated[["f1", "f2", "f3"]].corr()
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=0.95,
            target_column="target", precomputed_corr_matrix=precomputed,
        )
        selector.fit(df_correlated)
        dropped = set(selector.dropped_features)
        assert len({"f1", "f2"} & dropped) >= 1
        assert "f3" not in dropped

    def test_partial_precomputed_falls_back(self, df_correlated):
        partial = df_correlated[["f1", "f3"]].corr()
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=0.95,
            target_column="target", precomputed_corr_matrix=partial,
        )
        result = selector.fit_transform(df_correlated)
        assert len({"f1", "f2"} - set(result.selected_features)) >= 1


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
