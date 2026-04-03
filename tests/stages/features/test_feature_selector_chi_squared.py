import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import FeatureSelector, SelectionMethod


class TestChiSquaredSelection:
    @pytest.fixture
    def df_mixed(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        return pd.DataFrame({
            "strong_cat": np.where(target == 1, np.random.choice([0, 1, 2], n, p=[0.2, 0.3, 0.5]),
                                   np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])),
            "strong_num": target * 5.0 + np.random.randn(n),
            "medium_num": target * 2.0 + np.random.randn(n) * 2,
            "noise1": np.random.randn(n),
            "noise2": np.random.randn(n),
            "noise3": np.random.randn(n),
            "noise4": np.random.randn(n),
            "noise5": np.random.randn(n),
            "target": target,
        })

    def test_selects_top_k_features(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=3, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df_mixed)
        assert len(result.selected_features) == 3
        assert "strong_cat" in result.selected_features
        assert "strong_num" in result.selected_features

    def test_returns_correct_drop_reasons(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=3, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df_mixed)
        for feat in result.dropped_features:
            assert "chi_squared" in result.drop_reasons[feat]

    def test_respects_preserve_features(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=2, chi_squared_num_buckets=10,
            preserve_features=["noise1"],
        )
        result = selector.fit_transform(df_mixed)
        assert "noise1" in result.selected_features

    def test_no_drop_when_max_features_exceeds_count(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=100, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df_mixed)
        assert len(result.dropped_features) == 0

    def test_requires_target_column(self, df_mixed):
        selector = FeatureSelector(method=SelectionMethod.CHI_SQUARED, max_features=3)
        with pytest.raises(ValueError, match="target_column"):
            selector.fit_transform(df_mixed)

    def test_method_used_is_chi_squared(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=3, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df_mixed)
        assert result.method_used == SelectionMethod.CHI_SQUARED

    def test_importance_scores_are_chi2_statistics(self, df_mixed):
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=3, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df_mixed)
        assert result.importance_scores is not None
        assert all(v >= 0 for v in result.importance_scores.values())
        assert result.importance_scores["strong_cat"] > result.importance_scores.get("noise5", 0)

    def test_few_unique_values_handled(self):
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "binary": np.random.choice([0, 1], n),
            "constant": np.ones(n),
            "signal": target * 2.0 + np.random.randn(n) * 0.1,
            "target": target,
        })
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=2, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df)
        assert len(result.selected_features) == 2

    def test_handles_null_values(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        f1 = target * 3.0 + np.random.randn(n)
        f1[:20] = np.nan
        df = pd.DataFrame({"f1": f1, "f2": np.random.randn(n), "target": target})
        selector = FeatureSelector(
            method=SelectionMethod.CHI_SQUARED, target_column="target",
            max_features=1, chi_squared_num_buckets=10,
        )
        result = selector.fit_transform(df)
        assert len(result.selected_features) == 1


class TestChiSquaredStandaloneFunction:
    def test_run_chi_squared_selection_returns_result(self):
        from customer_retention.stages.features.feature_selector import run_chi_squared_selection

        np.random.seed(42)
        n = 300
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "strong": target * 5.0 + np.random.randn(n),
            "medium": target * 2.0 + np.random.randn(n),
            "noise1": np.random.randn(n),
            "noise2": np.random.randn(n),
            "noise3": np.random.randn(n),
            "target": target,
        })
        result = run_chi_squared_selection(df, "target", max_features=2)
        assert len(result.selected_features) == 2
        assert "strong" in result.selected_features
        assert result.method_used == SelectionMethod.CHI_SQUARED

    def test_explicit_feature_columns(self):
        from customer_retention.stages.features.feature_selector import run_chi_squared_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "meta": np.arange(n),
            "target": target,
        })
        result = run_chi_squared_selection(df, "target", max_features=1, feature_columns=["f1", "f2"])
        assert "meta" not in result.dropped_features
        assert "f1" in result.selected_features

    def test_temporal_column_excluded(self):
        from customer_retention.stages.features.feature_selector import run_chi_squared_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "as_of_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": target,
        })
        result = run_chi_squared_selection(df, "target", max_features=1, temporal_column="as_of_date")
        assert "as_of_date" not in result.dropped_features
        assert "as_of_date" not in result.selected_features


class TestDistributedChiSquaredSelection:
    def test_spark_chi_squared_returns_dropped_features(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        mock_spark_df = MagicMock()
        work_df = mock_spark_df.select.return_value.na.fill.return_value

        pct_row = {f"__p_{i}": [0.1 * j for j in range(1, 10)] for i in range(5)}
        work_df.agg.return_value.head.return_value = pct_row

        mock_bucketizer_cls = MagicMock()
        mock_bucketizer_cls.return_value.transform.return_value = work_df

        mock_assembler_cls = MagicMock()
        assembled_df = MagicMock()
        mock_assembler_cls.return_value.transform.return_value.select.return_value = assembled_df

        mock_selector_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.selectedFeatures = [0, 2]
        mock_selector_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_chi_squared_ml") as mock_imp:
            mock_F = MagicMock()
            mock_imp.return_value = (mock_selector_cls, mock_assembler_cls, mock_bucketizer_cls, mock_F)
            dropped, reasons, scores = _spark_chi_squared_selection(
                mock_spark_df, "target", feature_cols, num_top_features=2,
            )

        assert set(dropped) == {"f2", "f4", "f5"}
        assert all("chi_squared" in r for r in reasons.values())

    def test_spark_chi_squared_no_drop_when_all_selected(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        feature_cols = ["f1", "f2"]
        mock_spark_df = MagicMock()
        work_df = mock_spark_df.select.return_value.na.fill.return_value
        pct_row = {f"__p_{i}": [0.5] for i in range(2)}
        work_df.agg.return_value.head.return_value = pct_row

        with patch("customer_retention.stages.features.feature_selector._import_spark_chi_squared_ml") as mock_imp:
            mock_F = MagicMock()
            mock_selector_cls = MagicMock()
            mock_selector_cls.return_value.fit.return_value.selectedFeatures = [0, 1]
            mock_imp.return_value = (mock_selector_cls, MagicMock(), MagicMock(), mock_F)
            mock_imp.return_value[1].return_value.transform.return_value = work_df
            mock_imp.return_value[2].return_value.transform.return_value = work_df
            work_df.select.return_value = work_df
            dropped, reasons, scores = _spark_chi_squared_selection(
                mock_spark_df, "target", feature_cols, num_top_features=10,
            )
        assert dropped == []


class TestSelectionMethodEnum:
    def test_chi_squared_exists(self):
        assert SelectionMethod.CHI_SQUARED.value == "CHI_SQUARED"

    def test_lgbm_importance_exists(self):
        assert SelectionMethod.LGBM_IMPORTANCE.value == "LGBM_IMPORTANCE"
