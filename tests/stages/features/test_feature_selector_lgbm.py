import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import FeatureSelector, SelectionMethod


class TestLgbmImportanceSelection:
    @pytest.fixture
    def df_signal(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        data = {"target": target}
        data["strong1"] = target * 5.0 + np.random.randn(n) * 0.5
        data["strong2"] = target * 4.0 + np.random.randn(n) * 0.5
        data["medium"] = target * 2.0 + np.random.randn(n)
        for i in range(7):
            data[f"noise{i}"] = np.random.randn(n)
        return pd.DataFrame(data)

    def test_selects_top_k_features(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=3, lgbm_num_iterations=50,
        )
        result = selector.fit_transform(df_signal)
        assert len(result.selected_features) == 3
        assert "strong1" in result.selected_features
        assert "strong2" in result.selected_features

    def test_returns_importance_scores(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=3, lgbm_num_iterations=50,
        )
        result = selector.fit_transform(df_signal)
        assert result.importance_scores is not None
        assert len(result.importance_scores) == 10  # all features scored
        assert result.importance_scores["strong1"] > 0

    def test_drop_reasons_contain_lgbm(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=3, lgbm_num_iterations=50,
        )
        result = selector.fit_transform(df_signal)
        for reason in result.drop_reasons.values():
            assert "lgbm_importance" in reason

    def test_method_used_is_lgbm(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=3, lgbm_num_iterations=50,
        )
        result = selector.fit_transform(df_signal)
        assert result.method_used == SelectionMethod.LGBM_IMPORTANCE

    def test_no_drop_when_max_features_exceeds_count(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=100, lgbm_num_iterations=50,
        )
        result = selector.fit_transform(df_signal)
        assert len(result.dropped_features) == 0

    def test_respects_preserve_features(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=2, lgbm_num_iterations=50,
            preserve_features=["noise0"],
        )
        result = selector.fit_transform(df_signal)
        assert "noise0" in result.selected_features

    def test_requires_target_column(self, df_signal):
        selector = FeatureSelector(method=SelectionMethod.LGBM_IMPORTANCE, max_features=3)
        with pytest.raises(ValueError, match="target_column"):
            selector.fit_transform(df_signal)

    def test_handles_null_values(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        f1 = target * 3.0 + np.random.randn(n)
        f1[:20] = np.nan
        df = pd.DataFrame({"f1": f1, "f2": np.random.randn(n), "target": target})
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=1, lgbm_num_iterations=30,
        )
        result = selector.fit_transform(df)
        assert len(result.selected_features) == 1

    def test_all_noise_still_selects_max(self):
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            f"noise{i}": np.random.randn(n) for i in range(5)
        })
        df["target"] = target
        selector = FeatureSelector(
            method=SelectionMethod.LGBM_IMPORTANCE, target_column="target",
            max_features=3, lgbm_num_iterations=30,
        )
        result = selector.fit_transform(df)
        assert len(result.selected_features) == 3


class TestLgbmStandaloneFunction:
    def test_run_lgbm_importance_selection(self):
        from customer_retention.stages.features.feature_selector import run_lgbm_importance_selection

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
        result = run_lgbm_importance_selection(df, "target", max_features=2, num_iterations=50)
        assert len(result.selected_features) == 2
        assert "strong" in result.selected_features
        assert result.method_used == SelectionMethod.LGBM_IMPORTANCE
        assert result.importance_scores is not None

    def test_explicit_feature_columns(self):
        from customer_retention.stages.features.feature_selector import run_lgbm_importance_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "meta": np.arange(n),
            "target": target,
        })
        result = run_lgbm_importance_selection(
            df, "target", max_features=1, feature_columns=["f1", "f2"], num_iterations=30,
        )
        assert "meta" not in result.dropped_features
        assert "f1" in result.selected_features

    def test_temporal_column_excluded(self):
        from customer_retention.stages.features.feature_selector import run_lgbm_importance_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "as_of_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": target,
        })
        result = run_lgbm_importance_selection(
            df, "target", max_features=1, temporal_column="as_of_date", num_iterations=30,
        )
        assert "as_of_date" not in result.dropped_features
        assert "as_of_date" not in result.selected_features


class TestDistributedLgbmSelection:
    def test_spark_lgbm_returns_dropped_features(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_lgbm_importance_selection

        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        mock_spark_df = MagicMock()
        work_df = mock_spark_df.select.return_value.na.fill.return_value

        mock_assembler_cls = MagicMock()
        assembled = MagicMock()
        mock_assembler_cls.return_value.transform.return_value.select.return_value = assembled

        mock_lgbm_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.getFeatureImportances.return_value = np.array([100.0, 5.0, 80.0, 3.0, 2.0])
        mock_lgbm_cls.return_value.fit.return_value = mock_model

        mock_F = MagicMock()

        with patch("customer_retention.stages.features.feature_selector._import_spark_lgbm_ml") as mock_imp:
            mock_imp.return_value = (mock_lgbm_cls, mock_assembler_cls, mock_F)
            dropped, reasons, scores = _spark_lgbm_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=2,
            )

        assert set(dropped) == {"f2", "f4", "f5"}
        assert scores["f1"] == 100.0
        assert scores["f3"] == 80.0

    def test_spark_lgbm_all_selected_when_max_exceeds(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_lgbm_importance_selection

        feature_cols = ["f1", "f2"]
        mock_spark_df = MagicMock()
        work_df = mock_spark_df.select.return_value.na.fill.return_value
        mock_assembler_cls = MagicMock()
        mock_assembler_cls.return_value.transform.return_value.select.return_value = work_df

        mock_lgbm_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.getFeatureImportances.return_value = np.array([10.0, 5.0])
        mock_lgbm_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_lgbm_ml") as mock_imp:
            mock_imp.return_value = (mock_lgbm_cls, mock_assembler_cls, MagicMock())
            dropped, reasons, scores = _spark_lgbm_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=10,
            )
        assert dropped == []
        assert len(scores) == 2

    def test_spark_lgbm_configures_model_params(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_lgbm_importance_selection

        feature_cols = ["f1", "f2"]
        mock_spark_df = MagicMock()
        work_df = mock_spark_df.select.return_value.na.fill.return_value
        mock_assembler_cls = MagicMock()
        mock_assembler_cls.return_value.transform.return_value.select.return_value = work_df

        mock_lgbm_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.getFeatureImportances.return_value = np.array([1.0, 0.5])
        mock_lgbm_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_lgbm_ml") as mock_imp:
            mock_imp.return_value = (mock_lgbm_cls, mock_assembler_cls, MagicMock())
            _spark_lgbm_importance_selection(
                mock_spark_df, "target", feature_cols,
                num_top_features=1, num_iterations=100, num_leaves=31,
            )
        mock_lgbm_cls.assert_called_once_with(
            featuresCol="__lgbm_vec__", labelCol="target",
            numLeaves=31, numIterations=100, learningRate=0.1,
        )
