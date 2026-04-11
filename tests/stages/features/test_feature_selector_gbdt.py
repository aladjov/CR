import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import FeatureSelector, SelectionMethod


class TestGbdtImportanceSelection:
    @pytest.fixture
    def df_signal(self):
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        data = {"target": target}
        # Signal-to-noise ratios are intentionally modest so neither `strong1`
        # nor `strong2` can solve the problem in a single split — XGBoost's
        # level-wise histogram growth will distribute total-gain across both
        # strong features and `medium`, matching how a real prefilter sees
        # signal in the customer-retention feature matrix.
        data["strong1"] = target * 1.5 + np.random.randn(n)
        data["strong2"] = target * 1.3 + np.random.randn(n)
        data["medium"] = target * 0.6 + np.random.randn(n)
        for i in range(7):
            data[f"noise{i}"] = np.random.randn(n)
        return pd.DataFrame(data)

    def test_selects_top_k_features(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=3, gbdt_n_estimators=50,
        )
        result = selector.fit_transform(df_signal)
        assert len(result.selected_features) == 3
        assert "strong1" in result.selected_features
        assert "strong2" in result.selected_features

    def test_returns_importance_scores(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=3, gbdt_n_estimators=50,
        )
        result = selector.fit_transform(df_signal)
        assert result.importance_scores is not None
        assert len(result.importance_scores) == 10
        assert result.importance_scores["strong1"] > 0

    def test_drop_reasons_contain_gbdt(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=3, gbdt_n_estimators=50,
        )
        result = selector.fit_transform(df_signal)
        for reason in result.drop_reasons.values():
            assert "gbdt_importance" in reason

    def test_method_used_is_gbdt(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=3, gbdt_n_estimators=50,
        )
        result = selector.fit_transform(df_signal)
        assert result.method_used == SelectionMethod.GBDT_IMPORTANCE

    def test_no_drop_when_max_features_exceeds_count(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=100, gbdt_n_estimators=50,
        )
        result = selector.fit_transform(df_signal)
        assert len(result.dropped_features) == 0

    def test_respects_preserve_features(self, df_signal):
        selector = FeatureSelector(
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=2, gbdt_n_estimators=50,
            preserve_features=["noise0"],
        )
        result = selector.fit_transform(df_signal)
        assert "noise0" in result.selected_features

    def test_requires_target_column(self, df_signal):
        selector = FeatureSelector(method=SelectionMethod.GBDT_IMPORTANCE, max_features=3)
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
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=1, gbdt_n_estimators=30,
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
            method=SelectionMethod.GBDT_IMPORTANCE, target_column="target",
            max_features=3, gbdt_n_estimators=30,
        )
        result = selector.fit_transform(df)
        assert len(result.selected_features) == 3


class TestGbdtStandaloneFunction:
    def test_run_gbdt_importance_selection(self):
        from customer_retention.stages.features.feature_selector import run_gbdt_importance_selection

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
        result = run_gbdt_importance_selection(df, "target", max_features=2, n_estimators=50)
        assert len(result.selected_features) == 2
        assert "strong" in result.selected_features
        assert result.method_used == SelectionMethod.GBDT_IMPORTANCE
        assert result.importance_scores is not None

    def test_explicit_feature_columns(self):
        from customer_retention.stages.features.feature_selector import run_gbdt_importance_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "meta": np.arange(n),
            "target": target,
        })
        result = run_gbdt_importance_selection(
            df, "target", max_features=1, feature_columns=["f1", "f2"], n_estimators=30,
        )
        assert "meta" not in result.dropped_features
        assert "f1" in result.selected_features

    def test_temporal_column_excluded(self):
        from customer_retention.stages.features.feature_selector import run_gbdt_importance_selection

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "f1": target * 3.0 + np.random.randn(n),
            "f2": np.random.randn(n),
            "as_of_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": target,
        })
        result = run_gbdt_importance_selection(
            df, "target", max_features=1, temporal_column="as_of_date", n_estimators=30,
        )
        assert "as_of_date" not in result.dropped_features
        assert "as_of_date" not in result.selected_features


class TestDistributedGbdtSelection:
    def test_spark_gbdt_returns_dropped_features(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        mock_spark_df = MagicMock()

        mock_assembler_cls = MagicMock()
        assembled = MagicMock()
        mock_assembler_cls.return_value.transform.return_value.select.return_value = assembled

        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = {
            "f0": 100.0, "f1": 5.0, "f2": 80.0, "f3": 3.0, "f4": 2.0,
        }
        mock_xgb_cls.return_value.fit.return_value = mock_model

        mock_F = MagicMock()

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=4):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, mock_F)
            dropped, reasons, scores = _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=2,
            )

        assert set(dropped) == {"f2", "f4", "f5"}
        assert scores["f1"] == 100.0
        assert scores["f3"] == 80.0

    def test_spark_gbdt_all_selected_when_max_exceeds(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2"]
        mock_spark_df = MagicMock()
        mock_assembler_cls = MagicMock()

        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = {"f0": 10.0, "f1": 5.0}
        mock_xgb_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=2):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            dropped, reasons, scores = _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=10,
            )
        assert dropped == []
        assert len(scores) == 2
        assert all(v == 0.0 for v in scores.values())

    def test_spark_gbdt_configures_model_params(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2", "f3"]
        mock_spark_df = MagicMock()
        mock_assembler_cls = MagicMock()

        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = {"f0": 1.0, "f1": 0.5, "f2": 0.1}
        mock_xgb_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=8):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols,
                num_top_features=1, n_estimators=100, max_depth=4,
            )
        mock_xgb_cls.assert_called_once_with(
            features_col="__gbdt_vec__", label_col="target",
            num_workers=8,
            n_estimators=100, max_depth=4,
            learning_rate=0.1, verbosity=0,
        )

    def test_spark_gbdt_handles_missing_features_in_importance_dict(self):
        """XGBoost get_feature_importances() omits features never used in any split —
        they should default to 0.0, not raise KeyError."""
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2", "f3"]
        mock_spark_df = MagicMock()
        mock_assembler_cls = MagicMock()

        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = {"f0": 42.0}
        mock_xgb_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=1):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            dropped, reasons, scores = _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=1,
            )

        assert scores["f1"] == 42.0
        assert scores["f2"] == 0.0
        assert scores["f3"] == 0.0
        assert set(dropped) == {"f2", "f3"}

    def test_spark_gbdt_uses_total_gain_importance_type(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_gbdt_importance_selection

        feature_cols = ["f1", "f2"]
        mock_spark_df = MagicMock()
        mock_assembler_cls = MagicMock()

        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = {"f0": 1.0, "f1": 0.5}
        mock_xgb_cls.return_value.fit.return_value = mock_model

        with patch("customer_retention.stages.features.feature_selector._import_spark_gbdt_ml") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers", return_value=1):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            _spark_gbdt_importance_selection(
                mock_spark_df, "target", feature_cols, num_top_features=1,
            )

        mock_model.get_feature_importances.assert_called_once_with(importance_type="total_gain")
