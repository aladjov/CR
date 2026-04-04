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
    def test_spark_chi_squared_selects_top_n_by_score(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        mock_scores = {"f1": 50.0, "f2": 40.0, "f3": 10.0, "f4": 5.0, "f5": 1.0}

        with patch("customer_retention.stages.features.feature_selector._import_spark_chi_squared_bucketizer") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._spark_sql_chi_squared_scores", return_value=mock_scores):
            mock_F = MagicMock()
            mock_bucketizer = MagicMock()
            mock_imp.return_value = (mock_bucketizer, mock_F)
            mock_spark_df = MagicMock()
            work_df = mock_spark_df.select.return_value.na.fill.return_value
            work_df.agg.return_value.head.return_value = {f"__p_{i}": [0.5] for i in range(5)}
            mock_bucketizer.return_value.transform.return_value = work_df

            dropped, reasons, scores = _spark_chi_squared_selection(
                mock_spark_df, "target", feature_cols, num_top_features=2,
            )

        assert scores == mock_scores
        assert set(dropped) == {"f3", "f4", "f5"}
        assert all("chi_squared" in r for r in reasons.values())

    def test_spark_chi_squared_no_drop_when_all_selected(self):
        from unittest.mock import MagicMock

        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        dropped, reasons, scores = _spark_chi_squared_selection(
            MagicMock(), "target", ["f1", "f2"], num_top_features=10,
        )
        assert dropped == []
        assert reasons == {}

    def test_spark_chi_squared_degenerate_target_returns_zeros(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.stages.features.feature_selector import _spark_chi_squared_selection

        feature_cols = ["f1", "f2"]
        zero_scores = {"f1": 0.0, "f2": 0.0}

        with patch("customer_retention.stages.features.feature_selector._import_spark_chi_squared_bucketizer") as mock_imp, \
             patch("customer_retention.stages.features.feature_selector._spark_sql_chi_squared_scores", return_value=zero_scores):
            mock_F = MagicMock()
            mock_bucketizer = MagicMock()
            mock_imp.return_value = (mock_bucketizer, mock_F)
            mock_spark_df = MagicMock()
            work_df = mock_spark_df.select.return_value.na.fill.return_value
            work_df.agg.return_value.head.return_value = {f"__p_{i}": [0.5] for i in range(2)}
            mock_bucketizer.return_value.transform.return_value = work_df

            dropped, reasons, scores = _spark_chi_squared_selection(
                mock_spark_df, "target", feature_cols, num_top_features=1,
            )

        assert scores == zero_scores
        assert len(dropped) == 1


class TestSqlChiSquaredScores:
    def test_scores_rank_features_correctly(self):
        from sklearn.preprocessing import KBinsDiscretizer

        from customer_retention.stages.features.feature_selector import _sql_chi_squared_scores

        np.random.seed(42)
        n = 1000
        target = np.random.choice([0, 1], n)
        features = {
            "strong": target * 5.0 + np.random.randn(n),
            "medium": target * 2.0 + np.random.randn(n) * 2,
            "noise": np.random.randn(n),
        }
        df = pd.DataFrame({**features, "target": target})

        num_buckets = 10
        disc = KBinsDiscretizer(n_bins=num_buckets, encode="ordinal", strategy="quantile", subsample=None)
        feat_names = list(features.keys())
        X_binned = disc.fit_transform(df[feat_names].fillna(0).to_numpy())

        bucketed_df = pd.DataFrame(
            {f"__bkt_{c}": X_binned[:, i] for i, c in enumerate(feat_names)}
        )
        bucketed_df["target"] = target
        scores = _sql_chi_squared_scores(
            bucketed_df, "target", feat_names,
            [f"__bkt_{c}" for c in feat_names], num_buckets,
        )

        assert all(v >= 0 for v in scores.values())
        assert scores["strong"] > scores["medium"] > scores["noise"]

    def test_pearson_chi2_matches_scipy(self):
        from scipy.stats import chi2_contingency

        from customer_retention.stages.features.feature_selector import _sql_chi_squared_scores

        np.random.seed(42)
        n = 2000
        target = np.random.choice([0, 1], n)
        buckets = np.random.choice(range(5), n)
        df = pd.DataFrame({"__bkt_f1": buckets, "target": target})

        scores = _sql_chi_squared_scores(df, "target", ["f1"], ["__bkt_f1"], 10)

        ct = pd.crosstab(buckets, target)
        scipy_chi2, _, _, _ = chi2_contingency(ct)
        assert scores["f1"] == pytest.approx(scipy_chi2, rel=0.01)

    def test_scores_zero_for_constant_target(self):
        from customer_retention.stages.features.feature_selector import _sql_chi_squared_scores

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "__bkt_f1": np.random.choice([0, 1, 2], n),
            "__bkt_f2": np.random.choice([0, 1, 2], n),
            "target": np.ones(n),
        })
        scores = _sql_chi_squared_scores(
            df, "target", ["f1", "f2"], ["__bkt_f1", "__bkt_f2"], 10,
        )
        assert scores == {"f1": 0.0, "f2": 0.0}

    def test_handles_empty_buckets(self):
        from customer_retention.stages.features.feature_selector import _sql_chi_squared_scores

        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            "__bkt_f1": np.zeros(n),
            "target": target,
        })
        scores = _sql_chi_squared_scores(df, "target", ["f1"], ["__bkt_f1"], 10)
        assert scores["f1"] == pytest.approx(0.0, abs=1e-10)

    def test_batching_produces_same_scores(self):
        from customer_retention.stages.features.feature_selector import (
            _CHI_SQL_BATCH,
            _sql_chi_squared_scores,
        )

        np.random.seed(42)
        n = 300
        target = np.random.choice([0, 1], n)
        n_features = _CHI_SQL_BATCH + 5
        feat_names = [f"f{i}" for i in range(n_features)]
        bkt_names = [f"__bkt_f{i}" for i in range(n_features)]
        data = {bkt_names[i]: np.random.choice(range(10), n) for i in range(n_features)}
        data["target"] = target
        df = pd.DataFrame(data)
        scores = _sql_chi_squared_scores(df, "target", feat_names, bkt_names, 10)
        assert len(scores) == n_features
        assert all(v >= 0 for v in scores.values())


class TestSelectionMethodEnum:
    def test_chi_squared_exists(self):
        assert SelectionMethod.CHI_SQUARED.value == "CHI_SQUARED"

    def test_lgbm_importance_exists(self):
        assert SelectionMethod.LGBM_IMPORTANCE.value == "LGBM_IMPORTANCE"
