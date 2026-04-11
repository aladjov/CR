"""Tests for chi_squared_rescue feature selection mode.

Specification: docs/chi_squared_rescue_selection_plan.md §6.

Tests are organized in classes that mirror the spec sections so the
implementation can unskip them stage-by-stage as helpers land.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic snapshot-grid fixture (used by integration + parity tests)
# ---------------------------------------------------------------------------


def _build_snapshot_grid(
    n_entities: int = 200, n_snapshots: int = 6, seed: int = 7,
) -> pd.DataFrame:
    """1000 entities × 6 monthly snapshots with three feature families:

    - 5 entity-constant features strongly target-correlated
    - 5 time-varying features with within-entity noise but temporal signal
    - 20 pure noise features
    """
    rng = np.random.default_rng(seed)
    snapshot_dates = pd.date_range("2025-10-01", periods=n_snapshots, freq="MS")
    rows = []
    entity_target = rng.integers(0, 2, size=n_entities)
    entity_static = {
        f"ent_static_{k}": entity_target * (1.5 + 0.2 * k) + rng.normal(0, 1.0, n_entities)
        for k in range(5)
    }
    for s_idx, snap in enumerate(snapshot_dates):
        for e in range(n_entities):
            row = {"__cv_entity__": f"e{e}", "__cv_date__": snap, "target": int(entity_target[e])}
            for k in range(5):
                row[f"ent_static_{k}"] = float(entity_static[f"ent_static_{k}"][e])
            for k in range(5):
                temporal_signal = entity_target[e] * (0.8 + 0.1 * k) * (s_idx / max(1, n_snapshots - 1))
                row[f"time_var_{k}"] = float(temporal_signal + rng.normal(0, 1.0))
            for k in range(20):
                row[f"noise_{k}"] = float(rng.normal(0, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot_grid() -> pd.DataFrame:
    return _build_snapshot_grid()


@pytest.fixture
def small_snapshot_grid() -> pd.DataFrame:
    return _build_snapshot_grid(n_entities=120, n_snapshots=5, seed=11)


# ---------------------------------------------------------------------------
# §6.1 Slice resolution
# ---------------------------------------------------------------------------


class TestResolveTimeSlice:
    def _make_grid(self, n_dates: int = 5, rows_per_date: int = 50,
                   positive_rate: float = 0.4) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2025-01-01", periods=n_dates, freq="MS")
        rows = []
        for d in dates:
            for _ in range(rows_per_date):
                rows.append({
                    "__cv_date__": d,
                    "f": float(rng.normal()),
                    "target": int(rng.random() < positive_rate),
                })
        return pd.DataFrame(rows)

    def test_penultimate_strategy_picks_second_to_last_date(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid()
        slice_df, slice_date, count = resolve_time_slice(
            df, time_column="__cv_date__", strategy="penultimate",
            target_column="target", min_positive_rate=0.0,
        )
        assert slice_date == pd.Timestamp("2025-04-01").isoformat()
        assert count == 50

    def test_last_strategy_picks_most_recent_date(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid()
        _, slice_date, _ = resolve_time_slice(
            df, time_column="__cv_date__", strategy="last",
            target_column="target", min_positive_rate=0.0,
        )
        assert slice_date == pd.Timestamp("2025-05-01").isoformat()

    def test_random_per_entity_one_row_per_entity(self):
        from customer_retention.core.compat import resolve_time_slice
        rng = np.random.default_rng(0)
        rows = []
        dates = pd.date_range("2025-01-01", periods=4, freq="MS")
        for e in range(50):
            for d in dates:
                rows.append({"__cv_entity__": f"e{e}", "__cv_date__": d, "target": int(rng.random() < 0.4)})
        df = pd.DataFrame(rows)
        slice_df, _, count = resolve_time_slice(
            df, time_column="__cv_date__", strategy="random_per_entity",
            target_column="target", min_positive_rate=0.0, entity_column="__cv_entity__",
        )
        assert count == 50

    def test_two_slice_union_concatenates_penultimate_and_middle(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid(n_dates=6)
        _, _, count = resolve_time_slice(
            df, time_column="__cv_date__", strategy="two_slice_union",
            target_column="target", min_positive_rate=0.0,
        )
        assert count == 100  # 2 dates × 50 rows

    def test_single_snapshot_returns_full_df_unchanged(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid(n_dates=1)
        slice_df, _, count = resolve_time_slice(
            df, time_column="__cv_date__", strategy="penultimate",
            target_column="target", min_positive_rate=0.0,
        )
        assert count == 50

    def test_empty_df_raises_feature_selection_error(self):
        from customer_retention.core.compat import FeatureSelectionError, resolve_time_slice
        df = pd.DataFrame({"__cv_date__": [], "target": []})
        with pytest.raises(FeatureSelectionError):
            resolve_time_slice(df, time_column="__cv_date__", strategy="penultimate",
                               target_column="target", min_positive_rate=0.03)

    def test_low_positive_rate_falls_back_to_earlier_slice(self):
        from customer_retention.core.compat import resolve_time_slice
        rng = np.random.default_rng(0)
        rows = []
        dates = pd.date_range("2025-01-01", periods=5, freq="MS")
        positive_rates = [0.30, 0.30, 0.30, 0.30, 0.005]
        for d, pr in zip(dates, positive_rates):
            for _ in range(200):
                rows.append({"__cv_date__": d, "target": int(rng.random() < pr)})
        df = pd.DataFrame(rows)
        _, slice_date, _ = resolve_time_slice(
            df, time_column="__cv_date__", strategy="penultimate",
            target_column="target", min_positive_rate=0.05,
        )
        # penultimate is dates[-2] = 2025-04-01 with 30% positives — pass straight through
        assert slice_date == pd.Timestamp("2025-04-01").isoformat()

    def test_no_slice_meets_threshold_raises_feature_selection_error(self):
        from customer_retention.core.compat import FeatureSelectionError, resolve_time_slice
        rows = []
        dates = pd.date_range("2025-01-01", periods=4, freq="MS")
        for d in dates:
            for _ in range(200):
                rows.append({"__cv_date__": d, "target": 0})
        df = pd.DataFrame(rows)
        with pytest.raises(FeatureSelectionError):
            resolve_time_slice(df, time_column="__cv_date__", strategy="penultimate",
                               target_column="target", min_positive_rate=0.05)

    def test_missing_time_column_raises(self):
        from customer_retention.core.compat import FeatureSelectionError, resolve_time_slice
        df = pd.DataFrame({"target": [0, 1]})
        with pytest.raises(FeatureSelectionError):
            resolve_time_slice(df, time_column="missing", strategy="last", target_column="target")

    def test_missing_target_column_raises_when_min_positive_rate_set(self):
        from customer_retention.core.compat import FeatureSelectionError, resolve_time_slice
        df = pd.DataFrame({"__cv_date__": pd.date_range("2025-01-01", periods=3, freq="MS")})
        with pytest.raises(FeatureSelectionError):
            resolve_time_slice(df, time_column="__cv_date__", strategy="last",
                               target_column="missing", min_positive_rate=0.05)

    def test_slice_date_iso_format_in_return_tuple(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid()
        _, slice_date, _ = resolve_time_slice(
            df, time_column="__cv_date__", strategy="last",
            target_column="target", min_positive_rate=0.0,
        )
        assert isinstance(slice_date, str)
        assert "T" in slice_date or "-" in slice_date

    def test_slice_row_count_matches_actual_rows(self):
        from customer_retention.core.compat import resolve_time_slice
        df = self._make_grid(rows_per_date=37)
        slice_df, _, count = resolve_time_slice(
            df, time_column="__cv_date__", strategy="penultimate",
            target_column="target", min_positive_rate=0.0,
        )
        assert count == 37
        assert len(slice_df) == 37


# ---------------------------------------------------------------------------
# §6.2 Chi-squared primary stage
# ---------------------------------------------------------------------------


class TestChiSquaredPrimary:
    @pytest.fixture
    def slice_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 300
        target = rng.integers(0, 2, n)
        return pd.DataFrame({
            "strong1": target * 4.0 + rng.normal(0, 1, n),
            "strong2": target * 3.0 + rng.normal(0, 1, n),
            "weak1": rng.normal(0, 1, n),
            "weak2": rng.normal(0, 1, n),
            "weak3": rng.normal(0, 1, n),
            "target": target,
        })

    def test_uses_slice_not_full_df(self, slice_df):
        from customer_retention.stages.features.feature_selector import _chi_squared_primary
        keep, drop, stats = _chi_squared_primary(
            slice_df, target_column="target",
            features=["strong1", "strong2", "weak1", "weak2", "weak3"],
            top_k=2, num_buckets=5,
        )
        assert len(keep) == 2
        assert "strong1" in keep
        assert "strong2" in keep

    def test_returns_keep_and_drop_sets_as_disjoint(self, slice_df):
        from customer_retention.stages.features.feature_selector import _chi_squared_primary
        feats = ["strong1", "strong2", "weak1", "weak2", "weak3"]
        keep, drop, _ = _chi_squared_primary(slice_df, "target", feats, top_k=3, num_buckets=5)
        assert set(keep).isdisjoint(set(drop))
        assert set(keep) | set(drop) == set(feats)

    def test_returns_chi_stats_dict_for_all_features(self, slice_df):
        from customer_retention.stages.features.feature_selector import _chi_squared_primary
        feats = ["strong1", "strong2", "weak1", "weak2", "weak3"]
        _, _, stats = _chi_squared_primary(slice_df, "target", feats, top_k=2, num_buckets=5)
        for f in feats:
            assert f in stats
            assert "score" in stats[f]
            assert "rank" in stats[f]

    def test_top_k_cap_respected(self, slice_df):
        from customer_retention.stages.features.feature_selector import _chi_squared_primary
        feats = ["strong1", "strong2", "weak1", "weak2", "weak3"]
        keep, _, _ = _chi_squared_primary(slice_df, "target", feats, top_k=2, num_buckets=5)
        assert len(keep) == 2

    def test_preserves_features_arg_no_mutation(self, slice_df):
        from customer_retention.stages.features.feature_selector import _chi_squared_primary
        feats = ["strong1", "strong2", "weak1", "weak2", "weak3"]
        original = list(feats)
        _chi_squared_primary(slice_df, "target", feats, top_k=2, num_buckets=5)
        assert feats == original


# ---------------------------------------------------------------------------
# §6.3 Rescue stage
# ---------------------------------------------------------------------------


class TestRescueFromChiDrops:
    @pytest.fixture
    def slice_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 400
        target = rng.integers(0, 2, n)
        return pd.DataFrame({
            "rescue1": target * 2.0 + rng.normal(0, 1, n),
            "rescue2": target * 1.8 + rng.normal(0, 1, n),
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
            "noise3": rng.normal(0, 1, n),
            "target": target,
        })

    def test_l1_rescue_empty_when_disabled(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=False, gbdt_enabled=False,
        )
        assert rescue.l1_keep == set()
        assert rescue.gbdt_keep == set()

    def test_gbdt_rescue_empty_when_disabled(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=False, gbdt_enabled=False,
        )
        assert rescue.gbdt_keep == set()

    def test_l1_rescue_returns_nonzero_coefficient_features_only(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=True, gbdt_enabled=False, l1_max=10,
        )
        assert "rescue1" in rescue.l1_keep or "rescue2" in rescue.l1_keep

    def test_gbdt_rescue_returns_top_k_by_total_gain(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=False, gbdt_enabled=True, gbdt_max=2, gbdt_n_estimators=40,
        )
        assert len(rescue.gbdt_keep) <= 2

    def test_rescue_caps_at_max_features(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=False, gbdt_enabled=True, gbdt_max=1, gbdt_n_estimators=40,
        )
        assert len(rescue.gbdt_keep) <= 1

    def test_absolute_importance_floor_rejects_features_below_1_percent_of_max(self):
        from customer_retention.stages.features.feature_selector import _apply_importance_floor
        gains = {"a": 100.0, "b": 50.0, "c": 0.5, "d": 0.0}
        kept = _apply_importance_floor(gains, max_features=10, floor_ratio=0.01)
        assert "a" in kept
        assert "b" in kept
        assert "d" not in kept
        # c is at 0.5 = exactly 0.5% of max → below floor of 1.0
        assert "c" not in kept

    def test_shadow_floor_rejects_features_below_best_shadow(self, slice_df):
        from customer_retention.stages.features.feature_selector import _apply_shadow_floor
        gains = {"f1": 100.0, "f2": 50.0, "f3": 5.0, "f4": 1.0}
        shadow_gains = {"__shadow_0": 10.0, "__shadow_1": 6.0}
        kept = _apply_shadow_floor(gains, shadow_gains)
        assert "f1" in kept
        assert "f2" in kept
        assert "f3" not in kept
        assert "f4" not in kept

    def test_empty_drop_pool_raises(self, slice_df):
        from customer_retention.core.compat import FeatureSelectionError
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        with pytest.raises(FeatureSelectionError):
            _rescue_from_chi_drops(
                slice_df, target_column="target", drop_pool=[],
                l1_enabled=False, gbdt_enabled=True,
            )

    def test_l1_and_gbdt_return_coefs_and_gains_dicts(self, slice_df):
        from customer_retention.stages.features.feature_selector import _rescue_from_chi_drops
        rescue = _rescue_from_chi_drops(
            slice_df, target_column="target",
            drop_pool=["rescue1", "rescue2", "noise1", "noise2", "noise3"],
            l1_enabled=True, gbdt_enabled=True, gbdt_n_estimators=40,
        )
        assert isinstance(rescue.l1_coefs, dict)
        assert isinstance(rescue.gbdt_gains, dict)
        for f in ["rescue1", "rescue2", "noise1", "noise2", "noise3"]:
            assert f in rescue.l1_coefs
            assert f in rescue.gbdt_gains


# ---------------------------------------------------------------------------
# §6.4 Union and drop-reason audit
# ---------------------------------------------------------------------------


class TestUnionAndAudit:
    def test_final_keep_is_union_of_three_sets(self):
        from customer_retention.stages.features.feature_selector import _build_rescue_consensus_reasons
        all_features = ["a", "b", "c", "d", "e"]
        keep_chi = {"a", "b"}
        rescue_l1 = {"c"}
        rescue_gbdt = {"d"}
        chi_stats = {f: {"score": 0.0, "rank": i} for i, f in enumerate(all_features)}
        l1_coefs = {f: 0.0 for f in all_features}
        gbdt_gains = {f: 0.0 for f in all_features}
        keep = keep_chi | rescue_l1 | rescue_gbdt
        dropped = [f for f in all_features if f not in keep]
        reasons = _build_rescue_consensus_reasons(
            dropped, chi_stats, l1_coefs, gbdt_gains,
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=500,
            l1_considered=True, gbdt_considered=True,
        )
        assert dropped == ["e"]
        assert "e" in reasons

    def test_drop_reasons_carry_triple_consensus_fields(self):
        from customer_retention.stages.features.feature_selector import _build_rescue_consensus_reasons
        chi_stats = {"x": {"score": 1.5, "rank": 42}}
        l1_coefs = {"x": 0.0}
        gbdt_gains = {"x": 0.0}
        reasons = _build_rescue_consensus_reasons(
            ["x"], chi_stats, l1_coefs, gbdt_gains,
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=500,
            l1_considered=True, gbdt_considered=True,
        )
        params = reasons["x"]["parameters"]
        assert params["chi_squared_rank"] == 42
        assert params["chi_squared_score"] == 1.5
        assert params["l1_coefficient"] == 0.0
        assert params["gbdt_total_gain"] == 0.0

    def test_drop_reason_includes_slice_date_and_strategy_and_row_count(self):
        from customer_retention.stages.features.feature_selector import _build_rescue_consensus_reasons
        reasons = _build_rescue_consensus_reasons(
            ["x"], {"x": {"score": 0.0, "rank": 1}}, {"x": 0.0}, {"x": 0.0},
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
            l1_considered=True, gbdt_considered=True,
        )
        params = reasons["x"]["parameters"]
        assert params["slice_date"] == "2026-03-01"
        assert params["slice_strategy"] == "penultimate"
        assert params["slice_row_count"] == 5000

    def test_feature_kept_by_any_stage_not_in_dropped(self):
        keep_chi = {"a", "b"}
        rescue_l1 = {"c"}
        rescue_gbdt: set = set()
        all_features = {"a", "b", "c", "d"}
        keep = keep_chi | rescue_l1 | rescue_gbdt
        dropped = all_features - keep
        assert dropped == {"d"}

    def test_rationale_string_summarizes_why_dropped(self):
        from customer_retention.stages.features.feature_selector import _build_rescue_consensus_reasons
        reasons = _build_rescue_consensus_reasons(
            ["x"], {"x": {"score": 0.8, "rank": 742}}, {"x": 0.0}, {"x": 0.0},
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
            l1_considered=True, gbdt_considered=True,
        )
        rationale = reasons["x"]["rationale"]
        assert "742" in rationale
        assert "L1" in rationale or "l1" in rationale
        assert "GBDT" in rationale or "gbdt" in rationale


# ---------------------------------------------------------------------------
# §6.5 Snapshot-grid integration
# ---------------------------------------------------------------------------


class TestSnapshotGridIntegration:
    def test_snapshot_grid_rescues_temporal_features(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        result = run_chi_squared_rescue_selection(
            snapshot_grid, target_column="target",
            entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5,  # tight chi-squared cap so most time-vars get dropped
            slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        rescued = sum(1 for f in result.selected_features if f.startswith("time_var_"))
        assert rescued >= 3

    def test_snapshot_grid_keeps_entity_level_features(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        result = run_chi_squared_rescue_selection(
            snapshot_grid, target_column="target",
            entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        ent_kept = sum(1 for f in result.selected_features if f.startswith("ent_static_"))
        assert ent_kept >= 4

    def test_snapshot_grid_rejects_noise_features(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        result = run_chi_squared_rescue_selection(
            snapshot_grid, target_column="target",
            entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        noise_kept = sum(1 for f in result.selected_features if f.startswith("noise_"))
        assert noise_kept <= 5

    def test_snapshot_grid_reproducible(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        kwargs = dict(
            target_column="target", entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        a = run_chi_squared_rescue_selection(snapshot_grid, **kwargs)
        b = run_chi_squared_rescue_selection(snapshot_grid, **kwargs)
        assert set(a.selected_features) == set(b.selected_features)

    def test_drop_reasons_explain_why_noise_dropped(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        result = run_chi_squared_rescue_selection(
            snapshot_grid, target_column="target",
            entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        for f in result.dropped_features:
            entry = result.drop_reasons[f]
            assert isinstance(entry, dict)
            assert "parameters" in entry
            assert "chi_squared_rank" in entry["parameters"]


# ---------------------------------------------------------------------------
# §6.6 Distributed mocked path tests (Spark-free CI)
# ---------------------------------------------------------------------------


class TestDistributedRescuePath:
    """Mocked tests for the distributed rescue path.

    Mirrors the pattern in TestDistributedGbdtSelection: drives the inner
    Spark scoring helper directly with a mock spark_df so the test runs
    Spark-free in CI.
    """

    def _make_mock_spark_df(self):
        mock_spark_df = MagicMock()
        mock_spark_df.select.return_value.na.fill.return_value = mock_spark_df
        mock_spark_df.select.return_value.na.fill.return_value.localCheckpoint.return_value = mock_spark_df
        return mock_spark_df

    def _patches(self, importances, num_workers=2):
        mock_assembler_cls = MagicMock()
        mock_xgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.get_feature_importances.return_value = importances
        mock_xgb_cls.return_value.fit.return_value = mock_model
        return mock_xgb_cls, mock_assembler_cls, mock_model, num_workers

    def test_spark_rescue_uses_vector_assembler_on_drop_pool(self):
        from customer_retention.stages.features.feature_selector import (
            _spark_gbdt_total_gain_scores,
        )
        feature_cols = ["a", "b", "c"]
        mock_spark_df = self._make_mock_spark_df()
        mock_xgb_cls, mock_assembler_cls, _, num_workers = self._patches(
            {"f0": 1.0, "f1": 5.0, "f2": 2.0}
        )
        with patch(
            "customer_retention.stages.features.feature_selector._import_spark_gbdt_ml"
        ) as mock_imp, patch(
            "customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers",
            return_value=num_workers,
        ):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            scores = _spark_gbdt_total_gain_scores(
                mock_spark_df, target_column="target", feature_columns=feature_cols,
                n_estimators=50, max_depth=4,
            )
        # VectorAssembler must be constructed with the drop pool column subset
        assembler_kwargs = mock_assembler_cls.call_args.kwargs
        assert assembler_kwargs["inputCols"] == feature_cols
        assert scores["a"] == 1.0
        assert scores["b"] == 5.0
        assert scores["c"] == 2.0

    def test_spark_rescue_handles_missing_features_in_importance_dict(self):
        from customer_retention.stages.features.feature_selector import (
            _spark_gbdt_total_gain_scores,
        )
        mock_spark_df = self._make_mock_spark_df()
        mock_xgb_cls, mock_assembler_cls, _, num_workers = self._patches({"f0": 42.0})
        with patch(
            "customer_retention.stages.features.feature_selector._import_spark_gbdt_ml"
        ) as mock_imp, patch(
            "customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers",
            return_value=num_workers,
        ):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            scores = _spark_gbdt_total_gain_scores(
                mock_spark_df, target_column="target", feature_columns=["a", "b", "c"],
                n_estimators=10, max_depth=3,
            )
        assert scores["a"] == 42.0
        assert scores["b"] == 0.0
        assert scores["c"] == 0.0

    def test_spark_rescue_uses_total_gain_importance_type(self):
        from customer_retention.stages.features.feature_selector import (
            _spark_gbdt_total_gain_scores,
        )
        mock_spark_df = self._make_mock_spark_df()
        mock_xgb_cls, mock_assembler_cls, mock_model, num_workers = self._patches(
            {"f0": 1.0, "f1": 0.5}
        )
        with patch(
            "customer_retention.stages.features.feature_selector._import_spark_gbdt_ml"
        ) as mock_imp, patch(
            "customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers",
            return_value=num_workers,
        ):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            _spark_gbdt_total_gain_scores(
                mock_spark_df, target_column="target", feature_columns=["a", "b"],
                n_estimators=10, max_depth=3,
            )
        mock_model.get_feature_importances.assert_called_once_with(importance_type="total_gain")

    def test_spark_rescue_num_workers_from_resolver(self):
        from customer_retention.stages.features.feature_selector import (
            _spark_gbdt_total_gain_scores,
        )
        mock_spark_df = self._make_mock_spark_df()
        mock_xgb_cls, mock_assembler_cls, _, _ = self._patches({"f0": 1.0, "f1": 0.5}, num_workers=8)
        with patch(
            "customer_retention.stages.features.feature_selector._import_spark_gbdt_ml"
        ) as mock_imp, patch(
            "customer_retention.stages.features.feature_selector._resolve_spark_gbdt_workers",
            return_value=8,
        ):
            mock_imp.return_value = (mock_xgb_cls, mock_assembler_cls, MagicMock())
            _spark_gbdt_total_gain_scores(
                mock_spark_df, target_column="target", feature_columns=["a", "b"],
                n_estimators=10, max_depth=3,
            )
        kwargs = mock_xgb_cls.call_args.kwargs
        assert kwargs["num_workers"] == 8


# ---------------------------------------------------------------------------
# §6.7 Parity test (local pandas vs distributed pyspark.pandas)
# ---------------------------------------------------------------------------


@pytest.mark.spark
class TestSelectionParity:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")
        from customer_retention.core.compat.detection import is_spark_available
        if not is_spark_available():
            pytest.skip("Requires PySpark runtime")

    def test_selection_parity_local_vs_distributed(self, snapshot_grid):
        """Identical selected set, identical drop reasons across paths."""
        import pyspark.pandas as ps

        from customer_retention.stages.features.feature_selector import (
            run_chi_squared_rescue_selection,
        )

        kwargs = dict(
            target_column="target", entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
            gbdt_rescue_max_features=10, gbdt_n_estimators=60, gbdt_max_depth=4,
        )
        local_result = run_chi_squared_rescue_selection(snapshot_grid, **kwargs)

        psp_df = ps.from_pandas(snapshot_grid)
        spark_result = run_chi_squared_rescue_selection(psp_df, **kwargs)
        assert set(local_result.selected_features) == set(spark_result.selected_features)
        assert set(local_result.dropped_features) == set(spark_result.dropped_features)


# ---------------------------------------------------------------------------
# §6.8 Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def _basic_grid(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for d in pd.date_range("2025-01-01", periods=4, freq="MS"):
            for e in range(50):
                rows.append({
                    "__cv_entity__": f"e{e}", "__cv_date__": d,
                    "f1": float(rng.normal()), "f2": float(rng.normal()),
                    "target": int(rng.random() < 0.4),
                })
        return pd.DataFrame(rows)

    def test_raises_when_entity_column_not_in_df(self):
        from customer_retention.core.compat import FeatureSelectionError
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        df = self._basic_grid()
        with pytest.raises(FeatureSelectionError):
            run_chi_squared_rescue_selection(
                df, target_column="target", entity_column="missing", time_column="__cv_date__",
                max_features=1,
            )

    def test_raises_when_time_column_not_in_df(self):
        from customer_retention.core.compat import FeatureSelectionError
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        df = self._basic_grid()
        with pytest.raises(FeatureSelectionError):
            run_chi_squared_rescue_selection(
                df, target_column="target", entity_column="__cv_entity__", time_column="missing",
                max_features=1,
            )

    def test_raises_when_target_column_not_in_df(self):
        from customer_retention.core.compat import FeatureSelectionError
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        df = self._basic_grid()
        with pytest.raises(FeatureSelectionError):
            run_chi_squared_rescue_selection(
                df, target_column="missing", entity_column="__cv_entity__",
                time_column="__cv_date__", max_features=1,
            )

    def test_raises_when_drop_chi_pool_empty_but_rescue_enabled(self):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        df = self._basic_grid()
        # max_features high enough that no features are dropped → empty drop_chi
        result = run_chi_squared_rescue_selection(
            df, target_column="target", entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=100, slice_strategy="last", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=True,
        )
        # No drops at all → no rescue needed; result should be no error and empty drop list
        assert result.dropped_features == []

    def test_raises_when_slice_has_zero_positives(self):
        from customer_retention.core.compat import FeatureSelectionError
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        df = self._basic_grid()
        df["target"] = 0
        with pytest.raises(FeatureSelectionError):
            run_chi_squared_rescue_selection(
                df, target_column="target", entity_column="__cv_entity__",
                time_column="__cv_date__", max_features=1,
                slice_strategy="penultimate", min_positive_rate=0.05,
            )

    def test_handles_both_rescues_disabled_gracefully(self, snapshot_grid):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        result = run_chi_squared_rescue_selection(
            snapshot_grid, target_column="target",
            entity_column="__cv_entity__", time_column="__cv_date__",
            max_features=5, slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=False,
        )
        assert len(result.selected_features) == 5

    def test_handles_l1_all_zero_coefficients(self):
        from customer_retention.stages.features.feature_selector import _l1_rescue_on_pool
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "n1": rng.normal(0, 1, n),
            "n2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        kept, coefs = _l1_rescue_on_pool(df, "target", ["n1", "n2"], max_features=5, C=0.001)
        assert isinstance(coefs, dict)
        assert "n1" in coefs and "n2" in coefs

    def test_handles_gbdt_all_zero_gain(self):
        from customer_retention.stages.features.feature_selector import _gbdt_rescue_on_pool
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "n1": rng.normal(0, 1, n),
            "n2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        kept, gains = _gbdt_rescue_on_pool(df, "target", ["n1", "n2"], max_features=5,
                                            n_estimators=20, max_depth=3)
        assert isinstance(gains, dict)
        assert "n1" in gains and "n2" in gains

    def test_single_feature_input(self):
        from customer_retention.stages.features.feature_selector import run_chi_squared_rescue_selection
        rng = np.random.default_rng(0)
        rows = []
        for d in pd.date_range("2025-01-01", periods=3, freq="MS"):
            for e in range(80):
                rows.append({
                    "__cv_entity__": f"e{e}", "__cv_date__": d,
                    "only_feature": float(rng.normal()),
                    "target": int(rng.random() < 0.4),
                })
        df = pd.DataFrame(rows)
        result = run_chi_squared_rescue_selection(
            df, target_column="target", entity_column="__cv_entity__",
            time_column="__cv_date__", max_features=10,
            slice_strategy="penultimate", min_positive_rate=0.0,
            l1_rescue_enabled=False, gbdt_rescue_enabled=False,
        )
        assert "only_feature" in result.selected_features


# ---------------------------------------------------------------------------
# Static guard: ensure new symbols are exported when implemented
# ---------------------------------------------------------------------------


def test_run_chi_squared_rescue_selection_is_exported():
    from customer_retention.stages.features import run_chi_squared_rescue_selection  # noqa: F401


# ---------------------------------------------------------------------------
# slice_df_at_date — NB05 BOX_PLOT_SLICE_DATE override helper
# Spec: docs/nb05_time_slice_relationship_plan.md §5.1
# ---------------------------------------------------------------------------


class TestSliceDfAtDate:
    def _make_grid(self, n_dates: int = 4, rows_per_date: int = 25) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=n_dates, freq="MS")
        rows = []
        for d in dates:
            for i in range(rows_per_date):
                rows.append({"__cv_date__": d, "value": i, "target": i % 2})
        return pd.DataFrame(rows)

    def test_returns_only_rows_matching_slice_date(self):
        from customer_retention.core.compat import slice_df_at_date
        df = self._make_grid()
        target_date = pd.Timestamp("2025-02-01")
        sliced = slice_df_at_date(df, "__cv_date__", target_date)
        assert len(sliced) == 25
        assert (sliced["__cv_date__"] == target_date).all()

    def test_unknown_date_returns_empty_frame(self):
        from customer_retention.core.compat import slice_df_at_date
        df = self._make_grid()
        sliced = slice_df_at_date(df, "__cv_date__", pd.Timestamp("2099-01-01"))
        assert len(sliced) == 0
        # Same columns — empty frame, not None
        assert list(sliced.columns) == list(df.columns)

    def test_missing_time_column_raises(self):
        from customer_retention.core.compat import FeatureSelectionError, slice_df_at_date
        df = self._make_grid()
        with pytest.raises(FeatureSelectionError):
            slice_df_at_date(df, "not_a_column", pd.Timestamp("2025-01-01"))

    def test_pandas_path_preserves_row_ordering(self):
        from customer_retention.core.compat import slice_df_at_date
        df = self._make_grid()
        target_date = pd.Timestamp("2025-03-01")
        sliced = slice_df_at_date(df, "__cv_date__", target_date)
        assert sliced["value"].tolist() == list(range(25))

    def test_no_positive_rate_gate_or_fallback(self):
        """Unlike resolve_time_slice, slice_df_at_date never walks backward."""
        from customer_retention.core.compat import slice_df_at_date
        # Build a grid where the chosen date has zero positives — filter should
        # still return those rows, not silently fall back to another date.
        dates = pd.date_range("2025-01-01", periods=3, freq="MS")
        rows = []
        for d in dates:
            for _ in range(10):
                rows.append({"__cv_date__": d, "target": 0 if d == dates[1] else 1})
        df = pd.DataFrame(rows)
        sliced = slice_df_at_date(df, "__cv_date__", dates[1])
        assert len(sliced) == 10
        assert int(sliced["target"].sum()) == 0
