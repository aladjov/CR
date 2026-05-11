from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_variance


class TestBulkVariance:

    def test_matches_pandas_var(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100), "c": rng.standard_normal(100)})
        result = bulk_variance(df, ["a", "b", "c"])
        expected = df[["a", "b", "c"]].var()
        pd.testing.assert_series_equal(result, expected)

    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = bulk_variance(df, [])
        assert len(result) == 0

    def test_single_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = bulk_variance(df, ["x"])
        assert result["x"] == pytest.approx(df["x"].var())

    def test_constant_column_zero_variance(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = bulk_variance(df, ["a", "b"])
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] > 0

    def test_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
        result = bulk_variance(df, ["a", "b"])
        assert np.isnan(result["a"])
        assert result["b"] == pytest.approx(df["b"].var())

    def test_preserves_column_order(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"z": rng.standard_normal(50), "a": rng.standard_normal(50), "m": rng.standard_normal(50)})
        result = bulk_variance(df, ["z", "a", "m"])
        assert list(result.index) == ["z", "a", "m"]

    def test_many_columns_matches_pandas(self):
        """Verify batching does not lose columns when count exceeds _AGG_BATCH_SIZE."""
        rng = np.random.default_rng(123)
        ncols = 600
        df = pd.DataFrame({f"f_{i}": rng.standard_normal(50) for i in range(ncols)})
        cols = list(df.columns)
        result = bulk_variance(df, cols)
        expected = df.var()
        pd.testing.assert_series_equal(result, expected)


class TestBulkVarianceTimeColumn:
    """V4.4: within-snapshot variance via the new `time_column` kwarg.

    Motivating case: a feature that drifts monotonically per entity across
    snapshots (tenure proxy) appears high-variance under pooled estimation
    but has zero between-entity variance at any single snapshot. Restricting
    to one slice should reveal the absence of useful signal."""

    def _panel(self):
        # 3 entities x 4 snapshots. Feature `tenure` drifts +1 per
        # snapshot per entity (entity_offset + snapshot_idx). At any
        # single snapshot, tenure has identical value across entities
        # (zero between-entity variance), but pooled variance is ~5/4
        # because the same entity contributes 4 different values.
        rows = []
        for entity in range(3):
            for snap in range(4):
                rows.append({
                    "entity_id": f"E{entity}",
                    "as_of_date": snap,
                    # Static feature: same within entity across snapshots
                    "static": float(entity),
                    # Tenure proxy: entity-irrelevant, drifts with time
                    "tenure": float(snap),
                    # Mixed: real between-entity variance + cross-snapshot drift
                    "mixed": float(entity) + float(snap) * 0.1,
                })
        return pd.DataFrame(rows)

    def test_default_is_pooled_variance_unchanged(self):
        df = self._panel()
        result = bulk_variance(df, ["static", "tenure", "mixed"])
        expected = df[["static", "tenure", "mixed"]].var()
        pd.testing.assert_series_equal(result, expected)

    def test_time_column_restricts_to_latest_slice(self):
        df = self._panel()
        # Latest slice = as_of_date == 3 (max). At that slice:
        #   static = [0, 1, 2]               -> var = 1.0
        #   tenure = [3, 3, 3]               -> var = 0.0
        #   mixed  = [0.3, 1.3, 2.3]         -> var = 1.0
        result = bulk_variance(df, ["static", "tenure", "mixed"],
                               time_column="as_of_date")
        assert result["tenure"] == pytest.approx(0.0), (
            "tenure proxy must show zero within-snapshot variance"
        )
        assert result["static"] == pytest.approx(1.0)
        assert result["mixed"] == pytest.approx(1.0)

    def test_time_column_with_explicit_time_value(self):
        df = self._panel()
        # Same as test above but pin the slice explicitly to 0.
        result = bulk_variance(df, ["static", "tenure", "mixed"],
                               time_column="as_of_date", time_value=0)
        # At as_of_date == 0:
        #   static = [0, 1, 2]   -> var = 1.0
        #   tenure = [0, 0, 0]   -> var = 0.0
        #   mixed  = [0.0, 1.0, 2.0] -> var = 1.0
        assert result["tenure"] == pytest.approx(0.0)
        assert result["static"] == pytest.approx(1.0)
        assert result["mixed"] == pytest.approx(1.0)

    def test_pooled_inflates_tenure_proxy_variance(self):
        """Confirms the V4.4 motivation: pooled variance OVERESTIMATES
        tenure proxies' usefulness."""
        df = self._panel()
        pooled = bulk_variance(df, ["tenure"])
        within = bulk_variance(df, ["tenure"], time_column="as_of_date")
        # Pooled is ~1.45 (variance over [0,1,2,3,0,1,2,3,0,1,2,3]);
        # within-snapshot is 0.0. Ratio surfaces the bias.
        assert pooled["tenure"] > 1.0
        assert within["tenure"] == pytest.approx(0.0)
        assert pooled["tenure"] > within["tenure"]

    def test_time_column_missing_falls_back_to_pooled(self):
        # If the named time_column isn't in df, the kwarg is silently
        # ignored (pre-V4 behavior). Keeps existing call sites safe.
        df = self._panel()
        result = bulk_variance(df, ["static"], time_column="nonexistent")
        expected = df["static"].var()
        assert result["static"] == pytest.approx(expected)

    def test_time_column_empty_columns_returns_empty(self):
        df = self._panel()
        result = bulk_variance(df, [], time_column="as_of_date")
        assert len(result) == 0
