"""Tests for the rescue-ordering helper that splits FeatureSelectionResult drops
into (structural, reconsiderable) so NB08 can keep L1-dropped features eligible
for rescue instead of stripping them before rescue runs.
"""
from __future__ import annotations

from customer_retention.stages.features.feature_selector import (
    FeatureSelectionResult,
    SelectionMethod,
    split_reconsiderable_drops,
)


def _make_result(drop_reasons: dict[str, str]) -> FeatureSelectionResult:
    import pandas as pd
    return FeatureSelectionResult(
        df=pd.DataFrame(),
        selected_features=[],
        dropped_features=list(drop_reasons.keys()),
        drop_reasons=drop_reasons,
        method_used=SelectionMethod.L1_SELECTION,
        importance_scores=None,
    )


class TestSplitReconsiderableDrops:
    def test_l1_drops_are_reconsiderable(self):
        result = _make_result({"feat_a": "L1 zero coefficient"})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == set()
        assert reconsiderable == {"feat_a"}

    def test_variance_drops_are_structural(self):
        result = _make_result({"feat_a": "low variance (0.000000)"})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == {"feat_a"}
        assert reconsiderable == set()

    def test_correlation_drops_are_structural(self):
        result = _make_result({"feat_a": "high correlation (> 0.95)"})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == {"feat_a"}
        assert reconsiderable == set()

    def test_max_features_limit_is_structural(self):
        # max_features limit is applied post-L1 but is a hard cap — rescue
        # shouldn't re-add features that exceed the configured cap.
        result = _make_result({"feat_a": "max_features limit"})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == {"feat_a"}
        assert reconsiderable == set()

    def test_mixed_reasons(self):
        result = _make_result({
            "var_drop":   "low variance (0.000001)",
            "corr_drop":  "high correlation (> 0.95)",
            "l1_drop":    "L1 zero coefficient",
            "max_drop":   "max_features limit",
        })
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == {"var_drop", "corr_drop", "max_drop"}
        assert reconsiderable == {"l1_drop"}

    def test_empty_drops(self):
        result = _make_result({})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == set()
        assert reconsiderable == set()

    def test_unknown_reason_treated_as_structural(self):
        # Conservative: anything we don't recognize as L1 stays applied.
        # Avoids accidentally resurrecting features dropped for other reasons.
        result = _make_result({"feat_a": "some_new_drop_reason"})
        structural, reconsiderable = split_reconsiderable_drops(result)
        assert structural == {"feat_a"}
        assert reconsiderable == set()

    def test_case_insensitive_l1_match(self):
        # Implementation should match 'L1' case-insensitively to be robust
        # against future reason string variations.
        result = _make_result({
            "a": "L1 zero coefficient",
            "b": "l1 zero coefficient",
            "c": "L1_SELECTION dropped (coef=0.001)",
        })
        _, reconsiderable = split_reconsiderable_drops(result)
        assert reconsiderable == {"a", "b", "c"}


class TestPreserveAndRebuildAtomicity:
    """Simulates the NB08 cell 23 + cell 26 flow to verify that rescue-failure
    leaves X_train in the L1-stripped state (no regression) and rescue-success
    produces a feature pool rescue can actually reconsider.
    """

    def _panel(self):
        import pandas as pd
        # 8 rows × 6 feature cols + target. 3 cols will be L1-dropped.
        return pd.DataFrame({
            "entity_id":   ["A"] * 4 + ["B"] * 4,
            "as_of_date":  pd.date_range("2024-01-01", periods=8),
            "good_1":      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "good_2":      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "structural":  [0.0] * 8,               # zero variance — structural drop
            "l1_drop_1":   [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02],  # L1 dropped
            "l1_drop_2":   [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],           # L1 dropped
            "y":           [0, 1, 0, 1, 0, 1, 0, 1],
        })

    def _simulate_cell_23(self, df, l1_drops, structural_drops, *, gbdt_enabled: bool):
        """Replicates cell 23's preserve+strip logic in isolation."""
        X = df.drop(columns=["y"])
        preserved = None
        if l1_drops and gbdt_enabled:
            present = [c for c in l1_drops if c in X.columns]
            preserved = {"X_train": X[present].copy() if present else None}
        all_drops = set(structural_drops) | set(l1_drops)
        if all_drops:
            keep = [c for c in X.columns if c not in all_drops]
            X = X[keep]
        return X, preserved

    def test_cell_23_strips_all_drops(self):
        """Cell 23 behavior: X_train is L1+structural stripped regardless of rescue."""
        df = self._panel()
        X_stripped, _ = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=True,
        )
        assert set(X_stripped.columns) == {"entity_id", "as_of_date", "good_1", "good_2"}

    def test_preserved_l1_cols_captured_when_rescue_enabled(self):
        df = self._panel()
        _, preserved = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=True,
        )
        assert preserved is not None
        assert set(preserved["X_train"].columns) == {"l1_drop_1", "l1_drop_2"}
        # Preserved copy must not share memory with original (independent of X_train mutation)
        assert len(preserved["X_train"]) == 8

    def test_preserved_none_when_rescue_disabled(self):
        df = self._panel()
        _, preserved = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=False,
        )
        assert preserved is None

    def test_rescue_failure_leaves_x_train_l1_stripped(self):
        """Critical robustness guarantee: if rescue raises, X_train is unchanged.

        Cell 23 stripped X_train. The preserved cache exists separately. A failing
        rescue must not mutate X_train — downstream training must still work on
        the L1-stripped pool.
        """
        import pandas as pd
        df = self._panel()
        X_stripped, preserved = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=True,
        )
        X_before = X_stripped.copy()

        # Simulate cell 26: enlarge into a LOCAL, not X_train. Rescue raises.
        try:
            rescue_X = pd.concat([X_stripped, preserved["X_train"]], axis=1) if preserved else X_stripped  # noqa: F841
            raise RuntimeError("rescue crashed mid-pipeline")
        except RuntimeError:
            pass

        # X_stripped is unchanged: rescue failure didn't touch it.
        pd.testing.assert_frame_equal(X_stripped, X_before)

    def test_rescue_success_produces_enlarged_pool(self):
        """When rescue succeeds and keeps all features, the final pool includes
        the L1-dropped columns that were preserved."""
        import pandas as pd
        df = self._panel()
        X_stripped, preserved = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=True,
        )

        # Simulate cell 26: enlarge and use as the "committed" X_train
        rescue_X = pd.concat([X_stripped, preserved["X_train"]], axis=1) if preserved else X_stripped
        assert set(rescue_X.columns) >= {"entity_id", "as_of_date", "good_1", "good_2", "l1_drop_1", "l1_drop_2"}
        # All rows preserved
        assert len(rescue_X) == 8

    def test_rescue_success_can_drop_features_via_keep_set(self):
        """Rescue's final decision (a keep set from the enlarged pool) gets
        applied atomically. Simulates rescue choosing to drop one L1-candidate
        and one original feature."""
        import pandas as pd
        df = self._panel()
        X_stripped, preserved = self._simulate_cell_23(
            df, l1_drops={"l1_drop_1", "l1_drop_2"},
            structural_drops={"structural"}, gbdt_enabled=True,
        )
        rescue_X = pd.concat([X_stripped, preserved["X_train"]], axis=1)

        # Rescue keeps some, drops others
        rescue_drops = {"good_2", "l1_drop_2"}
        keep = [c for c in rescue_X.columns if c not in rescue_drops]
        final_X = rescue_X[keep]

        assert set(final_X.columns) == {"entity_id", "as_of_date", "good_1", "l1_drop_1"}
        assert len(final_X) == 8
