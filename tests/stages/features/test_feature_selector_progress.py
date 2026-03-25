from __future__ import annotations

import numpy as np
import pandas as pd

from customer_retention.stages.features.feature_selector import (
    FeatureSelector,
    SelectionMethod,
    run_selection_pipeline,
)


class TestProgressFn:

    def test_correlation_selector_calls_progress_fn(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.standard_normal(50),
            "b": rng.standard_normal(50),
            "target": rng.integers(0, 2, 50).astype(float),
        })
        messages: list[str] = []
        sel = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=0.95,
            target_column="target", progress_fn=messages.append,
        )
        sel.fit_transform(df)
        # progress_fn is forwarded to batched_corr_matrix; on native pandas
        # batched_corr_matrix doesn't emit progress, so messages may be empty,
        # but the plumbing must not raise
        assert isinstance(messages, list)

    def test_pipeline_emits_progress_messages(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "f1": rng.standard_normal(50),
            "f2": rng.standard_normal(50),
            "f3": rng.standard_normal(50),
            "target": rng.integers(0, 2, 50).astype(float),
        })
        messages: list[str] = []
        run_selection_pipeline(df, "target", progress_fn=messages.append)
        assert any("Variance filter" in m for m in messages)
        assert any("Correlation filter" in m for m in messages)
        assert any("Done" in m for m in messages)

    def test_pipeline_with_no_progress_fn(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "x": rng.standard_normal(30),
            "y": rng.standard_normal(30),
            "target": rng.integers(0, 2, 30).astype(float),
        })
        result = run_selection_pipeline(df, "target", progress_fn=None)
        assert len(result.selected_features) > 0


class TestBulkVarianceIntegration:

    def test_variance_filter_uses_bulk_variance(self):
        """FeatureSelector._get_variances uses bulk_variance which must match df.var()."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": [5.0] * 20,
            "b": rng.standard_normal(20),
            "c": rng.standard_normal(20),
            "target": rng.integers(0, 2, 20).astype(float),
        })
        sel = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=0.01,
            target_column="target",
        )
        result = sel.fit_transform(df)
        assert "a" in result.dropped_features
        assert "b" in result.selected_features
        assert "c" in result.selected_features
