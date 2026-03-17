from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import _ML_CORR_THRESHOLD, batched_corr_matrix


class TestCorrMatrixMLThreshold:

    def test_threshold_is_reasonable(self):
        assert 50 <= _ML_CORR_THRESHOLD <= 500

    def test_small_column_count_uses_pandas_path(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({f"c_{i}": rng.standard_normal(n) for i in range(10)})
        expected = df.corr()
        result = batched_corr_matrix(df, list(df.columns))
        pd.testing.assert_frame_equal(result, expected)

    def test_large_column_count_still_works_pandas(self):
        rng = np.random.default_rng(42)
        n = 30
        ncols = _ML_CORR_THRESHOLD + 20
        df = pd.DataFrame({f"c_{i}": rng.standard_normal(n) for i in range(ncols)})
        result = batched_corr_matrix(df, list(df.columns))
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_full_matrix_preserves_symmetry(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({f"f_{i}": rng.standard_normal(100) for i in range(50)})
        result = batched_corr_matrix(df, list(df.columns))
        np.testing.assert_allclose(result.to_numpy(), result.to_numpy().T, atol=1e-10)

    def test_full_matrix_diagonal_is_one(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({f"f_{i}": rng.standard_normal(100) for i in range(20)})
        result = batched_corr_matrix(df, list(df.columns))
        for col in df.columns:
            assert result.loc[col, col] == pytest.approx(1.0)
