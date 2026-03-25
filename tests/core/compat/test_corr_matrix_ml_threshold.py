from __future__ import annotations

import inspect
from pathlib import Path

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


class TestNoImputerInCorrML:
    """Imputer.fit() serializes an ML model that exceeds Spark Connect's 1 GB
    limit with many columns (MODEL_SIZE_OVERFLOW_EXCEPTION).  Null handling
    must use SQL-based percentile_approx + fillna instead."""

    def test_spark_corr_matrix_ml_does_not_import_imputer(self):
        from customer_retention.core.compat import _spark_corr_matrix_ml
        source = inspect.getsource(_spark_corr_matrix_ml)
        assert "Imputer" not in source, (
            "_spark_corr_matrix_ml must not use pyspark.ml.feature.Imputer — "
            "use F.percentile_approx + fillna to avoid MODEL_SIZE_OVERFLOW"
        )

    def test_compat_init_has_no_imputer_import(self):
        compat_init = (
            Path(__file__).resolve().parents[3]
            / "src" / "customer_retention" / "core" / "compat" / "__init__.py"
        )
        source = compat_init.read_text()
        assert "from pyspark.ml.feature import Imputer" not in source, (
            "core/compat/__init__.py must not import Imputer — "
            "use F.percentile_approx + fillna for null handling before correlation"
        )
