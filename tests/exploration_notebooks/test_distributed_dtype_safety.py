"""Static guards for the post-mortem of NB05 cell `f6986cb3` failing with
`TypeError: ufunc 'isnan' not supported for the input types`.

Root cause was `multi-column-slice.to_numpy()` on a `pyspark.pandas`
DataFrame yielding a `dtype=object` ndarray; the cell then ran
`np.isnan` on it. The fix moves the box-plot stats to a distributed
batched `.agg()` and routes single-column `.to_numpy()` calls used by
sklearn metrics through an explicit `.astype("float64")` so a nullable
`Int64` column can never become an object array.

These tests scan the notebook JSON to keep the regressions from
silently coming back through future notebook edits.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2] / "exploration_notebooks"


def _cell_source(notebook_name: str, cell_id: str) -> str:
    nb = json.loads((NB_DIR / notebook_name).read_text())
    cell = next((c for c in nb["cells"] if c.get("id") == cell_id), None)
    assert cell is not None, f"cell {cell_id} not in {notebook_name}"
    return "".join(cell["source"])


class TestNB05BoxPlotIsDistributed:
    def test_no_multi_column_to_numpy_with_isnan(self):
        src = _cell_source("05_relationship_analysis.ipynb", "f6986cb3")
        assert "np.isnan" not in src
        assert ".values" not in src
        assert "[plot_cols + [target]].to_numpy()" not in src

    def test_uses_batched_percentile_approx_on_spark(self):
        src = _cell_source("05_relationship_analysis.ipynb", "f6986cb3")
        assert "_BOX_AGG_BATCH = 100" in src
        assert "F.percentile_approx" in src
        assert "as_spark_df" in src
        assert "_is_spark_pandas(_box_df)" in src

    def test_plotly_uses_precomputed_quantiles(self):
        src = _cell_source("05_relationship_analysis.ipynb", "f6986cb3")
        assert "go.Box(" in src
        assert "q1=" in src and "median=" in src and "q3=" in src
        assert "lowerfence=" in src and "upperfence=" in src


class TestSingleColumnToNumpyCastsToFloat:
    def test_nb06_seg_rates_cast(self):
        src = _cell_source("06_feature_opportunities.ipynb", "b58df542")
        assert "_seg_stats[target].astype(\"float64\") * 100" in src

    def test_nb08_y_train_cast(self):
        src = _cell_source("08_baseline_experiments.ipynb", "acf253e6")
        assert "y_train.astype(\"float64\").to_numpy()" in src
        assert "_y_train_np" in src

    def test_nb11_y_true_cast(self):
        src = _cell_source("11_scoring_validation.ipynb", "ee54d054")
        assert "scoring_features[ORIGINAL_COLUMN].astype(\"float64\").to_numpy()" in src


class TestNB05LoadCoercesStringTargetToNumeric:
    """Silver_merged sometimes stores the binary target as `StringType` because
    of a NaN-padded-integer schema-inference fallthrough in the silver write
    path. NB05 must coerce the target to float64 at load so downstream
    `.mean()`, correlation, and effect-size calls don't trip pyspark.pandas's
    NumericType guard. Coercion failures raise — they don't silently mask a
    genuinely non-numeric target.
    """

    def test_load_findings_coerces_target_when_object_dtype(self):
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        assert "_t_dtype" in src
        assert "if \"object\" in _t_dtype or \"string\" in _t_dtype:" in src
        # Surgical fix replaced `df[_t] = df[_t].astype("float64")` (which
        # raises CAST_INVALID_INPUT under UC ANSI mode for non-numeric
        # categorical strings like 'Reseller') with a Spark `try_cast`
        # pattern that coerces uncastable values to NULL and surfaces the
        # offending rows via a one-shot agg. Test asserts the post-fix
        # shape so we don't silently regress to the bare-cast antipattern.
        assert "try_cast(" in src
        assert "as_spark_df(df)" in src
        assert "F.expr(" in src

    def test_load_findings_fails_fast_on_uncoercible_target(self):
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        # The post-fix flow still aborts when EVERY row fails to coerce
        # (target column is genuinely non-numeric) but with a richer
        # diagnostic message naming the silver_merged source. The bare
        # `raise RuntimeError(...)` was replaced by an aborts-on-100%-null
        # check; assert the operator-facing language stays put.
        assert "Target column" in src
        assert ("TemporalMerger" in src) or ("silver_merged" in src.lower())
