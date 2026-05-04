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
    """Post-FW-12, NB05's load cell delegates target classification + encoding
    to ``customer_retention.stages.profiling.target_validator`` (the same
    helper NB01's `validate_target_dtype` cell runs). The cell:

    * Reads the persisted ``findings.metadata['target_label_map']`` first.
    * Falls back to in-cell ``TARGET_LABEL_MAP`` only as an escape hatch.
    * Calls the shared ``apply_target_encoding`` helper so silver/gold
      derivation and the analysis cell apply the same mapping.
    * Calls ``validate_target_or_raise`` when no mapping is registered to
      surface the same actionable error NB01 emits.
    """

    def test_load_findings_uses_shared_target_validator(self):
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        # Imports the FW-12 helper module instead of inlining the
        # try_cast / case-when pattern.
        assert "from customer_retention.stages.profiling.target_validator import" in src
        assert "apply_target_encoding" in src
        assert "classify_target_dtype" in src

    def test_load_findings_reads_persisted_label_map_first(self):
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        # findings.metadata is the canonical source-of-truth set in NB01.
        assert "findings.metadata.get(\"target_label_map\")" in src
        # In-cell TARGET_LABEL_MAP remains as an escape-hatch override.
        assert "TARGET_LABEL_MAP: dict = {}" in src
        # Order matters: TARGET_LABEL_MAP wins on conflict, otherwise registered.
        assert "_effective_map = TARGET_LABEL_MAP or _registered_map" in src

    def test_load_findings_propagates_label_map_to_registry(self):
        """After NB05 initializes/loads the registry, it must persist the
        findings.metadata mapping into ``registry.gold.target_label_map``
        so codegen sees it."""
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        assert "registry.set_target_label_map(" in src
        assert "registry.get_target_label_map()" in src

    def test_load_findings_fails_fast_on_uncoercible_target(self):
        """No persisted mapping AND a multi-class string target → calls
        ``validate_target_or_raise`` which raises with the paste-ready
        ``registry.set_target_label_map(...)`` template."""
        src = _cell_source("05_relationship_analysis.ipynb", "09b14f1e")
        assert "validate_target_or_raise(df, _t)" in src
        assert "multi_class_string" in src
