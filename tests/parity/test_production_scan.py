from __future__ import annotations

from pathlib import Path

import pytest

from customer_retention.parity import ApplyOpKind
from customer_retention.parity.production_scan import (
    AuditScope,
    infer_dataset_from_path,
    scan_generated_pipeline,
    scan_production_source,
)

_FRAMEWORK_MODULES = (
    "customer_retention.analysis.auto_explorer.sampling",
    "customer_retention.stages.lifecycle.enrich",
    "customer_retention.stages.profiling.target_validator",
    "customer_retention.stages.profiling.time_window_aggregator",
    "customer_retention.stages.temporal.temporal_merger",
    "customer_retention.stages.modeling.data_splitter",
    "customer_retention.transforms.ops",
    "customer_retention.transforms.fitted",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_modules_imported():
    import importlib
    for mod in _FRAMEWORK_MODULES:
        importlib.import_module(mod)
    yield


def _write_source(tmp_path: Path, rel: str, source: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    return target


class TestDatasetInferenceFromPath:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("landing/landing_contract.py", "contract"),
            ("landing/landing_account.py", "account"),
            ("bronze/bronze_event_subscription.py", "subscription"),
            ("bronze/bronze_entity_account.py", "account"),
            ("bronze/bronze_entity_subscription_aggregated.py", "subscription"),
            ("silver/silver_featureset_cust_data.py", "<merged>"),
            ("gold/gold_features_cust_data.py", "<merged>"),
            ("training/ml_experiment.py", "<merged>"),
            ("target_derive/run_target_derive.py", "<merged>"),
            ("config.py", None),
            ("runner.py", None),
        ],
    )
    def test_inference(self, filename, expected):
        assert infer_dataset_from_path(Path(filename)) == expected


class TestScanProductionSourceLanding:
    def test_decorated_call_is_recorded(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "\n"
            "def run_landing(df, config):\n"
            "    return enrich_lifecycle_dataset(df, config)\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        assert len(entries) == 1
        assert entries[0].kind is ApplyOpKind.LIFECYCLE_ENRICH
        assert entries[0].dataset == "contract"

    def test_temporal_lookback_recorded(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_account.py",
            "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
            "\n"
            "def run_landing(df, intent):\n"
            "    return apply_temporal_lookback(df, 'feature_timestamp', intent)\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        assert any(
            e.kind is ApplyOpKind.TEMPORAL_LOOKBACK and e.dataset == "account"
            for e in entries
        )

    def test_helper_indirection_resolves(self, tmp_path):
        """The walker must follow locally-defined helpers per the design decision."""
        path = _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
            "\n"
            "def derive_temporal_columns(df, time_col, datetime_columns):\n"
            "    df, _ = derive_extra_datetime_features(df, time_column=time_col, datetime_columns=datetime_columns)\n"
            "    return df\n"
            "\n"
            "def run_landing(df, config):\n"
            "    df = enrich_lifecycle_dataset(df, config)\n"
            "    df = derive_temporal_columns(df, 'event_timestamp', ['col_a', 'col_b'])\n"
            "    return df\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        kinds = {e.kind for e in entries}
        assert ApplyOpKind.LIFECYCLE_ENRICH in kinds
        assert ApplyOpKind.DATETIME_DERIVE in kinds


class TestScanGeneratedPipeline:
    def test_scans_landing_directory_only_in_landing_scope(self, tmp_path):
        _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "def run_landing(df, c):\n"
            "    return enrich_lifecycle_dataset(df, c)\n"
        )
        _write_source(
            tmp_path, "bronze/bronze_entity_contract.py",
            "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
            "def run_bronze(df):\n"
            "    df, _ = derive_extra_datetime_features(df, time_column='ts', datetime_columns=[])\n"
            "    return df\n"
        )
        manifest = scan_generated_pipeline(tmp_path, scope=AuditScope.LANDING)
        kinds_by_dataset = {ds: manifest.kinds_for(ds) for ds in manifest.datasets()}
        assert ApplyOpKind.LIFECYCLE_ENRICH in kinds_by_dataset.get("contract", set())
        # Bronze entries excluded in LANDING scope
        assert ApplyOpKind.DATETIME_DERIVE not in kinds_by_dataset.get("contract", set())

    def test_all_scope_picks_up_every_stage(self, tmp_path):
        _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "def run_landing(df, c):\n"
            "    return enrich_lifecycle_dataset(df, c)\n"
        )
        _write_source(
            tmp_path, "silver/silver_featureset_cn.py",
            "from customer_retention.stages.temporal.temporal_merger import TemporalMerger\n"
            "def run_silver(spine, datasets):\n"
            "    return TemporalMerger().merge_all(spine, datasets)\n"
        )
        manifest = scan_generated_pipeline(tmp_path, scope=AuditScope.ALL)
        all_kinds = {e.kind for e in manifest.entries}
        assert ApplyOpKind.LIFECYCLE_ENRICH in all_kinds
        assert ApplyOpKind.SILVER_TEMPORAL_MERGE in all_kinds


class TestHistoryWindowRegressionScenario:
    """Locks in the fix for the production-only TEMPORAL_LOOKBACK bug.

    Pre-fix renderer would emit `apply_temporal_lookback` for INTERVAL_START_TIME
    datasets (contract, subscription) even though exploration gates the call off.
    This test simulates both pre-fix and post-fix renderer output and verifies the
    parity diff catches the gap.
    """

    def test_postfix_no_lookback_for_interval_dataset(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "\n"
            "def run_landing(df, config):\n"
            "    return enrich_lifecycle_dataset(df, config)\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        # Post-fix: only LIFECYCLE_ENRICH, no TEMPORAL_LOOKBACK
        kinds = {e.kind for e in entries}
        assert ApplyOpKind.LIFECYCLE_ENRICH in kinds
        assert ApplyOpKind.TEMPORAL_LOOKBACK not in kinds

    def test_prefix_simulation_emits_lookback_and_produces_gap(self, tmp_path):
        """Simulates the pre-fix bug: production emits TEMPORAL_LOOKBACK
        unconditionally; exploration skips it. The diff catches it."""
        from customer_retention.parity import (
            Manifest,
            ManifestEntry,
            SourceLocation,
            diff_manifests,
            fingerprint_kwargs,
        )
        from customer_retention.parity.gaps import GapKind

        prefix_landing = _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
            "\n"
            "def run_landing(df, config, intent):\n"
            "    df = enrich_lifecycle_dataset(df, config)\n"
            "    return apply_temporal_lookback(df, 'feature_timestamp', intent)\n"
        )
        production_manifest = Manifest(entries=tuple(
            scan_production_source(prefix_landing.read_text(), file_path=prefix_landing)
        ))
        # Exploration manifest: contract has only LIFECYCLE_ENRICH (lookback gated off)
        exploration_manifest = Manifest(entries=(
            ManifestEntry(
                dataset="contract",
                kind=ApplyOpKind.LIFECYCLE_ENRICH,
                kwargs_fingerprint=fingerprint_kwargs({}),
                call_order=0,
                source_location=SourceLocation(file=Path("01.ipynb"), line=10),
            ),
        ))
        gaps = diff_manifests(exploration_manifest, production_manifest)
        history_window_gap = next(
            (g for g in gaps
             if g.gap_kind is GapKind.PRODUCTION_ONLY
             and g.op_kind is ApplyOpKind.TEMPORAL_LOOKBACK), None
        )
        assert history_window_gap is not None
        assert history_window_gap.dataset == "contract"
        assert "landing_contract.py" in str(history_window_gap.production_location.file)


class TestKwargsCaptureProduction:
    def test_literal_kwargs_recorded(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_contract.py",
            "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
            "def run_landing(df):\n"
            "    df, _ = derive_extra_datetime_features(df, time_column='feature_timestamp', datetime_columns=['x', 'y'])\n"
            "    return df\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        entry = entries[0]
        fp = entry.kwargs_fingerprint
        assert fp["time_column"] == "feature_timestamp"
        assert fp["datetime_columns"] == ("x", "y")


class TestTemplateEmittedNames:
    """Renderer templates emit operations through conventional inline helpers
    (`apply_history_window`, `derive_temporal_columns`, etc.). The production
    scan must recognise these as the right apply kind even when they're
    locally-defined rather than decorated framework imports."""

    def test_apply_history_window_recorded_as_temporal_lookback(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_contract.py",
            "import pandas as pd\n"
            "\n"
            "def apply_history_window(df):\n"
            "    return df[df['feature_timestamp'] >= '2020-01-01']\n"
            "\n"
            "def run_landing(df):\n"
            "    return apply_history_window(df)\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        kinds = {e.kind for e in entries}
        assert ApplyOpKind.TEMPORAL_LOOKBACK in kinds

    def test_landing_template_emits_four_derive_kinds(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_x.py",
            "def derive_feature_timestamp(df): return df\n"
            "def derive_label_timestamp(df): return df\n"
            "def derive_label_available_flag(df): return df\n"
            "def derive_datetime_features(df): return df\n"
            "\n"
            "def run_landing(df):\n"
            "    df = derive_feature_timestamp(df)\n"
            "    df = derive_label_timestamp(df)\n"
            "    df = derive_label_available_flag(df)\n"
            "    df = derive_datetime_features(df)\n"
            "    return df\n"
        )
        kinds = {e.kind for e in scan_production_source(path.read_text(), file_path=path)}
        assert {
            ApplyOpKind.FEATURE_TIMESTAMP_DERIVE,
            ApplyOpKind.LABEL_TIMESTAMP_DERIVE,
            ApplyOpKind.LABEL_AVAILABLE_FLAG,
            ApplyOpKind.DATETIME_DERIVE,
        }.issubset(kinds)

    def test_compat_apply_sql_predicate_recorded_as_landing_filter(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_x.py",
            "from customer_retention.core.compat import apply_sql_predicate\n"
            "def run_landing(df):\n"
            "    return apply_sql_predicate(df, 'amount > 0')\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        kinds = {e.kind for e in entries}
        assert ApplyOpKind.LANDING_FILTER in kinds

    def test_silver_target_label_map_recorded(self, tmp_path):
        path = _write_source(
            tmp_path, "silver/silver_featureset_cn.py",
            "def apply_target_label_map(df): return df\n"
            "def run_silver(df):\n"
            "    return apply_target_label_map(df)\n"
        )
        kinds = {e.kind for e in scan_production_source(path.read_text(), file_path=path)}
        assert ApplyOpKind.SILVER_TARGET_LABEL_MAP in kinds

    def test_gold_transforms_recorded(self, tmp_path):
        path = _write_source(
            tmp_path, "gold/gold_features_cn.py",
            "def apply_transformations(df): return df\n"
            "def apply_encodings(df): return df\n"
            "def apply_feature_selection(df): return df\n"
            "def run_gold(df):\n"
            "    df = apply_transformations(df)\n"
            "    df = apply_encodings(df)\n"
            "    df = apply_feature_selection(df)\n"
            "    return df\n"
        )
        kinds = {e.kind for e in scan_production_source(path.read_text(), file_path=path)}
        assert ApplyOpKind.GOLD_TRANSFORMATION in kinds
        assert ApplyOpKind.GOLD_ENCODING in kinds
        assert ApplyOpKind.GOLD_FEATURE_SPEC_GATE in kinds


class TestHistoryWindowFixOnLiveRenderer:
    """End-to-end: run the actual `PipelineGenerator` against a fixture
    engagement and verify the audit reports zero `PRODUCTION_ONLY /
    TEMPORAL_LOOKBACK` gaps. This is the regression test that locks in the
    `should_apply_lookback` fix at the renderer level."""

    def test_real_pipeline_no_temporal_lookback_for_unflagged_dataset(self, tmp_path):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        findings_dir = Path("tests/fixtures/user_extensions/sps_mini")
        out = tmp_path / "generated"
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="parity_smoke",
        ).generate()

        manifest = scan_generated_pipeline(out, scope=AuditScope.LANDING)
        # The sps_mini fixture has no `lookback_periods` in its intent, so
        # `_build_history_window_config` returns None → no apply_history_window
        # emitted. The audit confirms that exactly.
        kinds = {e.kind for e in manifest.entries}
        assert ApplyOpKind.TEMPORAL_LOOKBACK not in kinds


class TestEmptyAndDegenerateSources:
    def test_empty_source_no_entries(self, tmp_path):
        path = _write_source(tmp_path, "landing/landing_x.py", "")
        assert scan_production_source(path.read_text(), file_path=path) == []

    def test_syntax_error_warns_and_returns_empty(self, tmp_path):
        path = _write_source(tmp_path, "landing/landing_x.py", "def broken((((")
        entries = scan_production_source(path.read_text(), file_path=path)
        assert entries == []

    def test_undecorated_calls_ignored(self, tmp_path):
        path = _write_source(
            tmp_path, "landing/landing_x.py",
            "import math\n"
            "def run_landing(df):\n"
            "    math.sqrt(df.value)\n"
            "    return df\n"
        )
        entries = scan_production_source(path.read_text(), file_path=path)
        assert entries == []
