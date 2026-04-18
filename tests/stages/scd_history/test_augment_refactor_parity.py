"""Parity tests: inline notebook implementation vs framework helpers.

Each test embeds the original notebook-cell logic as a reference
implementation, then asserts the new framework helpers produce
identical results.
"""
from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.project_context import CadenceInterval
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.analysis.auto_explorer.snapshot_grid import (
    DatasetGridVote,
    SnapshotGrid,
)
from customer_retention.core.compat import native_pd
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.scd_history import (
    AugmentationDiagnostics,
    AugmentationResult,
    SCDHistoryReconstructionConfig,
    augment_and_persist_parent_dataset,
    validate_scd_sources,
)

T0 = native_pd.Timestamp("2024-01-01")
DAY = native_pd.Timedelta(days=1)


def _scd_config(parent_table_dataset_name="case"):
    return SCDHistoryReconstructionConfig(
        enriched_view_name="test_view",
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        change_timestamp_column="CREATED_DATE",
        tracked_fields=("Status",),
        parent_table_dataset_name=parent_table_dataset_name,
    )


@pytest.fixture()
def namespace(tmp_path):
    ns = RunNamespace(root=tmp_path, run_id="parity-test")
    ns.setup()
    return ns


# ---------------------------------------------------------------------------
# 1. AugmentationResult backward-compat (2-tuple unpacking)
# ---------------------------------------------------------------------------

class TestAugmentationResultBackwardCompat:
    def test_two_tuple_unpacking_matches_old_return(self, namespace):
        parent = native_pd.DataFrame([
            {"CASE_ID": "A", "OWNER_NAME": "Alice"},
            {"CASE_ID": "B", "OWNER_NAME": "Bob"},
        ])
        state = native_pd.DataFrame([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "B", "as_of_date": T0, "Status": "Closed"},
        ])
        config = _scd_config()

        augmented, schema = augment_and_persist_parent_dataset(
            namespace=namespace, parent_dataset_name="case",
            parent_df=parent, state_view=state, config=config,
        )

        assert isinstance(augmented, native_pd.DataFrame)
        assert isinstance(schema, dict)
        assert set(augmented.columns) == {"CASE_ID", "OWNER_NAME", "as_of_date", "Status"}
        assert {k.lower() for k in schema} == {c.lower() for c in augmented.columns}

    def test_full_result_has_diagnostics_and_landing_path(self, namespace):
        parent = native_pd.DataFrame([
            {"CASE_ID": "A", "OWNER_NAME": "Alice"},
            {"CASE_ID": "B", "OWNER_NAME": "Bob"},
        ])
        state = native_pd.DataFrame([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "B", "as_of_date": T0, "Status": "Closed"},
        ])
        config = _scd_config()

        result = augment_and_persist_parent_dataset(
            namespace=namespace, parent_dataset_name="case",
            parent_df=parent, state_view=state, config=config,
        )

        assert isinstance(result, AugmentationResult)
        assert isinstance(result.diagnostics, AugmentationDiagnostics)
        assert result.landing_path
        assert result.diagnostics.parent_record_count == 2
        assert result.diagnostics.augmented_record_count == 2
        assert result.diagnostics.retention == 1.0


# ---------------------------------------------------------------------------
# 2. Diagnostics parity — inline computation vs framework
# ---------------------------------------------------------------------------

class TestDiagnosticsParity:
    """Compare the old inline diagnostic code (from the notebook cell)
    against the new ``AugmentationDiagnostics`` returned by the framework.
    """

    def _inline_diagnostics(self, parent_df, augmented, record_key, grid_dates):
        """Exact copy of the original notebook cell diagnostic code
        (with the _parent_raw fix applied).
        """
        _raw_cases = parent_df[record_key].nunique()
        _recon_cases = augmented[record_key].nunique()
        _retention = _recon_cases / _raw_cases if _raw_cases else 0.0
        _augmented_rows = len(augmented)
        _augmented_cols = len(augmented.columns)
        _grid_anchor_count = len(grid_dates)
        _fanout = _augmented_rows / _recon_cases if _recon_cases else 0.0
        return {
            "parent_record_count": _raw_cases,
            "augmented_record_count": _recon_cases,
            "retention": _retention,
            "augmented_rows": _augmented_rows,
            "augmented_cols": _augmented_cols,
            "grid_anchor_count": _grid_anchor_count,
            "fanout": _fanout,
        }

    def test_full_coverage(self, namespace):
        parent = native_pd.DataFrame([
            {"CASE_ID": "A", "OWNER": "Alice"},
            {"CASE_ID": "B", "OWNER": "Bob"},
            {"CASE_ID": "C", "OWNER": "Carol"},
        ])
        state = native_pd.DataFrame([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "A", "as_of_date": T0 + 30 * DAY, "Status": "Closed"},
            {"CASE_ID": "B", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "C", "as_of_date": T0, "Status": "Pending"},
        ])
        grid_dates = [T0, T0 + 30 * DAY]

        result = augment_and_persist_parent_dataset(
            namespace=namespace, parent_dataset_name="case",
            parent_df=parent, state_view=state, config=_scd_config(),
            grid_anchor_count=len(grid_dates),
        )
        inline = self._inline_diagnostics(
            parent, result.augmented_df, "CASE_ID", grid_dates,
        )
        fw = result.diagnostics

        assert fw.parent_record_count == inline["parent_record_count"]
        assert fw.augmented_record_count == inline["augmented_record_count"]
        assert fw.retention == pytest.approx(inline["retention"])
        assert fw.augmented_rows == inline["augmented_rows"]
        assert fw.augmented_cols == inline["augmented_cols"]
        assert fw.grid_anchor_count == inline["grid_anchor_count"]
        assert fw.fanout == pytest.approx(inline["fanout"])

    def test_partial_coverage_inner_join(self, namespace):
        parent = native_pd.DataFrame([
            {"CASE_ID": "A", "OWNER": "Alice"},
            {"CASE_ID": "B", "OWNER": "Bob"},
            {"CASE_ID": "Z", "OWNER": "Zoe"},
        ])
        state = native_pd.DataFrame([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "B", "as_of_date": T0, "Status": "Closed"},
        ])
        grid_dates = [T0]

        result = augment_and_persist_parent_dataset(
            namespace=namespace, parent_dataset_name="case",
            parent_df=parent, state_view=state, config=_scd_config(),
            grid_anchor_count=len(grid_dates),
        )
        inline = self._inline_diagnostics(
            parent, result.augmented_df, "CASE_ID", grid_dates,
        )
        fw = result.diagnostics

        assert fw.parent_record_count == 3
        assert fw.augmented_record_count == 2
        assert fw.retention == pytest.approx(2 / 3)
        assert fw.retention == pytest.approx(inline["retention"])
        assert fw.augmented_rows == inline["augmented_rows"]
        assert fw.fanout == pytest.approx(inline["fanout"])

    def test_empty_parent(self, namespace):
        parent = native_pd.DataFrame(columns=["CASE_ID", "OWNER"])
        state = native_pd.DataFrame(columns=["CASE_ID", "as_of_date", "Status"])

        result = augment_and_persist_parent_dataset(
            namespace=namespace, parent_dataset_name="case",
            parent_df=parent, state_view=state, config=_scd_config(),
        )
        assert result.diagnostics.retention == 0.0
        assert result.diagnostics.fanout == 0.0


# ---------------------------------------------------------------------------
# 3. AugmentationDiagnostics.assert_retention parity
# ---------------------------------------------------------------------------

class TestRetentionAssertParity:
    def test_above_floor_passes(self):
        diag = AugmentationDiagnostics(
            parent_record_count=100, augmented_record_count=90,
            retention=0.90, augmented_rows=900, augmented_cols=10,
            grid_anchor_count=10, fanout=10.0,
        )
        diag.assert_retention(0.70)

    def test_below_floor_raises_same_message_shape(self):
        diag = AugmentationDiagnostics(
            parent_record_count=100, augmented_record_count=60,
            retention=0.60, augmented_rows=600, augmented_cols=10,
            grid_anchor_count=10, fanout=10.0,
        )
        with pytest.raises(AssertionError, match=r"60\.0%.*below.*70%.*floor"):
            diag.assert_retention(0.70, parent_name="case")

    def test_default_floor_is_70_percent(self):
        diag = AugmentationDiagnostics(
            parent_record_count=100, augmented_record_count=69,
            retention=0.69, augmented_rows=690, augmented_cols=10,
            grid_anchor_count=10, fanout=10.0,
        )
        with pytest.raises(AssertionError, match="floor"):
            diag.assert_retention(parent_name="test")


# ---------------------------------------------------------------------------
# 4. summary_lines parity with inline print statements
# ---------------------------------------------------------------------------

class TestSummaryLinesParity:
    def test_matches_inline_format(self):
        diag = AugmentationDiagnostics(
            parent_record_count=6_526_928, augmented_record_count=4_306_666,
            retention=4_306_666 / 6_526_928, augmented_rows=117_042_960,
            augmented_cols=120, grid_anchor_count=89,
            fanout=117_042_960 / 4_306_666,
        )
        lines = diag.summary_lines("case")

        # The inline code produced exactly these two print() calls:
        inline_line1 = (
            f"SCD augmentation [case] retained {4_306_666:,} of "
            f"{6_526_928:,} parent records ({4_306_666 / 6_526_928:.1%})"
        )
        inline_line2 = (
            f"SCD augmentation [case] expanded to "
            f"{117_042_960:,} rows x 120 cols "
            f"across 89 grid anchors "
            f"(fan-out {117_042_960 / 4_306_666:.1f}x per retained parent)"
        )
        assert lines[0] == inline_line1
        assert lines[1] == inline_line2


# ---------------------------------------------------------------------------
# 5. SnapshotGrid.rehydrate_from_data_spans parity
# ---------------------------------------------------------------------------

class TestRehydrateGridParity:
    def _make_grid(self, dataset_names):
        votes = {
            name: DatasetGridVote(
                dataset_name=name, granularity=DatasetGranularity.EVENT_LEVEL,
            )
            for name in dataset_names
        }
        return SnapshotGrid(
            cadence_interval=CadenceInterval.MONTHLY,
            observation_window_days=180, purge_gap_days=37, label_window_days=30,
            dataset_votes=votes,
        )

    def _inline_rehydrate(self, snapshot_grid, spans, semantics):
        """Original inline rehydration logic from the notebook cell."""
        _full_data_votes = {}
        for _ds_name, (_start, _end) in spans.items():
            _ds_time_col = (semantics.get(_ds_name) or {}).get("time_column")
            _existing_vote = snapshot_grid.dataset_votes.get(_ds_name)
            if not _ds_time_col or _existing_vote is None:
                continue
            _existing_vote.data_span_start = str(_start)
            _existing_vote.data_span_end = str(_end)
            _full_data_votes[_ds_name] = _existing_vote
        if _full_data_votes:
            snapshot_grid.apply_votes(_full_data_votes)

    def test_identical_grid_dates(self):
        spans = {
            "account": ("2020-01-01", "2026-04-01"),
            "case": ("2018-06-01", "2026-04-01"),
        }
        semantics = {
            "account": {"time_column": "CREATED_DATE"},
            "case": {"time_column": "CREATED_DATE"},
        }

        grid_inline = self._make_grid(["account", "case"])
        self._inline_rehydrate(grid_inline, spans, semantics)

        grid_fw = self._make_grid(["account", "case"])
        grid_fw.rehydrate_from_data_spans(spans)

        assert grid_fw.grid_dates == grid_inline.grid_dates
        assert grid_fw.grid_start == grid_inline.grid_start
        assert grid_fw.grid_end == grid_inline.grid_end

    def test_unknown_dataset_skipped(self):
        grid = self._make_grid(["account"])
        status = grid.rehydrate_from_data_spans({
            "account": ("2020-01-01", "2026-04-01"),
            "unknown": ("2020-01-01", "2026-04-01"),
        })
        status_dict = dict(status)
        assert "OK" in status_dict["account"]
        assert "skipped" in status_dict["unknown"]

    def test_noop_when_grid_already_populated(self):
        grid = self._make_grid(["account"])
        grid.grid_dates = ["2025-01-01"]
        status = grid.rehydrate_from_data_spans({
            "account": ("2020-01-01", "2026-04-01"),
        })
        assert all("skipped" in msg for _, msg in status)
        assert grid.grid_dates == ["2025-01-01"]

    def test_noop_when_locked(self):
        grid = self._make_grid(["account"])
        grid.locked = True
        status = grid.rehydrate_from_data_spans({
            "account": ("2020-01-01", "2026-04-01"),
        })
        assert all("skipped" in msg for _, msg in status)


# ---------------------------------------------------------------------------
# 6. validate_scd_sources parity
# ---------------------------------------------------------------------------

class TestValidateScdSourcesParity:
    def _inline_validate(self, scd_sources, datasets, configs):
        """Original inline validation from the notebook cell."""
        for _parent_name in scd_sources:
            if _parent_name not in datasets:
                raise KeyError(
                    f"namespace.scd_history_sources references {_parent_name!r} "
                    f"which is not in `datasets`. Add it to the dataset "
                    f"registration first."
                )
            if _parent_name not in configs:
                raise KeyError(
                    f"namespace.scd_history_sources has an entry for "
                    f"{_parent_name!r} but SCD_RECONSTRUCTION_CONFIGS does not "
                    f"— add a SCDHistoryReconstructionConfig above."
                )

    def test_valid_sources_pass(self):
        scd = {"case": "path/to/history"}
        datasets = {"case": "path/to/case", "account": "path/to/account"}
        configs = {"case": _scd_config()}
        validate_scd_sources(scd, datasets, configs)
        self._inline_validate(scd, datasets, configs)

    def test_missing_dataset_same_error(self):
        scd = {"case": "path/to/history"}
        datasets = {"account": "path/to/account"}
        configs = {"case": _scd_config()}

        with pytest.raises(KeyError, match="case.*not in.*datasets"):
            validate_scd_sources(scd, datasets, configs)
        with pytest.raises(KeyError, match="case.*not in.*datasets"):
            self._inline_validate(scd, datasets, configs)

    def test_missing_config_same_error(self):
        scd = {"case": "path/to/history"}
        datasets = {"case": "path/to/case"}
        configs = {}

        with pytest.raises(KeyError, match="case.*SCD_RECONSTRUCTION_CONFIGS"):
            validate_scd_sources(scd, datasets, configs)
        with pytest.raises(KeyError, match="case.*SCD_RECONSTRUCTION_CONFIGS"):
            self._inline_validate(scd, datasets, configs)

    def test_empty_sources_is_noop(self):
        validate_scd_sources({}, {}, {})
