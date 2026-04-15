"""Tests for reconstruct_scd_history_at_grid.

Each test runs against pandas AND pyspark.pandas via the ``df_factory``
fixture parametrization in ``conftest.py``. The Spark-pandas variant is
skipped when pyspark is not importable.
"""
from __future__ import annotations

import logging

import pytest

from customer_retention.core.compat import native_pd
from customer_retention.stages.scd_history import (
    SCDHistoryReconstructionConfig,
    reconstruct_scd_history_at_grid,
)


def _base_config(**overrides) -> SCDHistoryReconstructionConfig:
    base = dict(
        enriched_view_name="reconstructed_case_history",
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        old_value_column="OLD_VALUE",
        change_timestamp_column="CREATED_DATE",
        unique_row_id_column="CASE_HISTORY_ID",
        tracked_fields=("Status", "Priority"),
        parent_table_dataset_name="case",
        parent_creation_timestamp_column="CREATED_DATE",
        parent_value_columns=(
            ("Status", "CASE_STATUS"),
            ("Priority", "PRIORITY"),
        ),
    )
    base.update(overrides)
    return SCDHistoryReconstructionConfig(**base)


def _to_native(df) -> native_pd.DataFrame:
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def _row_for(out: native_pd.DataFrame, case_id: str, anchor) -> native_pd.Series:
    anchor_ts = native_pd.Timestamp(anchor)
    matches = out[
        (out["CASE_ID"] == case_id)
        & (native_pd.to_datetime(out["as_of_date"]) == anchor_ts)
    ]
    assert len(matches) == 1, (
        f"expected exactly one row for {case_id} at {anchor_ts}, "
        f"got {len(matches)}"
    )
    return matches.iloc[0]


class TestSingleFieldReconstruction:
    def test_single_field_basic(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "A"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "A"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        out = out.sort_values(["CASE_ID", "as_of_date"]).reset_index(drop=True)

        # grid_dates: T0, T0+30, T0+60, T0+90, T0+120
        # Case A changes: day 1=Open, day 30=In Progress, day 60=On Hold,
        # day 90=In Progress, day 150=Closed (after grid range)
        assert _row_for(out, "A", grid_dates[0])["Status"] == "Open"  # parent fallback
        assert _row_for(out, "A", grid_dates[1])["Status"] == "In Progress"
        assert _row_for(out, "A", grid_dates[2])["Status"] == "On Hold"
        assert _row_for(out, "A", grid_dates[3])["Status"] == "In Progress"
        assert _row_for(out, "A", grid_dates[4])["Status"] == "In Progress"


class TestMultiFieldReconstruction:
    def test_multi_field_independent(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "B"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "B"]
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, _base_config(), parent)
        )

        anchor1 = _row_for(out, "B", grid_dates[1])
        assert anchor1["Status"] == "Open"
        assert anchor1["Priority"] == "Low"

        anchor2 = _row_for(out, "B", grid_dates[2])
        assert anchor2["Status"] == "In Progress"
        assert anchor2["Priority"] == "High"

        anchor4 = _row_for(out, "B", grid_dates[4])
        assert anchor4["Status"] == "Closed"
        assert anchor4["Priority"] == "High"


class TestParentFallback:
    def test_no_history_uses_parent_fallback(
        self, df_factory, synthetic_parent_records, grid_dates
    ):
        history = df_factory(
            [
                {
                    "CASE_HISTORY_ID": "h-1",
                    "CASE_ID": "A",
                    "FIELD": "Status",
                    "OLD_VALUE": None,
                    "NEW_VALUE": "Open",
                    "CREATED_DATE": grid_dates[0],
                }
            ]
        )
        parent = df_factory(synthetic_parent_records)
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        c_anchor = _row_for(out, "C", grid_dates[2])
        assert c_anchor["Status"] == "Pending"

    def test_no_history_returns_null_without_fallback(
        self, df_factory, synthetic_parent_records, grid_dates
    ):
        history = df_factory(
            [
                {
                    "CASE_HISTORY_ID": "h-1",
                    "CASE_ID": "A",
                    "FIELD": "Status",
                    "OLD_VALUE": None,
                    "NEW_VALUE": "Open",
                    "CREATED_DATE": grid_dates[0],
                }
            ]
        )
        parent = df_factory(synthetic_parent_records)
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        c_anchor = _row_for(out, "C", grid_dates[2])
        assert c_anchor["Status"] is None or native_pd.isna(c_anchor["Status"])


class TestBackwardAsOf:
    def test_change_at_exact_anchor_is_included(
        self, df_factory, synthetic_parent_records, grid_dates
    ):
        records = [
            {
                "CASE_HISTORY_ID": "h-1",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[2],
            }
        ]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "A"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        assert _row_for(out, "A", grid_dates[2])["Status"] == "Open"

    def test_change_after_anchor_not_included(
        self, df_factory, synthetic_parent_records, grid_dates
    ):
        records = [
            {
                "CASE_HISTORY_ID": "h-1",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "ParentInitial",
                "CREATED_DATE": grid_dates[3],
            }
        ]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "A"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        # At grid_dates[2] the change at grid_dates[3] is in the future →
        # the value should fall back to the parent column ("Open").
        assert _row_for(out, "A", grid_dates[2])["Status"] == "Open"


class TestUnknownFieldPolicy:
    def test_raise_on_unknown_field(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        # The fixture's Case A includes "Status" — config that excludes Status
        # would also need an unknown field. Use a config that mentions a field
        # not in the data instead, paired with on_unknown_field='raise'.
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "A"]
        history = df_factory(records)
        parent = df_factory(synthetic_parent_records)
        cfg = _base_config(
            tracked_fields=("Status", "NotPresent"),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            on_unknown_field="raise",
        )
        with pytest.raises(ValueError, match="NotPresent"):
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)

    def test_skip_unknown_field(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates, caplog
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "A"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "A"]
        )
        cfg = _base_config(
            tracked_fields=("Status", "NotPresent"),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            on_unknown_field="skip",
        )
        with caplog.at_level(
            logging.WARNING,
            logger="customer_retention.stages.scd_history.reconstruct",
        ):
            out = _to_native(
                reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
            )
        assert "Status" in out.columns
        assert "NotPresent" not in out.columns
        assert any("NotPresent" in rec.message for rec in caplog.records)


class TestColdStart:
    def test_cold_start_propagates_pre_grid_change(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "F"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "F"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        # The pre-grid change "PreGrid" is the most-recent value at every anchor
        for anchor in grid_dates:
            assert _row_for(out, "F", anchor)["Status"] == "PreGrid"


class TestSchema:
    def test_output_columns_match_tracked_fields(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        history = df_factory(synthetic_history_records)
        parent = df_factory(synthetic_parent_records)
        cfg = _base_config()

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        assert "CASE_ID" in out.columns
        assert "as_of_date" in out.columns
        for field_name in cfg.tracked_fields:
            assert field_name in out.columns
        # Raw change-log columns must NOT leak into the output
        assert "OLD_VALUE" not in out.columns
        assert "NEW_VALUE" not in out.columns
        assert "FIELD" not in out.columns


class TestTieBreaker:
    def test_tie_breaker_uses_unique_row_id(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "D"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "D"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        # Tiebreaker: larger CASE_HISTORY_ID ("D-status-zzz") wins
        assert _row_for(out, "D", grid_dates[1])["Status"] == "Second"

    def test_reconstruction_is_deterministic_across_reruns(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "D"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "D"]
        )
        cfg = _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),))

        out1 = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        out2 = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        sort_cols = ["CASE_ID", "as_of_date"]
        out1 = out1.sort_values(sort_cols).reset_index(drop=True)
        out2 = out2.sort_values(sort_cols).reset_index(drop=True)
        assert list(out1["Status"]) == list(out2["Status"])

    def test_unique_row_id_unset_no_collisions(
        self, df_factory, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        # Case A has no collisions — must work without a tiebreaker
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "A"]
        history = df_factory(records)
        parent = df_factory(
            [r for r in synthetic_parent_records if r["CASE_ID"] == "A"]
        )
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            unique_row_id_column=None,
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        # day 90 = "In Progress" is the last change <= grid_dates[4] (day 120)
        assert _row_for(out, "A", grid_dates[4])["Status"] == "In Progress"


class TestGridDateValidation:
    def test_grid_dates_none_rejected(self, synthetic_history_records, synthetic_parent_records):
        with pytest.raises(ValueError, match="grid_dates"):
            reconstruct_scd_history_at_grid(
                native_pd.DataFrame(synthetic_history_records),
                None,
                _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),)),
                native_pd.DataFrame(synthetic_parent_records),
            )

    def test_grid_dates_empty_rejected(self, synthetic_history_records, synthetic_parent_records):
        with pytest.raises(ValueError, match="grid_dates"):
            reconstruct_scd_history_at_grid(
                native_pd.DataFrame(synthetic_history_records),
                [],
                _base_config(tracked_fields=("Status",), parent_value_columns=(("Status", "CASE_STATUS"),)),
                native_pd.DataFrame(synthetic_parent_records),
            )


class TestPandasFallbackEdgeCases:
    def test_parent_fallback_missing_parent_column_raises(
        self, synthetic_history_records, grid_dates
    ):
        records = [r for r in synthetic_history_records if r["CASE_ID"] == "A"]
        history = native_pd.DataFrame(records)
        parent = native_pd.DataFrame([{"CASE_ID": "A"}])  # no CASE_STATUS column
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        with pytest.raises(ValueError, match="CASE_STATUS"):
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)


class TestParentColumnCaseInsensitiveResolution:
    """Snowflake-sourced parent tables mix unquoted UPPERCASE columns
    (``CASE_STATUS``, ``CASE_TYPE``) with quoted-identifier camelCase
    columns (``Origin``, ``Priority``) preserved from the Salesforce schema.
    The reconstruction must resolve ``parent_value_columns`` references
    case-insensitively so users do not have to know which case their
    upstream warehouse is using.
    """

    def test_uppercase_config_resolves_to_camelcase_parent(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "A",
                "FIELD": "Origin",
                "OLD_VALUE": None,
                "NEW_VALUE": "Email",
                "CREATED_DATE": grid_dates[2],
            }
        ])
        # Parent has the column as Salesforce-camelCase ``Origin``; the
        # config references it as Snowflake-style ``ORIGIN``.
        parent = df_factory([
            {"CASE_ID": "A", "Origin": "Web"},
            {"CASE_ID": "B", "Origin": "Phone"},
        ])
        cfg = _base_config(
            tracked_fields=("Origin",),
            parent_value_columns=(("Origin", "ORIGIN"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )

        # Case A: history change at grid_dates[2] propagates forward; before
        # that, the parent fallback ("Web") applies.
        assert _row_for(out, "A", grid_dates[0])["Origin"] == "Web"
        assert _row_for(out, "A", grid_dates[2])["Origin"] == "Email"
        # Case B: no history → parent fallback ("Phone") for every anchor.
        assert _row_for(out, "B", grid_dates[0])["Origin"] == "Phone"
        assert _row_for(out, "B", grid_dates[4])["Origin"] == "Phone"

    def test_camelcase_config_resolves_to_uppercase_parent(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[1],
            }
        ])
        parent = df_factory([
            {"CASE_ID": "A", "CASE_STATUS": "Pending"},
        ])
        # Config uses Salesforce camelCase ``case_status`` while the
        # parent has the unquoted Snowflake ``CASE_STATUS``.
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "case_status"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )

        assert _row_for(out, "A", grid_dates[0])["Status"] == "Pending"
        assert _row_for(out, "A", grid_dates[1])["Status"] == "Open"

    def test_mixed_case_resolution_across_multiple_columns(
        self, df_factory, grid_dates
    ):
        # Snowflake reality for SPS: CASE_STATUS / CASE_TYPE are unquoted
        # (UPPERCASE) but Origin / Priority are quoted (camelCase).
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[0],
            },
            {
                "CASE_HISTORY_ID": "h2",
                "CASE_ID": "A",
                "FIELD": "Type",
                "OLD_VALUE": None,
                "NEW_VALUE": "Support",
                "CREATED_DATE": grid_dates[0],
            },
            {
                "CASE_HISTORY_ID": "h3",
                "CASE_ID": "A",
                "FIELD": "Origin",
                "OLD_VALUE": None,
                "NEW_VALUE": "Web",
                "CREATED_DATE": grid_dates[0],
            },
            {
                "CASE_HISTORY_ID": "h4",
                "CASE_ID": "A",
                "FIELD": "Priority",
                "OLD_VALUE": None,
                "NEW_VALUE": "High",
                "CREATED_DATE": grid_dates[0],
            },
        ])
        parent = df_factory([
            {
                "CASE_ID": "A",
                "CASE_STATUS": "ParentStatus",
                "CASE_TYPE": "ParentType",
                "Origin": "ParentOrigin",
                "Priority": "ParentPriority",
            },
            {
                "CASE_ID": "B",
                "CASE_STATUS": "ParentStatus",
                "CASE_TYPE": "ParentType",
                "Origin": "ParentOrigin",
                "Priority": "ParentPriority",
            },
        ])
        cfg = _base_config(
            tracked_fields=("Status", "Type", "Origin", "Priority"),
            parent_value_columns=(
                ("Status", "CASE_STATUS"),
                ("Type", "CASE_TYPE"),
                ("Origin", "ORIGIN"),
                ("Priority", "PRIORITY"),
            ),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )

        # Case A: history at grid_dates[0] is the source of truth at every
        # subsequent anchor (no later changes).
        a_anchor = _row_for(out, "A", grid_dates[2])
        assert a_anchor["Status"] == "Open"
        assert a_anchor["Type"] == "Support"
        assert a_anchor["Origin"] == "Web"
        assert a_anchor["Priority"] == "High"
        # Case B: no history → parent fallback resolved case-insensitively.
        b_anchor = _row_for(out, "B", grid_dates[0])
        assert b_anchor["Status"] == "ParentStatus"
        assert b_anchor["Type"] == "ParentType"
        assert b_anchor["Origin"] == "ParentOrigin"
        assert b_anchor["Priority"] == "ParentPriority"

    def test_unresolvable_parent_column_still_raises(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "A",
                "FIELD": "Origin",
                "OLD_VALUE": None,
                "NEW_VALUE": "Email",
                "CREATED_DATE": grid_dates[1],
            }
        ])
        parent = df_factory([
            {"CASE_ID": "A", "OTHER_COL": "x"},
        ])
        cfg = _base_config(
            tracked_fields=("Origin",),
            parent_value_columns=(("Origin", "ORIGIN"),),
        )

        with pytest.raises(ValueError, match="ORIGIN"):
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)


class _FakeColumn:
    """Records the column expression for assertion."""
    def __init__(self, expr):
        self.expr = expr
    def __repr__(self):
        return f"<col {self.expr}>"
    def __eq__(self, other):
        return _FakeColumn(f"({self.expr}) == ({other!r})")
    def cast(self, _):
        return self
    def alias(self, name):
        return _FakeColumn(f"{self.expr} AS {name}")
    def asc(self):
        return _FakeColumn(f"{self.expr} ASC")
    def isin(self, values):
        return _FakeColumn(f"{self.expr} IN {tuple(values)}")


class _FakeSparkDF:
    """Records Spark DataFrame method calls without needing a session."""
    def __init__(self, columns, *, history=None, sparkSession=None):
        self._columns = list(columns)
        self.history = history if history is not None else []
        self.sparkSession = sparkSession or _FakeSparkSession()

    @property
    def columns(self):
        return list(self._columns)

    def withColumn(self, name, expr):  # noqa: N802 — Spark API
        self.history.append(("withColumn", name, str(expr)))
        new_cols = self._columns if name in self._columns else self._columns + [name]
        return _FakeSparkDF(new_cols, history=self.history, sparkSession=self.sparkSession)

    def filter(self, expr):
        self.history.append(("filter", str(expr)))
        return _FakeSparkDF(self._columns, history=self.history, sparkSession=self.sparkSession)

    def select(self, *cols):
        self.history.append(("select", tuple(str(c) for c in cols)))
        new_cols = []
        for c in cols:
            s = str(c)
            if " AS " in s:
                new_cols.append(s.split(" AS ")[-1])
            elif hasattr(c, "_jc"):
                new_cols.append(str(c).split(".")[-1])
            else:
                new_cols.append(s)
        return _FakeSparkDF(new_cols, history=self.history, sparkSession=self.sparkSession)

    def distinct(self):
        self.history.append(("distinct",))
        return _FakeSparkDF(self._columns, history=self.history, sparkSession=self.sparkSession)

    def collect(self):
        self.history.append(("collect",))
        # Return rows with the expected fields populated for distinct field discovery
        return []

    def unionByName(self, other):  # noqa: N802 — Spark API
        self.history.append(("unionByName",))
        return _FakeSparkDF(self._columns, history=self.history, sparkSession=self.sparkSession)

    def crossJoin(self, other):  # noqa: N802 — Spark API
        self.history.append(("crossJoin",))
        return _FakeSparkDF(self._columns + other._columns, history=self.history, sparkSession=self.sparkSession)

    def join(self, other, on=None, how=None):
        self.history.append(("join", on, how))
        return _FakeSparkDF(self._columns + other._columns, history=self.history, sparkSession=self.sparkSession)

    def drop(self, *cols):
        self.history.append(("drop", cols))
        remaining = [c for c in self._columns if c not in cols]
        return _FakeSparkDF(remaining, history=self.history, sparkSession=self.sparkSession)

    def dropDuplicates(self, _cols):  # noqa: N802 — Spark API
        self.history.append(("dropDuplicates",))
        return _FakeSparkDF(self._columns, history=self.history, sparkSession=self.sparkSession)


class _FakeSparkSession:
    def createDataFrame(self, data, schema=None):  # noqa: N802 — Spark API
        cols = schema if isinstance(schema, list) else ["as_of_date"]
        return _FakeSparkDF(cols)


class TestDistributedSparkExpressions:
    """Validates the Spark expression structure built by the distributed
    helpers — pyspark.sql.functions works without an active session so we can
    structurally verify the SQL semantics."""

    def test_resolve_effective_fields_native_spark_path(self, monkeypatch):
        from customer_retention.stages.scd_history import reconstruct as mod

        history = _FakeSparkDF(["FIELD"])
        monkeypatch.setattr(mod, "_is_spark_pandas", lambda obj: False)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: obj is history)
        monkeypatch.setattr(
            mod,
            "_spark_distinct_field_values",
            lambda df, col: {"Status", "Priority"},
        )
        cfg = _base_config()
        result = mod._resolve_effective_fields(history, cfg)
        assert set(result) == {"Status", "Priority"}

    def test_resolve_effective_fields_spark_pandas_path(self, monkeypatch):
        from customer_retention.stages.scd_history import reconstruct as mod

        marker = object()
        spark_df = _FakeSparkDF(["FIELD"])
        monkeypatch.setattr(mod, "_is_spark_pandas", lambda obj: obj is marker)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: False)
        monkeypatch.setattr(mod, "as_spark_df", lambda obj: spark_df)
        monkeypatch.setattr(
            mod, "_spark_distinct_field_values", lambda df, col: {"Status", "Priority"}
        )
        cfg = _base_config()
        result = mod._resolve_effective_fields(marker, cfg)
        assert set(result) == {"Status", "Priority"}

    def test_distinct_field_values_helper_collects_field_column(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _spark_distinct_field_values,
        )

        df = _FakeSparkDF(["FIELD"])
        result = _spark_distinct_field_values(df, "FIELD")
        assert result == set()
        ops = [h[0] for h in df.history]
        assert "select" in ops
        assert "distinct" in ops
        assert "collect" in ops

    def test_build_spine_distributed_broadcasts_grid(self, monkeypatch):
        pytest.importorskip("pyspark")
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.stages.scd_history.reconstruct import (
            _build_spine_distributed,
        )

        broadcast_calls: list = []
        monkeypatch.setattr(
            F, "broadcast", lambda arg: broadcast_calls.append(arg) or arg
        )

        history = _FakeSparkDF(["CASE_ID", "FIELD", "CREATED_DATE"])
        cfg = _base_config()
        grid = [native_pd.Timestamp("2024-01-01"), native_pd.Timestamp("2024-02-01")]

        result = _build_spine_distributed(history, grid, cfg, None)
        assert isinstance(result, _FakeSparkDF)
        ops = [h[0] for h in history.history]
        assert "select" in ops
        assert "distinct" in ops
        assert "crossJoin" in ops
        assert len(broadcast_calls) == 1, "the small grid must be broadcast-hinted"

    def test_build_spine_distributed_unions_parent_ids(self, monkeypatch):
        pytest.importorskip("pyspark")
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.stages.scd_history.reconstruct import (
            _build_spine_distributed,
        )

        monkeypatch.setattr(F, "broadcast", lambda arg: arg)

        session = _FakeSparkSession()
        history = _FakeSparkDF(["CASE_ID"], sparkSession=session)
        parent = _FakeSparkDF(["CASE_ID", "CASE_STATUS"], sparkSession=session)
        cfg = _base_config()
        grid = [native_pd.Timestamp("2024-01-01")]

        _build_spine_distributed(history, grid, cfg, parent)
        ops = [h[0] for h in history.history]
        assert "unionByName" in ops

    def test_history_anchor_union_without_row_id(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _build_history_anchor_union,
        )

        history = _FakeSparkDF(["CASE_ID", "FIELD", "NEW_VALUE", "CREATED_DATE"])
        spine = _FakeSparkDF(["CASE_ID", "as_of_date"])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            unique_row_id_column=None,
        )
        result = _build_history_anchor_union(history, spine, cfg, has_row_id=False)
        assert isinstance(result, _FakeSparkDF)

    def test_attach_field_values_without_row_id(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _attach_field_values_in_one_window_pass,
        )

        unioned = _FakeSparkDF(["__rec__", "__ts__", "__field__", "__value__", "__is_anchor__"])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            unique_row_id_column=None,
        )
        result = _attach_field_values_in_one_window_pass(
            unioned, cfg, ["Status"], has_row_id=False
        )
        assert isinstance(result, _FakeSparkDF)

    def test_attach_field_values_with_row_id(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _attach_field_values_in_one_window_pass,
        )

        unioned = _FakeSparkDF(
            ["__rec__", "__ts__", "__field__", "__value__", "__is_anchor__", "__row_id__"]
        )
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )
        result = _attach_field_values_in_one_window_pass(
            unioned, cfg, ["Status"], has_row_id=True
        )
        assert isinstance(result, _FakeSparkDF)

    def test_history_anchor_union_emits_uniform_schema(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _build_history_anchor_union,
        )

        history = _FakeSparkDF(
            ["CASE_ID", "FIELD", "NEW_VALUE", "CREATED_DATE", "CASE_HISTORY_ID"]
        )
        spine = _FakeSparkDF(["CASE_ID", "as_of_date"])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        result = _build_history_anchor_union(history, spine, cfg, has_row_id=True)
        assert isinstance(result, _FakeSparkDF)
        history_ops = [h[0] for h in history.history]
        spine_ops = [h[0] for h in spine.history]
        assert "select" in history_ops
        assert "select" in spine_ops

    def test_attach_field_values_uses_single_window_with_conditional_aggregates(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _attach_field_values_in_one_window_pass,
        )

        unioned = _FakeSparkDF(["__rec__", "__ts__", "__field__", "__value__", "__is_anchor__"])
        cfg = _base_config(
            tracked_fields=("Status", "Priority", "Type"),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        result = _attach_field_values_in_one_window_pass(
            unioned, cfg, ["Status", "Priority", "Type"], has_row_id=False
        )
        assert isinstance(result, _FakeSparkDF)
        ops = [h[0] for h in unioned.history]
        # Exactly ONE select call — all K conditional aggregates fold into a
        # single Spark stage
        assert ops.count("select") == 1

    def test_apply_parent_fallbacks_distributed_raises_on_missing_column(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _apply_parent_fallbacks_distributed,
        )

        per_field = _FakeSparkDF(["CASE_ID", "as_of_date", "Status"])
        parent = _FakeSparkDF(["CASE_ID"])  # missing CASE_STATUS
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        with pytest.raises(ValueError, match="CASE_STATUS"):
            _apply_parent_fallbacks_distributed(
                per_field, parent, cfg, [("Status", "CASE_STATUS")]
            )

    def test_apply_parent_fallbacks_distributed_one_join_for_all_fields(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history.reconstruct import (
            _apply_parent_fallbacks_distributed,
        )

        per_field = _FakeSparkDF(["CASE_ID", "as_of_date", "Status", "Priority"])
        parent = _FakeSparkDF(["CASE_ID", "CASE_STATUS", "PRIORITY"])
        cfg = _base_config(
            tracked_fields=("Status", "Priority"),
            parent_value_columns=(
                ("Status", "CASE_STATUS"),
                ("Priority", "PRIORITY"),
            ),
        )

        _apply_parent_fallbacks_distributed(
            per_field, parent, cfg,
            [("Status", "CASE_STATUS"), ("Priority", "PRIORITY")],
        )
        ops = [h[0] for h in per_field.history]
        # Exactly ONE join — both field fallbacks fold into a single shuffle
        assert ops.count("join") == 1


class TestDispatchToDistributed:
    def test_pyspark_pandas_routes_to_distributed_path(self, monkeypatch):
        from customer_retention.stages.scd_history import reconstruct as mod

        captured = {}
        fake_input = object()
        fake_parent = object()
        fake_spark_history = _FakeSparkDF(["CASE_ID", "FIELD"])
        fake_spark_parent = _FakeSparkDF(["CASE_ID", "CASE_STATUS"])

        def fake_is_spark_pandas(obj):
            return obj is fake_input or obj is fake_parent

        def fake_as_spark_df(obj):
            return fake_spark_history if obj is fake_input else fake_spark_parent

        def fake_resolve(history, _config):
            captured["resolved_with"] = history
            return ["Status"]

        def fake_distributed(history, dates, config, fields, parent):
            captured["distributed_called"] = True
            captured["distributed_history"] = history
            captured["distributed_parent"] = parent
            return "spark_result"

        def fake_as_pandas_api(obj):
            captured["wrapped"] = obj
            return "pyspark_pandas_result"

        monkeypatch.setattr(mod, "_is_spark_pandas", fake_is_spark_pandas)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: False)
        monkeypatch.setattr(mod, "as_spark_df", fake_as_spark_df)
        monkeypatch.setattr(mod, "_resolve_effective_fields", fake_resolve)
        monkeypatch.setattr(mod, "_reconstruct_distributed", fake_distributed)
        monkeypatch.setattr(mod, "as_pandas_api", fake_as_pandas_api)

        result = mod.reconstruct_scd_history_at_grid(
            fake_input, [native_pd.Timestamp("2024-01-01")], _base_config(), fake_parent
        )
        assert captured["distributed_called"] is True
        assert captured["distributed_history"] is fake_spark_history
        assert captured["distributed_parent"] is fake_spark_parent
        assert result == "pyspark_pandas_result"

    def test_native_spark_routes_to_distributed_directly(self, monkeypatch):
        from customer_retention.stages.scd_history import reconstruct as mod

        captured = {}
        fake_input = object()

        def fake_distributed(history, dates, config, fields, parent):
            captured["called"] = True
            return "spark_result"

        monkeypatch.setattr(mod, "_is_spark_pandas", lambda obj: False)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: obj is fake_input)
        monkeypatch.setattr(
            mod, "_resolve_effective_fields", lambda h, c: ["Status"]
        )
        monkeypatch.setattr(mod, "_reconstruct_distributed", fake_distributed)
        monkeypatch.setattr(
            mod, "as_pandas_api",
            lambda obj: pytest.fail("must NOT call as_pandas_api on native spark"),
        )

        result = mod.reconstruct_scd_history_at_grid(
            fake_input, [native_pd.Timestamp("2024-01-01")], _base_config()
        )
        assert result == "spark_result"
        assert captured["called"] is True


class TestPandasInternalHelpers:
    def test_all_parent_record_ids_pandas_without_parent(self):
        from customer_retention.stages.scd_history.reconstruct import (
            _all_parent_record_ids_pandas,
        )
        history = native_pd.DataFrame({"CASE_ID": ["a", "b", "a"]})
        cfg = _base_config()
        ids = _all_parent_record_ids_pandas(history, None, cfg)
        assert ids == ["a", "b"]


class TestReconstructDistributedFlow:
    def test_distributed_pipeline_no_parent_skips_fallback(self, monkeypatch):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history import reconstruct as mod

        history = _FakeSparkDF(["CASE_ID", "FIELD", "NEW_VALUE", "CREATED_DATE"])
        cfg = SCDHistoryReconstructionConfig(
            enriched_view_name="v",
            parent_record_key="CASE_ID",
            field_column="FIELD",
            new_value_column="NEW_VALUE",
            change_timestamp_column="CREATED_DATE",
            tracked_fields=("Status",),
        )
        grid = [native_pd.Timestamp("2024-01-01")]

        monkeypatch.setattr(
            mod, "_build_spine_distributed",
            lambda h, dates, c, p: _FakeSparkDF(["CASE_ID", "as_of_date"]),
        )
        monkeypatch.setattr(
            mod, "_build_history_anchor_union",
            lambda h, s, c, has_row_id: _FakeSparkDF(["__rec__", "__ts__", "__field__", "__value__", "__is_anchor__"]),
        )
        monkeypatch.setattr(
            mod, "_attach_field_values_in_one_window_pass",
            lambda u, c, fields, has_row_id: _FakeSparkDF(
                ["__rec__", "__ts__", "__is_anchor__", *fields]
            ),
        )
        monkeypatch.setattr(
            mod, "_apply_parent_fallbacks_distributed",
            lambda *args, **kwargs: pytest.fail("fallback must NOT run when spark_parent is None"),
        )

        result = mod._reconstruct_distributed(history, grid, cfg, ["Status"], None)
        assert isinstance(result, _FakeSparkDF)

    def test_distributed_pipeline_does_one_window_pass_and_one_fallback_join(self, monkeypatch):
        pytest.importorskip("pyspark")
        from customer_retention.stages.scd_history import reconstruct as mod

        history = _FakeSparkDF(["CASE_ID", "FIELD", "NEW_VALUE", "CREATED_DATE"])
        parent = _FakeSparkDF(["CASE_ID", "CASE_STATUS", "PRIORITY"])
        cfg = SCDHistoryReconstructionConfig(
            enriched_view_name="v",
            parent_record_key="CASE_ID",
            field_column="FIELD",
            new_value_column="NEW_VALUE",
            change_timestamp_column="CREATED_DATE",
            tracked_fields=("Status", "Priority"),
            parent_table_dataset_name="case",
            parent_value_columns=(("Status", "CASE_STATUS"), ("Priority", "PRIORITY")),
        )
        grid = [native_pd.Timestamp("2024-01-01")]

        call_counts = {"window_pass": 0, "fallback": 0}

        monkeypatch.setattr(
            mod, "_build_spine_distributed",
            lambda h, dates, c, p: _FakeSparkDF(["CASE_ID", "as_of_date"]),
        )
        monkeypatch.setattr(
            mod, "_build_history_anchor_union",
            lambda h, s, c, has_row_id: _FakeSparkDF(["__rec__", "__ts__", "__field__", "__value__", "__is_anchor__"]),
        )

        def fake_window_pass(unioned, c, fields, has_row_id):
            call_counts["window_pass"] += 1
            return _FakeSparkDF(["__rec__", "__ts__", "__is_anchor__", *fields])

        def fake_fallback(per_field, p, c, applicable):
            call_counts["fallback"] += 1
            assert len(applicable) == 2
            return per_field

        monkeypatch.setattr(mod, "_attach_field_values_in_one_window_pass", fake_window_pass)
        monkeypatch.setattr(mod, "_apply_parent_fallbacks_distributed", fake_fallback)

        result = mod._reconstruct_distributed(history, grid, cfg, ["Status", "Priority"], parent)
        assert isinstance(result, _FakeSparkDF)
        # ONE window pass for all K fields, ONE fallback for all K fields
        assert call_counts == {"window_pass": 1, "fallback": 1}


class TestParity:
    def test_pandas_and_spark_pandas_equivalence(
        self, synthetic_history_records, synthetic_parent_records, grid_dates
    ):
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession

        try:
            SparkSession.builder.master("local[1]").getOrCreate()
        except Exception as exc:
            pytest.skip(f"local Spark session unavailable: {exc}")

        import pyspark.pandas as ps

        cfg = _base_config()

        out_pd = _to_native(
            reconstruct_scd_history_at_grid(
                native_pd.DataFrame(synthetic_history_records),
                grid_dates,
                cfg,
                native_pd.DataFrame(synthetic_parent_records),
            )
        )
        out_ps = _to_native(
            reconstruct_scd_history_at_grid(
                ps.DataFrame(synthetic_history_records),
                grid_dates,
                cfg,
                ps.DataFrame(synthetic_parent_records),
            )
        )

        sort_cols = ["CASE_ID", "as_of_date"]
        out_pd = out_pd.sort_values(sort_cols).reset_index(drop=True)
        out_ps = out_ps.sort_values(sort_cols).reset_index(drop=True)

        assert len(out_pd) == len(out_ps)
        assert list(out_pd["CASE_ID"]) == list(out_ps["CASE_ID"])
        for f in cfg.tracked_fields:
            pd_vals = [None if native_pd.isna(v) else v for v in out_pd[f]]
            ps_vals = [None if native_pd.isna(v) else v for v in out_ps[f]]
            assert pd_vals == ps_vals, f"mismatch on {f}"


class TestSpinePrunedByParentCreationDate:
    """State before a parent exists is meaningless. When
    ``parent_creation_timestamp_column`` is configured and present on the
    parent, the spine only emits anchor rows where
    ``anchor_date >= parent.creation_date``. This is the main lever that
    keeps the state-view size linear in ``sum_of_parent_lifetimes`` rather
    than ``num_parents × num_anchors`` — the latter OOMs executor disk on
    SPS-scale datasets (5M cases × hundreds of anchors = ~1B rows).
    """

    def test_anchors_before_creation_date_are_excluded(
        self, df_factory, grid_dates
    ):
        late_parent_created_at = grid_dates[3]
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "LATE",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": late_parent_created_at,
            }
        ])
        parent = df_factory([
            {
                "CASE_ID": "LATE",
                "CREATED_DATE": late_parent_created_at,
                "CASE_STATUS": "Open",
                "PRIORITY": "Med",
            }
        ])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        out_late = out[out["CASE_ID"] == "LATE"]

        kept_anchors = sorted(native_pd.to_datetime(out_late["as_of_date"]).unique())
        expected = [native_pd.Timestamp(a) for a in grid_dates[3:]]
        assert kept_anchors == expected

    def test_no_creation_column_on_parent_keeps_all_anchors(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "X",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[2],
            }
        ])
        parent = df_factory([
            {"CASE_ID": "X", "CASE_STATUS": "Open"},
        ])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        kept = sorted(
            native_pd.to_datetime(out[out["CASE_ID"] == "X"]["as_of_date"]).unique()
        )
        assert kept == [native_pd.Timestamp(a) for a in grid_dates]

    def test_config_without_creation_column_keeps_all_anchors(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "X",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[2],
            }
        ])
        parent = df_factory([
            {"CASE_ID": "X", "CREATED_DATE": grid_dates[3], "CASE_STATUS": "Open"},
        ])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            parent_creation_timestamp_column=None,
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        kept = sorted(
            native_pd.to_datetime(out[out["CASE_ID"] == "X"]["as_of_date"]).unique()
        )
        assert kept == [native_pd.Timestamp(a) for a in grid_dates]

    def test_history_only_parent_keeps_all_anchors_when_parent_row_absent(
        self, df_factory, grid_dates
    ):
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "HIST_ONLY",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": grid_dates[0],
            }
        ])
        parent = df_factory([
            {
                "CASE_ID": "OTHER",
                "CREATED_DATE": grid_dates[4],
                "CASE_STATUS": "Closed",
            }
        ])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        hist_anchors = sorted(
            native_pd.to_datetime(
                out[out["CASE_ID"] == "HIST_ONLY"]["as_of_date"]
            ).unique()
        )
        assert hist_anchors == [native_pd.Timestamp(a) for a in grid_dates]

    def test_creation_column_resolved_case_insensitively(
        self, df_factory, grid_dates
    ):
        late_parent_created_at = grid_dates[2]
        history = df_factory([
            {
                "CASE_HISTORY_ID": "h1",
                "CASE_ID": "LATE",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": late_parent_created_at,
            }
        ])
        parent = df_factory([
            {
                "CASE_ID": "LATE",
                "created_date": late_parent_created_at,
                "CASE_STATUS": "Open",
            }
        ])
        cfg = _base_config(
            tracked_fields=("Status",),
            parent_value_columns=(("Status", "CASE_STATUS"),),
            parent_creation_timestamp_column="CREATED_DATE",
        )

        out = _to_native(
            reconstruct_scd_history_at_grid(history, grid_dates, cfg, parent)
        )
        kept = sorted(
            native_pd.to_datetime(out[out["CASE_ID"] == "LATE"]["as_of_date"]).unique()
        )
        expected = [native_pd.Timestamp(a) for a in grid_dates[2:]]
        assert kept == expected
