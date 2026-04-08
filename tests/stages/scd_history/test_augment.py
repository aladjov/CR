"""Tests for ``augment_parent_with_scd_state``.

Each test runs against pandas AND pyspark.pandas via the ``df_factory``
fixture in ``conftest.py``. The augment helper exists to bridge the
reconstructed-state view (one row per ``(parent_id, anchor_date)``) onto
the static parent attributes via a single distributed join. Its core
responsibility is to drop parent columns whose name case-insensitively
collides with a state-view column — Spark's default case-insensitive
resolver would otherwise raise ``[AMBIGUOUS_REFERENCE]`` downstream.
"""
from __future__ import annotations

import pytest

from customer_retention.core.compat import native_pd
from customer_retention.stages.scd_history import augment_parent_with_scd_state

T0 = native_pd.Timestamp("2024-01-01")
DAY = native_pd.Timedelta(days=1)


def _to_native(df) -> native_pd.DataFrame:
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


class TestBasicJoin:
    def test_no_collisions(self, df_factory):
        parent = df_factory([
            {"CASE_ID": "A", "OWNER_NAME": "Alice"},
            {"CASE_ID": "B", "OWNER_NAME": "Bob"},
        ])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
            {"CASE_ID": "A", "as_of_date": T0 + 30 * DAY, "Status": "Closed"},
            {"CASE_ID": "B", "as_of_date": T0, "Status": "Open"},
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert set(out.columns) == {"CASE_ID", "OWNER_NAME", "as_of_date", "Status"}
        assert len(out) == 3

    def test_inner_join_drops_unmatched_parents(self, df_factory):
        parent = df_factory([
            {"CASE_ID": "A", "OWNER_NAME": "Alice"},
            {"CASE_ID": "Z", "OWNER_NAME": "Zoe"},  # no state rows
        ])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert sorted(out["CASE_ID"].unique()) == ["A"]

    def test_left_join_keeps_unmatched_parents(self, df_factory):
        parent = df_factory([
            {"CASE_ID": "A", "OWNER_NAME": "Alice"},
            {"CASE_ID": "Z", "OWNER_NAME": "Zoe"},
        ])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Open"},
        ])

        out = _to_native(
            augment_parent_with_scd_state(parent, state, "CASE_ID", join_type="left")
        )

        assert sorted(out["CASE_ID"].unique()) == ["A", "Z"]


class TestCaseInsensitiveCollisionDrop:
    def test_single_uppercase_parent_collides_with_titlecase_state(self, df_factory):
        # SPS-shaped scenario: parent has ``ORIGIN`` (Snowflake convention),
        # state view has ``Origin`` (Salesforce convention). Spark resolves
        # both names to the same column case-insensitively → AMBIGUOUS_REFERENCE
        # unless we drop the parent column before the join.
        parent = df_factory([
            {"CASE_ID": "A", "ORIGIN": "Web", "OWNER_NAME": "Alice"},
        ])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Origin": "Email"},
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert "ORIGIN" not in out.columns
        assert "Origin" in out.columns
        assert "OWNER_NAME" in out.columns
        assert out.iloc[0]["Origin"] == "Email"

    def test_multiple_collisions(self, df_factory):
        parent = df_factory([
            {
                "CASE_ID": "A",
                "ORIGIN": "Web",
                "PRIORITY": "Low",
                "OWNER_NAME": "Alice",
            },
        ])
        state = df_factory([
            {
                "CASE_ID": "A",
                "as_of_date": T0,
                "Origin": "Email",
                "Priority": "High",
            },
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert "ORIGIN" not in out.columns
        assert "PRIORITY" not in out.columns
        assert "OWNER_NAME" in out.columns
        assert out.iloc[0]["Origin"] == "Email"
        assert out.iloc[0]["Priority"] == "High"

    def test_no_case_insensitive_duplicates_in_output(self, df_factory):
        parent = df_factory([
            {"CASE_ID": "A", "ORIGIN": "Web", "PRIORITY": "Low"},
        ])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Origin": "Email", "Priority": "High"},
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        lowered = [str(c).lower() for c in out.columns]
        assert len(lowered) == len(set(lowered)), (
            f"output columns have case-insensitive duplicates: {list(out.columns)}"
        )

    def test_join_key_excluded_from_collision_set(self, df_factory):
        # The join key (``CASE_ID``) appears on both sides by definition; the
        # helper must NOT treat that as a collision and drop it from the parent.
        parent = df_factory([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = df_factory([{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert "CASE_ID" in out.columns

    def test_collision_when_state_uppercase_parent_titlecase(self, df_factory):
        # Symmetric to the SPS case: the parent has the canonical-cased name
        # and the state view has the uppercase name. Parent still loses.
        parent = df_factory([{"CASE_ID": "A", "Origin": "Web"}])
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "ORIGIN": "Email"},
        ])

        out = _to_native(augment_parent_with_scd_state(parent, state, "CASE_ID"))

        assert "Origin" not in out.columns
        assert "ORIGIN" in out.columns
        assert out.iloc[0]["ORIGIN"] == "Email"


class TestValidation:
    def test_missing_join_key_in_parent_raises(self, df_factory):
        parent = df_factory([{"OTHER_ID": "A", "Status": "Open"}])
        state = df_factory([{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}])

        with pytest.raises(ValueError, match="CASE_ID"):
            augment_parent_with_scd_state(parent, state, "CASE_ID")

    def test_missing_join_key_in_state_view_raises(self, df_factory):
        parent = df_factory([{"CASE_ID": "A"}])
        state = df_factory([{"OTHER_ID": "A", "as_of_date": T0, "Status": "Open"}])

        with pytest.raises(ValueError, match="CASE_ID"):
            augment_parent_with_scd_state(parent, state, "CASE_ID")


class TestFailFastDuplicateGuards:
    """Fail-fast guards: surface upstream column-name dupes at the augment
    cell instead of letting them propagate to NB01 as ``[AMBIGUOUS_REFERENCE]``.
    """

    def test_parent_with_case_insensitive_duplicate_raises(self):
        # Native pandas only — pandas allows duplicate column names; pyspark
        # rejects them at construction time. The fail-fast guard runs in
        # both backends and surfaces the same error message at the cell.
        parent = native_pd.DataFrame(
            [{"CASE_ID": "A", "x": 1, "y": 2}],
        )
        # Force a case-insensitive duplicate by overwriting columns directly.
        parent.columns = ["CASE_ID", "ORIGIN", "Origin"]
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        with pytest.raises(ValueError, match="parent_df.*ORIGIN"):
            augment_parent_with_scd_state(parent, state, "CASE_ID")

    def test_state_view_with_case_insensitive_duplicate_raises(self):
        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "x": 1, "y": 2}],
        )
        state.columns = ["CASE_ID", "Status", "STATUS"]

        with pytest.raises(ValueError, match="state_view.*STATUS|state_view.*Status"):
            augment_parent_with_scd_state(parent, state, "CASE_ID")

    def test_collision_drops_all_case_variants_from_parent(self):
        # Pandas-only: parent has BOTH ``ORIGIN`` and ``Origin``, state has
        # ``Origin``. Both parent variants must be dropped — the state view
        # is the source of truth — and the output must be free of duplicates.
        parent = native_pd.DataFrame(
            [{"CASE_ID": "A", "x": 1, "y": 2, "z": "Alice"}]
        )
        parent.columns = ["CASE_ID", "ORIGIN", "Origin", "OWNER_NAME"]
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Origin": "Email"}]
        )

        out = augment_parent_with_scd_state(parent, state, "CASE_ID")

        cols_lower = [str(c).lower() for c in out.columns]
        assert cols_lower.count("origin") == 1, (
            f"output should have exactly one origin column, got {list(out.columns)}"
        )
        assert "OWNER_NAME" in out.columns
        assert out.iloc[0]["Origin"] == "Email"
