"""Tests for ``augment_parent_with_scd_state`` and ``augment_and_persist_parent_dataset``.

Each test runs against pandas AND pyspark.pandas via the ``df_factory``
fixture in ``conftest.py``. The augment helper exists to bridge the
reconstructed-state view (one row per ``(parent_id, anchor_date)``) onto
the static parent attributes via a single distributed join. Its core
responsibility is to drop parent columns whose name case-insensitively
collides with a state-view column — Spark's default case-insensitive
resolver would otherwise raise ``[AMBIGUOUS_REFERENCE]`` downstream.

``augment_and_persist_parent_dataset`` is the single supported NB00 entry
point: it composes the augment helper with ``save_active_dataset`` so the
augmented schema lands on the canonical Delta path that NB01 reads from.
"""
from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.active_dataset_store import (
    load_active_dataset,
    save_active_dataset,
)
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat import native_pd
from customer_retention.stages.scd_history import (
    SCDHistoryReconstructionConfig,
    augment_and_persist_parent_dataset,
    augment_parent_with_scd_state,
)

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


class TestIdempotencyOnRerun:
    """The augment cell is rerun-safe: when a user re-executes only the
    0.11.5 cell after a previous successful run, ``datasets["case"]`` still
    points to the previously-augmented view (it carries ``as_of_date`` and
    every tracked field). The helper must drop ALL state-view columns from
    the parent — not just the configured tracked fields — so that the
    re-run produces a clean output instead of duplicating ``as_of_date``.
    """

    def test_rerun_with_previously_augmented_parent(self, df_factory):
        # Simulate the second invocation: ``parent`` already carries the
        # output schema of a prior augmentation run.
        previously_augmented_parent = df_factory([
            {
                "CASE_ID": "A",
                "OWNER_NAME": "Alice",
                "as_of_date": T0,
                "Status": "Open",
                "Origin": "Web",
            },
            {
                "CASE_ID": "A",
                "OWNER_NAME": "Alice",
                "as_of_date": T0 + 30 * DAY,
                "Status": "Closed",
                "Origin": "Web",
            },
        ])
        # Fresh state view from a re-reconstruction step.
        state = df_factory([
            {"CASE_ID": "A", "as_of_date": T0, "Status": "Reopened", "Origin": "Phone"},
            {"CASE_ID": "A", "as_of_date": T0 + 30 * DAY, "Status": "Closed", "Origin": "Phone"},
        ])

        out = _to_native(
            augment_parent_with_scd_state(
                previously_augmented_parent, state, "CASE_ID"
            )
        )

        # No case-insensitive duplicates anywhere.
        cols_lower = [str(c).lower() for c in out.columns]
        assert len(cols_lower) == len(set(cols_lower)), (
            f"output has duplicates after re-run: {list(out.columns)}"
        )
        # State view is the source of truth — its values win.
        assert "as_of_date" in out.columns
        assert "OWNER_NAME" in out.columns
        # Re-running and re-augmenting should yield the state-view values.
        assert set(out["Status"].dropna()) == {"Reopened", "Closed"}
        assert set(out["Origin"].dropna()) == {"Phone"}


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


def _scd_config(
    *,
    parent_table_dataset_name: str | None = None,
    enriched_view_name: str = "case_with_state_history",
) -> SCDHistoryReconstructionConfig:
    return SCDHistoryReconstructionConfig(
        enriched_view_name=enriched_view_name,
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        change_timestamp_column="CREATED_DATE",
        tracked_fields=("Status",),
        parent_table_dataset_name=parent_table_dataset_name,
    )


@pytest.fixture()
def namespace(tmp_path):
    ns = RunNamespace(root=tmp_path, run_id="scd-test1234")
    ns.setup()
    return ns


class TestAugmentAndPersistParentDataset:
    """The single NB00 entry point: augment + overwrite landing Delta + return schema.

    The whole class exists because the previous user-cell pattern of
    ``join + register_temp_view`` left the augmented schema invisible to
    NB01's ``load_active_dataset`` path — temp views are session-scoped,
    landing Delta is the canonical sink.
    """

    def test_persists_to_landing_delta_and_returns_augmented(self, namespace):
        parent = native_pd.DataFrame([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )
        config = _scd_config(parent_table_dataset_name="case")

        augmented, schema = augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=config,
        )

        landing = namespace.landing_table_dir("case")
        assert landing.is_dir(), f"landing Delta not created at {landing}"

        loaded = load_active_dataset(namespace, "case")
        assert "CASE_ID" in loaded.columns
        assert "OWNER_NAME" in loaded.columns
        assert "as_of_date" in loaded.columns
        assert "Status" in loaded.columns
        assert loaded.iloc[0]["Status"] == "Open"
        assert {str(c).lower() for c in augmented.columns} == {
            str(c).lower() for c in loaded.columns
        }
        assert isinstance(schema, dict)

    def test_returned_schema_keys_match_landing_delta_columns(self, namespace):
        parent = native_pd.DataFrame([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        _, schema = augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
        )

        loaded = load_active_dataset(namespace, "case")
        assert {k.lower() for k in schema} == {
            str(c).lower() for c in loaded.columns
        }

    def test_sps_scenario_drops_origin_priority_as_of_date_then_persists(
        self, namespace
    ):
        # Reproduces the production failure shape: Snowflake parent with
        # ``ORIGIN``, ``PRIORITY``, AND a pre-existing ``as_of_date`` column
        # (e.g. from a prior augment run). All three must be dropped before
        # the join, and the persisted Delta must have no case-insensitive
        # duplicates.
        parent = native_pd.DataFrame(
            [
                {
                    "CASE_ID": "A",
                    "ORIGIN": "Web",
                    "PRIORITY": "Low",
                    "OWNER_NAME": "Alice",
                    "as_of_date": T0 - DAY,
                },
            ]
        )
        state = native_pd.DataFrame(
            [
                {
                    "CASE_ID": "A",
                    "as_of_date": T0,
                    "Origin": "Email",
                    "Priority": "High",
                    "Status": "Open",
                },
            ]
        )

        augmented, _ = augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(parent_table_dataset_name="case"),
        )

        loaded = load_active_dataset(namespace, "case")
        cols_lower = [str(c).lower() for c in loaded.columns]
        assert len(cols_lower) == len(set(cols_lower)), (
            f"persisted Delta has case-insensitive dupes: {list(loaded.columns)}"
        )
        assert "ORIGIN" not in loaded.columns
        assert "PRIORITY" not in loaded.columns
        assert "Origin" in loaded.columns
        assert "Priority" in loaded.columns
        assert "OWNER_NAME" in loaded.columns
        assert loaded.iloc[0]["Origin"] == "Email"
        assert loaded.iloc[0]["Priority"] == "High"
        # The state-view ``as_of_date`` wins, parent's stale value is dropped.
        assert native_pd.Timestamp(loaded.iloc[0]["as_of_date"]) == T0

    def test_rerun_overwrites_landing_delta_idempotently(self, namespace):
        parent = native_pd.DataFrame(
            [{"CASE_ID": "A", "OWNER_NAME": "Alice"}]
        )
        state_first = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )
        state_second = native_pd.DataFrame(
            [
                {"CASE_ID": "A", "as_of_date": T0, "Status": "Reopened"},
                {"CASE_ID": "A", "as_of_date": T0 + 30 * DAY, "Status": "Closed"},
            ]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state_first,
            config=_scd_config(),
        )
        # Re-run feeds the previously-augmented landing Delta back in as the
        # parent (simulating a user re-execution of the cell).
        previously_persisted = load_active_dataset(namespace, "case")

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=previously_persisted,
            state_view=state_second,
            config=_scd_config(),
        )

        final = load_active_dataset(namespace, "case")
        cols_lower = [str(c).lower() for c in final.columns]
        assert len(cols_lower) == len(set(cols_lower)), (
            f"persisted Delta has dupes after re-run: {list(final.columns)}"
        )
        assert set(final["Status"].dropna()) == {"Reopened", "Closed"}
        assert "OWNER_NAME" in final.columns

    def test_empty_dataset_name_raises(self, namespace):
        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        with pytest.raises(ValueError, match="parent_dataset_name"):
            augment_and_persist_parent_dataset(
                namespace=namespace,
                parent_dataset_name="",
                parent_df=parent,
                state_view=state,
                config=_scd_config(),
            )

    def test_whitespace_dataset_name_raises(self, namespace):
        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        with pytest.raises(ValueError, match="parent_dataset_name"):
            augment_and_persist_parent_dataset(
                namespace=namespace,
                parent_dataset_name="   ",
                parent_df=parent,
                state_view=state,
                config=_scd_config(),
            )

    def test_config_dataset_name_mismatch_raises(self, namespace):
        # The config says the parent dataset is "case", but the cell passes
        # "support_ticket" — this is the copy-paste failure mode.
        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        with pytest.raises(
            ValueError, match=r"parent_table_dataset_name.*case.*support_ticket",
        ):
            augment_and_persist_parent_dataset(
                namespace=namespace,
                parent_dataset_name="support_ticket",
                parent_df=parent,
                state_view=state,
                config=_scd_config(parent_table_dataset_name="case"),
            )

    def test_config_dataset_name_unset_is_allowed(self, namespace):
        # When ``parent_table_dataset_name`` is None, the facade trusts the
        # caller — no mismatch check fires.
        parent = native_pd.DataFrame([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(parent_table_dataset_name=None),
        )

        assert namespace.landing_table_dir("case").is_dir()

    def test_landing_schema_drift_after_write_raises(
        self, namespace, monkeypatch
    ):
        # Simulate a Delta-write reordering or silent column drop by
        # monkey-patching the schema-readback to return fewer columns.
        from customer_retention.stages.scd_history import augment as augment_module

        parent = native_pd.DataFrame(
            [{"CASE_ID": "A", "OWNER_NAME": "Alice"}]
        )
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        original = augment_module._read_landing_columns

        def _drop_a_column(landing_path):
            cols = original(landing_path)
            return [c for c in cols if c != "OWNER_NAME"]

        monkeypatch.setattr(augment_module, "_read_landing_columns", _drop_a_column)

        with pytest.raises(ValueError, match="landing Delta schema drift"):
            augment_and_persist_parent_dataset(
                namespace=namespace,
                parent_dataset_name="case",
                parent_df=parent,
                state_view=state,
                config=_scd_config(),
            )

    def test_join_type_left_keeps_unmatched(self, namespace):
        parent = native_pd.DataFrame(
            [
                {"CASE_ID": "A", "OWNER_NAME": "Alice"},
                {"CASE_ID": "Z", "OWNER_NAME": "Zoe"},
            ]
        )
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
            join_type="left",
        )

        loaded = load_active_dataset(namespace, "case")
        assert sorted(loaded["CASE_ID"].unique()) == ["A", "Z"]

    def test_grid_anchor_count_derives_target_partitions(self, namespace, monkeypatch):
        # When caller passes grid_anchor_count the facade derives a partition
        # hint and forwards it to save_active_dataset. The hint stops the
        # write path from defaulting to get_default_parallelism() which
        # under-partitions multi-billion-row SCD fan-outs and spills OOM.
        from customer_retention.stages.scd_history import augment as augment_module

        monkeypatch.setattr(
            augment_module, "get_default_parallelism", lambda: 8
        )
        captured: dict = {}

        def fake_save(namespace, dataset_name, df, target_partitions=None):
            captured["target_partitions"] = target_partitions
            return namespace.landing_table_dir(dataset_name)

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.save_active_dataset",
            fake_save,
        )
        monkeypatch.setattr(
            augment_module, "_assert_landing_delta_matches_schema",
            lambda *args, **kwargs: None,
        )

        parent = native_pd.DataFrame([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
            grid_anchor_count=12,
        )

        # 12 anchors × 8 cores × multiplier(8) = 768 partitions
        assert captured["target_partitions"] == 12 * 8 * 8

    def test_explicit_target_partitions_wins_over_grid_anchor_count(
        self, namespace, monkeypatch
    ):
        from customer_retention.stages.scd_history import augment as augment_module

        monkeypatch.setattr(
            augment_module, "get_default_parallelism", lambda: 8
        )
        captured: dict = {}

        def fake_save(namespace, dataset_name, df, target_partitions=None):
            captured["target_partitions"] = target_partitions
            return namespace.landing_table_dir(dataset_name)

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.save_active_dataset",
            fake_save,
        )
        monkeypatch.setattr(
            augment_module, "_assert_landing_delta_matches_schema",
            lambda *args, **kwargs: None,
        )

        parent = native_pd.DataFrame([{"CASE_ID": "A", "OWNER_NAME": "Alice"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
            grid_anchor_count=12,
            target_partitions=200,
        )

        assert captured["target_partitions"] == 200

    def test_partition_hint_capped_at_max(self, namespace, monkeypatch):
        from customer_retention.stages.scd_history import augment as augment_module

        monkeypatch.setattr(
            augment_module, "get_default_parallelism", lambda: 64
        )
        captured: dict = {}

        def fake_save(namespace, dataset_name, df, target_partitions=None):
            captured["target_partitions"] = target_partitions
            return namespace.landing_table_dir(dataset_name)

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.save_active_dataset",
            fake_save,
        )
        monkeypatch.setattr(
            augment_module, "_assert_landing_delta_matches_schema",
            lambda *args, **kwargs: None,
        )

        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        # 500 anchors × 64 cores × 8 = 256000 — capped at 4096
        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
            grid_anchor_count=500,
        )

        assert captured["target_partitions"] == augment_module._AUGMENT_MAX_PARTITIONS

    def test_partition_hint_not_below_cores(self, namespace, monkeypatch):
        from customer_retention.stages.scd_history import augment as augment_module

        monkeypatch.setattr(
            augment_module, "get_default_parallelism", lambda: 32
        )
        captured: dict = {}

        def fake_save(namespace, dataset_name, df, target_partitions=None):
            captured["target_partitions"] = target_partitions
            return namespace.landing_table_dir(dataset_name)

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.active_dataset_store.save_active_dataset",
            fake_save,
        )
        monkeypatch.setattr(
            augment_module, "_assert_landing_delta_matches_schema",
            lambda *args, **kwargs: None,
        )

        parent = native_pd.DataFrame([{"CASE_ID": "A"}])
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        # grid_anchor_count=0 → no derivation → None hint passed through
        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
            grid_anchor_count=0,
        )
        assert captured["target_partitions"] is None

    def test_existing_landing_delta_is_overwritten(self, namespace):
        # If NB00 wrote a stale landing Delta earlier in the run (e.g. via
        # the key-resolution save), the facade must overwrite it cleanly.
        save_active_dataset(
            namespace,
            "case",
            native_pd.DataFrame(
                [{"CASE_ID": "A", "OWNER_NAME": "Alice", "stale_col": 1}]
            ),
        )

        parent = native_pd.DataFrame(
            [{"CASE_ID": "A", "OWNER_NAME": "Alice"}]
        )
        state = native_pd.DataFrame(
            [{"CASE_ID": "A", "as_of_date": T0, "Status": "Open"}]
        )

        augment_and_persist_parent_dataset(
            namespace=namespace,
            parent_dataset_name="case",
            parent_df=parent,
            state_view=state,
            config=_scd_config(),
        )

        loaded = load_active_dataset(namespace, "case")
        # Stale column from the prior write is gone — overwrite, not append.
        assert "stale_col" not in loaded.columns
        assert "Status" in loaded.columns
