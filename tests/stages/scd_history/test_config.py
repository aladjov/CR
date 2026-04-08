"""Tests for SCDHistoryReconstructionConfig."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from customer_retention.stages.scd_history.config import (
    SCDHistoryReconstructionConfig,
)


def _minimal_kwargs(**overrides):
    base = dict(
        enriched_view_name="sps_reconstructed_case_history",
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        change_timestamp_column="CREATED_DATE",
        tracked_fields=("Status", "Priority"),
    )
    base.update(overrides)
    return base


class TestConstruction:
    def test_minimal_valid_config(self):
        cfg = SCDHistoryReconstructionConfig(**_minimal_kwargs())
        assert cfg.enriched_view_name == "sps_reconstructed_case_history"
        assert cfg.parent_record_key == "CASE_ID"
        assert cfg.field_column == "FIELD"
        assert cfg.new_value_column == "NEW_VALUE"
        assert cfg.change_timestamp_column == "CREATED_DATE"
        assert cfg.tracked_fields == ("Status", "Priority")
        assert cfg.old_value_column is None
        assert cfg.parent_table_dataset_name is None
        assert cfg.parent_creation_timestamp_column is None
        assert cfg.parent_value_columns == ()
        assert cfg.on_unknown_field == "raise"
        assert cfg.output_record_key is None
        assert cfg.unique_row_id_column is None

    def test_full_config(self):
        cfg = SCDHistoryReconstructionConfig(
            **_minimal_kwargs(
                tracked_fields=("Status", "Priority", "Type", "Origin"),
                old_value_column="OLD_VALUE",
                parent_table_dataset_name="case",
                parent_creation_timestamp_column="CREATED_DATE",
                parent_value_columns=(
                    ("Status", "CASE_STATUS"),
                    ("Priority", "PRIORITY"),
                ),
                on_unknown_field="skip",
                output_record_key="case_id",
                unique_row_id_column="CASE_HISTORY_ID",
            )
        )
        assert cfg.old_value_column == "OLD_VALUE"
        assert cfg.parent_table_dataset_name == "case"
        assert cfg.parent_creation_timestamp_column == "CREATED_DATE"
        assert cfg.parent_value_columns == (
            ("Status", "CASE_STATUS"),
            ("Priority", "PRIORITY"),
        )
        assert cfg.on_unknown_field == "skip"
        assert cfg.output_record_key == "case_id"
        assert cfg.unique_row_id_column == "CASE_HISTORY_ID"

    def test_is_frozen(self):
        cfg = SCDHistoryReconstructionConfig(**_minimal_kwargs())
        with pytest.raises(FrozenInstanceError):
            cfg.parent_record_key = "OTHER"  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize(
        "field_name",
        [
            "enriched_view_name",
            "parent_record_key",
            "field_column",
            "new_value_column",
            "change_timestamp_column",
        ],
    )
    def test_empty_required_string_rejected(self, field_name):
        with pytest.raises(ValueError, match=field_name):
            SCDHistoryReconstructionConfig(**_minimal_kwargs(**{field_name: ""}))

    def test_empty_tracked_fields_rejected(self):
        with pytest.raises(ValueError, match="tracked_fields"):
            SCDHistoryReconstructionConfig(**_minimal_kwargs(tracked_fields=()))

    def test_parent_value_columns_referencing_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="tracked_fields"):
            SCDHistoryReconstructionConfig(
                **_minimal_kwargs(
                    tracked_fields=("Status",),
                    parent_table_dataset_name="case",
                    parent_value_columns=(("Priority", "PRIORITY"),),
                )
            )

    def test_parent_value_columns_without_parent_table_rejected(self):
        with pytest.raises(ValueError, match="parent_table_dataset_name"):
            SCDHistoryReconstructionConfig(
                **_minimal_kwargs(
                    parent_value_columns=(("Status", "CASE_STATUS"),),
                )
            )

    def test_invalid_on_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="on_unknown_field"):
            SCDHistoryReconstructionConfig(
                **_minimal_kwargs(on_unknown_field="explode")
            )

    @pytest.mark.parametrize("policy", ["raise", "skip", "warn"])
    def test_valid_on_unknown_field_accepted(self, policy):
        cfg = SCDHistoryReconstructionConfig(
            **_minimal_kwargs(on_unknown_field=policy)
        )
        assert cfg.on_unknown_field == policy

    def test_unique_row_id_column_empty_rejected(self):
        with pytest.raises(ValueError, match="unique_row_id_column"):
            SCDHistoryReconstructionConfig(
                **_minimal_kwargs(unique_row_id_column="")
            )

    def test_unique_row_id_column_whitespace_rejected(self):
        with pytest.raises(ValueError, match="unique_row_id_column"):
            SCDHistoryReconstructionConfig(
                **_minimal_kwargs(unique_row_id_column="   ")
            )

    def test_unique_row_id_column_none_accepted(self):
        cfg = SCDHistoryReconstructionConfig(
            **_minimal_kwargs(unique_row_id_column=None)
        )
        assert cfg.unique_row_id_column is None

    def test_unique_row_id_column_non_empty_accepted(self):
        cfg = SCDHistoryReconstructionConfig(
            **_minimal_kwargs(unique_row_id_column="CASE_HISTORY_ID")
        )
        assert cfg.unique_row_id_column == "CASE_HISTORY_ID"


class TestRoundTrip:
    def test_to_dict_round_trip_full(self):
        cfg = SCDHistoryReconstructionConfig(
            **_minimal_kwargs(
                tracked_fields=("Status", "Priority", "Type"),
                old_value_column="OLD_VALUE",
                parent_table_dataset_name="case",
                parent_creation_timestamp_column="CREATED_DATE",
                parent_value_columns=(
                    ("Status", "CASE_STATUS"),
                    ("Priority", "PRIORITY"),
                ),
                unique_row_id_column="CASE_HISTORY_ID",
                on_unknown_field="warn",
            )
        )
        as_dict = cfg.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["tracked_fields"] == ["Status", "Priority", "Type"]
        assert as_dict["parent_value_columns"] == [
            ["Status", "CASE_STATUS"],
            ["Priority", "PRIORITY"],
        ]
        assert as_dict["unique_row_id_column"] == "CASE_HISTORY_ID"

        rebuilt = SCDHistoryReconstructionConfig.from_dict(as_dict)
        assert rebuilt == cfg
        assert isinstance(rebuilt.tracked_fields, tuple)
        assert isinstance(rebuilt.parent_value_columns, tuple)
        assert all(
            isinstance(pair, tuple) for pair in rebuilt.parent_value_columns
        )

    def test_from_dict_minimal(self):
        cfg = SCDHistoryReconstructionConfig.from_dict(
            {
                "enriched_view_name": "v",
                "parent_record_key": "K",
                "field_column": "F",
                "new_value_column": "N",
                "change_timestamp_column": "T",
                "tracked_fields": ["A", "B"],
            }
        )
        assert cfg.tracked_fields == ("A", "B")
        assert cfg.on_unknown_field == "raise"
        assert cfg.unique_row_id_column is None

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(TypeError):
            SCDHistoryReconstructionConfig.from_dict(
                {
                    "enriched_view_name": "v",
                    "parent_record_key": "K",
                    "field_column": "F",
                    "new_value_column": "N",
                    "change_timestamp_column": "T",
                    "tracked_fields": ["A"],
                    "bogus_field": 1,
                }
            )
