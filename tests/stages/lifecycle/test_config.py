"""Tests for LifecycleEnrichmentConfig."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig


def _minimal_kwargs(**overrides):
    base = dict(
        enriched_view_name="sps_enriched_contract",
        parent_entity_key="ACCOUNT_ID",
        sub_entity_key="CONTRACT_ID",
        valid_from_column="CONTRACT_START_DATE",
        valid_to_columns=("BILLING_TERMINATION_DATE",),
    )
    base.update(overrides)
    return base


class TestConstruction:
    def test_minimal_valid_config(self):
        cfg = LifecycleEnrichmentConfig(**_minimal_kwargs())
        assert cfg.enriched_view_name == "sps_enriched_contract"
        assert cfg.parent_entity_key == "ACCOUNT_ID"
        assert cfg.sub_entity_key == "CONTRACT_ID"
        assert cfg.valid_from_column == "CONTRACT_START_DATE"
        assert cfg.valid_to_columns == ("BILLING_TERMINATION_DATE",)
        assert cfg.status_column is None
        assert cfg.terminal_status_values == ()
        assert cfg.drop_columns == ()
        assert cfg.on_corrupt_row == "raise"
        assert cfg.event_type_column == "event_type"
        assert cfg.event_timestamp_column == "event_timestamp"

    def test_full_config(self):
        cfg = LifecycleEnrichmentConfig(
            **_minimal_kwargs(
                status_column="CONTRACT_STATUS",
                terminal_status_values=("Cancelled", "Terminated"),
                drop_columns=("BILLING_TERMINATION_DATE", "TAKE_RATE"),
                on_corrupt_row="skip",
            )
        )
        assert cfg.status_column == "CONTRACT_STATUS"
        assert cfg.terminal_status_values == ("Cancelled", "Terminated")
        assert cfg.drop_columns == ("BILLING_TERMINATION_DATE", "TAKE_RATE")
        assert cfg.on_corrupt_row == "skip"

    def test_is_frozen(self):
        cfg = LifecycleEnrichmentConfig(**_minimal_kwargs())
        with pytest.raises(FrozenInstanceError):
            cfg.parent_entity_key = "OTHER"  # type: ignore[misc]


class TestValidation:
    def test_empty_enriched_view_name_rejected(self):
        with pytest.raises(ValueError, match="enriched_view_name"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(enriched_view_name=""))

    def test_empty_parent_entity_key_rejected(self):
        with pytest.raises(ValueError, match="parent_entity_key"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(parent_entity_key=""))

    def test_empty_sub_entity_key_rejected(self):
        with pytest.raises(ValueError, match="sub_entity_key"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(sub_entity_key=""))

    def test_empty_valid_from_column_rejected(self):
        with pytest.raises(ValueError, match="valid_from_column"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(valid_from_column=""))

    def test_empty_valid_to_columns_rejected(self):
        with pytest.raises(ValueError, match="valid_to_columns"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(valid_to_columns=()))

    def test_terminal_status_values_without_status_column_rejected(self):
        with pytest.raises(ValueError, match="status_column"):
            LifecycleEnrichmentConfig(
                **_minimal_kwargs(
                    status_column=None,
                    terminal_status_values=("Cancelled",),
                )
            )

    def test_status_column_without_terminal_values_rejected(self):
        with pytest.raises(ValueError, match="terminal_status_values"):
            LifecycleEnrichmentConfig(
                **_minimal_kwargs(
                    status_column="CONTRACT_STATUS",
                    terminal_status_values=(),
                )
            )

    def test_invalid_on_corrupt_row_rejected(self):
        with pytest.raises(ValueError, match="on_corrupt_row"):
            LifecycleEnrichmentConfig(**_minimal_kwargs(on_corrupt_row="explode"))

    def test_valid_from_column_in_drop_columns_rejected(self):
        with pytest.raises(ValueError, match="valid_from_column"):
            LifecycleEnrichmentConfig(
                **_minimal_kwargs(
                    drop_columns=("CONTRACT_START_DATE", "TAKE_RATE"),
                )
            )

    @pytest.mark.parametrize("policy", ["raise", "skip", "warn"])
    def test_valid_on_corrupt_row_accepted(self, policy):
        cfg = LifecycleEnrichmentConfig(**_minimal_kwargs(on_corrupt_row=policy))
        assert cfg.on_corrupt_row == policy


class TestRoundTrip:
    def test_to_dict_round_trip(self):
        cfg = LifecycleEnrichmentConfig(
            **_minimal_kwargs(
                status_column="CONTRACT_STATUS",
                terminal_status_values=("Cancelled", "Terminated", "Expired"),
                drop_columns=("BTD", "TAKE_RATE"),
                on_corrupt_row="warn",
            )
        )
        as_dict = cfg.to_dict()
        assert isinstance(as_dict, dict)
        # Tuple fields serialise as lists (json-friendly)
        assert as_dict["valid_to_columns"] == ["BILLING_TERMINATION_DATE"]
        assert as_dict["terminal_status_values"] == ["Cancelled", "Terminated", "Expired"]
        assert as_dict["drop_columns"] == ["BTD", "TAKE_RATE"]

        rebuilt = LifecycleEnrichmentConfig.from_dict(as_dict)
        assert rebuilt == cfg
        assert isinstance(rebuilt.valid_to_columns, tuple)
        assert isinstance(rebuilt.terminal_status_values, tuple)
        assert isinstance(rebuilt.drop_columns, tuple)

    def test_from_dict_minimal(self):
        cfg = LifecycleEnrichmentConfig.from_dict(
            {
                "enriched_view_name": "v",
                "parent_entity_key": "P",
                "sub_entity_key": "S",
                "valid_from_column": "F",
                "valid_to_columns": ["T1", "T2"],
            }
        )
        assert cfg.valid_to_columns == ("T1", "T2")
        assert cfg.on_corrupt_row == "raise"

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(TypeError):
            LifecycleEnrichmentConfig.from_dict(
                {
                    "enriched_view_name": "v",
                    "parent_entity_key": "P",
                    "sub_entity_key": "S",
                    "valid_from_column": "F",
                    "valid_to_columns": ["T"],
                    "bogus_field": 1,
                }
            )
