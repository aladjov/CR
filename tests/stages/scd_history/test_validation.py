"""Tests for ``validate_scd_sources`` — NB00 §0.11.5 preflight check.

The validator guards the congruence between two NB00-local dicts that must
stay in lock-step:

* ``namespace.scd_history_sources``  — parent-name → history-table path.
* ``SCD_RECONSTRUCTION_CONFIGS``     — parent-name → reconstruction config.

A mismatched state (one half commented, the other half live) silently
degrades to "no augmentation happens" because the NB00 loop iterates
``scd_history_sources.items()``. The validator must fail loud in both
directions: source-without-config AND config-without-source.

The tests also cover the two legitimate states: both-empty (no SCD
reconstruction in this pipeline) and both-congruent (augmentation runs).
"""
from __future__ import annotations

import pytest

from customer_retention.stages.scd_history import SCDHistoryReconstructionConfig
from customer_retention.stages.scd_history.validation import validate_scd_sources


def _oracle_congruent(sources: dict, configs: dict) -> bool:
    """First-principles: the two dicts are congruent iff key-sets are equal."""
    return set(sources.keys()) == set(configs.keys())


def _make_config(parent_name: str) -> SCDHistoryReconstructionConfig:
    return SCDHistoryReconstructionConfig(
        enriched_view_name=f"{parent_name}_with_state",
        parent_record_key=f"{parent_name.upper()}_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        old_value_column="OLD_VALUE",
        change_timestamp_column="CREATED_DATE",
        unique_row_id_column=f"{parent_name.upper()}_HISTORY_ID",
        tracked_fields=("Status",),
        parent_table_dataset_name=parent_name,
        parent_creation_timestamp_column="CREATED_DATE",
        parent_value_columns=(),
    )


class TestBothEmpty:
    def test_both_empty_does_not_raise(self):
        validate_scd_sources({}, datasets={"case": "path"}, configs={})


class TestBothCongruent:
    def test_single_parent(self):
        datasets = {"case": "t.case", "account": "t.account"}
        sources = {"case": "t.case_history"}
        configs = {"case": _make_config("case")}
        validate_scd_sources(sources, datasets, configs)

    def test_multiple_parents(self):
        datasets = {"case": "t.case", "opportunity": "t.opp", "account": "t.account"}
        sources = {"case": "t.case_history", "opportunity": "t.opp_history"}
        configs = {"case": _make_config("case"), "opportunity": _make_config("opportunity")}
        validate_scd_sources(sources, datasets, configs)


class TestSourceWithoutConfig:
    def test_source_entry_without_matching_config_raises(self):
        datasets = {"case": "t.case"}
        sources = {"case": "t.case_history"}
        configs: dict = {}
        with pytest.raises(KeyError, match=r"SCD_RECONSTRUCTION_CONFIGS"):
            validate_scd_sources(sources, datasets, configs)

    def test_source_for_unknown_dataset_raises(self):
        datasets: dict = {}
        sources = {"case": "t.case_history"}
        configs = {"case": _make_config("case")}
        with pytest.raises(KeyError, match=r"not in `datasets`"):
            validate_scd_sources(sources, datasets, configs)


class TestConfigWithoutSource:
    """Regression for engagement_e4ad6e1b run ``spschurn-e4ad6e1b``.

    Run artifact shape: ``_namespace.scd_history_sources = {}`` (the
    ``case_history`` line was commented out), but ``SCD_RECONSTRUCTION_CONFIGS
    = {"case": CASE_HISTORY_RECONSTRUCTION}`` stayed live. The NB00 loop
    silently ran zero iterations, ``landing/case`` was never augmented, and
    sanity probe 1.6 FAILed with the six tracked SCD fields missing.
    """

    def test_config_entry_without_matching_source_raises(self):
        datasets = {"case": "t.case"}
        sources: dict = {}
        configs = {"case": _make_config("case")}
        with pytest.raises(KeyError, match=r"scd_history_sources"):
            validate_scd_sources(sources, datasets, configs)

    def test_config_raise_names_offending_parent(self):
        datasets = {"case": "t.case"}
        sources: dict = {}
        configs = {"case": _make_config("case")}
        with pytest.raises(KeyError, match=r"'case'"):
            validate_scd_sources(sources, datasets, configs)

    def test_partial_overlap_config_extra_raises(self):
        datasets = {"case": "t.case", "opportunity": "t.opp"}
        sources = {"case": "t.case_history"}
        configs = {"case": _make_config("case"), "opportunity": _make_config("opportunity")}
        with pytest.raises(KeyError, match=r"'opportunity'"):
            validate_scd_sources(sources, datasets, configs)

    def test_engagement_e4ad6e1b_regression_shape(self):
        """Exact shape from run ``spschurn-e4ad6e1b``.

        Parent dataset registered in ``datasets``, the reconstruction config
        kept in ``SCD_RECONSTRUCTION_CONFIGS``, and the sources dict emptied
        by commenting out the single entry. The validator must raise before
        the augmentation loop runs (which would otherwise be a silent no-op).
        """
        datasets = {
            "account": "t.account",
            "case": "t.case",
            "contract": "t.contract",
            "subscription": "t.subscription",
        }
        sources: dict = {}
        configs = {"case": _make_config("case")}
        assert not _oracle_congruent(sources, configs)
        with pytest.raises(KeyError):
            validate_scd_sources(sources, datasets, configs)


class TestOracleAgreement:
    """The validator must raise exactly when the oracle reports incongruent."""

    @pytest.mark.parametrize(
        "sources,configs",
        [
            ({}, {}),
            ({"case": "t.case_history"}, {"case": "cfg"}),
            ({"case": "t.case_history", "opp": "t.opp_hist"}, {"case": "cfg", "opp": "cfg"}),
        ],
    )
    def test_validator_and_oracle_agree_on_pass(self, sources, configs):
        datasets = {k: f"t.{k}" for k in set(sources) | set(configs)}
        real_configs = {k: _make_config(k) for k in configs}
        assert _oracle_congruent(sources, configs)
        validate_scd_sources(sources, datasets, real_configs)

    @pytest.mark.parametrize(
        "sources,configs",
        [
            ({}, {"case": "cfg"}),
            ({"case": "t.case_history"}, {}),
            ({"case": "t.case_history"}, {"opp": "cfg"}),
            ({"case": "t.case_history", "opp": "t.opp_hist"}, {"case": "cfg"}),
        ],
    )
    def test_validator_and_oracle_agree_on_raise(self, sources, configs):
        datasets = {k: f"t.{k}" for k in set(sources) | set(configs)}
        real_configs = {k: _make_config(k) for k in configs}
        assert not _oracle_congruent(sources, configs)
        with pytest.raises(KeyError):
            validate_scd_sources(sources, datasets, real_configs)
