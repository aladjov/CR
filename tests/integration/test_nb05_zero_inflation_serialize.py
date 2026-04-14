from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer.findings import (
    ExplorationFindings,
    apply_zero_inflation_opt_in,
)


def make_findings(names):
    return {
        name: ExplorationFindings(source_path=f"/fake/{name}.csv", source_format="csv")
        for name in names
    }


def test_apply_zero_inflation_opt_in_sets_per_dataset_field():
    findings = make_findings(["case", "subscription"])
    apply_zero_inflation_opt_in(
        findings,
        {"case": ["RESOLUTION_TARGET_DATE_TIME"], "subscription": []},
    )
    assert findings["case"].zero_inflation_opt_in == ["RESOLUTION_TARGET_DATE_TIME"]
    assert findings["subscription"].zero_inflation_opt_in == []


def test_apply_zero_inflation_opt_in_yaml_round_trip(tmp_path: Path):
    findings = make_findings(["case"])
    apply_zero_inflation_opt_in(findings, {"case": ["RESOLUTION_TARGET_DATE_TIME"]})
    path = tmp_path / "case_findings.yaml"
    findings["case"].save(path)
    reloaded = ExplorationFindings.load(path)
    assert reloaded.zero_inflation_opt_in == ["RESOLUTION_TARGET_DATE_TIME"]


def test_apply_zero_inflation_opt_in_overwrites_existing():
    findings = make_findings(["case"])
    findings["case"].zero_inflation_opt_in = ["LEGACY_COL"]
    apply_zero_inflation_opt_in(findings, {"case": ["NEW_COL"]})
    assert findings["case"].zero_inflation_opt_in == ["NEW_COL"]


def test_apply_zero_inflation_opt_in_empty_list_clears():
    findings = make_findings(["case"])
    findings["case"].zero_inflation_opt_in = ["LEGACY_COL"]
    apply_zero_inflation_opt_in(findings, {"case": []})
    assert findings["case"].zero_inflation_opt_in == []


def test_apply_zero_inflation_opt_in_leaves_untouched_datasets_alone():
    findings = make_findings(["case", "subscription"])
    findings["subscription"].zero_inflation_opt_in = ["KEEP_ME"]
    apply_zero_inflation_opt_in(findings, {"case": ["A"]})
    assert findings["subscription"].zero_inflation_opt_in == ["KEEP_ME"]


def test_apply_zero_inflation_opt_in_unknown_dataset_raises():
    findings = make_findings(["case"])
    with pytest.raises(KeyError, match="unknown_dataset"):
        apply_zero_inflation_opt_in(findings, {"unknown_dataset": ["X"]})


def test_apply_zero_inflation_opt_in_copies_list_not_references():
    findings = make_findings(["case"])
    opt_in = ["ORIGINAL"]
    apply_zero_inflation_opt_in(findings, {"case": opt_in})
    opt_in.append("MUTATED")
    assert findings["case"].zero_inflation_opt_in == ["ORIGINAL"]


def test_apply_zero_inflation_opt_in_empty_config_is_noop():
    findings = make_findings(["case"])
    findings["case"].zero_inflation_opt_in = ["EXISTING"]
    apply_zero_inflation_opt_in(findings, {})
    assert findings["case"].zero_inflation_opt_in == ["EXISTING"]


def test_apply_zero_inflation_opt_in_none_config_raises():
    findings = make_findings(["case"])
    with pytest.raises(TypeError):
        apply_zero_inflation_opt_in(findings, None)
