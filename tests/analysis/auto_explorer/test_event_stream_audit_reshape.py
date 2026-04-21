from __future__ import annotations

import pytest

from customer_retention.core.compat import native_pd

reshape_event_stream_to_snapshot = pytest.importorskip(
    "customer_retention.analysis.auto_explorer.field_availability_audit",
).__dict__.get("reshape_event_stream_to_snapshot")
audit_surprises_against_drop_list = pytest.importorskip(
    "customer_retention.analysis.auto_explorer.field_availability_audit",
).__dict__.get("audit_surprises_against_drop_list")

if reshape_event_stream_to_snapshot is None or audit_surprises_against_drop_list is None:
    pytest.skip(
        "fix not landed yet: reshape_event_stream_to_snapshot / "
        "audit_surprises_against_drop_list not exported from field_availability_audit",
        allow_module_level=True,
    )


def _oracle_snapshot(events: native_pd.DataFrame) -> native_pd.DataFrame:
    starts = events[events["event_type"] == "start"].copy()
    terms = events[events["event_type"] == "terminate"]
    if len(terms) == 0:
        starts["TERMINATION_DATE"] = native_pd.NaT
        return starts.reset_index(drop=True)
    term_max = (
        terms.groupby("CONTRACT_ID")["event_timestamp"]
        .max()
        .rename("TERMINATION_DATE")
        .reset_index()
    )
    return starts.merge(term_max, on="CONTRACT_ID", how="left").reset_index(drop=True)


def _doubled_event_stream() -> native_pd.DataFrame:
    return native_pd.DataFrame({
        "CONTRACT_ID": ["C1", "C2", "C3", "C4", "C4", "C5", "C5", "C6", "C6"],
        "ACCOUNT_ID":  ["A1", "A1", "A2", "A2", "A2", "A3", "A3", "A3", "A3"],
        "event_type":  ["start", "start", "start",
                        "start", "terminate",
                        "start", "terminate",
                        "start", "terminate"],
        "event_timestamp": native_pd.to_datetime([
            "2023-01-01", "2023-02-01", "2023-03-01",
            "2023-04-01", "2023-10-15",
            "2023-05-01", "2024-01-20",
            "2023-06-01", "2024-02-28",
        ]),
        "STATUS_AT_START": ["Active", "Active", "Active",
                            "Active", "Cancelled",
                            "Active", "Cancelled",
                            "Active", "Cancelled"],
    })


def test_reshape_collapses_doubled_stream_to_one_row_per_unit():
    events = _doubled_event_stream()
    snap = reshape_event_stream_to_snapshot(
        events,
        unit_id_column="CONTRACT_ID",
        event_type_column="event_type",
        terminate_value="terminate",
        event_timestamp_column="event_timestamp",
    )
    expected = _oracle_snapshot(events)
    assert len(snap) == 6
    assert snap["CONTRACT_ID"].nunique() == 6
    assert set(snap["CONTRACT_ID"]) == set(expected["CONTRACT_ID"])


def test_reshape_termination_date_only_on_terminated_units():
    events = _doubled_event_stream()
    snap = reshape_event_stream_to_snapshot(
        events,
        unit_id_column="CONTRACT_ID",
        event_type_column="event_type",
        terminate_value="terminate",
        event_timestamp_column="event_timestamp",
    )
    by_id = snap.set_index("CONTRACT_ID")["TERMINATION_DATE"]
    assert by_id["C1"] is native_pd.NaT or native_pd.isna(by_id["C1"])
    assert by_id["C2"] is native_pd.NaT or native_pd.isna(by_id["C2"])
    assert by_id["C3"] is native_pd.NaT or native_pd.isna(by_id["C3"])
    assert by_id["C4"] == native_pd.Timestamp("2023-10-15")
    assert by_id["C5"] == native_pd.Timestamp("2024-01-20")
    assert by_id["C6"] == native_pd.Timestamp("2024-02-28")


def test_reshape_picks_max_when_multiple_terminate_events():
    events = native_pd.DataFrame({
        "CONTRACT_ID": ["C1", "C1", "C1"],
        "event_type": ["start", "terminate", "terminate"],
        "event_timestamp": native_pd.to_datetime([
            "2023-01-01", "2023-06-01", "2024-01-01",
        ]),
    })
    snap = reshape_event_stream_to_snapshot(
        events,
        unit_id_column="CONTRACT_ID",
        event_type_column="event_type",
        terminate_value="terminate",
        event_timestamp_column="event_timestamp",
    )
    assert len(snap) == 1
    assert snap["TERMINATION_DATE"].iloc[0] == native_pd.Timestamp("2024-01-01")


def test_reshape_all_active_yields_all_null_termination():
    events = native_pd.DataFrame({
        "CONTRACT_ID": ["C1", "C2", "C3"],
        "event_type": ["start", "start", "start"],
        "event_timestamp": native_pd.to_datetime([
            "2023-01-01", "2023-02-01", "2023-03-01",
        ]),
    })
    snap = reshape_event_stream_to_snapshot(
        events,
        unit_id_column="CONTRACT_ID",
        event_type_column="event_type",
        terminate_value="terminate",
        event_timestamp_column="event_timestamp",
    )
    assert len(snap) == 3
    assert snap["TERMINATION_DATE"].isna().all()


def test_reshape_fail_fast_on_missing_required_column():
    events = native_pd.DataFrame({
        "CONTRACT_ID": ["C1"],
        "event_type": ["start"],
    })
    with pytest.raises(KeyError, match="event_timestamp"):
        reshape_event_stream_to_snapshot(
            events,
            unit_id_column="CONTRACT_ID",
            event_type_column="event_type",
            terminate_value="terminate",
            event_timestamp_column="event_timestamp",
        )


def test_reshape_regression_doubled_event_stream_not_overcounted():
    events = _doubled_event_stream()
    snap = reshape_event_stream_to_snapshot(
        events,
        unit_id_column="CONTRACT_ID",
        event_type_column="event_type",
        terminate_value="terminate",
        event_timestamp_column="event_timestamp",
    )
    n_active = int(snap["TERMINATION_DATE"].isna().sum())
    n_terminated = int(snap["TERMINATION_DATE"].notna().sum())
    assert n_active == 3
    assert n_terminated == 3


def _profile(name, score, dataset="contract", recommendation="exclude"):
    from customer_retention.analysis.auto_explorer.field_availability_audit import (
        FieldLeadLagProfile,
    )
    return FieldLeadLagProfile(
        field_name=name, source_dataset=dataset,
        analysis_tier="contract_level", anchor_type="contract_termination",
        suspicion_score=score, recommendation=recommendation,
    )


def _audit_result(profiles, exclusions):
    from customer_retention.analysis.auto_explorer.field_availability_audit import (
        FieldAvailabilityAuditConfig,
        FieldAvailabilityAuditResult,
    )
    from customer_retention.analysis.auto_explorer.service_unit_detector import (
        ServiceUnitConfig,
    )
    su = ServiceUnitConfig(
        dataset_name="contract", unit_id_column="CONTRACT_ID",
        entity_column="ACCOUNT_ID", anchor_date_column="TERMINATION_DATE",
    )
    cfg = FieldAvailabilityAuditConfig(service_unit=su, suspicion_threshold=0.5)
    return FieldAvailabilityAuditResult(
        config=cfg, field_profiles=profiles, recommended_exclusions=exclusions,
    )


def test_audit_surprises_flags_columns_above_threshold_not_in_drop_list():
    result = _audit_result(
        profiles=[
            _profile("BILLING_TERMINATION_DATE", 0.95),
            _profile("ACCOUNT_NAME", 0.10, recommendation="safe"),
        ],
        exclusions=["BILLING_TERMINATION_DATE"],
    )
    surprises = audit_surprises_against_drop_list(
        result, drop_columns=[], threshold=0.5,
    )
    assert "BILLING_TERMINATION_DATE" in surprises
    assert "ACCOUNT_NAME" not in surprises


def test_audit_surprises_empty_when_all_above_threshold_dropped():
    result = _audit_result(
        profiles=[_profile("CANCEL_REASON", 0.92)],
        exclusions=["CANCEL_REASON"],
    )
    surprises = audit_surprises_against_drop_list(
        result, drop_columns=["CANCEL_REASON"], threshold=0.5,
    )
    assert surprises == set()


def test_audit_surprises_includes_value_distribution_flags():
    profile = _profile("BILLING_METHOD", 0.40, recommendation="safe")
    profile.value_suspicion_score = 0.85
    result = _audit_result(profiles=[profile], exclusions=[])
    surprises = audit_surprises_against_drop_list(
        result, drop_columns=[], threshold=0.5,
    )
    assert "BILLING_METHOD" in surprises


def test_audit_surprises_regression_no_artifact_means_loud_failure():
    """Mirrors the run-artifact gap: audit never ran → no result object → callers
    must not silently treat 'no surprises' as PASS. Helper takes a plain result;
    the cycle notebook surfaces 'audit dir missing' as a separate FAIL check.
    The contract here: passing None must fail loud, not return empty set."""
    with pytest.raises((TypeError, AttributeError)):
        audit_surprises_against_drop_list(None, drop_columns=[], threshold=0.5)
