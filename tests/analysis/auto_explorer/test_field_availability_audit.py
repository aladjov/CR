from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.field_availability_audit import (
    FieldAvailabilityAuditConfig,
    FieldAvailabilityAuditor,
    FieldAvailabilityAuditResult,
    FieldLeadLagProfile,
    _population_suspicion_score,
    build_account_anchors,
)
from customer_retention.analysis.auto_explorer.service_unit_detector import (
    DatasetLinkage,
    ServiceUnitConfig,
)
from customer_retention.core.compat import pd


def _su_config(**overrides) -> ServiceUnitConfig:
    defaults = dict(
        dataset_name="contract",
        unit_id_column="CONTRACT_ID",
        entity_column="ACCOUNT_ID",
        anchor_date_column="CONTRACT_END_DATE",
        status_column="CONTRACT_STATUS",
        terminated_statuses=["Cancelled", "Terminated"],
        start_date_column="CONTRACT_START_DATE",
    )
    defaults.update(overrides)
    return ServiceUnitConfig(**defaults)


def _contract_df() -> pd.DataFrame:
    return pd.DataFrame({
        "CONTRACT_ID": ["C1", "C2", "C3", "C4", "C5"],
        "ACCOUNT_ID": ["A1", "A1", "A2", "A3", "A3"],
        "CONTRACT_START_DATE": pd.to_datetime([
            "2023-01-01", "2023-06-01", "2023-01-01", "2023-03-01", "2023-09-01",
        ]),
        "CONTRACT_END_DATE": pd.to_datetime([
            "2024-01-01", "2024-06-01", "2025-01-01", "2024-03-01", "2024-09-01",
        ]),
        "CONTRACT_STATUS": ["Cancelled", "Active", "Active", "Terminated", "Terminated"],
        "CANCELLATION_REASON": ["Price", None, None, "Service", "Quality"],
        "DOCUMENT_PLAN": ["Basic", "Premium", "Premium", "Basic", "Basic"],
    })


def _account_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ACCOUNT_ID": ["A1", "A2", "A3", "A4"],
        "ACCOUNT_NAME": ["Foo", "Bar", "Baz", "Qux"],
        "INDUSTRY": ["Tech", "Retail", None, "Finance"],
        "CANCEL_FLAG": [None, None, "Yes", None],
    })


def _event_df() -> pd.DataFrame:
    return pd.DataFrame({
        "EVENT_ID": list(range(12)),
        "ACCOUNT_ID": ["A1"] * 4 + ["A2"] * 4 + ["A3"] * 4,
        "EVENT_DATE": pd.to_datetime([
            "2023-06-01", "2023-09-01", "2023-12-15", "2024-01-05",
            "2023-06-01", "2023-09-01", "2024-06-01", "2024-09-01",
            "2023-06-01", "2023-12-01", "2024-02-25", "2024-03-05",
        ]),
        "EVENT_TYPE": [
            "login", "login", "support", "support",
            "login", "login", "login", "login",
            "login", "support", "cancellation_call", "exit_survey",
        ],
        "RESOLUTION_CODE": [
            None, None, "R1", "R2",
            None, None, None, None,
            None, None, "RC1", "RC2",
        ],
    })


# ---------------------------------------------------------------------------
# Account anchor builder tests
# ---------------------------------------------------------------------------


class TestBuildAccountAnchors:
    def test_partial_and_full_termination(self):
        df = _contract_df()
        cfg = _su_config()
        anchors = build_account_anchors(df, cfg)
        a1 = anchors[anchors["ACCOUNT_ID"] == "A1"].iloc[0]
        assert a1["terminated_units"] == 1
        assert a1["total_units"] == 2
        assert a1["is_fully_terminated"] == False  # noqa: E712
        assert a1["full_termination_date"] is None or pd.isna(a1["full_termination_date"])

    def test_fully_terminated_account(self):
        df = _contract_df()
        cfg = _su_config()
        anchors = build_account_anchors(df, cfg)
        a3 = anchors[anchors["ACCOUNT_ID"] == "A3"].iloc[0]
        assert a3["terminated_units"] == 2
        assert a3["total_units"] == 2
        assert a3["is_fully_terminated"] == True  # noqa: E712
        assert pd.notna(a3["full_termination_date"])

    def test_full_termination_date_is_max(self):
        df = _contract_df()
        cfg = _su_config()
        anchors = build_account_anchors(df, cfg)
        a3 = anchors[anchors["ACCOUNT_ID"] == "A3"].iloc[0]
        assert pd.Timestamp(a3["full_termination_date"]) == pd.Timestamp("2024-09-01")

    def test_first_partial_cancel_is_min(self):
        df = _contract_df()
        cfg = _su_config()
        anchors = build_account_anchors(df, cfg)
        a3 = anchors[anchors["ACCOUNT_ID"] == "A3"].iloc[0]
        assert pd.Timestamp(a3["first_partial_cancel_date"]) == pd.Timestamp("2024-03-01")

    def test_single_terminated_contract(self):
        df = _contract_df()
        cfg = _su_config()
        anchors = build_account_anchors(df, cfg)
        a1 = anchors[anchors["ACCOUNT_ID"] == "A1"].iloc[0]
        assert pd.Timestamp(a1["first_partial_cancel_date"]) == pd.Timestamp("2024-01-01")

    def test_no_terminated_contracts(self):
        df = pd.DataFrame({
            "CONTRACT_ID": ["C1"], "ACCOUNT_ID": ["A1"],
            "CONTRACT_START_DATE": pd.to_datetime(["2023-01-01"]),
            "CONTRACT_END_DATE": pd.to_datetime(["2025-01-01"]),
            "CONTRACT_STATUS": ["Active"],
        })
        anchors = build_account_anchors(df, _su_config())
        assert len(anchors) == 0

    def test_no_status_column_uses_anchor_notnull(self):
        df = pd.DataFrame({
            "CONTRACT_ID": ["C1", "C2"], "ACCOUNT_ID": ["A1", "A1"],
            "CONTRACT_START_DATE": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "CONTRACT_END_DATE": [pd.Timestamp("2024-01-01"), None],
        })
        cfg = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        anchors = build_account_anchors(df, cfg)
        assert len(anchors) == 1
        assert anchors.iloc[0]["terminated_units"] == 1


# ---------------------------------------------------------------------------
# Self-probe tests
# ---------------------------------------------------------------------------


class TestSelfProbe:
    def test_leaky_field_detected(self):
        df = _contract_df()
        cfg = FieldAvailabilityAuditConfig(service_unit=_su_config(), min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        profiles = auditor._probe_self(df, cfg.service_unit)

        reason_profile = next(p for p in profiles if p.field_name == "CANCELLATION_REASON")
        assert reason_profile.pct_non_null_terminated > 0.5
        assert reason_profile.pct_non_null_active == 0.0
        assert reason_profile.suspicion_score > 0.7

    def test_safe_field_not_flagged(self):
        df = _contract_df()
        cfg = FieldAvailabilityAuditConfig(service_unit=_su_config(), min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        profiles = auditor._probe_self(df, cfg.service_unit)

        doc_profile = next(p for p in profiles if p.field_name == "DOCUMENT_PLAN")
        assert doc_profile.suspicion_score < 0.5

    def test_skip_columns_excluded(self):
        df = _contract_df()
        cfg = FieldAvailabilityAuditConfig(service_unit=_su_config(), min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        profiles = auditor._probe_self(df, cfg.service_unit)
        probed_names = {p.field_name for p in profiles}
        assert "CONTRACT_ID" not in probed_names
        assert "ACCOUNT_ID" not in probed_names
        assert "CONTRACT_END_DATE" not in probed_names
        assert "CONTRACT_STATUS" not in probed_names

    def test_evidence_populated(self):
        df = _contract_df()
        cfg = FieldAvailabilityAuditConfig(service_unit=_su_config(), min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        profiles = auditor._probe_self(df, cfg.service_unit)
        for p in profiles:
            assert len(p.evidence) >= 2


# ---------------------------------------------------------------------------
# Entity-level probe tests
# ---------------------------------------------------------------------------


class TestEntityProbe:
    def test_leaky_entity_field(self):
        contract_df = _contract_df()
        account_df = _account_df()
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        profiles = auditor._probe_entity_dataset(account_df, "account", anchors, su_cfg)

        cancel_profile = next((p for p in profiles if p.field_name == "CANCEL_FLAG"), None)
        assert cancel_profile is not None
        assert cancel_profile.analysis_tier == "account_level"
        assert cancel_profile.anchor_type == "full_termination"

    def test_safe_entity_field(self):
        contract_df = _contract_df()
        account_df = _account_df()
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        profiles = auditor._probe_entity_dataset(account_df, "account", anchors, su_cfg)

        name_profile = next(p for p in profiles if p.field_name == "ACCOUNT_NAME")
        assert name_profile.suspicion_score < 0.5

    def test_entity_column_skipped(self):
        contract_df = _contract_df()
        account_df = _account_df()
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        profiles = auditor._probe_entity_dataset(account_df, "account", anchors, su_cfg)
        probed_names = {p.field_name for p in profiles}
        assert "ACCOUNT_ID" not in probed_names


# ---------------------------------------------------------------------------
# Event-level probe tests
# ---------------------------------------------------------------------------


class TestEventProbe:
    def test_event_lead_lag_buckets(self):
        contract_df = _contract_df()
        event_df = _event_df()
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        linkage = DatasetLinkage(tier="account_only", link_method="entity_column")
        profiles = auditor._probe_event_dataset(
            event_df, "events", "EVENT_DATE", anchors, su_cfg, linkage,
        )
        assert len(profiles) > 0
        for p in profiles:
            assert p.analysis_tier == "account_level"
            if p.coverage > 0:
                bucket_sum = (p.pct_before_90d or 0) + (p.pct_30_to_90d or 0) + (p.pct_0_to_30d or 0) + (p.pct_after_anchor or 0)
                assert abs(bucket_sum - 1.0) < 0.02

    def test_resolution_code_late_population(self):
        contract_df = _contract_df()
        event_df = _event_df()
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        linkage = DatasetLinkage(tier="account_only", link_method="entity_column")
        profiles = auditor._probe_event_dataset(
            event_df, "events", "EVENT_DATE", anchors, su_cfg, linkage,
        )
        res_profile = next((p for p in profiles if p.field_name == "RESOLUTION_CODE"), None)
        assert res_profile is not None
        assert res_profile.pct_0_to_30d is not None or res_profile.pct_after_anchor is not None

    def test_all_null_field(self):
        contract_df = _contract_df()
        event_df = _event_df().copy()
        event_df["EMPTY_COL"] = None
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        linkage = DatasetLinkage(tier="account_only", link_method="entity_column")
        profiles = auditor._probe_event_dataset(
            event_df, "events", "EVENT_DATE", anchors, su_cfg, linkage,
        )
        empty_profile = next(p for p in profiles if p.field_name == "EMPTY_COL")
        assert empty_profile.coverage == 0.0
        assert empty_profile.suspicion_score == 0.0


# ---------------------------------------------------------------------------
# Full auditor run tests
# ---------------------------------------------------------------------------


class TestAuditorRun:
    def test_full_run(self):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        linkage = {
            "contract": DatasetLinkage(tier="contract_linked", link_method="self"),
            "account": DatasetLinkage(tier="account_only", link_method="entity_column"),
        }
        result = auditor.run(
            service_unit_df=_contract_df(),
            probe_dfs={"contract": _contract_df(), "account": _account_df()},
            dataset_linkage=linkage,
        )
        assert result.total_terminated_units == 3
        assert result.fully_terminated_accounts >= 1
        assert len(result.field_profiles) > 0

    def test_recommended_exclusions_populated(self):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1, suspicion_threshold=0.5)
        auditor = FieldAvailabilityAuditor(cfg)
        linkage = {
            "contract": DatasetLinkage(tier="contract_linked", link_method="self"),
        }
        result = auditor.run(
            service_unit_df=_contract_df(),
            probe_dfs={"contract": _contract_df()},
            dataset_linkage=linkage,
        )
        assert "CANCELLATION_REASON" in result.recommended_exclusions

    def test_too_few_terminated_raises(self):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=10000)
        auditor = FieldAvailabilityAuditor(cfg)
        with pytest.raises(ValueError, match="Only .* terminated units"):
            auditor.run(
                service_unit_df=_contract_df(),
                probe_dfs={},
                dataset_linkage={},
            )

    def test_no_terminated_raises(self):
        df = pd.DataFrame({
            "CONTRACT_ID": ["C1"], "ACCOUNT_ID": ["A1"],
            "CONTRACT_START_DATE": pd.to_datetime(["2023-01-01"]),
            "CONTRACT_END_DATE": pd.to_datetime(["2025-01-01"]),
            "CONTRACT_STATUS": ["Active"],
        })
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        with pytest.raises(ValueError, match="No terminated"):
            auditor.run(service_unit_df=df, probe_dfs={}, dataset_linkage={})

    def test_classification_thresholds(self):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1, suspicion_threshold=0.5)
        auditor = FieldAvailabilityAuditor(cfg)
        assert auditor._classify(FieldLeadLagProfile(
            field_name="x", source_dataset="y", analysis_tier="a", anchor_type="b",
            suspicion_score=0.8,
        )) == "exclude"
        assert auditor._classify(FieldLeadLagProfile(
            field_name="x", source_dataset="y", analysis_tier="a", anchor_type="b",
            suspicion_score=0.55,
        )) == "investigate"
        assert auditor._classify(FieldLeadLagProfile(
            field_name="x", source_dataset="y", analysis_tier="a", anchor_type="b",
            suspicion_score=0.3,
        )) == "safe"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_three_files(self, tmp_path):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            total_accounts=100, fully_terminated_accounts=20,
            partially_terminated_accounts=10, active_accounts=70,
            total_terminated_units=30,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="CANCEL_REASON", source_dataset="contract",
                    analysis_tier="contract_level", anchor_type="contract_termination",
                    pct_non_null_terminated=0.9, pct_non_null_active=0.0,
                    suspicion_score=0.95, recommendation="exclude",
                    evidence=["Only populated for terminated"],
                ),
            ],
            suspicious_fields=["CANCEL_REASON"],
            recommended_exclusions=["CANCEL_REASON"],
        )
        paths = result.save(tmp_path / "audit")
        assert (tmp_path / "audit" / "audit_config.yaml").exists()
        assert (tmp_path / "audit" / "column_profiles.yaml").exists()
        assert (tmp_path / "audit" / "recommendations.yaml").exists()
        assert len(paths) == 3

    def test_load_roundtrip(self, tmp_path):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        original = FieldAvailabilityAuditResult(
            config=cfg,
            total_accounts=100, fully_terminated_accounts=20,
            partially_terminated_accounts=10, active_accounts=70,
            total_terminated_units=30,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="CANCEL_REASON", source_dataset="contract",
                    analysis_tier="contract_level", anchor_type="contract_termination",
                    pct_non_null_terminated=0.9, pct_non_null_active=0.0,
                    suspicion_score=0.95, recommendation="exclude",
                    evidence=["Only populated for terminated"],
                ),
                FieldLeadLagProfile(
                    field_name="PLAN_TYPE", source_dataset="contract",
                    analysis_tier="contract_level", anchor_type="contract_termination",
                    pct_non_null_terminated=0.8, pct_non_null_active=0.85,
                    population_ratio=0.94, suspicion_score=0.05,
                    recommendation="safe", evidence=["Similar across cohorts"],
                ),
            ],
            suspicious_fields=["CANCEL_REASON"],
            recommended_exclusions=["CANCEL_REASON"],
        )
        out_dir = tmp_path / "audit"
        original.save(out_dir)
        loaded = FieldAvailabilityAuditResult.load(out_dir)
        assert loaded.total_accounts == 100
        assert loaded.fully_terminated_accounts == 20
        assert loaded.total_terminated_units == 30
        assert len(loaded.field_profiles) == 2
        assert loaded.field_profiles[0].field_name == "CANCEL_REASON"
        assert loaded.field_profiles[0].suspicion_score == 0.95
        assert "Only populated for terminated" in loaded.field_profiles[0].evidence

    def test_recommendations_yaml_structure(self, tmp_path):
        import yaml

        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="F1", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.9, recommendation="exclude",
                    evidence=["Bad field"],
                ),
                FieldLeadLagProfile(
                    field_name="F2", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.6, recommendation="investigate",
                    evidence=["Maybe bad"],
                ),
                FieldLeadLagProfile(
                    field_name="F3", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.1, recommendation="safe",
                ),
            ],
        )
        result.save(tmp_path / "audit")
        recs = yaml.safe_load((tmp_path / "audit" / "recommendations.yaml").read_text())
        assert len(recs["recommended_exclusions"]) == 1
        assert recs["recommended_exclusions"][0]["field_name"] == "F1"
        assert "Bad field" in recs["recommended_exclusions"][0]["rationale"]
        assert len(recs["investigate"]) == 1
        assert recs["safe_fields_count"] == 1

    def test_inf_population_ratio_serialized_as_null(self, tmp_path):
        import yaml

        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="F1", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    population_ratio=float("inf"),
                    suspicion_score=0.9, recommendation="exclude",
                ),
            ],
        )
        result.save(tmp_path / "audit")
        profiles = yaml.safe_load((tmp_path / "audit" / "column_profiles.yaml").read_text())
        assert profiles["columns"][0]["population_ratio"] is None


# ---------------------------------------------------------------------------
# Scoring helper tests
# ---------------------------------------------------------------------------


class TestScoringHelpers:
    def test_both_zero(self):
        assert _population_suspicion_score(0.0, 0.0, 0.0) == 0.0

    def test_inf_ratio(self):
        score = _population_suspicion_score(float("inf"), 0.5, 0.0)
        assert score > 0.7

    def test_high_ratio(self):
        score = _population_suspicion_score(3.0, 0.9, 0.3)
        assert score > 0.5

    def test_normal_ratio(self):
        score = _population_suspicion_score(1.0, 0.5, 0.5)
        assert score < 0.3

    def test_low_ratio(self):
        score = _population_suspicion_score(0.5, 0.3, 0.6)
        assert score == 0.0
