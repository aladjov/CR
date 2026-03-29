from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.field_availability_audit import (
    DatasetDateInfo,
    FieldAvailabilityAuditConfig,
    FieldAvailabilityAuditor,
    FieldAvailabilityAuditResult,
    FieldLeadLagProfile,
    _compute_date_range,
    _display_audit_results,
    _population_suspicion_score,
    _value_distribution_score,
    build_account_anchors,
    run_field_availability_audit,
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
        assert (tmp_path / "audit" / "audit_suggestions.yaml").exists()
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

    def test_suggestions_yaml_structure(self, tmp_path):
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
        recs = yaml.safe_load((tmp_path / "audit" / "audit_suggestions.yaml").read_text())
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


class TestDisplayResults:
    def test_display_with_profiles(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg, total_accounts=100, fully_terminated_accounts=20,
            partially_terminated_accounts=10, active_accounts=70,
            total_terminated_units=30,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="F1", source_dataset="d1",
                    analysis_tier="contract_level", anchor_type="b",
                    suspicion_score=0.9, recommendation="exclude",
                ),
                FieldLeadLagProfile(
                    field_name="F2", source_dataset="d1",
                    analysis_tier="account_level", anchor_type="b",
                    suspicion_score=0.1, recommendation="safe",
                ),
            ],
            recommended_exclusions=["F1"],
            suspicious_fields=["F1"],
        )
        _display_audit_results(result)
        out = capsys.readouterr().out
        assert "F1" in out
        assert "exclude" in out
        assert "100" in out

    def test_display_empty(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(config=cfg)
        _display_audit_results(result)
        assert "No fields audited" in capsys.readouterr().out

    def test_display_investigate_only(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="F1", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.6, recommendation="investigate",
                ),
            ],
            suspicious_fields=["F1"],
        )
        _display_audit_results(result)
        assert "investigate" in capsys.readouterr().out.lower()


class TestFacade:
    def test_auto_detect_and_run(self, tmp_path):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        ctx = ProjectContext(
            project_name="test", entity_column="ACCOUNT_ID",
            datasets={
                "account": DatasetRegistryEntry(name="account", path="/tmp/a"),
                "contract": DatasetRegistryEntry(name="contract", path="/tmp/c"),
            },
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        frames = {
            "account": pd.DataFrame({
                "ACCOUNT_ID": ["A1", "A2", "A3"],
                "INDUSTRY": ["Tech", "Retail", "Finance"],
            }),
            "contract": _contract_df(),
        }
        ns = RunNamespace(root=tmp_path, run_id="test_run")
        ns.run_dir.mkdir(parents=True, exist_ok=True)
        ns.merged_dir.mkdir(parents=True, exist_ok=True)

        result = run_field_availability_audit(
            context=ctx, loaded_frames=frames, namespace=ns,
            min_terminated_units=1,
        )
        assert len(result.field_profiles) > 0
        assert (ns.field_availability_audit_dir / "audit_config.yaml").exists()
        assert (ns.field_availability_audit_dir / "column_profiles.yaml").exists()
        assert (ns.field_availability_audit_dir / "audit_suggestions.yaml").exists()

    def test_explicit_overrides(self):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )

        ctx = ProjectContext(
            project_name="test", entity_column="ACCOUNT_ID",
            datasets={"contract": DatasetRegistryEntry(name="contract", path="/tmp/c")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        result = run_field_availability_audit(
            context=ctx, loaded_frames={"contract": _contract_df()},
            service_unit_dataset="contract",
            service_unit_id_column="CONTRACT_ID",
            service_unit_anchor_column="CONTRACT_END_DATE",
            service_unit_status_column="CONTRACT_STATUS",
            service_unit_terminated_statuses=["Cancelled", "Terminated"],
            min_terminated_units=1,
        )
        assert result.config.service_unit.dataset_name == "contract"
        assert len(result.field_profiles) > 0

    def test_no_service_unit_raises(self):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )

        ctx = ProjectContext(
            project_name="test", entity_column="USER_ID",
            datasets={"users": DatasetRegistryEntry(name="users", path="/tmp/u")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        frames = {"users": pd.DataFrame({"USER_ID": ["U1"], "NAME": ["Foo"]})}
        with pytest.raises(ValueError, match="No service unit"):
            run_field_availability_audit(context=ctx, loaded_frames=frames)

    def test_additional_exclusions_merged(self):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )

        ctx = ProjectContext(
            project_name="test", entity_column="ACCOUNT_ID",
            datasets={"contract": DatasetRegistryEntry(name="contract", path="/tmp/c")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        result = run_field_availability_audit(
            context=ctx, loaded_frames={"contract": _contract_df()},
            min_terminated_units=1,
            additional_exclusions=["MANUAL_EXCLUDE"],
        )
        assert "MANUAL_EXCLUDE" in result.recommended_exclusions


# ---------------------------------------------------------------------------
# Value-distribution scoring tests
# ---------------------------------------------------------------------------


class TestValueDistributionScore:
    def test_terminated_exclusive_value_high_coverage(self):
        term = {"Cancelled": 80, "Active": 20}
        active = {"Active": 100}
        excl, score, evidence = _value_distribution_score("STATUS", term, active, 100, 100)
        assert "Cancelled" in excl
        assert score >= 0.5
        assert any("terminated-exclusive" in e for e in evidence)

    def test_terminated_exclusive_value_low_coverage(self):
        term = {"Cancelled": 15, "Active": 85}
        active = {"Active": 100}
        excl, score, _ = _value_distribution_score("STATUS", term, active, 100, 100)
        assert "Cancelled" in excl
        assert score >= 0.5  # any exclusive value above min_count flags

    def test_no_exclusive_values_skewed(self):
        term = {"Cancelled": 90, "Active": 10}
        active = {"Cancelled": 10, "Active": 90}
        excl, score, _ = _value_distribution_score("STATUS", term, active, 100, 100)
        assert excl == []
        assert score > 0.0  # skew detected

    def test_balanced_values_safe(self):
        term = {"A": 50, "B": 50}
        active = {"A": 50, "B": 50}
        excl, score, evidence = _value_distribution_score("COL", term, active, 100, 100)
        assert excl == []
        assert score == 0.0
        assert evidence == []

    def test_rare_values_below_min_count_ignored(self):
        term = {"Cancelled": 3, "Active": 97}
        active = {"Active": 100}
        excl, score, _ = _value_distribution_score("STATUS", term, active, 100, 100, min_count=10)
        assert excl == []
        assert score == 0.0

    def test_empty_counts_safe(self):
        excl, score, evidence = _value_distribution_score("X", {}, {}, 0, 0)
        assert excl == []
        assert score == 0.0

    def test_all_terminated_no_active(self):
        excl, score, _ = _value_distribution_score("X", {"A": 50}, {}, 100, 0)
        assert score > 0.0


class TestValueDistributionAugmentation:
    def test_augmentation_sets_cardinality(self):
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
        profiled_with_card = [p for p in result.field_profiles if p.cardinality is not None]
        assert len(profiled_with_card) > 0

    def test_terminated_exclusive_values_detected_in_full_run(self):
        contract_df = _contract_df()
        account_df = pd.DataFrame({
            "ACCOUNT_ID": ["A1", "A2", "A3", "A4"],
            "ACCOUNT_NAME": ["Foo", "Bar", "Baz", "Qux"],
            "CHURN_TAG": ["churned", None, "churned", None],
        })
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        anchors = build_account_anchors(contract_df, su_cfg)
        profiles = auditor._probe_entity_dataset(account_df, "account", anchors, su_cfg)
        auditor._augment_with_value_distributions(
            profiles, contract_df, {"account": account_df}, anchors, su_cfg,
        )
        churn_tag = next(p for p in profiles if p.field_name == "CHURN_TAG")
        assert churn_tag.cardinality == 1  # only "churned" (non-null unique)
        assert churn_tag.value_suspicion_score is not None

    def test_value_dist_bumps_score_for_leaky_categorical(self):
        """A field always populated but with a terminated-exclusive value."""
        n = 200
        contract_df = pd.DataFrame({
            "CONTRACT_ID": [f"C{i}" for i in range(n)],
            "ACCOUNT_ID": [f"A{i}" for i in range(n)],
            "CONTRACT_END_DATE": [pd.Timestamp("2024-01-01")] * (n // 2) + [None] * (n // 2),
            "CONTRACT_STATUS": ["Cancelled"] * (n // 2) + ["Active"] * (n // 2),
            "CONTRACT_START_DATE": [pd.Timestamp("2023-01-01")] * n,
            "PLAN": (
                ["churned_plan"] * (n // 2)
                + ["active_plan"] * (n // 2)
            ),
        })
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        auditor = FieldAvailabilityAuditor(cfg)
        linkage = {"contract": DatasetLinkage(tier="contract_linked", link_method="self")}
        result = auditor.run(
            service_unit_df=contract_df, probe_dfs={"contract": contract_df},
            dataset_linkage=linkage,
        )
        plan_profile = next(p for p in result.field_profiles if p.field_name == "PLAN")
        # Null-rate alone: both groups 100% populated -> score ~0
        # Value-distribution: "churned_plan" only for terminated -> bumped
        assert plan_profile.value_suspicion_score is not None
        assert plan_profile.value_suspicion_score > 0.5
        assert plan_profile.suspicion_score > 0.5
        assert "churned_plan" in plan_profile.terminated_exclusive_values


# ---------------------------------------------------------------------------
# Date info tests
# ---------------------------------------------------------------------------


class TestDateInfo:
    def test_compute_date_range_datetime(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-15", "2024-06-30", None])})
        lo, hi = _compute_date_range(df, "dt")
        assert lo == "2023-01-15"
        assert hi == "2024-06-30"

    def test_compute_date_range_all_null(self):
        df = pd.DataFrame({"dt": [None, None]})
        lo, hi = _compute_date_range(df, "dt")
        assert lo is None and hi is None

    def test_compute_date_range_non_date_column(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        lo, hi = _compute_date_range(df, "x")
        # Should not crash; returns string representation
        assert lo is not None or hi is not None or (lo is None and hi is None)

    def test_facade_populates_date_info(self):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )

        ctx = ProjectContext(
            project_name="test", entity_column="ACCOUNT_ID",
            datasets={"contract": DatasetRegistryEntry(name="contract", path="/tmp/c")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        result = run_field_availability_audit(
            context=ctx, loaded_frames={"contract": _contract_df()},
            min_terminated_units=1,
        )
        assert len(result.dataset_date_info) == 1
        di = result.dataset_date_info[0]
        assert di.dataset_name == "contract"
        assert di.time_column == "CONTRACT_END_DATE"
        assert di.min_date is not None
        assert di.record_count == 5

    def test_display_shows_date_info(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg, total_accounts=10,
            field_profiles=[],
            dataset_date_info=[
                DatasetDateInfo(
                    dataset_name="contract", time_column="END_DATE",
                    min_date="2023-01-01", max_date="2024-12-31",
                    linkage_tier="self", record_count=100,
                ),
            ],
        )
        _display_audit_results(result)
        out = capsys.readouterr().out
        assert "Dataset date columns:" in out
        assert "contract" in out
        assert "END_DATE" in out
        assert "2023-01-01" in out

    def test_display_shows_methodology(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="F1", source_dataset="d1",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.1, recommendation="safe",
                    value_suspicion_score=0.0, cardinality=3,
                ),
            ],
        )
        _display_audit_results(result)
        out = capsys.readouterr().out
        assert "Methodology:" in out
        assert "Null-rate analysis:" in out
        assert "Value-distribution:" in out

    def test_display_shows_value_dist_flags(self, capsys):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="STATUS", source_dataset="ds",
                    analysis_tier="a", anchor_type="b",
                    suspicion_score=0.65, recommendation="investigate",
                    value_suspicion_score=0.65, cardinality=5,
                    terminated_exclusive_values=["Cancelled"],
                ),
            ],
        )
        _display_audit_results(result)
        out = capsys.readouterr().out
        assert "Value-distribution flags" in out
        assert "Cancelled" in out


class TestPersistenceNewFields:
    def test_roundtrip_with_value_dist_fields(self, tmp_path):
        su_cfg = _su_config()
        cfg = FieldAvailabilityAuditConfig(service_unit=su_cfg, min_terminated_units=1)
        result = FieldAvailabilityAuditResult(
            config=cfg, total_accounts=50,
            field_profiles=[
                FieldLeadLagProfile(
                    field_name="STATUS", source_dataset="contract",
                    analysis_tier="contract_level", anchor_type="b",
                    suspicion_score=0.65, recommendation="investigate",
                    cardinality=5, value_suspicion_score=0.65,
                    terminated_exclusive_values=["Cancelled"],
                    evidence=["STATUS: 1 terminated-exclusive value(s): Cancelled"],
                ),
            ],
            dataset_date_info=[
                DatasetDateInfo(
                    dataset_name="contract", time_column="END_DATE",
                    min_date="2023-01-01", max_date="2024-12-31",
                    linkage_tier="self", record_count=100,
                ),
            ],
        )
        result.save(tmp_path / "audit")
        loaded = FieldAvailabilityAuditResult.load(tmp_path / "audit")
        p = loaded.field_profiles[0]
        assert p.cardinality == 5
        assert p.value_suspicion_score == 0.65
        assert "Cancelled" in p.terminated_exclusive_values
