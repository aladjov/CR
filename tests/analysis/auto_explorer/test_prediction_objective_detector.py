from __future__ import annotations

import pandas as pd

from customer_retention.analysis.auto_explorer.prediction_objective_detector import (
    PredictionObjectiveAssessment,
    PredictionObjectiveDetector,
    derive_objective_support,
)
from customer_retention.analysis.auto_explorer.project_context import (
    ObjectivePriority,
    ObjectiveSpec,
    ObjectiveSupport,
    ObjectiveSupportEntry,
    PredictionAnchor,
    PredictionObjective,
)
from customer_retention.stages.profiling.temporal_coverage import TemporalComparison


def _churn_df():
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(100)],
        "churned": [1 if i < 30 else 0 for i in range(100)],
        "age": [20 + i % 50 for i in range(100)],
    })


def _renewal_df():
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(100)],
        "contract_end_date": pd.to_datetime([f"2025-{(i % 12) + 1:02d}-15" for i in range(100)]),
        "renewed": [1 if i < 60 else 0 for i in range(100)],
        "subscription_type": ["monthly" if i % 2 == 0 else "annual" for i in range(100)],
    })


def _disengagement_df():
    dates = []
    ids = []
    for cid in range(50):
        for day in range(0, 240, 3):
            dates.append(pd.Timestamp("2024-01-01") + pd.Timedelta(days=day))
            ids.append(f"C{cid:04d}")
    return pd.DataFrame({
        "customer_id": ids,
        "event_timestamp": dates,
        "action": ["login"] * len(ids),
    })


def _minimal_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    })


class TestImmediateRiskDetection:
    def test_detects_churn_target(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        immediate = [r for r in results if r.objective == PredictionObjective.IMMEDIATE_RISK]
        assert len(immediate) == 1
        assert immediate[0].feasible is True

    def test_confidence_positive(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        immediate = [r for r in results if r.objective == PredictionObjective.IMMEDIATE_RISK][0]
        assert immediate.confidence > 0.0

    def test_suggested_anchor_is_now(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        immediate = [r for r in results if r.objective == PredictionObjective.IMMEDIATE_RISK][0]
        assert immediate.suggested_anchor == PredictionAnchor.NOW

    def test_has_evidence(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        immediate = [r for r in results if r.objective == PredictionObjective.IMMEDIATE_RISK][0]
        assert len(immediate.evidence) > 0


class TestRenewalRiskDetection:
    def test_detects_renewal_columns(self):
        df = _renewal_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "renewed", None,
        )
        renewal = [r for r in results if r.objective == PredictionObjective.RENEWAL_RISK]
        assert len(renewal) == 1
        assert renewal[0].feasible is True

    def test_suggested_anchor_is_contract(self):
        df = _renewal_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "renewed", None,
        )
        renewal = [r for r in results if r.objective == PredictionObjective.RENEWAL_RISK][0]
        assert renewal.suggested_anchor == PredictionAnchor.CONTRACT

    def test_not_feasible_without_contract_columns(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        renewal = [r for r in results if r.objective == PredictionObjective.RENEWAL_RISK]
        assert len(renewal) == 0 or renewal[0].feasible is False


class TestDisengagementDetection:
    def test_detects_from_event_data(self):
        df = _disengagement_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", None, "event_timestamp",
        )
        disengage = [r for r in results if r.objective == PredictionObjective.DISENGAGEMENT]
        assert len(disengage) == 1
        assert disengage[0].feasible is True

    def test_suggested_anchor_is_inactivity(self):
        df = _disengagement_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", None, "event_timestamp",
        )
        disengage = [r for r in results if r.objective == PredictionObjective.DISENGAGEMENT][0]
        assert disengage.suggested_anchor == PredictionAnchor.INACTIVITY

    def test_not_feasible_with_short_span(self):
        df = pd.DataFrame({
            "customer_id": ["C001", "C001", "C002", "C002"],
            "event_timestamp": pd.to_datetime([
                "2024-01-01", "2024-01-15",
                "2024-01-01", "2024-01-20",
            ]),
            "action": ["login"] * 4,
        })
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", None, "event_timestamp",
        )
        disengage = [r for r in results if r.objective == PredictionObjective.DISENGAGEMENT]
        assert len(disengage) == 0 or disengage[0].feasible is False

    def test_has_evidence(self):
        df = _disengagement_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", None, "event_timestamp",
        )
        disengage = [r for r in results if r.objective == PredictionObjective.DISENGAGEMENT][0]
        assert len(disengage.evidence) > 0


class TestMultipleFeasibleObjectives:
    def test_churn_and_renewal_both_feasible(self):
        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(100)],
            "churned": [1 if i < 30 else 0 for i in range(100)],
            "contract_end_date": pd.to_datetime([f"2025-{(i % 12) + 1:02d}-15" for i in range(100)]),
            "subscription_type": ["monthly" if i % 2 == 0 else "annual" for i in range(100)],
        })
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        feasible = [r for r in results if r.feasible]
        objectives = {r.objective for r in feasible}
        assert PredictionObjective.IMMEDIATE_RISK in objectives
        assert PredictionObjective.RENEWAL_RISK in objectives


class TestNoFeasibleObjectives:
    def test_minimal_data(self):
        df = _minimal_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "id", None, None,
        )
        feasible = [r for r in results if r.feasible]
        assert len(feasible) == 0


class TestAssessmentStructure:
    def test_assessment_fields(self):
        assessment = PredictionObjectiveAssessment(
            objective=PredictionObjective.IMMEDIATE_RISK,
            feasible=True,
            confidence=0.9,
            evidence=["Binary target detected"],
            suggested_anchor=PredictionAnchor.NOW,
            parameters={},
        )
        assert assessment.objective == PredictionObjective.IMMEDIATE_RISK
        assert assessment.feasible is True
        assert assessment.confidence == 0.9
        assert len(assessment.evidence) == 1
        assert assessment.suggested_anchor == PredictionAnchor.NOW

    def test_parameters_populated_for_immediate_risk(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        immediate = [r for r in results if r.objective == PredictionObjective.IMMEDIATE_RISK][0]
        assert isinstance(immediate.parameters, dict)


class TestDetectorWithNoneColumns:
    def test_none_target_column(self):
        df = _disengagement_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", None, "event_timestamp",
        )
        assert isinstance(results, list)

    def test_none_time_column(self):
        df = _churn_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "customer_id", "churned", None,
        )
        assert isinstance(results, list)

    def test_both_none(self):
        df = _minimal_df()
        results = PredictionObjectiveDetector().detect_feasible_objectives(
            df, "id", None, None,
        )
        assert isinstance(results, list)


def _stable_comparison():
    return TemporalComparison(
        entity_ratio=0.85,
        event_ratio=0.25,
        events_per_entity_change=-0.05,
        numeric_columns_drifted_pct=0.1,
        categorical_shift_detected=False,
        regime_shift_detected=False,
        regime_shift_severity="low",
        target_rate_delta=0.02,
        representativeness_score=0.8,
    )


def _drifted_comparison():
    return TemporalComparison(
        entity_ratio=0.4,
        event_ratio=0.15,
        events_per_entity_change=-0.5,
        numeric_columns_drifted_pct=0.6,
        categorical_shift_detected=True,
        regime_shift_detected=True,
        regime_shift_severity="high",
        target_rate_delta=-0.15,
        representativeness_score=0.3,
    )


def _immediate_risk_spec():
    return ObjectiveSpec(
        objective=PredictionObjective.IMMEDIATE_RISK,
        priority=ObjectivePriority.PRIMARY,
        anchor=PredictionAnchor.NOW,
    )


def _disengagement_spec():
    return ObjectiveSpec(
        objective=PredictionObjective.DISENGAGEMENT,
        priority=ObjectivePriority.SECONDARY,
        anchor=PredictionAnchor.INACTIVITY,
    )


def _renewal_risk_spec():
    return ObjectiveSpec(
        objective=PredictionObjective.RENEWAL_RISK,
        priority=ObjectivePriority.SECONDARY,
        anchor=PredictionAnchor.CONTRACT,
    )


class TestDeriveObjectiveSupport:
    def test_returns_objective_support(self):
        comparison = _stable_comparison()
        objectives = [_immediate_risk_spec()]
        result = derive_objective_support(comparison, objectives)
        assert isinstance(result, ObjectiveSupport)

    def test_scope_and_lifecycle(self):
        result = derive_objective_support(_stable_comparison(), [_immediate_risk_spec()])
        assert result.scope == "dataset"
        assert result.lifecycle_stage == "evidence"

    def test_one_entry_per_objective(self):
        objectives = [_immediate_risk_spec(), _disengagement_spec(), _renewal_risk_spec()]
        result = derive_objective_support(_stable_comparison(), objectives)
        assert len(result.entries) == 3
        assert PredictionObjective.IMMEDIATE_RISK in result.entries
        assert PredictionObjective.DISENGAGEMENT in result.entries
        assert PredictionObjective.RENEWAL_RISK in result.entries

    def test_entry_has_signals(self):
        result = derive_objective_support(_stable_comparison(), [_immediate_risk_spec()])
        entry = result.entries[PredictionObjective.IMMEDIATE_RISK]
        assert isinstance(entry.signals, dict)
        assert len(entry.signals) > 0

    def test_entry_has_implications(self):
        result = derive_objective_support(_stable_comparison(), [_immediate_risk_spec()])
        entry = result.entries[PredictionObjective.IMMEDIATE_RISK]
        assert isinstance(entry.implications, list)
        assert len(entry.implications) > 0

    def test_entry_strength_valid(self):
        result = derive_objective_support(_stable_comparison(), [_immediate_risk_spec()])
        entry = result.entries[PredictionObjective.IMMEDIATE_RISK]
        assert entry.strength in ("weak", "medium", "strong")

    def test_stable_data_immediate_risk_strong(self):
        result = derive_objective_support(_stable_comparison(), [_immediate_risk_spec()])
        entry = result.entries[PredictionObjective.IMMEDIATE_RISK]
        assert entry.strength in ("medium", "strong")

    def test_drifted_data_weakens_support(self):
        result = derive_objective_support(_drifted_comparison(), [_immediate_risk_spec()])
        entry = result.entries[PredictionObjective.IMMEDIATE_RISK]
        assert entry.strength in ("weak", "medium")

    def test_disengagement_uses_coverage_signals(self):
        result = derive_objective_support(_stable_comparison(), [_disengagement_spec()])
        entry = result.entries[PredictionObjective.DISENGAGEMENT]
        assert isinstance(entry.signals, dict)
        assert entry.strength in ("weak", "medium", "strong")

    def test_renewal_risk_regime_shift_impact(self):
        result = derive_objective_support(_drifted_comparison(), [_renewal_risk_spec()])
        entry = result.entries[PredictionObjective.RENEWAL_RISK]
        assert entry.strength in ("weak", "medium")

    def test_empty_objectives_returns_empty(self):
        result = derive_objective_support(_stable_comparison(), [])
        assert len(result.entries) == 0

    def test_all_entries_are_support_entries(self):
        objectives = [_immediate_risk_spec(), _disengagement_spec()]
        result = derive_objective_support(_stable_comparison(), objectives)
        for entry in result.entries.values():
            assert isinstance(entry, ObjectiveSupportEntry)
