from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from customer_retention.analysis.auto_explorer.project_context import (
    ObjectiveSpec,
    ObjectiveSupport,
    ObjectiveSupportEntry,
    PredictionAnchor,
    PredictionObjective,
)
from customer_retention.core.compat import head_as_list, is_datetime64_any_dtype, native_pd, pd, safe_to_datetime
from customer_retention.stages.profiling.temporal_coverage import TemporalComparison

_IMMEDIATE_RISK_PATTERNS = [
    "churn", "churned", "cancel", "cancelled", "terminate", "terminated",
    "unsubscribe", "unsubscribed", "downgrade", "downgraded", "attrition",
    "exit", "exited", "leave", "left", "close", "closed", "discontinue",
    "retained", "retention",
]

_RENEWAL_COLUMN_PATTERNS = [
    "contract", "subscription", "renewal", "expiry", "expire",
    "plan", "term", "renew",
]


@dataclass
class PredictionObjectiveAssessment:
    objective: PredictionObjective
    feasible: bool
    confidence: float
    evidence: list[str]
    suggested_anchor: PredictionAnchor
    parameters: dict = field(default_factory=dict)


class PredictionObjectiveDetector:
    def detect_feasible_objectives(
        self,
        target_df: pd.DataFrame,
        entity_column: str,
        target_column: Optional[str],
        time_column: Optional[str],
    ) -> list[PredictionObjectiveAssessment]:
        results = []
        immediate = self._assess_immediate_risk(target_df, target_column)
        if immediate is not None:
            results.append(immediate)
        renewal = self._assess_renewal_risk(target_df, target_column)
        if renewal is not None:
            results.append(renewal)
        disengagement = self._assess_disengagement(target_df, entity_column, time_column)
        if disengagement is not None:
            results.append(disengagement)
        return results

    def _assess_immediate_risk(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
    ) -> Optional[PredictionObjectiveAssessment]:
        if target_column is None or target_column not in df.columns:
            return None

        target_lower = target_column.lower()
        pattern_match = any(p in target_lower for p in _IMMEDIATE_RISK_PATTERNS)
        series = df[target_column].dropna()
        n_unique = series.nunique()
        is_binary = n_unique == 2

        if not pattern_match and not is_binary:
            return None

        evidence = []
        confidence = 0.0

        if pattern_match:
            evidence.append(f"Target column '{target_column}' matches churn/cancel pattern")
            confidence += 0.5

        if is_binary:
            evidence.append(f"Binary target with {n_unique} unique values")
            confidence += 0.3

        if n_unique <= 5:
            evidence.append(f"Low cardinality target ({n_unique} classes)")
            confidence += 0.1

        positive_rate = series.mean() if is_binary else None
        params = {}
        if positive_rate is not None:
            params["positive_rate"] = round(float(positive_rate), 4)
            evidence.append(f"Positive class rate: {positive_rate:.1%}")
            confidence += 0.1

        confidence = min(confidence, 1.0)
        feasible = confidence >= 0.3

        return PredictionObjectiveAssessment(
            objective=PredictionObjective.IMMEDIATE_RISK,
            feasible=feasible,
            confidence=confidence,
            evidence=evidence,
            suggested_anchor=PredictionAnchor.NOW,
            parameters=params,
        )

    def _assess_renewal_risk(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
    ) -> Optional[PredictionObjectiveAssessment]:
        renewal_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in _RENEWAL_COLUMN_PATTERNS):
                renewal_columns.append(col)

        if not renewal_columns:
            return PredictionObjectiveAssessment(
                objective=PredictionObjective.RENEWAL_RISK,
                feasible=False,
                confidence=0.0,
                evidence=["No contract/subscription/renewal columns found"],
                suggested_anchor=PredictionAnchor.CONTRACT,
            )

        evidence = [f"Renewal-related columns found: {', '.join(renewal_columns)}"]
        confidence = 0.5

        date_columns = [c for c in renewal_columns if _looks_like_date_column(df, c)]
        if date_columns:
            evidence.append(f"Date-like renewal columns: {', '.join(date_columns)}")
            confidence += 0.3

        if target_column and target_column in df.columns:
            target_lower = target_column.lower()
            if any(p in target_lower for p in ["renew", "renewed", "retention"]):
                evidence.append(f"Target '{target_column}' matches renewal pattern")
                confidence += 0.2

        confidence = min(confidence, 1.0)

        return PredictionObjectiveAssessment(
            objective=PredictionObjective.RENEWAL_RISK,
            feasible=True,
            confidence=confidence,
            evidence=evidence,
            suggested_anchor=PredictionAnchor.CONTRACT,
            parameters={"renewal_columns": renewal_columns},
        )

    def _assess_disengagement(
        self,
        df: pd.DataFrame,
        entity_column: str,
        time_column: Optional[str],
    ) -> Optional[PredictionObjectiveAssessment]:
        if time_column is None or time_column not in df.columns:
            return None
        if entity_column not in df.columns:
            return None

        try:
            ts = safe_to_datetime(df[time_column])
        except Exception:
            return None

        valid_ts = ts.dropna()
        if len(valid_ts) < 10:
            return None

        span = valid_ts.max() - valid_ts.min()
        span_days = span.days

        if span_days < 60:
            return PredictionObjectiveAssessment(
                objective=PredictionObjective.DISENGAGEMENT,
                feasible=False,
                confidence=0.0,
                evidence=[f"Temporal span too short ({span_days} days, need 60+)"],
                suggested_anchor=PredictionAnchor.INACTIVITY,
            )

        n_entities = df[entity_column].nunique()
        n_rows = len(df)
        avg_events = n_rows / n_entities if n_entities > 0 else 0

        evidence = [
            f"Temporal span: {span_days} days",
            f"Entities: {n_entities}, avg events per entity: {avg_events:.1f}",
        ]
        confidence = 0.4

        if avg_events > 3:
            evidence.append("Sufficient event density for disengagement tracking")
            confidence += 0.3

        if span_days > 180:
            evidence.append("Long history supports inactivity pattern detection")
            confidence += 0.2

        confidence = min(confidence, 1.0)

        return PredictionObjectiveAssessment(
            objective=PredictionObjective.DISENGAGEMENT,
            feasible=True,
            confidence=confidence,
            evidence=evidence,
            suggested_anchor=PredictionAnchor.INACTIVITY,
            parameters={"span_days": span_days, "avg_events_per_entity": round(avg_events, 2)},
        )


def derive_objective_support(
    comparison: TemporalComparison,
    objectives: list[ObjectiveSpec],
) -> ObjectiveSupport:
    entries = {}
    for spec in objectives:
        if spec.objective == PredictionObjective.IMMEDIATE_RISK:
            entries[spec.objective] = _support_immediate_risk(comparison)
        elif spec.objective == PredictionObjective.DISENGAGEMENT:
            entries[spec.objective] = _support_disengagement(comparison)
        elif spec.objective == PredictionObjective.RENEWAL_RISK:
            entries[spec.objective] = _support_renewal_risk(comparison)
    return ObjectiveSupport(entries=entries)


def _support_immediate_risk(c: TemporalComparison) -> ObjectiveSupportEntry:
    signals = {
        "entity_ratio": c.entity_ratio,
        "event_ratio": c.event_ratio,
        "numeric_drift_pct": c.numeric_columns_drifted_pct,
        "regime_shift": c.regime_shift_detected,
        "representativeness": c.representativeness_score,
    }
    if c.target_rate_delta is not None:
        signals["target_rate_delta"] = c.target_rate_delta

    implications = []
    if c.entity_ratio >= 0.7:
        implications.append("Recent window covers most entities — short-window features reliable")
    else:
        implications.append("Low recent entity coverage — short-window features may miss many entities")

    if c.regime_shift_detected:
        implications.append(
            f"Regime shift ({c.regime_shift_severity}) detected — "
            "recent behavior may not match historical patterns"
        )

    if c.numeric_columns_drifted_pct > 0.3:
        implications.append("High feature drift — model trained on full history may underperform on recent data")

    if c.target_rate_delta is not None and abs(c.target_rate_delta) > 0.1:
        implications.append(f"Target rate shifted by {c.target_rate_delta:+.2f} — class balance changing")

    strength = _assess_strength(
        c.representativeness_score,
        c.regime_shift_detected,
        c.numeric_columns_drifted_pct,
    )
    return ObjectiveSupportEntry(signals=signals, implications=implications, strength=strength)


def _support_disengagement(c: TemporalComparison) -> ObjectiveSupportEntry:
    signals = {
        "entity_ratio": c.entity_ratio,
        "events_per_entity_change": c.events_per_entity_change,
        "regime_shift": c.regime_shift_detected,
        "representativeness": c.representativeness_score,
    }

    implications = []
    if c.events_per_entity_change < -0.3:
        implications.append("Entity density dropping — disengagement trend may already be present in data")
    elif c.events_per_entity_change > 0.3:
        implications.append("Entity density rising — trend features may capture engagement growth, not decay")
    else:
        implications.append("Stable entity density — trend-based features should be reliable")

    if c.entity_ratio < 0.5:
        implications.append("Many entities absent from recent window — long-term coverage gaps affect trend visibility")
    else:
        implications.append("Good entity continuity across time windows")

    if c.regime_shift_detected:
        implications.append("Regime shift may fragment long-term behavioral trends")

    strength = _assess_strength(
        c.representativeness_score,
        c.regime_shift_detected,
        c.numeric_columns_drifted_pct,
    )
    return ObjectiveSupportEntry(signals=signals, implications=implications, strength=strength)


def _support_renewal_risk(c: TemporalComparison) -> ObjectiveSupportEntry:
    signals = {
        "entity_ratio": c.entity_ratio,
        "regime_shift": c.regime_shift_detected,
        "regime_shift_severity": c.regime_shift_severity,
        "representativeness": c.representativeness_score,
    }

    implications = []
    if c.regime_shift_detected:
        implications.append(
            f"Regime shift (severity: {c.regime_shift_severity}) — "
            "historical contract patterns may not reflect current behavior"
        )
    else:
        implications.append("No regime shift — long-horizon contract features should be stable")

    if c.numeric_columns_drifted_pct > 0.3:
        implications.append("Feature drift affects contract-cycle feature reliability")

    if c.entity_ratio < 0.5:
        implications.append("Low recent entity coverage — renewal predictions may lack recent behavioral context")

    regime_penalty = 0.15 if c.regime_shift_detected else 0.0
    if c.regime_shift_severity == "high":
        regime_penalty = 0.25

    strength = _assess_strength(
        c.representativeness_score,
        c.regime_shift_detected,
        c.numeric_columns_drifted_pct,
        regime_penalty=regime_penalty,
    )
    return ObjectiveSupportEntry(signals=signals, implications=implications, strength=strength)


def _assess_strength(
    representativeness: float,
    regime_shift: bool,
    drift_pct: float,
    regime_penalty: float = 0.0,
) -> str:
    score = representativeness
    if regime_shift:
        score -= max(regime_penalty, 0.1)
    if drift_pct > 0.3:
        score -= 0.15
    if score >= 0.6:
        return "strong"
    if score >= 0.35:
        return "medium"
    return "weak"


def _looks_like_date_column(df: pd.DataFrame, col: str) -> bool:
    if is_datetime64_any_dtype(df[col]):
        return True
    sample = df[col].dropna().head(20)
    if len(sample) == 0:
        return False
    parseable = 0
    for val in head_as_list(sample, 20):
        try:
            native_pd.to_datetime(val, format="mixed")
            parseable += 1
        except (ValueError, TypeError):
            pass
    return parseable / len(sample) > 0.5
