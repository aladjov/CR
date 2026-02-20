from __future__ import annotations

from dataclasses import dataclass

from .project_context import CadenceInterval, IntentConfig, PredictionObjective, SplitStrategy, TemporalPosture
from .snapshot_grid import CADENCE_DAYS


@dataclass
class IntentSuggestion:
    config: IntentConfig
    formula_explanations: dict[str, str]


class IntentDefaultsEngine:
    def suggest(
        self,
        objective: PredictionObjective,
        posture: TemporalPosture,
        prediction_horizon: int = 90,
        data_span_days: int | None = None,
        renewal_cycle_days: int | None = None,
    ) -> IntentSuggestion:
        dispatch = {
            (PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE): self._immediate_reactive,
            (PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE): self._immediate_stable,
            (PredictionObjective.DISENGAGEMENT, TemporalPosture.REACTIVE): self._disengagement_reactive,
            (PredictionObjective.DISENGAGEMENT, TemporalPosture.STABLE): self._disengagement_stable,
            (PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE): self._renewal_reactive,
            (PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE): self._renewal_stable,
        }
        handler = dispatch[(objective, posture)]
        recent, recent_formula = handler(prediction_horizon, data_span_days, renewal_cycle_days)
        label_window, label_formula = self._label_window(objective, prediction_horizon, renewal_cycle_days)
        purge, purge_formula = self._purge_gap(objective, prediction_horizon, label_window)
        horizons, horizons_formula = self._prediction_horizons(prediction_horizon)
        cadence, cadence_formula = self._cadence(objective, posture, prediction_horizon, renewal_cycle_days)
        split, split_formula = self._split_strategy(objective)
        config = IntentConfig(
            prediction_horizons=horizons,
            recent_window_days=recent,
            observation_window_days=recent,
            purge_gap_days=purge,
            label_window_days=label_window,
            temporal_split=True,
            cadence_interval=cadence,
            split_strategy=split,
        )
        return IntentSuggestion(
            config=config,
            formula_explanations={
                "prediction_horizons": horizons_formula,
                "recent_window_days": recent_formula,
                "observation_window_days": f"equals recent_window_days = {recent}",
                "purge_gap_days": purge_formula,
                "label_window_days": label_formula,
                "cadence_interval": cadence_formula,
                "split_strategy": split_formula,
            },
        )

    def _immediate_reactive(self, h, _span, _cycle):
        val = 4 * h
        return val, f"4 \u00d7 H = 4 \u00d7 {h} = {val}"

    def _immediate_stable(self, h, _span, _cycle):
        val = max(180, 3 * h)
        return val, f"max(180, 3 \u00d7 H) = max(180, {3 * h}) = {val}"

    def _disengagement_reactive(self, h, _span, _cycle):
        val = 2 * h
        return val, f"2 \u00d7 H = 2 \u00d7 {h} = {val}"

    def _disengagement_stable(self, h, span, _cycle):
        if span is not None:
            raw = int(0.2 * span)
            val = max(180, min(raw, 365))
            return val, f"clamp(0.2 \u00d7 {span}, 180, 365) = clamp({raw}, 180, 365) = {val}"
        return 270, "fallback midpoint(180, 365) = 270"

    def _renewal_reactive(self, _h, _span, cycle):
        val = cycle if cycle is not None else 365
        formula = f"cycle = {cycle}" if cycle is not None else "fallback = 365"
        return val, formula

    def _renewal_stable(self, _h, _span, cycle):
        if cycle is not None:
            raw = 2 * cycle
            val = max(365, min(raw, 730))
            return val, f"clamp(2 \u00d7 {cycle}, 365, 730) = clamp({raw}, 365, 730) = {val}"
        return 548, "fallback midpoint(365, 730) = 548"

    def _label_window(self, objective, h, cycle):
        if objective == PredictionObjective.RENEWAL_RISK:
            val = cycle if cycle is not None else 180
            formula = f"cycle = {cycle}" if cycle is not None else "fallback = 180"
            return val, formula
        return h, f"H = {h}"

    def _purge_gap(self, objective, h, label_window):
        if objective == PredictionObjective.IMMEDIATE_RISK:
            val = h + 14
            return val, f"H + 14 = {h} + 14 = {val}"
        if objective == PredictionObjective.DISENGAGEMENT:
            val = h + 21
            return val, f"H + 21 = {h} + 21 = {val}"
        val = label_window + 21
        return val, f"label_window + 21 = {label_window} + 21 = {val}"

    def _prediction_horizons(self, h):
        if h >= 60:
            vals = [h // 3, h * 2 // 3, h]
            return vals, f"[H//3, 2H//3, H] = {vals}"
        vals = [h // 2, h]
        return vals, f"[H//2, H] = {vals}"

    def _split_strategy(self, objective):
        if objective == PredictionObjective.RENEWAL_RISK:
            return SplitStrategy.COHORT_BASED, "cohort-based (renewal contracts define cohorts)"
        return SplitStrategy.TEMPORAL, "temporal (time-ordered split required)"

    def _cadence(self, objective, posture, h, cycle):
        if objective == PredictionObjective.IMMEDIATE_RISK:
            return self._cadence_immediate(posture, h)
        if objective == PredictionObjective.DISENGAGEMENT:
            return self._cadence_disengagement(posture)
        return self._cadence_renewal(posture, cycle)

    def _cadence_immediate(self, posture, h):
        if posture == TemporalPosture.REACTIVE and h <= 30:
            return CadenceInterval.DAILY, f"daily (reactive, H={h} \u2264 30d)"
        return CadenceInterval.WEEKLY, f"weekly (immediate risk, H={h})"

    def _cadence_disengagement(self, posture):
        if posture == TemporalPosture.REACTIVE:
            return CadenceInterval.WEEKLY, "weekly (disengagement, reactive)"
        return CadenceInterval.MONTHLY, "monthly (disengagement, stable)"

    def _cadence_renewal(self, posture, cycle):
        tier = self._classify_cycle(cycle)
        if posture == TemporalPosture.REACTIVE:
            return self._cadence_renewal_reactive(tier)
        return self._cadence_renewal_stable(tier)

    def _classify_cycle(self, cycle):
        if cycle is None:
            return "mixed"
        if cycle <= 31:
            return "monthly"
        if cycle <= 92:
            return "quarterly"
        return "yearly"

    def _cadence_renewal_reactive(self, tier):
        mapping = {
            "monthly": (CadenceInterval.WEEKLY, "weekly (renewal reactive, monthly cycle)"),
            "quarterly": (CadenceInterval.BIWEEKLY, "biweekly (renewal reactive, quarterly cycle)"),
            "yearly": (CadenceInterval.MONTHLY, "monthly (renewal reactive, yearly cycle)"),
            "mixed": (CadenceInterval.WEEKLY, "weekly (renewal reactive, mixed/unknown cycle)"),
        }
        return mapping[tier]

    def _cadence_renewal_stable(self, tier):
        mapping = {
            "monthly": (CadenceInterval.WEEKLY, "weekly (renewal stable, monthly cycle)"),
            "quarterly": (CadenceInterval.MONTHLY, "monthly (renewal stable, quarterly cycle)"),
            "yearly": (CadenceInterval.MONTHLY, "monthly (renewal stable, yearly cycle)"),
            "mixed": (CadenceInterval.MONTHLY, "monthly (renewal stable, mixed/unknown cycle)"),
        }
        return mapping[tier]


def format_dataset_preview(
    temporal_span_days: int,
    cadence_interval: CadenceInterval,
    unique_entities: int,
    avg_rows_per_entity: float,
) -> str:
    cadence_days = CADENCE_DAYS[cadence_interval]
    span_periods = temporal_span_days // cadence_days
    unit = cadence_interval.value
    total_rows = int(unique_entities * avg_rows_per_entity)
    return (
        f"**Dataset preview:** ~{span_periods:,} {unit} periods of history, "
        f"~{unique_entities:,} unique entities with ~{avg_rows_per_entity:.1f} "
        f"events each, yielding ~{total_rows:,} rows before train/gap/test splitting."
    )
