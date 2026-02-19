"""Automatic timestamp scenario detection for ML datasets.

This module provides high-level scenario detection that determines the
appropriate timestamp strategy for a given dataset. It wraps the
TimestampDiscoveryEngine and translates its results into actionable
configurations.

Scenarios:
    - production: Dataset has explicit feature and label timestamps
    - production_derived: Timestamps exist but need derivation
    - partial: Only feature timestamp found, label derived from window
    - derived: Timestamps can be computed from other columns (e.g., tenure)
    - synthetic: No temporal information, must use synthetic timestamps

Example:
    >>> from customer_retention.stages.temporal import ScenarioDetector
    >>> detector = ScenarioDetector()
    >>> scenario, config, discovery = detector.detect(df, "churn")
    >>> print(f"Scenario: {scenario}")  # e.g., "production"
    >>> print(f"Strategy: {config.strategy.value}")  # e.g., "production"
"""

from datetime import datetime
from typing import Any, Optional

from .timestamp_discovery import TimestampDiscoveryEngine, TimestampDiscoveryResult
from .timestamp_manager import TimestampConfig, TimestampStrategy


class ScenarioDetector:
    """Detects the timestamp scenario for a dataset and provides configuration.

    The ScenarioDetector analyzes a dataset to determine which timestamp
    handling strategy is appropriate, returning both a human-readable scenario
    name and a TimestampConfig ready for use with TimestampManager.

    Example:
        >>> detector = ScenarioDetector()
        >>> scenario, config, result = detector.detect(df, "churn")
        >>> # Use config with TimestampManager
        >>> from customer_retention.stages.temporal import TimestampManager
        >>> manager = TimestampManager(config)
        >>> df_with_timestamps = manager.ensure_timestamps(df)
    """
    def __init__(self, reference_date: Optional[datetime] = None, label_window_days: int = 180):
        self.label_window_days = label_window_days
        self.discovery_engine = TimestampDiscoveryEngine(reference_date, label_window_days)

    _TARGET_PATTERNS = (
        "churned", "churn", "retained", "retention", "attrition",
        "unsubscribe", "cancel", "terminate",
        "target", "label", "outcome",
    )

    def _resolve_target_column(self, df: Any, target_column: Optional[str]) -> Optional[str]:
        if target_column and target_column in df.columns:
            return target_column
        if target_column is not None:
            return target_column
        for col in df.columns:
            if any(p in col.lower() for p in self._TARGET_PATTERNS):
                if df[col].nunique() <= 10:
                    return col
        return None

    def _effective_window(self, df: Any, target_column: Optional[str]) -> int:
        resolved = self._resolve_target_column(df, target_column)
        if resolved and resolved in df.columns and len(df) > 0 and df[resolved].notna().all():
            return 0
        return self.label_window_days

    def detect(
        self, df: Any, target_column: str
    ) -> tuple[str, TimestampConfig, TimestampDiscoveryResult]:
        discovery_result = self.discovery_engine.discover(df, target_column)
        window = self._effective_window(df, target_column)

        has_explicit_feature = discovery_result.feature_timestamp and not discovery_result.feature_timestamp.is_derived
        has_explicit_label = discovery_result.label_timestamp and not discovery_result.label_timestamp.is_derived
        label_derived_from_feature = (
            discovery_result.label_timestamp and
            discovery_result.label_timestamp.is_derived and
            discovery_result.feature_timestamp and
            discovery_result.feature_timestamp.column_name in discovery_result.label_timestamp.source_columns
        )

        if has_explicit_feature and has_explicit_label:
            return self._configure_production_scenario(discovery_result, window)
        elif has_explicit_feature and label_derived_from_feature:
            return self._configure_partial_scenario(discovery_result, window)
        elif discovery_result.feature_timestamp and discovery_result.label_timestamp:
            both_derived = (
                discovery_result.feature_timestamp.is_derived
                and discovery_result.label_timestamp.is_derived
            )
            if both_derived and discovery_result.derivable_options:
                return self._configure_derivable_scenario(discovery_result)
            return self._configure_production_scenario(discovery_result, window)
        elif discovery_result.feature_timestamp:
            return self._configure_partial_scenario(discovery_result, window)
        elif discovery_result.derivable_options:
            return self._configure_derivable_scenario(discovery_result)
        return self._configure_synthetic_scenario(discovery_result)

    def _configure_production_scenario(
        self, result: TimestampDiscoveryResult, window: int
    ) -> tuple[str, TimestampConfig, TimestampDiscoveryResult]:
        feature_col = result.feature_timestamp.column_name if result.feature_timestamp else None
        label_col = result.label_timestamp.column_name if result.label_timestamp else None

        derivation_config = {}
        if result.feature_timestamp and result.feature_timestamp.is_derived:
            derivation_config["feature_derivation"] = {
                "formula": result.feature_timestamp.derivation_formula,
                "sources": result.feature_timestamp.source_columns,
            }
        if result.label_timestamp and result.label_timestamp.is_derived:
            derivation_config["label_derivation"] = {
                "formula": result.label_timestamp.derivation_formula,
                "sources": result.label_timestamp.source_columns,
            }

        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column=feature_col if not (result.feature_timestamp and result.feature_timestamp.is_derived) else None,
            label_timestamp_column=label_col if not (result.label_timestamp and result.label_timestamp.is_derived) else None,
            observation_window_days=window,
            derivation_config=derivation_config if derivation_config else None,
        )

        scenario = "production" if not derivation_config else "production_derived"
        return (scenario, config, result)

    def _configure_partial_scenario(
        self, result: TimestampDiscoveryResult, window: int
    ) -> tuple[str, TimestampConfig, TimestampDiscoveryResult]:
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column=result.feature_timestamp.column_name if result.feature_timestamp else None,
            label_timestamp_column=None,
            observation_window_days=window,
            derive_label_from_feature=True,
        )
        return ("partial", config, result)

    def _configure_derivable_scenario(
        self, result: TimestampDiscoveryResult
    ) -> tuple[str, TimestampConfig, TimestampDiscoveryResult]:
        best_derivable = max(result.derivable_options, key=lambda c: c.confidence)

        config = TimestampConfig(
            strategy=TimestampStrategy.DERIVED,
            derivation_config={
                "feature_derivation": {
                    "formula": best_derivable.derivation_formula,
                    "sources": best_derivable.source_columns,
                }
            },
            observation_window_days=self.label_window_days,
        )
        return ("derived", config, result)

    def _configure_synthetic_scenario(
        self, result: TimestampDiscoveryResult
    ) -> tuple[str, TimestampConfig, TimestampDiscoveryResult]:
        config = TimestampConfig(
            strategy=TimestampStrategy.SYNTHETIC_INDEX,
            observation_window_days=self.label_window_days,
            synthetic_base_date="2024-01-01",
        )
        return ("synthetic", config, result)

    def get_scenario_summary(self, scenario: str, config: TimestampConfig, result: TimestampDiscoveryResult) -> dict:
        return {
            "scenario": scenario,
            "strategy": config.strategy.value,
            "feature_timestamp_column": config.feature_timestamp_column,
            "label_timestamp_column": config.label_timestamp_column,
            "observation_window_days": config.observation_window_days,
            "requires_derivation": config.derivation_config is not None,
            "requires_synthetic": result.requires_synthetic,
            "recommendation": result.recommendation,
            "datetime_columns_found": result.discovery_report.get("datetime_columns_found", 0),
            "derivable_timestamps_found": result.discovery_report.get("derivable_timestamps_found", 0),
        }


def auto_detect_and_configure(df: Any, target_column: str) -> tuple[str, TimestampConfig]:
    detector = ScenarioDetector()
    scenario, config, _ = detector.detect(df, target_column)
    return scenario, config
