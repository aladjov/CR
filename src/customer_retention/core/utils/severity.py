from dataclasses import dataclass
from typing import Optional

from ..components.enums import Severity


@dataclass
class ThresholdConfig:
    critical: Optional[float] = None
    high: Optional[float] = None
    warning: Optional[float] = None
    medium: Optional[float] = None
    low: Optional[float] = None
    ascending: bool = True


def classify_by_thresholds(value: float, config: ThresholdConfig) -> Severity:
    if config.ascending:
        if config.critical is not None and value >= config.critical:
            return Severity.CRITICAL
        if config.high is not None and value >= config.high:
            return Severity.HIGH
        if config.warning is not None and value >= config.warning:
            return Severity.WARNING
        if config.medium is not None and value >= config.medium:
            return Severity.MEDIUM
        if config.low is not None and value >= config.low:
            return Severity.LOW
    else:
        if config.critical is not None and value <= config.critical:
            return Severity.CRITICAL
        if config.high is not None and value <= config.high:
            return Severity.HIGH
        if config.warning is not None and value <= config.warning:
            return Severity.WARNING
        if config.medium is not None and value <= config.medium:
            return Severity.MEDIUM
        if config.low is not None and value <= config.low:
            return Severity.LOW
    return Severity.INFO


def severity_recommendation(severity: Severity, context: str, action_critical: str = "investigate immediately",
                            action_warning: str = "monitor closely", action_info: str = "no action needed") -> str:
    recommendations = {
        Severity.CRITICAL: f"CRITICAL: {context}. {action_critical}.",
        Severity.HIGH: f"HIGH: {context}. {action_critical}.",
        Severity.WARNING: f"WARNING: {context}. {action_warning}.",
        Severity.MEDIUM: f"MEDIUM: {context}. {action_warning}.",
        Severity.LOW: f"LOW: {context}. {action_info}.",
        Severity.INFO: f"INFO: {context}. {action_info}.",
    }
    return recommendations.get(severity, f"INFO: {context}.")
