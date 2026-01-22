"""Automatic validation rule generation from exploration findings."""

from typing import Dict, Any
from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType


PERCENTAGE_PATTERNS = ["rate", "pct", "percent", "ratio"]
SKIP_TYPES = [ColumnType.IDENTIFIER, ColumnType.DATETIME, ColumnType.TEXT,
              ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL,
              ColumnType.CATEGORICAL_CYCLICAL, ColumnType.UNKNOWN]


class RuleGenerator:

    @staticmethod
    def for_column(col: ColumnFinding) -> Dict[str, Dict[str, Any]]:
        if col.inferred_type in SKIP_TYPES:
            return {}

        if col.inferred_type == ColumnType.BINARY:
            return {col.name: {"type": "binary", "valid_values": [0, 1]}}

        if col.inferred_type == ColumnType.TARGET:
            distinct = col.type_metrics.get("distinct_count", 0)
            if distinct == 2:
                return {col.name: {"type": "binary", "valid_values": [0, 1]}}
            return {}

        if col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            return RuleGenerator._numeric_rule(col)

        return {}

    @staticmethod
    def _numeric_rule(col: ColumnFinding) -> Dict[str, Dict[str, Any]]:
        name_lower = col.name.lower()
        metrics = col.type_metrics
        min_val = metrics.get("min")
        max_val = metrics.get("max")

        if any(p in name_lower for p in PERCENTAGE_PATTERNS):
            if max_val is not None and max_val <= 1:
                return {col.name: {"type": "percentage", "min": 0, "max": 1}}
            return {col.name: {"type": "percentage", "min": 0, "max": 100}}

        if min_val is not None and min_val >= 0:
            return {col.name: {"type": "non_negative"}}

        return {}

    @staticmethod
    def from_findings(findings: ExplorationFindings) -> Dict[str, Dict[str, Any]]:
        rules = {}
        for col in findings.columns.values():
            rules.update(RuleGenerator.for_column(col))
        return rules
