from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from customer_retention.stages.temporal.timestamp_discovery import (
    DatetimeOrderAnalyzer,
    TimestampDiscoveryEngine,
)


@dataclass
class EntityTimestampResult:
    column_name: Optional[str]
    source_columns: list[str] = field(default_factory=list)
    method: str = "none"


class EntityFeatureTimestampDeriver:
    def __init__(self):
        self._engine = TimestampDiscoveryEngine()
        self._order_analyzer = DatetimeOrderAnalyzer()

    def derive(self, df: pd.DataFrame, target_column: Optional[str] = None) -> EntityTimestampResult:
        discovery = self._engine.discover(df, target_column=target_column)
        if discovery.feature_timestamp is not None:
            candidate = discovery.feature_timestamp
            return EntityTimestampResult(
                column_name=candidate.column_name,
                source_columns=candidate.source_columns or [candidate.column_name],
                method="direct" if not candidate.is_derived else "coalesced",
            )
        coalesced = self._order_analyzer.derive_last_action_date(df)
        if coalesced is not None and coalesced.notna().any():
            ordered = self._order_analyzer.analyze_datetime_ordering(df)
            return EntityTimestampResult(
                column_name="feature_timestamp",
                source_columns=ordered,
                method="coalesced",
            )
        return EntityTimestampResult(column_name=None, source_columns=[], method="none")

    def apply(self, df: pd.DataFrame, result: EntityTimestampResult) -> pd.DataFrame:
        out = df.copy()
        if result.method == "none":
            return out
        if result.method == "direct" and result.column_name in df.columns:
            out["feature_timestamp"] = df[result.column_name]
        elif result.method == "coalesced":
            coalesced = self._order_analyzer.derive_last_action_date(df)
            if coalesced is not None:
                out["feature_timestamp"] = coalesced.values
        return out
