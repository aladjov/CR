from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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

    def derive(self, df: Any, target_column: Optional[str] = None,
               known_datetime_columns: Optional[list[str]] = None,
               datetime_stats: Optional[dict] = None) -> EntityTimestampResult:
        if datetime_stats is None and known_datetime_columns is not None:
            from customer_retention.core.compat.bulk_profiling import bulk_datetime_discovery_stats
            valid_cols = [c for c in known_datetime_columns if c in df.columns]
            datetime_stats = bulk_datetime_discovery_stats(df, valid_cols) if valid_cols else {}
        discovery = self._engine.discover(df, target_column=target_column, datetime_stats=datetime_stats)
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

    def apply(self, df: Any, result: EntityTimestampResult) -> Any:
        out = df.copy()
        if result.method == "none":
            return out
        if result.method == "direct" and result.column_name in df.columns:
            out["feature_timestamp"] = df[result.column_name]
        elif result.method == "coalesced":
            coalesced = self._order_analyzer.derive_last_action_date(df)
            if coalesced is not None:
                out["feature_timestamp"] = coalesced
        return out
