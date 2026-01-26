from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from customer_retention.core.compat import DataFrame


class TargetLevel(Enum):
    ENTITY_LEVEL = "entity_level"
    EVENT_LEVEL = "event_level"
    UNKNOWN = "unknown"
    MISSING = "missing"


class AggregationMethod(Enum):
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    FIRST = "first"


@dataclass
class TargetDistribution:
    value_counts: Dict[int, int]
    total: int

    @property
    def as_percentages(self) -> Dict[int, float]:
        return {k: v / self.total * 100 for k, v in self.value_counts.items()}

    def get_label(self, value: int) -> str:
        return {1: "Churned", 0: "Retained"}.get(value, str(value))


@dataclass
class TargetLevelResult:
    target_column: str
    entity_column: str
    level: TargetLevel
    suggested_aggregation: Optional[AggregationMethod]
    event_distribution: Optional[TargetDistribution] = None
    entity_distribution: Optional[TargetDistribution] = None
    variation_pct: float = 0.0
    is_binary: bool = False
    entity_target_column: Optional[str] = None
    aggregation_used: Optional[AggregationMethod] = None
    messages: List[str] = field(default_factory=list)


class TargetLevelAnalyzer:
    ENTITY_LEVEL_THRESHOLD = 5.0
    TARGET_KEYWORDS = ['churn', 'unsub', 'cancel', 'retain', 'active', 'lost', 'leave', 'target']

    def __init__(self, variation_threshold: float = 5.0):
        self.variation_threshold = variation_threshold

    def detect_level(self, df: DataFrame, target_column: str, entity_column: str) -> TargetLevelResult:
        if target_column is None or entity_column is None:
            return TargetLevelResult(
                target_column=target_column or "", entity_column=entity_column or "",
                level=TargetLevel.UNKNOWN, suggested_aggregation=None,
                messages=["Target or entity column not specified"])

        if target_column not in df.columns:
            return TargetLevelResult(
                target_column=target_column, entity_column=entity_column,
                level=TargetLevel.MISSING, suggested_aggregation=None,
                messages=[f"Target column '{target_column}' not found in data"])

        event_counts = df[target_column].value_counts().to_dict()
        event_dist = TargetDistribution(value_counts=event_counts, total=len(df))

        target_per_entity = df.groupby(entity_column)[target_column].nunique()
        total_entities = len(target_per_entity)
        variation_pct = ((target_per_entity > 1).sum() / total_entities * 100) if total_entities > 0 else 0
        is_binary = len(event_counts) == 2

        if variation_pct < self.variation_threshold:
            entity_target = df.groupby(entity_column)[target_column].first()
            entity_dist = TargetDistribution(value_counts=entity_target.value_counts().to_dict(), total=len(entity_target))
            return TargetLevelResult(
                target_column=target_column, entity_column=entity_column, level=TargetLevel.ENTITY_LEVEL,
                suggested_aggregation=None, event_distribution=event_dist, entity_distribution=entity_dist,
                variation_pct=variation_pct, is_binary=is_binary,
                messages=["Target is consistent within entities (entity-level)"])

        return TargetLevelResult(
            target_column=target_column, entity_column=entity_column, level=TargetLevel.EVENT_LEVEL,
            suggested_aggregation=self._suggest_aggregation(event_counts, is_binary),
            event_distribution=event_dist, variation_pct=variation_pct, is_binary=is_binary,
            messages=[f"Target varies within entities ({variation_pct:.1f}% have variation)",
                      f"Suggested aggregation: {self._suggest_aggregation(event_counts, is_binary).value}"])

    def aggregate_to_entity(self, df: DataFrame, target_column: str, entity_column: str,
                           time_column: Optional[str] = None,
                           method: AggregationMethod = AggregationMethod.MAX) -> Tuple[DataFrame, TargetLevelResult]:
        result = self.detect_level(df, target_column, entity_column)

        if result.level == TargetLevel.ENTITY_LEVEL:
            result.entity_target_column = target_column
            return df, result

        if result.level in [TargetLevel.MISSING, TargetLevel.UNKNOWN]:
            return df, result

        entity_target_col = f"{target_column}_entity"
        entity_target = self._compute_entity_target(df, target_column, entity_column, time_column, method, result)

        entity_dist = TargetDistribution(value_counts=entity_target.value_counts().to_dict(), total=len(entity_target))
        entity_target_map = entity_target.reset_index()
        entity_target_map.columns = [entity_column, entity_target_col]
        df_result = df.merge(entity_target_map, on=entity_column, how="left")

        result.entity_distribution = entity_dist
        result.entity_target_column = entity_target_col
        result.aggregation_used = method
        result.messages.append(f"Created entity-level target: {entity_target_col}")
        return df_result, result

    def _compute_entity_target(self, df: DataFrame, target_column: str, entity_column: str,
                               time_column: Optional[str], method: AggregationMethod,
                               result: TargetLevelResult):
        agg_funcs = {
            AggregationMethod.MAX: lambda: df.groupby(entity_column)[target_column].max(),
            AggregationMethod.MEAN: lambda: df.groupby(entity_column)[target_column].mean(),
            AggregationMethod.SUM: lambda: df.groupby(entity_column)[target_column].sum(),
        }
        if method in agg_funcs:
            return agg_funcs[method]()

        if method == AggregationMethod.LAST:
            if time_column is None:
                result.messages.append("Warning: 'last' aggregation without time_column uses row order")
                return df.groupby(entity_column)[target_column].last()
            return df.sort_values(time_column).groupby(entity_column)[target_column].last()

        if method == AggregationMethod.FIRST:
            if time_column is None:
                return df.groupby(entity_column)[target_column].first()
            return df.sort_values(time_column).groupby(entity_column)[target_column].first()

        return df.groupby(entity_column)[target_column].max()

    def _suggest_aggregation(self, value_counts: Dict[int, int], is_binary: bool) -> AggregationMethod:
        return AggregationMethod.MAX

    def print_analysis(self, result: TargetLevelResult):
        print("=" * 70 + "\nTARGET LEVEL ANALYSIS\n" + "=" * 70)
        print(f"\nColumn: {result.target_column}\nLevel: {result.level.value.upper()}")

        if result.level == TargetLevel.EVENT_LEVEL:
            print(f"\n⚠️  EVENT-LEVEL TARGET DETECTED\n   {result.variation_pct:.1f}% of entities have varying target values")
            if result.event_distribution:
                print("\n   Event-level distribution:")
                for val, count in sorted(result.event_distribution.value_counts.items()):
                    print(f"      {result.target_column}={val}: {count:,} events ({result.event_distribution.as_percentages[val]:.1f}%)")
            if result.suggested_aggregation:
                print(f"\n   Suggested aggregation: {result.suggested_aggregation.value}")

        elif result.level == TargetLevel.ENTITY_LEVEL:
            print("\n✓ Target is already at entity-level")
            if result.entity_distribution:
                print("\n   Entity-level distribution:")
                for val, count in sorted(result.entity_distribution.value_counts.items()):
                    pct = result.entity_distribution.as_percentages[val]
                    label = result.entity_distribution.get_label(val)
                    print(f"      {label} ({result.target_column}={val}): {count:,} entities ({pct:.1f}%)")

        if result.aggregation_used:
            print(f"\n   Aggregation applied: {result.aggregation_used.value}")
            print(f"   Entity target column: {result.entity_target_column}")
            if result.entity_distribution:
                print("\n   Entity-level distribution (after aggregation):")
                for val, count in sorted(result.entity_distribution.value_counts.items()):
                    pct = result.entity_distribution.as_percentages[val]
                    label = result.entity_distribution.get_label(val)
                    print(f"      {label} ({result.entity_target_column}={val}): {count:,} entities ({pct:.1f}%)")
        print()


class TargetColumnDetector:
    TARGET_KEYWORDS = ['churn', 'unsub', 'cancel', 'retain', 'active', 'lost', 'leave', 'target']

    def detect(self, findings, df: DataFrame, override: Optional[str] = None) -> Tuple[Optional[str], str]:
        from customer_retention.core.config.column_config import ColumnType

        if override == "DEFER_TO_MULTI_DATASET":
            return None, "deferred"
        if override is not None:
            return override, "override"

        for col_name, col_info in findings.columns.items():
            if col_info.inferred_type == ColumnType.TARGET:
                return col_name, "auto-detected"

        for col_name, col_info in findings.columns.items():
            if col_info.inferred_type == ColumnType.BINARY:
                if any(kw in col_name.lower() for kw in self.TARGET_KEYWORDS):
                    return col_name, "binary-candidate"

        return None, "not-found"

    def print_detection(self, target_column: Optional[str], method: str,
                        other_candidates: Optional[List[str]] = None):
        messages = {
            "deferred": "\n⏳ Target deferred to multi-dataset notebook (05)\n   Analysis will proceed without target-based comparisons",
            "override": f"\n🔧 Using override target: {target_column}",
            "auto-detected": f"\n🔍 Auto-detected target: {target_column}",
            "not-found": "\n🔍 No target column detected"
        }
        if method == "binary-candidate":
            print(f"\n🔍 No explicit target detected, using binary candidate: {target_column}")
            if other_candidates:
                print(f"   Other candidates: {other_candidates}")
        else:
            print(messages.get(method, ""))
