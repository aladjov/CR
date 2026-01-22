from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from customer_retention.core.compat import pd
from customer_retention.core.config import DataSourceConfig


@dataclass
class SCDResult:
    """Result of SCD analysis for a column."""
    column_name: str
    changes_detected: bool
    entities_with_change: int
    change_percentage: float
    max_changes: int
    avg_changes_per_entity: float
    scd_type_recommendation: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SCDAnalyzer:
    """Analyzes Slowly Changing Dimension patterns in data."""

    def __init__(self, entity_key: Optional[str] = None):
        """
        Initialize SCD Analyzer.

        Args:
            entity_key: Column name that identifies unique entities (e.g., customer_id)
        """
        self.entity_key = entity_key

    def analyze(self, df: pd.DataFrame, columns: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze SCD patterns in dataframe.

        Args:
            df: DataFrame with multi-row per entity data
            columns: List of columns to analyze (if None, analyze all except entity_key)

        Returns:
            Dictionary mapping column names to SCD metrics
        """
        if self.entity_key is None:
            raise ValueError("entity_key must be set to analyze SCD patterns")

        if self.entity_key not in df.columns:
            raise ValueError(f"entity_key '{self.entity_key}' not found in dataframe")

        # Determine columns to analyze
        if columns is None:
            columns = [col for col in df.columns if col != self.entity_key]

        results = {}

        for column in columns:
            metrics = self._analyze_column(df, column)
            results[column] = metrics

        return results

    def _analyze_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze SCD pattern for a single column."""
        # Group by entity and count distinct values per entity
        entity_changes = df.groupby(self.entity_key)[column].nunique()

        # Entities with more than 1 value = changes detected
        entities_with_change = (entity_changes > 1).sum()
        total_entities = len(entity_changes)

        change_percentage = (entities_with_change / total_entities * 100) if total_entities > 0 else 0.0

        # Max changes for any entity
        max_changes = int(entity_changes.max() - 1) if len(entity_changes) > 0 else 0

        # Average changes per entity (only for entities with changes)
        avg_changes = float(entity_changes[entity_changes > 1].mean() - 1) if entities_with_change > 0 else 0.0

        metrics = {
            "changes_detected": bool(entities_with_change > 0),  # Convert numpy bool to Python bool
            "entities_with_change": int(entities_with_change),
            "total_entities": int(total_entities),
            "change_percentage": round(change_percentage, 2),
            "max_changes": max_changes,
            "avg_changes_per_entity": round(avg_changes, 2),
        }

        # Add SCD type recommendation
        metrics["scd_type_recommendation"] = self.recommend_scd_type(metrics)

        return metrics

    def recommend_scd_type(self, metrics: Dict[str, Any]) -> str:
        """
        Recommend SCD type based on change patterns.

        Returns:
            String describing recommended SCD type
        """
        if not metrics["changes_detected"]:
            return "Type 0 (Static - Never changes)"

        change_pct = metrics["change_percentage"]
        avg_changes = metrics.get("avg_changes_per_entity", 0)

        # Type 1: Rare changes, only current value matters
        if change_pct < 10 and avg_changes < 2:
            return "Type 1 (Overwrite - Rare changes, history not important)"

        # Type 2: Frequent changes, history matters
        elif change_pct >= 30 or avg_changes >= 3:
            return "Type 2 (Track History - Frequent changes, full history needed)"

        # Type 3: Moderate changes, only previous value matters
        elif change_pct < 30 and avg_changes < 3:
            return "Type 3 (Keep Previous - Only previous value matters)"

        # Default
        return "Type 2 (Track History - Moderate to frequent changes)"

    def analyze_with_config(self, df: pd.DataFrame, config: DataSourceConfig) -> Dict[str, Dict[str, Any]]:
        """
        Analyze SCD patterns using configuration.

        Args:
            df: DataFrame to analyze
            config: DataSourceConfig with entity key information

        Returns:
            Dictionary of SCD metrics per column
        """
        # Use primary key as entity key
        self.entity_key = config.primary_key

        # Analyze all columns except primary key
        columns = [col.name for col in config.columns if col.name != config.primary_key]

        return self.analyze(df, columns)

    def to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert SCD analysis results to a summary DataFrame."""
        rows = []
        for column_name, metrics in results.items():
            row = {"column": column_name}
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)
