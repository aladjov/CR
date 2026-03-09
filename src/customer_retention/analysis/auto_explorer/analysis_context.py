from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from customer_retention.core.compat import pd
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta

from .active_dataset_store import load_active_dataset, load_silver_merged
from .findings import ExplorationFindings
from .layered_recommendations import RecommendationRegistry
from .recommendation_builder import RecommendationBuilder
from .session import load_notebook_findings, resolve_target_column


class AnalysisContext:
    def __init__(
        self, notebook_name: str, *,
        prefer_merged: bool = False, prefer_aggregated: bool = True,
        exclude_aggregated: bool = False,
    ):
        self._notebook_name = notebook_name
        self._builder: Optional[RecommendationBuilder] = None
        self.findings_path, self.namespace, self.dataset_name = load_notebook_findings(
            notebook_name, prefer_merged=prefer_merged,
            prefer_aggregated=prefer_aggregated,
            exclude_aggregated=exclude_aggregated,
        )
        self.findings = ExplorationFindings.load(self.findings_path)
        self.target = resolve_target_column(self.namespace, self.findings)
        self.recommendations_path = self._resolve_recommendations_path()
        self.registry = self._load_or_create_registry()
        self.data_source = self.findings.source_path
        self.df = self._load_dataframe()

    def _resolve_recommendations_path(self) -> str:
        if self.namespace and self.dataset_name is None:
            return str(self.namespace.merged_recommendations_path)
        return self.findings_path.replace("_findings.yaml", "_recommendations.yaml")

    def _load_or_create_registry(self) -> RecommendationRegistry:
        path = Path(self.recommendations_path)
        if path.exists():
            with path.open("r") as f:
                return RecommendationRegistry.from_dict(yaml.safe_load(f))
        reg = RecommendationRegistry()
        reg.init_bronze(self.findings.source_path)
        entity_col = (self.findings.identifier_columns[0]
                      if self.findings.identifier_columns else "entity_id")
        reg.init_silver(entity_col)
        reg.init_gold(self.target or "target")
        return reg

    def _load_dataframe(self) -> pd.DataFrame:
        if self.dataset_name is None and self.namespace:
            self.data_source = "silver_merged"
            df = get_delta(force_local=True).read(str(self.namespace.silver_merged_path))
            return df.to_pandas() if hasattr(df, "to_pandas") else df
        if "_aggregated" in self.findings_path:
            return self._load_aggregated()
        if self.namespace and self.dataset_name:
            self.data_source = self.dataset_name
            return load_active_dataset(self.namespace, self.dataset_name)
        return pd.DataFrame()

    def _load_aggregated(self) -> pd.DataFrame:
        source_path = Path(self.findings.source_path)
        if not source_path.is_absolute():
            source_path = Path("..") / source_path
        self.data_source = f"aggregated:{source_path.name}"
        if source_path.is_dir():
            df = get_delta(force_local=True).read(str(source_path))
            return df.to_pandas() if hasattr(df, "to_pandas") else df
        if source_path.is_file():
            return pd.read_parquet(source_path)
        if self.namespace and self.dataset_name:
            return load_silver_merged(self.namespace, self.dataset_name, DatasetGranularity.EVENT_LEVEL)
        return pd.DataFrame()

    @property
    def builder(self) -> RecommendationBuilder:
        if self._builder is None:
            self._builder = RecommendationBuilder(self.findings, self._notebook_name)
            self._builder.registry = self.registry
        return self._builder

    def run(self, step: Any) -> None:
        step.analyze(self)

    def save(self) -> None:
        self.findings.save(self.findings_path)
        self.registry.save(self.recommendations_path)

    def __enter__(self) -> AnalysisContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.save()
