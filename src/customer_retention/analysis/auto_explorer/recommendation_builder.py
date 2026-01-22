from typing import TYPE_CHECKING, Any, Dict, List

from .layered_recommendations import LayeredRecommendation, RecommendationRegistry

if TYPE_CHECKING:
    from .findings import ExplorationFindings


class BronzeBuilder:
    def __init__(self, parent: "RecommendationBuilder"):
        self._parent = parent
        self._registry = parent.registry
        self._notebook = parent.notebook
        if self._registry.bronze is None:
            self._registry.init_bronze(parent.findings.source_path)

    def impute_nulls(self, column: str, strategy: str, reason: str) -> "BronzeBuilder":
        self._registry.add_bronze_null(column, strategy, reason, self._notebook)
        return self

    def cap_outliers(self, column: str, method: str, reason: str = "", **kwargs) -> "BronzeBuilder":
        params = {"method": method, **kwargs}
        self._registry.add_bronze_outlier(column, "cap", params, reason, self._notebook)
        return self

    def drop_column(self, column: str, reason: str) -> "BronzeBuilder":
        rec = LayeredRecommendation(
            id=f"bronze_drop_{column}", layer="bronze", category="filtering",
            action="drop", target_column=column, parameters={},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.bronze.filtering.append(rec)
        return self

    def convert_type(self, column: str, target_type: str, reason: str) -> "BronzeBuilder":
        rec = LayeredRecommendation(
            id=f"bronze_type_{column}", layer="bronze", category="type",
            action="cast", target_column=column, parameters={"target_type": target_type},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.bronze.type_conversions.append(rec)
        return self


class SilverBuilder:
    def __init__(self, parent: "RecommendationBuilder"):
        self._parent = parent
        self._registry = parent.registry
        self._notebook = parent.notebook
        if self._registry.silver is None:
            entity_col = (parent.findings.identifier_columns[0]
                          if parent.findings.identifier_columns else "id")
            time_col = (parent.findings.datetime_columns[0]
                        if parent.findings.datetime_columns else None)
            self._registry.init_silver(entity_col, time_col)

    def aggregate(self, column: str, aggregation: str, windows: List[str], reason: str) -> "SilverBuilder":
        self._registry.add_silver_aggregation(column, aggregation, windows, reason, self._notebook)
        return self

    def join(self, dataset: str, join_key: str, join_type: str, reason: str) -> "SilverBuilder":
        rec = LayeredRecommendation(
            id=f"silver_join_{dataset}", layer="silver", category="join",
            action="join", target_column=join_key,
            parameters={"dataset": dataset, "join_type": join_type},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.silver.joins.append(rec)
        return self

    def derive(self, column_name: str, formula: str, reason: str) -> "SilverBuilder":
        rec = LayeredRecommendation(
            id=f"silver_derive_{column_name}", layer="silver", category="derived",
            action="compute", target_column=column_name, parameters={"formula": formula},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.silver.derived_columns.append(rec)
        return self


class GoldBuilder:
    def __init__(self, parent: "RecommendationBuilder"):
        self._parent = parent
        self._registry = parent.registry
        self._notebook = parent.notebook
        if self._registry.gold is None:
            target = parent.findings.target_column or "target"
            self._registry.init_gold(target)

    def encode(self, column: str, method: str, reason: str, **kwargs) -> "GoldBuilder":
        rec = LayeredRecommendation(
            id=f"gold_encode_{column}", layer="gold", category="encoding",
            action=method, target_column=column, parameters={"method": method, **kwargs},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.gold.encoding.append(rec)
        return self

    def scale(self, column: str, method: str, reason: str) -> "GoldBuilder":
        rec = LayeredRecommendation(
            id=f"gold_scale_{column}", layer="gold", category="scaling",
            action=method, target_column=column, parameters={"method": method},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.gold.scaling.append(rec)
        return self

    def select(self, column: str, include: bool, reason: str) -> "GoldBuilder":
        action = "include" if include else "exclude"
        rec = LayeredRecommendation(
            id=f"gold_select_{column}", layer="gold", category="selection",
            action=action, target_column=column, parameters={"include": include},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.gold.feature_selection.append(rec)
        return self

    def transform(self, column: str, method: str, reason: str) -> "GoldBuilder":
        rec = LayeredRecommendation(
            id=f"gold_transform_{column}", layer="gold", category="transformation",
            action=method, target_column=column, parameters={"method": method},
            rationale=reason, source_notebook=self._notebook
        )
        self._registry.gold.transformations.append(rec)
        return self


class RecommendationBuilder:
    def __init__(self, findings: "ExplorationFindings", notebook: str):
        self.findings = findings
        self.notebook = notebook
        self.registry = RecommendationRegistry()

    def bronze(self) -> BronzeBuilder:
        return BronzeBuilder(self)

    def silver(self) -> SilverBuilder:
        return SilverBuilder(self)

    def gold(self) -> GoldBuilder:
        return GoldBuilder(self)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return self.registry.all_recommendations

    def to_dict(self) -> Dict[str, Any]:
        return self.registry.to_dict()
