import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _to_native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


NUMERIC_AGGREGATIONS = ("sum", "mean", "max", "min", "count", "std", "median", "first", "last")
CATEGORICAL_AGGREGATIONS = ("mode", "nunique", "mode_ratio", "entropy", "value_counts")
ALL_AGGREGATIONS = NUMERIC_AGGREGATIONS + CATEGORICAL_AGGREGATIONS


@dataclass
class LayeredRecommendation:
    id: str
    layer: str
    category: str
    action: str
    target_column: str
    parameters: Dict[str, Any]
    rationale: str
    source_notebook: str
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    fit_artifact_id: Optional[str] = None


@dataclass
class BronzeRecommendations:
    source_file: str
    null_handling: List[LayeredRecommendation] = field(default_factory=list)
    outlier_handling: List[LayeredRecommendation] = field(default_factory=list)
    type_conversions: List[LayeredRecommendation] = field(default_factory=list)
    deduplication: List[LayeredRecommendation] = field(default_factory=list)
    filtering: List[LayeredRecommendation] = field(default_factory=list)
    text_processing: List[LayeredRecommendation] = field(default_factory=list)
    modeling_strategy: List[LayeredRecommendation] = field(default_factory=list)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return (self.null_handling + self.outlier_handling + self.type_conversions +
                self.deduplication + self.filtering + self.text_processing + self.modeling_strategy)


@dataclass
class SilverRecommendations:
    entity_column: str
    time_column: Optional[str] = None
    joins: List[LayeredRecommendation] = field(default_factory=list)
    aggregations: List[LayeredRecommendation] = field(default_factory=list)
    derived_columns: List[LayeredRecommendation] = field(default_factory=list)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return self.joins + self.aggregations + self.derived_columns


@dataclass
class GoldRecommendations:
    target_column: str
    encoding: List[LayeredRecommendation] = field(default_factory=list)
    scaling: List[LayeredRecommendation] = field(default_factory=list)
    feature_selection: List[LayeredRecommendation] = field(default_factory=list)
    transformations: List[LayeredRecommendation] = field(default_factory=list)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return self.encoding + self.scaling + self.feature_selection + self.transformations


class RecommendationRegistry:
    def __init__(self):
        self.sources: Dict[str, BronzeRecommendations] = {}
        self.bronze: Optional[BronzeRecommendations] = None
        self.silver: Optional[SilverRecommendations] = None
        self.gold: Optional[GoldRecommendations] = None
        self.fit_artifacts: Dict[str, str] = {}
        self._id_counter = 0

    def link_fit_artifact(self, recommendation_id: str, artifact_id: str) -> None:
        self.fit_artifacts[recommendation_id] = artifact_id

    def get_fit_artifact(self, recommendation_id: str) -> Optional[str]:
        return self.fit_artifacts.get(recommendation_id)

    @property
    def source_names(self) -> List[str]:
        return list(self.sources.keys())

    def add_source(self, name: str, source_file: str) -> None:
        self.sources[name] = BronzeRecommendations(source_file=source_file)

    def get_source_recommendations(self, name: str) -> List[LayeredRecommendation]:
        if name in self.sources:
            return self.sources[name].all_recommendations
        return []

    def init_bronze(self, source_file: str) -> None:
        self.bronze = BronzeRecommendations(source_file=source_file)

    def init_silver(self, entity_column: str, time_column: Optional[str] = None) -> None:
        self.silver = SilverRecommendations(entity_column=entity_column, time_column=time_column)

    def init_gold(self, target_column: str) -> None:
        self.gold = GoldRecommendations(target_column=target_column)

    def add_bronze_null(self, column: str, strategy: str, rationale: str, source_notebook: str,
                        source: Optional[str] = None) -> None:
        rec = self._create_recommendation("bronze", "null", "impute", column,
                                          {"strategy": strategy}, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].null_handling.append(rec)
        elif self.bronze:
            self.bronze.null_handling.append(rec)

    def add_bronze_outlier(self, column: str, action: str, parameters: Dict, rationale: str,
                           source_notebook: str, source: Optional[str] = None) -> None:
        rec = self._create_recommendation("bronze", "outlier", action, column,
                                          parameters, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].outlier_handling.append(rec)
        elif self.bronze:
            self.bronze.outlier_handling.append(rec)

    def add_bronze_text_processing(self, column: str, embedding_model: str,
                                    variance_threshold: float, n_components: int,
                                    rationale: str, source_notebook: str,
                                    source: Optional[str] = None) -> None:
        params = {
            "embedding_model": embedding_model,
            "variance_threshold": variance_threshold,
            "n_components": n_components,
            "approach": "pca"
        }
        rec = self._create_recommendation("bronze", "text", "embed_reduce", column,
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].text_processing.append(rec)
        elif self.bronze:
            self.bronze.text_processing.append(rec)

    def add_bronze_filtering(self, column: str, condition: str, action: str, rationale: str,
                             source_notebook: str, source: Optional[str] = None) -> None:
        rec = self._create_recommendation("bronze", "filtering", action, column,
                                          {"condition": condition}, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].filtering.append(rec)
        elif self.bronze:
            self.bronze.filtering.append(rec)

    def add_bronze_modeling_strategy(self, strategy: str, column: str, parameters: Dict,
                                      rationale: str, source_notebook: str,
                                      source: Optional[str] = None) -> None:
        rec = self._create_recommendation("bronze", "modeling", strategy, column,
                                          parameters, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def add_bronze_deduplication(self, key_column: str, strategy: str, rationale: str,
                                  source_notebook: str, conflict_columns: Optional[List[str]] = None,
                                  source: Optional[str] = None) -> None:
        params = {"strategy": strategy}
        if conflict_columns:
            params["conflict_columns"] = conflict_columns
        rec = self._create_recommendation("bronze", "deduplication", strategy, key_column,
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].deduplication.append(rec)
        elif self.bronze:
            self.bronze.deduplication.append(rec)

    def add_bronze_consistency(self, column: str, issue_type: str, action: str,
                                variants: List[str], rationale: str, source_notebook: str,
                                source: Optional[str] = None) -> None:
        params = {"issue_type": issue_type, "variants": variants}
        rec = self._create_recommendation("bronze", "consistency", action, column,
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].type_conversions.append(rec)
        elif self.bronze:
            self.bronze.type_conversions.append(rec)

    def add_bronze_imbalance_strategy(self, target_column: str, imbalance_ratio: float,
                                       minority_class: Any, strategy: str, rationale: str,
                                       source_notebook: str, source: Optional[str] = None) -> None:
        params = {"imbalance_ratio": imbalance_ratio, "minority_class": minority_class}
        rec = self._create_recommendation("bronze", "imbalance", strategy, target_column,
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def add_silver_derived(self, column: str, expression: str, feature_type: str,
                           rationale: str, source_notebook: str) -> None:
        params = {"expression": expression, "feature_type": feature_type}
        rec = self._create_recommendation("silver", "derived", feature_type, column,
                                          params, rationale, source_notebook)
        self.silver.derived_columns.append(rec)

    def add_gold_transformation(self, column: str, transform: str, parameters: Dict,
                                 rationale: str, source_notebook: str) -> None:
        rec = self._create_recommendation("gold", "transformation", transform, column,
                                          parameters, rationale, source_notebook)
        self.gold.transformations.append(rec)

    def add_silver_aggregation(self, column: str, aggregation: str, windows: List[str],
                               rationale: str, source_notebook: str) -> None:
        params = {"aggregation": aggregation, "windows": windows}
        rec = self._create_recommendation("silver", "aggregation", aggregation, column,
                                          params, rationale, source_notebook)
        self.silver.aggregations.append(rec)

    def add_silver_join(self, left_source: str, right_source: str, join_keys: List[str],
                        join_type: str, rationale: str, source_notebook: str = "") -> None:
        params = {
            "left_source": left_source,
            "right_source": right_source,
            "join_keys": join_keys,
            "join_type": join_type
        }
        rec = self._create_recommendation("silver", "join", "join", "_merge",
                                          params, rationale, source_notebook)
        self.silver.joins.append(rec)

    def add_gold_encoding(self, column: str, method: str, rationale: str,
                          source_notebook: str) -> None:
        rec = self._create_recommendation("gold", "encoding", method, column,
                                          {"method": method}, rationale, source_notebook)
        self.gold.encoding.append(rec)

    def add_gold_scaling(self, column: str, method: str, rationale: str,
                         source_notebook: str) -> None:
        rec = self._create_recommendation("gold", "scaling", method, column,
                                          {"method": method}, rationale, source_notebook)
        self.gold.scaling.append(rec)

    def add_gold_drop_multicollinear(self, column: str, correlated_with: str, correlation: float,
                                      rationale: str, source_notebook: str) -> None:
        params = {"correlated_with": correlated_with, "correlation": correlation}
        rec = self._create_recommendation("gold", "feature_selection", "drop_multicollinear", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_weak(self, column: str, effect_size: float, correlation: float,
                           rationale: str, source_notebook: str) -> None:
        params = {"effect_size": effect_size, "correlation": correlation}
        rec = self._create_recommendation("gold", "feature_selection", "drop_weak", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_prioritize_feature(self, column: str, effect_size: float, correlation: float,
                                     rationale: str, source_notebook: str) -> None:
        params = {"effect_size": effect_size, "correlation": correlation}
        rec = self._create_recommendation("gold", "feature_selection", "prioritize", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_silver_ratio(self, column: str, numerator: str, denominator: str,
                         rationale: str, source_notebook: str) -> None:
        params = {"feature_type": "ratio", "numerator": numerator, "denominator": denominator,
                  "expression": f"{numerator} / {denominator}"}
        rec = self._create_recommendation("silver", "derived", "ratio", column,
                                          params, rationale, source_notebook)
        self.silver.derived_columns.append(rec)

    def add_silver_interaction(self, column: str, features: List[str],
                                rationale: str, source_notebook: str) -> None:
        params = {"feature_type": "interaction", "features": features,
                  "expression": " * ".join(features)}
        rec = self._create_recommendation("silver", "derived", "interaction", column,
                                          params, rationale, source_notebook)
        self.silver.derived_columns.append(rec)

    def add_silver_temporal_config(self, source_dataset: str, columns: List[str],
                                    lag_windows: int, lag_window_days: int,
                                    aggregations: List[str], feature_groups: List[str],
                                    rationale: str, source_notebook: str) -> None:
        params = {
            "columns": columns, "lag_windows": lag_windows, "lag_window_days": lag_window_days,
            "aggregations": aggregations, "feature_groups": feature_groups
        }
        rec = self._create_recommendation("silver", "temporal", "temporal_aggregation", source_dataset,
                                          params, rationale, source_notebook)
        self.silver.aggregations.append(rec)

    def add_bronze_segmentation_strategy(self, strategy: str, confidence: float, n_segments: int,
                                          silhouette_score: float, rationale: str,
                                          source_notebook: str, source: Optional[str] = None) -> None:
        params = {"confidence": confidence, "n_segments": n_segments, "silhouette_score": silhouette_score}
        rec = self._create_recommendation("bronze", "segmentation", strategy, "target",
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def add_bronze_feature_capacity(self, epv: float, capacity_status: str, recommended_features: int,
                                     current_features: int, rationale: str,
                                     source_notebook: str, source: Optional[str] = None) -> None:
        params = {"epv": epv, "capacity_status": capacity_status,
                  "recommended_features": recommended_features, "current_features": current_features}
        rec = self._create_recommendation("bronze", "capacity", "feature_capacity", "features",
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def add_bronze_model_type(self, model_type: str, max_features_linear: int,
                               max_features_regularized: int, max_features_tree: int,
                               rationale: str, source_notebook: str,
                               source: Optional[str] = None) -> None:
        params = {"max_features_linear": max_features_linear,
                  "max_features_regularized": max_features_regularized,
                  "max_features_tree": max_features_tree}
        rec = self._create_recommendation("bronze", "model_selection", model_type, "model",
                                          params, rationale, source_notebook)
        if source and source in self.sources:
            self.sources[source].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        recs = []
        for source_bronze in self.sources.values():
            recs.extend(source_bronze.all_recommendations)
        if self.bronze:
            recs.extend(self.bronze.all_recommendations)
        if self.silver:
            recs.extend(self.silver.all_recommendations)
        if self.gold:
            recs.extend(self.gold.all_recommendations)
        return recs

    def get_by_layer(self, layer: str) -> List[LayeredRecommendation]:
        if layer == "bronze":
            recs = []
            for source_bronze in self.sources.values():
                recs.extend(source_bronze.all_recommendations)
            if self.bronze:
                recs.extend(self.bronze.all_recommendations)
            return recs
        if layer == "silver" and self.silver:
            return self.silver.all_recommendations
        if layer == "gold" and self.gold:
            return self.gold.all_recommendations
        return []

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.sources:
            result["sources"] = {name: self._layer_to_dict(bronze)
                                 for name, bronze in self.sources.items()}
        if self.bronze:
            result["bronze"] = self._layer_to_dict(self.bronze)
        if self.silver:
            result["silver"] = self._layer_to_dict(self.silver)
        if self.gold:
            result["gold"] = self._layer_to_dict(self.gold)
        if self.fit_artifacts:
            result["fit_artifacts"] = self.fit_artifacts.copy()
        return result

    def compute_recommendations_hash(self, length: int = 8) -> str:
        hashable_data = self._build_hashable_gold_data()
        serialized = json.dumps(hashable_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).hexdigest()[:length]

    def _build_hashable_gold_data(self) -> Dict[str, Any]:
        if not self.gold:
            return {}
        return {
            "transformations": self._recs_to_hashable(self.gold.transformations),
            "encoding": self._recs_to_hashable(self.gold.encoding),
            "scaling": self._recs_to_hashable(self.gold.scaling),
            "feature_selection": self._recs_to_hashable(self.gold.feature_selection),
        }

    def _recs_to_hashable(self, recs: List[LayeredRecommendation]) -> List[Dict]:
        return sorted(
            [{"column": r.target_column, "action": r.action, "params": r.parameters} for r in recs],
            key=lambda x: (x["column"], x["action"])
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationRegistry":
        registry = cls()
        if "sources" in data:
            for name, bronze_data in data["sources"].items():
                registry.sources[name] = cls._bronze_from_dict(bronze_data)
        if "bronze" in data:
            registry.bronze = cls._bronze_from_dict(data["bronze"])
        if "silver" in data:
            registry.silver = cls._silver_from_dict(data["silver"])
        if "gold" in data:
            registry.gold = cls._gold_from_dict(data["gold"])
        if "fit_artifacts" in data:
            registry.fit_artifacts = data["fit_artifacts"].copy()
        return registry

    def _create_recommendation(self, layer: str, category: str, action: str, column: str,
                               parameters: Dict, rationale: str,
                               source_notebook: str) -> LayeredRecommendation:
        self._id_counter += 1
        rec_id = f"{layer}_{category}_{column}"
        return LayeredRecommendation(
            id=rec_id, layer=layer, category=category, action=action,
            target_column=column, parameters=_to_native(parameters),
            rationale=rationale, source_notebook=source_notebook
        )

    def _layer_to_dict(self, layer_obj) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(layer_obj).items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                result[key] = value
            elif isinstance(value, list):
                result[key] = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in value]
            else:
                result[key] = value
        return result

    @classmethod
    def _bronze_from_dict(cls, data: Dict) -> BronzeRecommendations:
        return BronzeRecommendations(
            source_file=data["source_file"],
            null_handling=[cls._rec_from_dict(r) for r in data.get("null_handling", [])],
            outlier_handling=[cls._rec_from_dict(r) for r in data.get("outlier_handling", [])],
            type_conversions=[cls._rec_from_dict(r) for r in data.get("type_conversions", [])],
            deduplication=[cls._rec_from_dict(r) for r in data.get("deduplication", [])],
            filtering=[cls._rec_from_dict(r) for r in data.get("filtering", [])],
            text_processing=[cls._rec_from_dict(r) for r in data.get("text_processing", [])],
            modeling_strategy=[cls._rec_from_dict(r) for r in data.get("modeling_strategy", [])]
        )

    @classmethod
    def _silver_from_dict(cls, data: Dict) -> SilverRecommendations:
        return SilverRecommendations(
            entity_column=data["entity_column"],
            time_column=data.get("time_column"),
            joins=[cls._rec_from_dict(r) for r in data.get("joins", [])],
            aggregations=[cls._rec_from_dict(r) for r in data.get("aggregations", [])],
            derived_columns=[cls._rec_from_dict(r) for r in data.get("derived_columns", [])]
        )

    @classmethod
    def _gold_from_dict(cls, data: Dict) -> GoldRecommendations:
        return GoldRecommendations(
            target_column=data["target_column"],
            encoding=[cls._rec_from_dict(r) for r in data.get("encoding", [])],
            scaling=[cls._rec_from_dict(r) for r in data.get("scaling", [])],
            feature_selection=[cls._rec_from_dict(r) for r in data.get("feature_selection", [])],
            transformations=[cls._rec_from_dict(r) for r in data.get("transformations", [])]
        )

    @classmethod
    def _rec_from_dict(cls, data: Dict) -> LayeredRecommendation:
        return LayeredRecommendation(**data)
