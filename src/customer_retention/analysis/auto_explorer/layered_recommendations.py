import decimal
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from customer_retention.runtime.flags import is_user_extensions_disabled

_logger = logging.getLogger(__name__)


def _to_native(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    mod = type(value).__module__
    if mod.startswith('pandas'):
        return str(value)
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
class FeatureSelectionConfig:
    variance_threshold: float
    correlation_threshold: float
    analyzed_features: List[str]


@dataclass
class GoldRecommendations:
    target_column: str
    encoding: List[LayeredRecommendation] = field(default_factory=list)
    scaling: List[LayeredRecommendation] = field(default_factory=list)
    feature_selection: List[LayeredRecommendation] = field(default_factory=list)
    transformations: List[LayeredRecommendation] = field(default_factory=list)
    feature_selection_config: Optional[FeatureSelectionConfig] = None
    # FW-12: ``{value: 0|1}`` mapping that silver merge applies during the
    # target-column write so gold sees a numeric column. Set via
    # ``RecommendationRegistry.set_target_label_map``. Empty dict (default)
    # means no mapping — target is expected to be numeric already.
    target_label_map: Dict[Any, int] = field(default_factory=dict)
    target_label_map_rationale: str = ""

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return self.encoding + self.scaling + self.feature_selection + self.transformations


@dataclass
class LandingRecommendations:
    filters: List[LayeredRecommendation] = field(default_factory=list)
    lifecycle_enrichments: List[LayeredRecommendation] = field(default_factory=list)
    drop_columns: List[LayeredRecommendation] = field(default_factory=list)

    @property
    def all_recommendations(self) -> List[LayeredRecommendation]:
        return self.filters + self.lifecycle_enrichments + self.drop_columns


_BRONZE_AGG_OVERRIDE_KEYS = frozenset(
    {"per_grid_date_mode", "value_counts_columns", "windows", "categorical_value_counts"}
)


class RecommendationRegistry:
    def __init__(self, disable_user_extensions: Optional[bool] = None):
        self.sources: Dict[str, BronzeRecommendations] = {}
        self.bronze: Optional[BronzeRecommendations] = None
        self.silver: Optional[SilverRecommendations] = None
        self.gold: Optional[GoldRecommendations] = None
        self.landing: Optional[LandingRecommendations] = None
        self.fit_artifacts: Dict[str, str] = {}
        self.bronze_aggregations: Dict[str, Dict[str, Any]] = {}
        self._id_counter = 0
        self._ext_disabled: bool = is_user_extensions_disabled(disable_user_extensions)
        self._discarded_landing: List[LayeredRecommendation] = []

    @property
    def user_extensions_disabled(self) -> bool:
        return self._ext_disabled

    def save(self, path) -> None:
        p = path if isinstance(path, Path) else Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path) -> "RecommendationRegistry":
        p = path if isinstance(path, Path) else Path(str(path))
        with p.open("r") as f:
            return cls.from_dict(yaml.safe_load(f))

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

    def set_target_label_map(
        self,
        target_column: str,
        mapping: Dict[Any, int],
        rationale: str = "",
        source_notebook: str = "",
    ) -> None:
        """Register a ``{value: 0|1}`` mapping for the target column (FW-12).

        Silver merge applies the mapping during the target write so gold ships
        with a numeric column. NB05's ``load_findings`` cell reads the mapping
        from the registry first, falls back to the in-cell ``TARGET_LABEL_MAP``
        only when no mapping is registered.

        Validates that:

        * ``target_column`` matches the gold target (or initializes gold).
        * Every mapping value is 0 or 1 (binary classification only).
        * The mapping is non-empty.

        ``rationale`` and ``source_notebook`` are recorded on the gold
        recommendations object for the audit trail; they appear in the
        rendered recommendations.yaml so reviewers can trace why the
        mapping was chosen.
        """
        if not mapping:
            raise ValueError(
                "set_target_label_map: mapping must be non-empty. "
                "Pass at least two entries (one for each binary class)."
            )
        bad_labels = sorted({v for v in mapping.values() if v not in (0, 1)})
        if bad_labels:
            raise ValueError(
                f"set_target_label_map: mapping values must be 0 or 1; "
                f"got: {bad_labels!r}. Multi-class classification is not "
                f"supported by the framework — pick a binary collapse "
                f"(e.g. churn-positive vs all-others)."
            )
        if self.gold is None:
            self.init_gold(target_column)
        elif self.gold.target_column != target_column:
            raise ValueError(
                f"set_target_label_map: target_column mismatch. Registry "
                f"already initialized with gold.target_column="
                f"{self.gold.target_column!r}; got {target_column!r}. "
                f"Re-init via init_gold(target_column) first if you mean "
                f"to switch targets."
            )
        self.gold.target_label_map = dict(mapping)
        self.gold.target_label_map_rationale = (
            rationale if rationale else f"set via {source_notebook or 'unknown notebook'}"
        )

    def get_target_label_map(self) -> Dict[Any, int]:
        """Return the registered ``{value: 0|1}`` map or empty dict."""
        if self.gold is None:
            return {}
        return dict(self.gold.target_label_map or {})

    def init_landing(self) -> None:
        self.landing = LandingRecommendations()

    def add_landing_filter(self, dataset: str, predicate: str, rationale: str,
                           source_notebook: str) -> None:
        params = {"dataset": dataset, "predicate": predicate}
        rec = self._create_recommendation("landing", "landing_filtering", "filter", dataset,
                                          params, rationale, source_notebook)
        if self._ext_disabled:
            self._discarded_landing.append(rec)
            _logger.info(
                "user-extensions disabled; landing filter for dataset=%s dropped", dataset
            )
            return
        if self.landing is None:
            self.init_landing()
        self.landing.filters.append(rec)

    def add_landing_lifecycle_enrichment(self, dataset: str, config: Any, rationale: str,
                                         source_notebook: str) -> None:
        serialized = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        params = {"dataset": dataset, "config": serialized}
        rec = self._create_recommendation("landing", "lifecycle_enrichment", "enrich", dataset,
                                          params, rationale, source_notebook)
        if self._ext_disabled:
            self._discarded_landing.append(rec)
            _logger.info(
                "user-extensions disabled; landing lifecycle enrichment for dataset=%s dropped",
                dataset,
            )
            return
        if self.landing is None:
            self.init_landing()
        self.landing.lifecycle_enrichments.append(rec)

    def add_landing_drop_columns(
        self,
        dataset: str,
        columns,
        rationale: str,
        source_notebook: str,
    ) -> None:
        """Register case-EXACT column drops at landing time (PA-3).

        Mirrors ``add_landing_filter`` / ``add_landing_lifecycle_enrichment``
        — emits a ``LandingRecommendations.drop_columns`` entry that
        ``FindingsParser._apply_landing_recommendations`` merges into
        ``config.landing[<dataset>].drop_columns``. The generator's
        landing template already supports per-column case-EXACT drops
        (``df.select(*[c for c in df.columns if c not in _drop_set])``);
        this API lets NB02's case-collision detector emit a declarative
        recommendation instead of forcing the operator to paste
        ``LANDING_DROP_COLUMNS_OVERRIDES`` into NB10 user_code.

        Idempotent on duplicate calls — only the first add for
        ``(dataset, column)`` is kept; later calls are a no-op so a
        re-running detector cell does not multiply the registry.
        """
        cols = list(dict.fromkeys(str(c) for c in columns if c))
        if not cols:
            raise ValueError("add_landing_drop_columns requires at least one column")
        if self._ext_disabled:
            params = {"dataset": dataset, "columns": cols}
            rec = self._create_recommendation(
                "landing", "drop_columns", "drop", dataset, params,
                rationale, source_notebook,
            )
            self._discarded_landing.append(rec)
            _logger.info(
                "user-extensions disabled; landing drop_columns for dataset=%s dropped",
                dataset,
            )
            return
        if self.landing is None:
            self.init_landing()
        existing = self._existing_landing_drop_columns(dataset)
        new_cols = [c for c in cols if c not in existing]
        if not new_cols:
            return
        params = {"dataset": dataset, "columns": new_cols}
        rec = self._create_recommendation(
            "landing", "drop_columns", "drop", dataset, params,
            rationale, source_notebook,
        )
        self.landing.drop_columns.append(rec)

    def _existing_landing_drop_columns(self, dataset: str) -> set:
        if self.landing is None:
            return set()
        out: set = set()
        for rec in self.landing.drop_columns:
            if rec.parameters.get("dataset") != dataset:
                continue
            for c in rec.parameters.get("columns") or ():
                out.add(c)
        return out

    def add_bronze_value_counts(
        self,
        dataset: str,
        columns,
        windows=None,
        rationale: str = "",
        source_notebook: str = "",
        per_grid_date_mode: bool = True,
    ) -> None:
        cols = tuple(columns)
        if not cols:
            raise ValueError("add_bronze_value_counts requires at least one column")
        entry: Dict[str, Any] = self.bronze_aggregations.setdefault(dataset, {})
        existing_cols = tuple(entry.get("value_counts_columns") or ())
        entry["value_counts_columns"] = tuple(dict.fromkeys(existing_cols + cols))
        entry["per_grid_date_mode"] = bool(per_grid_date_mode)
        if windows is not None:
            entry["windows"] = tuple(windows)
        rec = self._create_recommendation(
            "bronze", "aggregation", "value_counts", dataset,
            {"columns": list(entry["value_counts_columns"]),
             "windows": list(entry.get("windows") or ()),
             "per_grid_date_mode": entry["per_grid_date_mode"]},
            rationale, source_notebook,
        )
        if dataset in self.sources:
            self.sources[dataset].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def add_bronze_aggregation_override(
        self,
        dataset: str,
        overrides: Dict[str, Any],
        rationale: str = "",
        source_notebook: str = "",
    ) -> None:
        unknown = set(overrides) - _BRONZE_AGG_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"unknown bronze aggregation override key(s) for '{dataset}': {sorted(unknown)}. "
                f"Recognized: {sorted(_BRONZE_AGG_OVERRIDE_KEYS)}"
            )
        entry: Dict[str, Any] = self.bronze_aggregations.setdefault(dataset, {})
        for k, v in overrides.items():
            if k == "value_counts_columns":
                entry[k] = tuple(v)
            elif k == "windows":
                entry[k] = tuple(v)
            else:
                entry[k] = v
        rec = self._create_recommendation(
            "bronze", "aggregation", "override", dataset,
            {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in entry.items()},
            rationale, source_notebook,
        )
        if dataset in self.sources:
            self.sources[dataset].modeling_strategy.append(rec)
        elif self.bronze:
            self.bronze.modeling_strategy.append(rec)

    def merge_bronze_aggregation_overrides(
        self, nb10_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for name, cfg in (self.bronze_aggregations or {}).items():
            merged[name] = {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
            }
        for name, cfg in (nb10_overrides or {}).items():
            bucket = merged.setdefault(name, {})
            for k, v in cfg.items():
                bucket[k] = list(v) if isinstance(v, tuple) else v
        return merged

    def add_bronze_null(self, column: str, strategy: str, rationale: str, source_notebook: str,
                        source: Optional[str] = None) -> None:
        # `action` mirrors the strategy intent ("drop" → action=drop, otherwise "impute")
        # so yaml inspection matches the parser's `_map_bronze_null` decision (which
        # branches on `parameters.strategy`). Without this, the yaml `action: impute`
        # silently masks columns that will be DROPped at codegen time.
        action = "drop" if strategy == "drop" else "impute"
        rec = self._create_recommendation("bronze", "null", action, column,
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
                           rationale: str, source_notebook: str,
                           source_columns: Optional[List[str]] = None) -> None:
        params: Dict[str, object] = {"expression": expression, "feature_type": feature_type}
        if source_columns:
            params["source_columns"] = list(source_columns)
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

    def set_feature_selection_config(self, variance_threshold: float,
                                      correlation_threshold: float,
                                      analyzed_features: List[str]) -> None:
        if self.gold:
            self.gold.feature_selection_config = FeatureSelectionConfig(
                variance_threshold=variance_threshold,
                correlation_threshold=correlation_threshold,
                analyzed_features=sorted(analyzed_features),
            )

    def add_gold_drop_multicollinear(self, column: str, correlated_with: str, correlation: float,
                                      rationale: str, source_notebook: str,
                                      *,
                                      slice_date: Optional[str] = None,
                                      slice_strategy: Optional[str] = None,
                                      slice_row_count: Optional[int] = None) -> None:
        params: Dict[str, Any] = {"correlated_with": correlated_with, "correlation": correlation}
        if slice_date is not None:
            params["slice_date"] = slice_date
        if slice_strategy is not None:
            params["slice_strategy"] = slice_strategy
        if slice_row_count is not None:
            params["slice_row_count"] = int(slice_row_count)
        rec = self._create_recommendation("gold", "feature_selection", "drop_multicollinear", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_weak(self, column: str, effect_size: float, correlation: float,
                           rationale: str, source_notebook: str,
                           *,
                           slice_date: Optional[str] = None,
                           slice_strategy: Optional[str] = None,
                           slice_row_count: Optional[int] = None) -> None:
        params: Dict[str, Any] = {"effect_size": effect_size, "correlation": correlation}
        if slice_date is not None:
            params["slice_date"] = slice_date
        if slice_strategy is not None:
            params["slice_strategy"] = slice_strategy
        if slice_row_count is not None:
            params["slice_row_count"] = int(slice_row_count)
        rec = self._create_recommendation("gold", "feature_selection", "drop_weak", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_prioritize_feature(self, column: str, effect_size: float, correlation: float,
                                     rationale: str, source_notebook: str) -> None:
        params = {"effect_size": effect_size, "correlation": correlation}
        rec = self._create_recommendation("gold", "feature_selection", "prioritize", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_l1_zero(self, column: str, l1_coefficient: float,
                               rationale: str, source_notebook: str) -> None:
        params = {"l1_coefficient": l1_coefficient}
        rec = self._create_recommendation("gold", "feature_selection", "drop_l1_zero", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_availability(self, column: str, issue_type: str, coverage_pct: float,
                                    rationale: str, source_notebook: str) -> None:
        params = {"issue_type": issue_type, "coverage_pct": coverage_pct}
        rec = self._create_recommendation("gold", "feature_selection", "drop_availability", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_zero_variance(self, column: str, variance: float,
                                     rationale: str, source_notebook: str) -> None:
        params = {"variance": variance}
        rec = self._create_recommendation("gold", "feature_selection", "drop_zero_variance", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_chi_squared(self, column: str, rationale: str,
                                   source_notebook: str) -> None:
        rec = self._create_recommendation("gold", "feature_selection", "drop_chi_squared", column,
                                          {}, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_gbdt_importance(self, column: str, importance: float,
                                       rationale: str, source_notebook: str) -> None:
        params = {"importance": importance}
        rec = self._create_recommendation("gold", "feature_selection", "drop_gbdt_importance", column,
                                          params, rationale, source_notebook)
        self.gold.feature_selection.append(rec)

    def add_gold_drop_rescue_consensus(self, column: str, chi_rank: int, chi_score: float,
                                        l1_coefficient: Optional[float], gbdt_total_gain: Optional[float],
                                        slice_date: str, slice_strategy: str, slice_row_count: int,
                                        rationale: str, source_notebook: str,
                                        l1_considered: bool = True, gbdt_considered: bool = True) -> None:
        params: Dict[str, Any] = {
            "chi_squared_rank": int(chi_rank),
            "chi_squared_score": float(chi_score),
            "l1_coefficient": float(l1_coefficient) if l1_coefficient is not None else 0.0,
            "l1_considered": bool(l1_considered),
            "gbdt_total_gain": float(gbdt_total_gain) if gbdt_total_gain is not None else 0.0,
            "gbdt_considered": bool(gbdt_considered),
            "slice_date": slice_date,
            "slice_strategy": slice_strategy,
            "slice_row_count": int(slice_row_count),
        }
        rec = self._create_recommendation("gold", "feature_selection", "drop_rescue_consensus", column,
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

    def add_silver_composite(self, column: str, columns: List[str],
                             rationale: str, source_notebook: str) -> None:
        params = {"feature_type": "composite", "columns": columns,
                  "expression": f"mean({', '.join(columns)})"}
        rec = self._create_recommendation("silver", "derived", "composite", column,
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
        if self.landing:
            recs.extend(self.landing.all_recommendations)
        if self.bronze:
            recs.extend(self.bronze.all_recommendations)
        if self.silver:
            recs.extend(self.silver.all_recommendations)
        if self.gold:
            recs.extend(self.gold.all_recommendations)
        return recs

    def get_by_layer(self, layer: str) -> List[LayeredRecommendation]:
        if layer == "landing" and self.landing:
            return self.landing.all_recommendations
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
        if self.landing:
            result["landing"] = self._layer_to_dict(self.landing)
        if self.bronze:
            result["bronze"] = self._layer_to_dict(self.bronze)
        if self.silver:
            result["silver"] = self._layer_to_dict(self.silver)
        if self.gold:
            result["gold"] = self._layer_to_dict(self.gold)
        if self.fit_artifacts:
            result["fit_artifacts"] = self.fit_artifacts.copy()
        if self.bronze_aggregations:
            result["bronze_aggregations"] = {
                name: {
                    k: (list(v) if isinstance(v, tuple) else _to_native(v))
                    for k, v in cfg.items()
                }
                for name, cfg in self.bronze_aggregations.items()
            }
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
            "target_label_map": sorted(
                ((str(k), int(v)) for k, v in (self.gold.target_label_map or {}).items()),
                key=lambda kv: kv[0],
            ),
        }

    def _recs_to_hashable(self, recs: List[LayeredRecommendation]) -> List[Dict]:
        return sorted(
            [{"column": r.target_column, "action": r.action, "params": r.parameters} for r in recs],
            key=lambda x: (x["column"], x["action"])
        )

    @classmethod
    def merge(cls, registries: List["RecommendationRegistry"]) -> "RecommendationRegistry":
        if not registries:
            return cls()
        result = cls()
        primary = next((r for r in registries if r.gold is not None), None)
        primary = primary or next((r for r in registries if r.silver), None)
        for reg in registries:
            result.sources.update(reg.sources)
            result.fit_artifacts.update(reg.fit_artifacts)
            for name, cfg in (reg.bronze_aggregations or {}).items():
                bucket = result.bronze_aggregations.setdefault(name, {})
                bucket.update(cfg)
        if primary:
            result.silver, result.gold, result.bronze = primary.silver, primary.gold, primary.bronze
            result.landing = primary.landing
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationRegistry":
        registry = cls()
        if "sources" in data:
            for name, bronze_data in data["sources"].items():
                registry.sources[name] = cls._bronze_from_dict(bronze_data)
        if "landing" in data:
            registry.landing = cls._landing_from_dict(data["landing"])
        if "bronze" in data:
            registry.bronze = cls._bronze_from_dict(data["bronze"])
        if "silver" in data:
            registry.silver = cls._silver_from_dict(data["silver"])
        if "gold" in data:
            registry.gold = cls._gold_from_dict(data["gold"])
        if "fit_artifacts" in data:
            registry.fit_artifacts = data["fit_artifacts"].copy()
        if "bronze_aggregations" in data:
            for name, cfg in (data["bronze_aggregations"] or {}).items():
                normalized = dict(cfg)
                if "value_counts_columns" in normalized:
                    normalized["value_counts_columns"] = tuple(normalized["value_counts_columns"])
                if "windows" in normalized:
                    normalized["windows"] = tuple(normalized["windows"])
                registry.bronze_aggregations[name] = normalized
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
                result[key] = [_to_native(item) for item in value]
            elif isinstance(value, list):
                result[key] = [_to_native(asdict(r)) if hasattr(r, '__dataclass_fields__') else _to_native(r) for r in value]
            else:
                result[key] = _to_native(value)
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
    def _landing_from_dict(cls, data: Dict) -> LandingRecommendations:
        return LandingRecommendations(
            filters=[cls._rec_from_dict(r) for r in data.get("filters", [])],
            lifecycle_enrichments=[cls._rec_from_dict(r) for r in data.get("lifecycle_enrichments", [])],
            drop_columns=[cls._rec_from_dict(r) for r in data.get("drop_columns", [])],
        )

    @classmethod
    def _gold_from_dict(cls, data: Dict) -> GoldRecommendations:
        return GoldRecommendations(
            target_column=data["target_column"],
            encoding=[cls._rec_from_dict(r) for r in data.get("encoding", [])],
            scaling=[cls._rec_from_dict(r) for r in data.get("scaling", [])],
            feature_selection=[cls._rec_from_dict(r) for r in data.get("feature_selection", [])],
            transformations=[cls._rec_from_dict(r) for r in data.get("transformations", [])],
            feature_selection_config=FeatureSelectionConfig(**data["feature_selection_config"]) if data.get("feature_selection_config") else None,
            target_label_map=dict(data.get("target_label_map") or {}),
            target_label_map_rationale=data.get("target_label_map_rationale", ""),
        )

    @classmethod
    def _rec_from_dict(cls, data: Dict) -> LayeredRecommendation:
        return LayeredRecommendation(**data)
