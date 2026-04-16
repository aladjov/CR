import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    _numeric_column_names,
    batched_corr_matrix,
    bulk_variance,
    isna,
)
from customer_retention.stages.modeling.feature_profile import SelectionTraceRecorder

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata


class SelectionMethod(Enum):
    VARIANCE = "VARIANCE"
    CORRELATION = "CORRELATION"
    MUTUAL_INFO = "MUTUAL_INFO"
    IMPORTANCE = "IMPORTANCE"
    RECURSIVE = "RECURSIVE"
    L1_SELECTION = "L1_SELECTION"
    CHI_SQUARED = "CHI_SQUARED"
    GBDT_IMPORTANCE = "GBDT_IMPORTANCE"


@dataclass
class FeatureSelectionResult:
    df: DataFrame
    selected_features: List[str]
    dropped_features: List[str]
    drop_reasons: Dict[str, str]
    method_used: SelectionMethod
    importance_scores: Optional[Dict[str, float]] = None


def split_reconsiderable_drops(result: FeatureSelectionResult) -> tuple[set, set]:
    """Partition drops into (structural, reconsiderable) for rescue ordering.

    When rescue runs after L1, the conventional NB08 flow strips X_train down
    to L1's keeps — leaving rescue with nothing to rescue. This helper lets the
    caller distinguish:

    - **Structural** drops: variance, correlation, max_features limit, and any
      unknown-reason drop. These are safe to apply unconditionally because they
      indicate the column itself isn't meaningful (constant, redundant, capped).
    - **Reconsiderable** drops: L1 zero-coefficient drops. These represent
      model-opinion decisions that rescue can overrule. Keeping them out of
      X_train pre-rescue means rescue can still evaluate and pick them.

    Returns ``(structural_set, reconsiderable_set)``. Conservative default:
    any drop whose reason doesn't contain ``"L1"`` (case-insensitive) is
    treated as structural.
    """
    structural: set = set()
    reconsiderable: set = set()
    for feature, reason in result.drop_reasons.items():
        if "l1" in str(reason).lower():
            reconsiderable.add(feature)
        else:
            structural.add(feature)
    return structural, reconsiderable


@dataclass
class AvailabilityRecommendation:
    column: str
    issue_type: str
    coverage_pct: float
    first_valid_date: Optional[str]
    last_valid_date: Optional[str]
    options: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "coverage_pct": self.coverage_pct,
            "first_valid_date": self.first_valid_date,
            "last_valid_date": self.last_valid_date,
            "options": self.options,
        }


class FeatureSelector:
    def __init__(self, method: SelectionMethod = SelectionMethod.VARIANCE, variance_threshold: float = 0.01, correlation_threshold: float = 0.95, target_column: Optional[str] = None, preserve_features: Optional[List[str]] = None, max_features: Optional[int] = None, apply_correlation_filter: bool = False, precomputed_corr_matrix: Optional[Any] = None, l1_C: float = 1.0, l1_ratio: float = 1.0, progress_fn: Optional[Callable[[str], None]] = None, precomputed_variances: Optional[Any] = None, precomputed_medians: Optional[Dict[str, float]] = None, precomputed_non_null: Optional[Dict[str, int]] = None, correlation_candidates: Optional[List[str]] = None, chi_squared_num_buckets: int = 10, gbdt_n_estimators: int = 200, gbdt_max_depth: int = 6, trace_recorder: Optional[SelectionTraceRecorder] = None):
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_column = target_column
        self.preserve_features = preserve_features or []
        self.max_features = max_features
        self.apply_correlation_filter = apply_correlation_filter
        self._precomputed_corr_matrix = precomputed_corr_matrix
        self.l1_C = l1_C
        self.l1_ratio = l1_ratio
        self._progress_fn = progress_fn
        self._precomputed_medians = precomputed_medians
        self._precomputed_non_null = precomputed_non_null
        self._correlation_candidates = correlation_candidates
        self.chi_squared_num_buckets = chi_squared_num_buckets
        self.gbdt_n_estimators = gbdt_n_estimators
        self.gbdt_max_depth = gbdt_max_depth
        self.trace_recorder = trace_recorder

        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.drop_reasons: Dict[str, str] = {}
        self.importance_scores: Optional[Dict[str, float]] = None
        self._is_fitted = False
        self._cached_variances: Optional[Any] = precomputed_variances

    def fit(self, df: DataFrame) -> "FeatureSelector":
        feature_cols = [c for c in df.columns if c != self.target_column]
        numeric_set = _numeric_column_names(df, feature_cols)

        self.selected_features = feature_cols.copy()
        self.dropped_features = []
        self.drop_reasons = {}

        if self.method == SelectionMethod.VARIANCE:
            self._apply_variance_selection(df, feature_cols, numeric_set)
        elif self.method == SelectionMethod.CORRELATION:
            self._apply_correlation_selection(df, feature_cols, numeric_set)
        elif self.method == SelectionMethod.L1_SELECTION:
            self._apply_l1_selection(df, feature_cols, numeric_set)
        elif self.method == SelectionMethod.CHI_SQUARED:
            self._apply_chi_squared_selection(df, feature_cols, numeric_set)
        elif self.method == SelectionMethod.GBDT_IMPORTANCE:
            self._apply_gbdt_importance_selection(df, feature_cols, numeric_set)

        if self.apply_correlation_filter and self.method != SelectionMethod.CORRELATION:
            self._apply_correlation_selection(df, self.selected_features.copy(), numeric_set)

        if self.max_features and len(self.selected_features) > self.max_features:
            variances = self._get_variances(df, self.selected_features, numeric_set)
            to_keep = set(variances.sort_values(ascending=False).head(self.max_features).index.tolist())
            for feature in [f for f in self.selected_features if f not in to_keep and f not in self.preserve_features]:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = "max_features limit"

        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> FeatureSelectionResult:
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        cols_to_keep = self.selected_features.copy()
        if self.target_column and self.target_column in df.columns:
            cols_to_keep.append(self.target_column)

        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        result_df = df[cols_to_keep].copy()

        return FeatureSelectionResult(
            df=result_df,
            selected_features=self.selected_features.copy(),
            dropped_features=self.dropped_features.copy(),
            drop_reasons=self.drop_reasons.copy(),
            method_used=self.method,
            importance_scores=dict(self.importance_scores) if self.importance_scores else None,
        )

    def fit_transform(self, df: DataFrame) -> FeatureSelectionResult:
        self.fit(df)
        return self.transform(df)

    def _get_variances(self, df: DataFrame, features: List[str], numeric_set: set) -> Any:
        numeric_features = [f for f in features if f in numeric_set]
        if not numeric_features:
            import pandas as _pd
            return _pd.Series(dtype=float)
        if self._cached_variances is not None:
            cached_cols = set(self._cached_variances.index)
            missing = [f for f in numeric_features if f not in cached_cols]
            if not missing:
                return self._cached_variances[numeric_features]
        self._cached_variances = bulk_variance(df, numeric_features)
        return self._cached_variances

    def _apply_variance_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        candidates = [f for f in features if f not in self.preserve_features and f in numeric_set]
        scores: Dict[str, float] = {}
        decisions: Dict[str, str] = {}
        reasons: Dict[str, Optional[str]] = {}
        preserve_in_features = [f for f in features if f in self.preserve_features]
        for f in preserve_in_features:
            scores[f] = float("nan")
            decisions[f] = "preserved"
            reasons[f] = "in_preserve_list"
        if not candidates:
            self._record_stage(
                stage="variance", score_name="variance", scores=scores,
                threshold=self.variance_threshold, decisions=decisions, reasons=reasons,
                input_count=len(features), output_count=len(self.selected_features),
            )
            return
        variances = self._get_variances(df, candidates, numeric_set)
        for feature in candidates:
            variance = variances[feature]
            scores[feature] = float("nan") if isna(variance) else float(variance)
            if isna(variance) or variance < self.variance_threshold:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = f"low variance ({variance:.6f})"
                decisions[feature] = "dropped"
                reasons[feature] = "low_variance"
            else:
                decisions[feature] = "kept"
                reasons[feature] = None
        self._record_stage(
            stage="variance", score_name="variance", scores=scores,
            threshold=self.variance_threshold, decisions=decisions, reasons=reasons,
            input_count=len(features), output_count=len(self.selected_features),
        )

    def _record_stage(
        self, *, stage: str, score_name: str, scores: Dict[str, float],
        threshold: Optional[float], decisions: Dict[str, str],
        reasons: Dict[str, Optional[str]], input_count: int, output_count: int,
        companion_map: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.record_stage(
            stage=stage, score_name=score_name, scores=scores,
            threshold=threshold, decisions=decisions, reasons=reasons,
            stage_input_count=input_count, stage_output_count=output_count,
            companion_map=companion_map,
        )

    def _apply_correlation_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        numeric_features = [f for f in features if f in numeric_set and f in self.selected_features]
        if len(numeric_features) < 2:
            return

        pre = self._precomputed_corr_matrix
        if pre is not None and set(numeric_features).issubset(pre.columns):
            corr_matrix = pre.loc[numeric_features, numeric_features].abs()
        else:
            corr_matrix = batched_corr_matrix(
                df, numeric_features, progress_fn=self._progress_fn,
                precomputed_medians=self._precomputed_medians,
                precomputed_non_null=self._precomputed_non_null,
                cross_columns=self._correlation_candidates,
            ).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        variances = self._get_variances(df, numeric_features, numeric_set)

        to_drop: set = set()
        companion_map: Dict[str, str] = {}
        max_pair_corr: Dict[str, float] = {}
        for column in upper.columns:
            correlated = upper.index[upper[column] > self.correlation_threshold].tolist()
            for corr_feature in correlated:
                pair_corr = float(upper.at[corr_feature, column])
                if corr_feature in self.preserve_features:
                    loser = column if column not in self.preserve_features else None
                    winner = corr_feature
                elif column in self.preserve_features:
                    loser = corr_feature
                    winner = column
                else:
                    if variances[column] >= variances[corr_feature]:
                        loser, winner = corr_feature, column
                    else:
                        loser, winner = column, corr_feature
                if loser is not None:
                    to_drop.add(loser)
                    if pair_corr > max_pair_corr.get(loser, 0.0):
                        max_pair_corr[loser] = pair_corr
                        companion_map[loser] = winner

        for feature in to_drop:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = f"high correlation (> {self.correlation_threshold})"

        scores: Dict[str, float] = {}
        decisions: Dict[str, str] = {}
        reasons: Dict[str, Optional[str]] = {}
        for feature in numeric_features:
            if feature in self.preserve_features:
                scores[feature] = float("nan")
                decisions[feature] = "preserved"
                reasons[feature] = "in_preserve_list"
            elif feature in to_drop:
                scores[feature] = max_pair_corr.get(feature, float("nan"))
                decisions[feature] = "dropped"
                reasons[feature] = "high_correlation"
            else:
                scores[feature] = float(upper[feature].max()) if feature in upper.columns else 0.0
                decisions[feature] = "kept"
                reasons[feature] = None
        self._record_stage(
            stage="correlation", score_name="abs_pearson_r", scores=scores,
            threshold=self.correlation_threshold, decisions=decisions, reasons=reasons,
            input_count=len(numeric_features), output_count=len(self.selected_features),
            companion_map=companion_map,
        )

    def _apply_l1_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for L1_SELECTION")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        numeric_features = [f for f in features if f not in self.preserve_features and f in numeric_set]
        if not numeric_features:
            return

        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            from customer_retention.core.compat import as_spark_df
            spark_df = as_spark_df(df[numeric_features + [self.target_column]])
            dropped, reasons, scores = _spark_l1_selection(spark_df, self.target_column, numeric_features, reg_param=1.0 / self.l1_C, elastic_net_param=self.l1_ratio)
            self.importance_scores = scores
            drop_threshold_val: Optional[float] = None
            for feature in dropped:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = reasons[feature]
            self._record_l1_trace(features, numeric_features, scores, set(dropped), drop_threshold_val)
            return

        work_df = df[numeric_features + [self.target_column]]
        work_df = work_df.dropna(subset=[self.target_column])
        if len(work_df) < 10:
            raise ValueError(
                f"L1_SELECTION requires at least 10 rows with non-null target; "
                f"got {len(work_df)} after dropping NaN targets"
            )

        X = work_df[numeric_features].fillna(0).to_numpy()
        y = work_df[self.target_column].to_numpy()

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X)
        model = LogisticRegression(solver="saga", C=self.l1_C, max_iter=2000, l1_ratio=self.l1_ratio)
        model.fit(X_scaled, y)

        coefs = np.abs(model.coef_)
        max_coefs = coefs.max(axis=0) if coefs.ndim == 2 else coefs

        self.importance_scores = {numeric_features[i]: float(max_coefs[i]) for i in range(len(numeric_features))}
        max_signal = float(max_coefs.max()) if len(max_coefs) else 0.0
        drop_threshold = max_signal * 0.01
        zero_features = [numeric_features[i] for i in range(len(numeric_features)) if max_coefs[i] <= drop_threshold]
        if len(zero_features) == len(numeric_features):
            self._record_l1_trace(features, numeric_features, self.importance_scores, set(), drop_threshold)
            return
        for feature in zero_features:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = "L1 zero coefficient"
        self._record_l1_trace(features, numeric_features, self.importance_scores, set(zero_features), drop_threshold)

    def _record_l1_trace(
        self, all_features: List[str], numeric_features: List[str],
        coef_scores: Dict[str, float], dropped: set,
        drop_threshold: Optional[float],
    ) -> None:
        if self.trace_recorder is None:
            return
        scores: Dict[str, float] = {}
        decisions: Dict[str, str] = {}
        reasons: Dict[str, Optional[str]] = {}
        for feature in all_features:
            if feature in self.preserve_features:
                scores[feature] = float("nan")
                decisions[feature] = "preserved"
                reasons[feature] = "in_preserve_list"
            elif feature in numeric_features:
                scores[feature] = float(coef_scores.get(feature, 0.0))
                if feature in dropped:
                    decisions[feature] = "dropped"
                    reasons[feature] = "below_threshold"
                else:
                    decisions[feature] = "kept"
                    reasons[feature] = None
            else:
                scores[feature] = float("nan")
                decisions[feature] = "not_evaluated"
                reasons[feature] = "not_numeric"
        self._record_stage(
            stage="l1", score_name="l1_abs_coef", scores=scores,
            threshold=drop_threshold, decisions=decisions, reasons=reasons,
            input_count=len(all_features), output_count=len(self.selected_features),
        )

    def _apply_chi_squared_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for CHI_SQUARED")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        candidates = [f for f in features if f not in self.preserve_features and f in numeric_set]
        if not candidates or (self.max_features and self.max_features >= len(candidates)):
            self._record_importance_trace(
                stage="chi_squared", score_name="chi2_stat",
                all_features=features, candidates=candidates, scores={}, dropped=set(),
                threshold=None,
            )
            return

        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            from customer_retention.core.compat import as_spark_df
            spark_df = as_spark_df(df[candidates + [self.target_column]])
            dropped, reasons, scores = _spark_chi_squared_selection(
                spark_df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                num_buckets=self.chi_squared_num_buckets,
                progress_fn=self._progress_fn,
            )
        else:
            dropped, reasons, scores = _local_chi_squared_selection(
                df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                num_buckets=self.chi_squared_num_buckets,
            )
        self.importance_scores = scores
        for feature in dropped:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = reasons[feature]
        self._record_importance_trace(
            stage="chi_squared", score_name="chi2_stat",
            all_features=features, candidates=candidates, scores=scores, dropped=set(dropped),
            threshold=None,
        )

    def _record_importance_trace(
        self, *, stage: str, score_name: str, all_features: List[str],
        candidates: List[str], scores: Dict[str, float], dropped: set,
        threshold: Optional[float],
    ) -> None:
        if self.trace_recorder is None:
            return
        out_scores: Dict[str, float] = {}
        decisions: Dict[str, str] = {}
        reasons: Dict[str, Optional[str]] = {}
        for feature in all_features:
            if feature in self.preserve_features:
                out_scores[feature] = float("nan")
                decisions[feature] = "preserved"
                reasons[feature] = "in_preserve_list"
            elif feature in candidates:
                out_scores[feature] = float(scores.get(feature, 0.0))
                if feature in dropped:
                    decisions[feature] = "dropped"
                    reasons[feature] = f"{stage}_drop"
                else:
                    decisions[feature] = "kept"
                    reasons[feature] = None
            else:
                out_scores[feature] = float("nan")
                decisions[feature] = "not_evaluated"
                reasons[feature] = "not_numeric"
        self._record_stage(
            stage=stage, score_name=score_name, scores=out_scores,
            threshold=threshold, decisions=decisions, reasons=reasons,
            input_count=len(all_features), output_count=len(self.selected_features),
        )

    def _apply_gbdt_importance_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for GBDT_IMPORTANCE")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        candidates = [f for f in features if f not in self.preserve_features and f in numeric_set]
        if not candidates or (self.max_features and self.max_features >= len(candidates)):
            self._record_importance_trace(
                stage="gbdt_importance", score_name="gbdt_gain",
                all_features=features, candidates=candidates, scores={}, dropped=set(),
                threshold=None,
            )
            return

        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            from customer_retention.core.compat import as_spark_df
            spark_df = as_spark_df(df[candidates + [self.target_column]])
            dropped, reasons, scores = _spark_gbdt_importance_selection(
                spark_df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                n_estimators=self.gbdt_n_estimators,
                max_depth=self.gbdt_max_depth,
                progress_fn=self._progress_fn,
            )
        else:
            dropped, reasons, scores = _local_gbdt_importance_selection(
                df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                n_estimators=self.gbdt_n_estimators,
                max_depth=self.gbdt_max_depth,
                progress_fn=self._progress_fn,
            )
        self.importance_scores = scores
        for feature in dropped:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = reasons[feature]
        self._record_importance_trace(
            stage="gbdt_importance", score_name="gbdt_gain",
            all_features=features, candidates=candidates, scores=scores, dropped=set(dropped),
            threshold=None,
        )

    def get_availability_recommendations(self, availability: Optional["FeatureAvailabilityMetadata"]) -> List[AvailabilityRecommendation]:
        if availability is None:
            return []
        recommendations: List[AvailabilityRecommendation] = []
        problem_columns = availability.new_tracking + availability.retired_tracking + availability.partial_window
        for col in problem_columns:
            feat_info = availability.features.get(col)
            if feat_info is None:
                continue
            recommendations.append(AvailabilityRecommendation(
                column=col,
                issue_type=feat_info.availability_type,
                coverage_pct=feat_info.coverage_pct,
                first_valid_date=feat_info.first_valid_date,
                last_valid_date=feat_info.last_valid_date,
                options=self._build_availability_options(col, feat_info.availability_type, feat_info.first_valid_date, feat_info.last_valid_date, feat_info.coverage_pct),
            ))
        return recommendations

    def _build_availability_options(self, col: str, issue_type: str, first_date: Optional[str], last_date: Optional[str], coverage_pct: float) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        options.append({
            "type": "remove",
            "description": f"Remove '{col}' from feature selection (recommended for most cases)",
            "preserves_data": False,
            "recommended": True,
        })
        options.append({
            "type": "add_indicator",
            "description": f"Create '{col}_available' indicator column to flag valid observations",
            "preserves_data": True,
        })
        if issue_type == "new_tracking":
            options.append({
                "type": "filter_window",
                "description": f"Filter training data to start from {first_date}",
                "preserves_data": True,
            })
            options.append({
                "type": "segment_by_cohort",
                "description": f"Train separate models: pre-{first_date} cohort (without feature) vs post-{first_date} cohort (with feature)",
                "preserves_data": True,
            })
        elif issue_type == "retired":
            options.append({
                "type": "filter_window",
                "description": f"Filter test/scoring data to end at {last_date}",
                "preserves_data": True,
            })
            options.append({
                "type": "segment_by_cohort",
                "description": "Use feature only for historical scoring; train fallback model without it for future predictions",
                "preserves_data": True,
            })
        elif issue_type == "partial_window":
            options.append({
                "type": "filter_window",
                "description": f"Use data only within {first_date} to {last_date}",
                "preserves_data": True,
            })
            options.append({
                "type": "segment_by_availability",
                "description": "Train separate models: one using this feature (within window), one without (outside window)",
                "preserves_data": True,
            })
        if coverage_pct >= 30:
            options.append({
                "type": "impute",
                "description": f"Impute missing values (median/mode) - {coverage_pct:.0f}% coverage may be sufficient",
                "preserves_data": True,
            })
        return options


def _import_spark_ml():
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    return LogisticRegression, VectorAssembler, F


def _spark_l1_selection(
    spark_df: Any, target_column: str, feature_columns: List[str],
    reg_param: float = 1.0, max_iter: int = 2000, elastic_net_param: float = 1.0,
) -> tuple:
    LR, VectorAssembler, F = _import_spark_ml()

    work_df = spark_df.select([F.col(c).cast("double").alias(c) for c in feature_columns] + [F.col(target_column).cast("double").alias(target_column)])
    work_df = work_df.na.drop(subset=[target_column]).na.fill(0.0, subset=feature_columns)

    stats_exprs = []
    for c in feature_columns:
        stats_exprs.extend([F.mean(c).alias(f"__mean__{c}"), F.stddev(c).alias(f"__std__{c}")])
    stats_row = work_df.agg(*stats_exprs).head()

    scaled_cols = []
    for c in feature_columns:
        mean_val = float(stats_row[f"__mean__{c}"] or 0.0)
        std_val = float(stats_row[f"__std__{c}"] or 1.0)
        if std_val == 0.0:
            std_val = 1.0
        scaled_cols.append(((F.col(c) - F.lit(mean_val)) / F.lit(std_val)).alias(c))
    scaled_cols.append(F.col(target_column))
    work_df = work_df.select(scaled_cols)

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="__scaled__", handleInvalid="keep")
    assembled = assembler.transform(work_df).select("__scaled__", target_column)
    del work_df
    lr = LR(featuresCol="__scaled__", labelCol=target_column, elasticNetParam=elastic_net_param, regParam=reg_param, maxIter=max_iter)
    model = lr.fit(assembled)
    coefs = np.abs(model.coefficients.toArray())
    del model, assembled
    scores: Dict[str, float] = {col: float(coefs[i]) for i, col in enumerate(feature_columns)}
    zero_cols = [col for i, col in enumerate(feature_columns) if coefs[i] == 0.0]
    if len(zero_cols) == len(feature_columns):
        return [], {}, scores
    dropped: List[str] = []
    reasons: Dict[str, str] = {}
    for col in zero_cols:
        dropped.append(col)
        reasons[col] = "L1 zero coefficient"
    return dropped, reasons, scores


# ---------------------------------------------------------------------------
# Chi-squared selection (statistical, univariate)
# ---------------------------------------------------------------------------

def _import_spark_chi_squared_bucketizer():
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.feature import Bucketizer
    return Bucketizer, F


_CHI_BUCKET_BATCH = 200
_CHI_CHECKPOINT_INTERVAL = 600
_CHI_SQL_BATCH = 10  # legacy default, use _chi_sql_batch_size() instead


def _chi_sql_batch_size(num_buckets: int) -> int:
    exprs_per_feature = 2 * (num_buckets + 1)
    return max(5, 400 // exprs_per_feature)


def _sql_chi_squared_scores(
    df: Any, target_column: str, feature_columns: List[str],
    bucketed_names: List[str], num_buckets: int,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    is_pandas = hasattr(df, "iloc") and not hasattr(df, "to_spark")
    if is_pandas:
        return _pandas_chi_squared_scores(df, target_column, feature_columns, bucketed_names, num_buckets)
    return _spark_sql_chi_squared_scores(df, target_column, feature_columns, bucketed_names, num_buckets, progress_fn=progress_fn)


def _pandas_chi_squared_scores(
    df: Any, target_column: str, feature_columns: List[str],
    bucketed_names: List[str], num_buckets: int,
) -> Dict[str, float]:
    target = df[target_column].to_numpy()
    N = len(target)
    N1 = float(target.sum())
    N0 = N - N1
    if N1 == 0 or N0 == 0:
        return {c: 0.0 for c in feature_columns}
    scores: Dict[str, float] = {}
    for feat_col, bkt_col in zip(feature_columns, bucketed_names):
        bkt = df[bkt_col].to_numpy()
        chi2 = 0.0
        for b in range(num_buckets + 1):
            mask = bkt == b
            nb = int(mask.sum())
            if nb == 0:
                continue
            nb1 = float(target[mask].sum())
            nb0 = nb - nb1
            e_b0, e_b1 = nb * N0 / N, nb * N1 / N
            if e_b0 > 0:
                chi2 += (nb0 - e_b0) ** 2 / e_b0
            if e_b1 > 0:
                chi2 += (nb1 - e_b1) ** 2 / e_b1
        scores[feat_col] = chi2
    return scores


def _spark_sql_chi_squared_scores(
    work_df: Any, target_column: str, feature_columns: List[str],
    bucketed_names: List[str], num_buckets: int,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    import pyspark.sql.functions as F  # noqa: N812

    log = progress_fn or (lambda _msg: None)
    N = work_df.count()
    target_sum_row = work_df.agg(F.sum(F.col(target_column).cast("double"))).head()
    N1 = float(target_sum_row[0] or 0)
    N0 = N - N1
    if N1 == 0 or N0 == 0:
        return {c: 0.0 for c in feature_columns}

    n_features = len(feature_columns)
    batch_size = _chi_sql_batch_size(num_buckets)
    n_batches = (n_features + batch_size - 1) // batch_size
    log(f"      Scoring {n_features} features in {n_batches} batches "
        f"(batch_size={batch_size})...")

    scores: Dict[str, float] = {}
    report_every = max(1, n_batches // 10)
    for bi, start in enumerate(range(0, n_features, batch_size)):
        batch_feats = feature_columns[start:start + batch_size]
        batch_bkts = bucketed_names[start:start + batch_size]
        exprs = []
        for fi, bkt_col in enumerate(batch_bkts):
            for b in range(num_buckets + 1):
                exprs.append(F.sum(F.when(F.col(bkt_col) == b, F.lit(1)).otherwise(F.lit(0))).alias(f"nb_{fi}_{b}"))
                exprs.append(F.sum(F.when(F.col(bkt_col) == b, F.col(target_column)).otherwise(F.lit(0))).alias(f"nb1_{fi}_{b}"))
        row = work_df.agg(*exprs).head()
        for fi, feat_col in enumerate(batch_feats):
            chi2 = 0.0
            for b in range(num_buckets + 1):
                nb = int(row[f"nb_{fi}_{b}"] or 0)
                if nb == 0:
                    continue
                nb1 = float(row[f"nb1_{fi}_{b}"] or 0)
                nb0 = nb - nb1
                e_b0, e_b1 = nb * N0 / N, nb * N1 / N
                if e_b0 > 0:
                    chi2 += (nb0 - e_b0) ** 2 / e_b0
                if e_b1 > 0:
                    chi2 += (nb1 - e_b1) ** 2 / e_b1
            scores[feat_col] = chi2
        done = bi + 1
        if done % report_every == 0 or done == n_batches:
            log(f"      Scoring: {done}/{n_batches} batches "
                f"({done * 100 // n_batches}%)")
    return scores


def _spark_chi_squared_selection(
    spark_df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 1000, num_buckets: int = 10,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple:
    Bucketizer, F = _import_spark_chi_squared_bucketizer()
    log = progress_fn or (lambda _msg: None)

    n_features = len(feature_columns)
    if num_top_features >= n_features:
        return [], {}, None

    log(f"    Preparing {n_features} features...")
    work_df = spark_df.select(
        [F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c) for c in feature_columns]
        + [F.col(target_column).cast("double").alias(target_column)]
    )
    work_df = work_df.na.fill(0.0, subset=feature_columns)
    if n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)

    # --- Phase 1: quantile splits ---
    quantiles = [i / num_buckets for i in range(1, num_buckets)]
    n_q = (n_features + _CHI_BUCKET_BATCH - 1) // _CHI_BUCKET_BATCH
    log(f"    Phase 1/3: quantile splits ({n_q} batches)...")
    all_splits: Dict[str, list] = {}
    report_q = max(1, n_q // 5)
    for qi, start in enumerate(range(0, n_features, _CHI_BUCKET_BATCH)):
        batch = feature_columns[start:start + _CHI_BUCKET_BATCH]
        exprs = [F.percentile_approx(c, quantiles).alias(f"__p_{i}")
                 for i, c in enumerate(batch)]
        row = work_df.agg(*exprs).head()
        for i, c in enumerate(batch):
            all_splits[c] = row[f"__p_{i}"]
        done = qi + 1
        if done % report_q == 0 or done == n_q:
            log(f"      Quantiles: {done}/{n_q} batches")

    # --- Phase 2: bucketizer (checkpoint every _CHI_CHECKPOINT_INTERVAL cols) ---
    n_bkt = (n_features + _CHI_BUCKET_BATCH - 1) // _CHI_BUCKET_BATCH
    log(f"    Phase 2/3: bucketizer ({n_bkt} batches, "
        f"checkpoint every {_CHI_CHECKPOINT_INTERVAL} features)...")
    bucketed_names: List[str] = []
    cols_since_checkpoint = 0
    for start in range(0, n_features, _CHI_BUCKET_BATCH):
        batch = feature_columns[start:start + _CHI_BUCKET_BATCH]
        b_input, b_output, b_splits_arr = [], [], []
        for c in batch:
            pcts = all_splits.get(c)
            unique = sorted(set(pcts)) if pcts else []
            if len(unique) < 2:
                unique = [0.0, 1.0]
            b_input.append(c)
            bkt_name = f"__bkt_{c}"
            b_output.append(bkt_name)
            bucketed_names.append(bkt_name)
            b_splits_arr.append([float("-inf")] + unique + [float("inf")])
        bkt = Bucketizer(inputCols=b_input, outputCols=b_output,
                         splitsArray=b_splits_arr, handleInvalid="keep")
        work_df = bkt.transform(work_df)
        cols_since_checkpoint += len(batch)
        needs_more = start + _CHI_BUCKET_BATCH < n_features
        if needs_more and cols_since_checkpoint >= _CHI_CHECKPOINT_INTERVAL:
            work_df = work_df.localCheckpoint(eager=True)
            cols_since_checkpoint = 0
            log(f"      Bucketizer: checkpoint after {start + len(batch)}/{n_features} features")

    if cols_since_checkpoint > 0 and n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)
    log(f"      Bucketizer: done ({n_features} features bucketed)")

    # --- Phase 3: chi-squared scoring ---
    log("    Phase 3/3: chi-squared scoring...")
    scores = _spark_sql_chi_squared_scores(
        work_df, target_column, feature_columns, bucketed_names, num_buckets,
        progress_fn=progress_fn,
    )
    del work_df

    sorted_features = sorted(scores, key=scores.get, reverse=True)
    selected = set(sorted_features[:num_top_features])
    dropped = [c for c in feature_columns if c not in selected]
    reasons = {c: "chi_squared below threshold" for c in dropped}
    return dropped, reasons, scores


def _local_chi_squared_selection(
    df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 1000, num_buckets: int = 10,
) -> tuple:
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import KBinsDiscretizer

    X = df[feature_columns].fillna(0).to_numpy()
    y = df[target_column].to_numpy()
    k = min(num_top_features, len(feature_columns))

    discretizer = KBinsDiscretizer(n_bins=num_buckets, encode="ordinal", strategy="quantile", subsample=None)
    X_binned = discretizer.fit_transform(X)

    selector = SelectKBest(chi2, k=k)
    selector.fit(X_binned, y)
    mask = selector.get_support()
    chi2_scores = selector.scores_

    scores = {feature_columns[i]: float(chi2_scores[i]) if not np.isnan(chi2_scores[i]) else 0.0
              for i in range(len(feature_columns))}
    dropped = [feature_columns[i] for i in range(len(feature_columns)) if not mask[i]]
    reasons = {c: "chi_squared below threshold" for c in dropped}
    return dropped, reasons, scores


# ---------------------------------------------------------------------------
# Gradient-Boosted Decision Tree (GBDT) importance-based selection
#
# Uses XGBoost as the backend for both distributed (Spark) and local paths —
# matches the default downstream training model (xgboost) so feature importance
# ranking is parity-consistent between exploration and production. The Spark
# path uses xgboost.spark.SparkXGBClassifier which stays fully distributed and
# works on PySpark 4.0 / Unity Catalog shared clusters, unlike SynapseML's
# LightGBMClassifier (see microsoft/SynapseML#2452).
# ---------------------------------------------------------------------------

def _import_spark_gbdt_ml():
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.feature import VectorAssembler
    from xgboost.spark import SparkXGBClassifier
    return SparkXGBClassifier, VectorAssembler, F


def _resolve_spark_gbdt_workers() -> int:
    from customer_retention.core.compat.detection import get_default_parallelism
    return max(1, get_default_parallelism() or 1)


def _spark_gbdt_importance_selection(
    spark_df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 300, n_estimators: int = 200, max_depth: int = 6,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple:
    SparkXGBClassifier, VectorAssembler, F = _import_spark_gbdt_ml()
    log = progress_fn or (lambda _msg: None)

    n_features = len(feature_columns)
    if num_top_features >= n_features:
        return [], {}, {c: 0.0 for c in feature_columns}

    log(f"    Preparing {n_features} features (cast + fill null)...")
    work_df = spark_df.select(
        [F.col(c).cast("double").alias(c) for c in feature_columns]
        + [F.col(target_column).cast("double").alias(target_column)]
    )
    work_df = work_df.na.fill(0.0, subset=feature_columns)
    if n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)

    log(f"    Assembling {n_features} features into vector...")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="__gbdt_vec__", handleInvalid="keep")
    assembled = assembler.transform(work_df).select("__gbdt_vec__", target_column)

    num_workers = _resolve_spark_gbdt_workers()
    log(f"    Training XGBoost ({n_estimators} estimators, max_depth {max_depth}, {num_workers} workers)...")
    xgb_estimator = SparkXGBClassifier(
        features_col="__gbdt_vec__", label_col=target_column,
        num_workers=num_workers,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, verbosity=0,
    )
    model = xgb_estimator.fit(assembled)
    scores_dict = model.get_feature_importances(importance_type="total_gain")
    del work_df, assembled
    log("    Training done, ranking features by gain importance...")

    importances = np.array([float(scores_dict.get(f"f{i}", 0.0)) for i in range(n_features)])
    scores: Dict[str, float] = {feature_columns[i]: float(importances[i]) for i in range(n_features)}
    top_indices = set(np.argsort(importances)[-num_top_features:])
    dropped = [feature_columns[i] for i in range(n_features) if i not in top_indices]
    reasons = {c: f"gbdt_importance below top-{num_top_features}" for c in dropped}
    return dropped, reasons, scores


def _local_gbdt_importance_selection(
    df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 300, n_estimators: int = 200, max_depth: int = 6,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple:
    import xgboost as xgb

    log = progress_fn or (lambda _msg: None)
    n_features = len(feature_columns)
    if num_top_features >= n_features:
        return [], {}, {c: 0.0 for c in feature_columns}

    log(f"    Preparing {n_features} features...")
    X = df[feature_columns].fillna(0).to_numpy()
    y = df[target_column].to_numpy()

    log(f"    Training XGBoost ({n_estimators} estimators, max_depth {max_depth})...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, importance_type="total_gain",
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    log("    Training done, ranking features by gain importance...")

    scores: Dict[str, float] = {feature_columns[i]: float(importances[i]) for i in range(n_features)}
    top_indices = set(np.argsort(importances)[-num_top_features:])
    dropped = [feature_columns[i] for i in range(n_features) if i not in top_indices]
    reasons = {c: f"gbdt_importance below top-{num_top_features}" for c in dropped}
    return dropped, reasons, scores


def extract_precomputed_stats(findings: Any) -> Dict[str, Any]:
    """Extract medians, non-null counts, and variances from ExplorationFindings.

    Returns dict with keys 'medians', 'non_null', 'variances' — pass as
    ``**extract_precomputed_stats(findings)``-style kwargs or individually
    to ``run_selection_pipeline``.
    """
    medians: Dict[str, float] = {}
    non_null: Dict[str, int] = {}
    variances: Dict[str, float] = {}
    total = getattr(findings, "row_count", 0) or 0
    for name, col in getattr(findings, "columns", {}).items():
        um = getattr(col, "universal_metrics", {})
        tm = getattr(col, "type_metrics", {})
        nc = um.get("null_count", 0)
        if total > 0:
            non_null[name] = total - int(nc)
        std = tm.get("std")
        if std is not None:
            variances[name] = float(std) ** 2
        med = tm.get("median")
        if med is not None:
            medians[name] = float(med)
    import pandas as _pd
    return {
        "precomputed_medians": medians or None,
        "precomputed_non_null": non_null or None,
        "precomputed_variances": _pd.Series(variances) if variances else None,
    }


_L1_SAMPLE_ROWS = 400_000


def _thin_by_temporal_stride(ps_df: Any, temporal_column: str, target_rows: int, log: Any) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    from customer_retention.core.compat import as_spark_df
    from customer_retention.core.compat.spark_backend import _as_pandas_api

    spark_df = as_spark_df(ps_df)
    total = spark_df.count()
    if total <= target_rows:
        return ps_df

    dates = [row[0] for row in spark_df.select(temporal_column).distinct().orderBy(temporal_column).collect()]
    n_dates = len(dates)
    if n_dates <= 4:
        return ps_df

    rows_per_date = total / n_dates
    target_dates = max(4, int(target_rows / rows_per_date))
    stride = max(1, n_dates // target_dates)
    kept = [dates[i] for i in range(n_dates - 1, -1, -stride)]

    thinned = spark_df.filter(F.col(temporal_column).isin(kept)).localCheckpoint(eager=True)
    log(f"    L1 temporal thinning: {total:,} -> ~{len(kept) * int(rows_per_date):,} rows "
        f"({n_dates} -> {len(kept)} dates, stride {stride})")
    return _as_pandas_api(thinned)


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def run_selection_pipeline(
    df: DataFrame, target_column: str,
    variance_threshold: float = 0.01, correlation_threshold: float = 0.95,
    l1_enabled: bool = False, max_features: Optional[int] = None,
    preserve_features: Optional[List[str]] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    precomputed_corr_matrix: Optional[Any] = None,
    l1_C: float = 1.0, l1_ratio: float = 1.0,
    precomputed_variances: Optional[Any] = None,
    precomputed_medians: Optional[Dict[str, float]] = None,
    precomputed_non_null: Optional[Dict[str, int]] = None,
    temporal_column: Optional[str] = None,
    l1_sample_rows: int = _L1_SAMPLE_ROWS,
    candidate_features: Optional[List[str]] = None,
    trace_recorder: Optional[SelectionTraceRecorder] = None,
) -> FeatureSelectionResult:
    log = progress_fn or (lambda msg: print(msg))
    all_dropped: List[str] = []
    all_reasons: Dict[str, str] = {}
    importance_scores: Optional[Dict[str, float]] = None
    current_df = df
    pipeline_start = time.monotonic()
    n_features_initial = len([c for c in df.columns if c != target_column])

    if candidate_features is not None:
        _candidate_set = set(candidate_features)
        _non_target = [c for c in df.columns if c != target_column]
        _base_features = [f for f in _non_target if f not in _candidate_set]
        filter_preserve = list(set((preserve_features or []) + _base_features))
        _skip_filters = len(candidate_features) == 0
    else:
        filter_preserve = preserve_features
        _skip_filters = False

    stages: List[str] = []
    if not _skip_filters:
        stages += ["variance", "correlation"]
    if l1_enabled:
        stages.append("L1")
    total_stages = len(stages)

    if candidate_features is not None and not _skip_filters:
        log(f"Feature selection pipeline: {len(candidate_features)} candidate features "
            f"({len(_base_features)} base protected), {total_stages} stages")
    else:
        log(f"Feature selection pipeline: {n_features_initial} features, {total_stages} stages")

    shared_variances = precomputed_variances
    result_var = None
    result_corr = None

    if not _skip_filters:
        t0 = time.monotonic()
        log(f"  [1/{total_stages}] Variance filter (threshold={variance_threshold})...")
        var_selector = FeatureSelector(
            method=SelectionMethod.VARIANCE, variance_threshold=variance_threshold,
            target_column=target_column, preserve_features=filter_preserve,
            precomputed_variances=precomputed_variances,
            trace_recorder=trace_recorder,
        )
        result_var = var_selector.fit_transform(current_df)
        shared_variances = var_selector._cached_variances
        current_df = result_var.df
        all_dropped.extend(result_var.dropped_features)
        all_reasons.update(result_var.drop_reasons)
        elapsed = time.monotonic() - t0
        remaining = len(result_var.selected_features)
        log(f"    {_format_elapsed(elapsed)} — dropped {len(result_var.dropped_features)}, remaining {remaining}")

        t0 = time.monotonic()
        log(f"  [2/{total_stages}] Correlation filter (threshold={correlation_threshold}, {remaining} features)...")
        result_corr = FeatureSelector(
            method=SelectionMethod.CORRELATION, correlation_threshold=correlation_threshold,
            target_column=target_column, preserve_features=filter_preserve,
            precomputed_corr_matrix=precomputed_corr_matrix, progress_fn=log,
            precomputed_variances=shared_variances,
            precomputed_medians=precomputed_medians,
            precomputed_non_null=precomputed_non_null,
            correlation_candidates=candidate_features,
            trace_recorder=trace_recorder,
        ).fit_transform(current_df)
        current_df = result_corr.df
        all_dropped.extend(result_corr.dropped_features)
        all_reasons.update(result_corr.drop_reasons)
        elapsed = time.monotonic() - t0
        remaining = len(result_corr.selected_features)
        log(f"    {_format_elapsed(elapsed)} — dropped {len(result_corr.dropped_features)}, remaining {remaining}")

    last_method = SelectionMethod.CORRELATION if not _skip_filters else SelectionMethod.L1_SELECTION
    remaining = len([c for c in current_df.columns if c != target_column])

    if l1_enabled:
        import gc
        del result_var, result_corr, shared_variances
        gc.collect()

        from customer_retention.core.compat import _is_spark_pandas
        if temporal_column and temporal_column in current_df.columns:
            if _is_spark_pandas(current_df):
                current_df = _thin_by_temporal_stride(current_df, temporal_column, l1_sample_rows, log)
            current_df = current_df.drop(columns=[temporal_column])

        t0 = time.monotonic()
        log(f"  [{total_stages}/{total_stages}] L1 selection ({remaining} features, max_features={max_features})...")
        last_method = SelectionMethod.L1_SELECTION
        result_l1 = FeatureSelector(
            method=SelectionMethod.L1_SELECTION, target_column=target_column,
            preserve_features=preserve_features, max_features=max_features,
            l1_C=l1_C, l1_ratio=l1_ratio,
            trace_recorder=trace_recorder,
        ).fit_transform(current_df)
        current_df = result_l1.df
        all_dropped.extend(result_l1.dropped_features)
        all_reasons.update(result_l1.drop_reasons)
        importance_scores = result_l1.importance_scores
        elapsed = time.monotonic() - t0
        remaining = len(result_l1.selected_features)
        log(f"    {_format_elapsed(elapsed)} — dropped {len(result_l1.dropped_features)}, remaining {remaining}")

    total_elapsed = time.monotonic() - pipeline_start
    selected = [c for c in current_df.columns if c != target_column]
    log(f"  Done: {n_features_initial} -> {len(selected)} features ({len(all_dropped)} dropped) in {_format_elapsed(total_elapsed)}")

    return FeatureSelectionResult(
        df=current_df, selected_features=selected, dropped_features=all_dropped,
        drop_reasons=all_reasons, method_used=last_method, importance_scores=importance_scores,
    )


def run_chi_squared_selection(
    df: DataFrame, target_column: str, max_features: int = 1000,
    num_buckets: int = 10, feature_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    trace_recorder: Optional[SelectionTraceRecorder] = None,
) -> FeatureSelectionResult:
    log = progress_fn or (lambda msg: print(msg))
    t0 = time.monotonic()

    exclude = {target_column}
    if temporal_column:
        exclude.add(temporal_column)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in exclude]
    work_df = df[feature_columns + [target_column]]
    if temporal_column and temporal_column in work_df.columns:
        work_df = work_df.drop(columns=[temporal_column])

    n_initial = len(feature_columns)
    log(f"  Chi-squared selection: {n_initial} features, selecting top {max_features}...")

    selector = FeatureSelector(
        method=SelectionMethod.CHI_SQUARED, target_column=target_column,
        max_features=max_features, chi_squared_num_buckets=num_buckets,
        progress_fn=log, trace_recorder=trace_recorder,
    )
    result = selector.fit_transform(work_df)
    elapsed = time.monotonic() - t0
    log(f"    {_format_elapsed(elapsed)} — dropped {len(result.dropped_features)}, "
        f"remaining {len(result.selected_features)}")
    return result


def run_gbdt_importance_selection(
    df: DataFrame, target_column: str, max_features: int = 300,
    n_estimators: int = 200, max_depth: int = 6,
    feature_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    trace_recorder: Optional[SelectionTraceRecorder] = None,
) -> FeatureSelectionResult:
    log = progress_fn or (lambda msg: print(msg))
    t0 = time.monotonic()

    exclude = {target_column}
    if temporal_column:
        exclude.add(temporal_column)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in exclude]
    work_df = df[feature_columns + [target_column]]
    if temporal_column and temporal_column in work_df.columns:
        work_df = work_df.drop(columns=[temporal_column])

    n_initial = len(feature_columns)
    log(f"  XGBoost GBDT importance selection: {n_initial} features, selecting top {max_features}...")

    selector = FeatureSelector(
        method=SelectionMethod.GBDT_IMPORTANCE, target_column=target_column,
        max_features=max_features, gbdt_n_estimators=n_estimators,
        gbdt_max_depth=max_depth, progress_fn=log,
        trace_recorder=trace_recorder,
    )
    result = selector.fit_transform(work_df)
    elapsed = time.monotonic() - t0
    log(f"    {_format_elapsed(elapsed)} — dropped {len(result.dropped_features)}, "
        f"remaining {len(result.selected_features)}")
    return result


# ---------------------------------------------------------------------------
# Chi-squared rescue selection (snapshot-grid IID-aware union of selectors)
#
# See docs/chi_squared_rescue_selection_plan.md for the design.  The flow is:
#   1. Slice the training data to one IID-respecting time slice (penultimate
#      snapshot by default) using compat.resolve_time_slice.
#   2. Run chi-squared on the slice as the primary selector.
#   3. Rescue chi-squared drops via L1 and/or XGBoost gain importance trained
#      on the same slice, restricted to the chi-squared drop pool.
#   4. Union the three keep sets and audit drop reasons with triple-consensus
#      stats (chi rank + L1 coef + GBDT total_gain) so the recommendation
#      registry can persist them as drop_rescue_consensus actions.
#
# All Spark dispatch reuses the existing distributed batched primitives in
# this file:  _spark_chi_squared_selection (Phase 1/2/3 batched aggs +
# bucketizer with checkpoints), _spark_gbdt_importance_selection (cast/fill
# + checkpoint + VectorAssembler + SparkXGBClassifier), and
# _spark_l1_selection (batched mean/stddev aggs + scaled select).  The rescue
# helpers never collect feature rows to the driver — they hand the slice to
# these distributed primitives instead.
# ---------------------------------------------------------------------------


@dataclass
class _RescueResult:
    l1_keep: set
    gbdt_keep: set
    l1_coefs: Dict[str, float]
    gbdt_gains: Dict[str, float]


def _chi_squared_primary(
    df: Any, target_column: str, features: List[str],
    top_k: int, num_buckets: int = 10,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple[set, set, Dict[str, Dict[str, float]]]:
    """Run chi-squared on the training slice; return (keep, drop, stats).

    ``stats`` maps every input feature to ``{"score": float, "rank": int}``.
    The Spark path stays distributed via the existing batched
    ``_spark_chi_squared_selection`` primitive — slice rows are never
    collected for chi-squared scoring.
    """
    if not features:
        return set(), set(), {}

    # No-op guard: when top_k covers the entire feature pool, every feature
    # will be kept regardless of chi-squared score. Skip the scoring work
    # (avoids bucketizer construction + one Spark job per batch of columns).
    # Emit NaN scores so the trace can distinguish "not evaluated" from
    # "genuinely-zero chi²".
    if top_k >= len(features):
        if progress_fn:
            progress_fn(
                f"    chi-squared: top_k={top_k} >= n_features={len(features)}, "
                f"skipping scoring (all features kept)"
            )
        nan_val = float("nan")
        stats = {f: {"score": nan_val, "rank": i + 1} for i, f in enumerate(features)}
        return set(features), set(), stats

    from customer_retention.core.compat import _is_spark_pandas

    if _is_spark_pandas(df):
        from customer_retention.core.compat import as_spark_df
        spark_df = as_spark_df(df[features + [target_column]])
        _, _, scores = _spark_chi_squared_selection(
            spark_df, target_column, features,
            num_top_features=len(features),  # always score all; we cut top_k below
            num_buckets=num_buckets, progress_fn=progress_fn,
        )
    else:
        _, _, scores = _local_chi_squared_selection(
            df, target_column, features,
            num_top_features=len(features), num_buckets=num_buckets,
        )
    scores = scores or {f: 0.0 for f in features}
    ordered = sorted(features, key=lambda f: float(scores.get(f, 0.0)), reverse=True)
    keep = set(ordered[:top_k])
    drop = set(ordered[top_k:])
    stats = {
        f: {"score": float(scores.get(f, 0.0)), "rank": rank + 1}
        for rank, f in enumerate(ordered)
    }
    return keep, drop, stats


def _l1_rescue_on_pool(
    df: Any, target_column: str, drop_pool: List[str],
    max_features: int = 50, C: float = 1.0,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple[set, Dict[str, float]]:
    """L1 rescue on the chi-squared drop pool.  Returns (kept, coefs)."""
    if not drop_pool:
        return set(), {}

    from customer_retention.core.compat import _is_spark_pandas

    if _is_spark_pandas(df):
        from customer_retention.core.compat import as_spark_df
        spark_df = as_spark_df(df[drop_pool + [target_column]])
        _, _, scores = _spark_l1_selection(
            spark_df, target_column, drop_pool, reg_param=1.0 / C,
        )
    else:
        scores = _local_l1_scores(df, target_column, drop_pool, C=C)
    coefs = {f: float(scores.get(f, 0.0)) for f in drop_pool}
    if not coefs:
        return set(), coefs
    max_coef = max(coefs.values()) if coefs else 0.0
    if max_coef <= 0.0:
        return set(), coefs
    floor = max_coef * 0.01
    ranked = sorted(drop_pool, key=lambda f: coefs[f], reverse=True)
    kept: set = set()
    for f in ranked:
        if coefs[f] <= floor:
            break
        if len(kept) >= max_features:
            break
        kept.add(f)
    return kept, coefs


def _local_l1_scores(
    df: Any, target_column: str, feature_columns: List[str], C: float = 1.0,
) -> Dict[str, float]:
    work_df = df[feature_columns + [target_column]].dropna(subset=[target_column])
    if len(work_df) < 10:
        return {f: 0.0 for f in feature_columns}
    X = work_df[feature_columns].fillna(0).to_numpy()
    y = work_df[target_column].to_numpy()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(solver="saga", C=C, max_iter=2000, l1_ratio=1.0)
    model.fit(X_scaled, y)
    coefs = np.abs(model.coef_)
    max_coefs = coefs.max(axis=0) if coefs.ndim == 2 else coefs
    return {feature_columns[i]: float(max_coefs[i]) for i in range(len(feature_columns))}


def _gbdt_rescue_on_pool(
    df: Any, target_column: str, drop_pool: List[str],
    max_features: int = 100, n_estimators: int = 200, max_depth: int = 6,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple[set, Dict[str, float]]:
    """XGBoost gain rescue on the chi-squared drop pool.

    Returns ``(kept, gains)``. Spark path stays distributed via the existing
    ``_spark_gbdt_importance_selection`` primitive (VectorAssembler +
    SparkXGBClassifier with checkpoint and ``num_workers`` from the
    parallelism resolver). No row collection.
    """
    if not drop_pool:
        return set(), {}

    from customer_retention.core.compat import _is_spark_pandas

    if _is_spark_pandas(df):
        from customer_retention.core.compat import as_spark_df
        spark_df = as_spark_df(df[drop_pool + [target_column]])
        scores = _spark_gbdt_total_gain_scores(
            spark_df, target_column, drop_pool,
            n_estimators=n_estimators, max_depth=max_depth,
            progress_fn=progress_fn,
        )
    else:
        scores = _local_gbdt_total_gain_scores(
            df, target_column, drop_pool,
            n_estimators=n_estimators, max_depth=max_depth,
            progress_fn=progress_fn,
        )
    gains = {f: float(scores.get(f, 0.0)) for f in drop_pool}
    kept = _apply_importance_floor(gains, max_features=max_features, floor_ratio=0.01)
    return kept, gains


def _local_gbdt_total_gain_scores(
    df: Any, target_column: str, feature_columns: List[str],
    n_estimators: int, max_depth: int,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """Train XGBoost and return raw total_gain importances per feature.

    Unlike ``_local_gbdt_importance_selection``, this never short-circuits
    when ``num_top_features >= n_features`` — the rescue stage needs the
    real gains regardless of how the keep cap is set.
    """
    import xgboost as xgb

    log = progress_fn or (lambda _msg: None)
    if not feature_columns:
        return {}
    X = df[feature_columns].fillna(0).to_numpy()
    y = df[target_column].to_numpy()
    log(f"    GBDT rescue: {len(feature_columns)} features × {len(X)} rows...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, importance_type="total_gain",
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)
    booster = model.get_booster()
    raw = booster.get_score(importance_type="total_gain")
    scores: Dict[str, float] = {f: 0.0 for f in feature_columns}
    for k, v in raw.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_columns):
                scores[feature_columns[idx]] = float(v)
    return scores


def _spark_gbdt_total_gain_scores(
    spark_df: Any, target_column: str, feature_columns: List[str],
    n_estimators: int, max_depth: int,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """Distributed XGBoost gain importances — never short-circuits.

    Reuses the existing chunked + checkpointed prep pattern from
    ``_spark_gbdt_importance_selection`` (cast/fill, localCheckpoint above
    _CHI_BUCKET_BATCH features, VectorAssembler, SparkXGBClassifier with
    num_workers from the parallelism resolver).
    """
    SparkXGBClassifier, VectorAssembler, F = _import_spark_gbdt_ml()
    log = progress_fn or (lambda _msg: None)
    if not feature_columns:
        return {}
    n_features = len(feature_columns)
    log(f"    GBDT rescue (distributed): preparing {n_features} features...")
    work_df = spark_df.select(
        [F.col(c).cast("double").alias(c) for c in feature_columns]
        + [F.col(target_column).cast("double").alias(target_column)]
    )
    work_df = work_df.na.fill(0.0, subset=feature_columns)
    if n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="__gbdt_rescue_vec__", handleInvalid="keep",
    )
    assembled = assembler.transform(work_df).select("__gbdt_rescue_vec__", target_column)
    num_workers = _resolve_spark_gbdt_workers()
    log(f"    GBDT rescue: training XGBoost ({n_estimators} est, depth {max_depth}, "
        f"{num_workers} workers)...")
    estimator = SparkXGBClassifier(
        features_col="__gbdt_rescue_vec__", label_col=target_column,
        num_workers=num_workers,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, verbosity=0,
    )
    model = estimator.fit(assembled)
    raw = model.get_feature_importances(importance_type="total_gain")
    del work_df, assembled
    scores: Dict[str, float] = {f: 0.0 for f in feature_columns}
    for k, v in raw.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_columns):
                scores[feature_columns[idx]] = float(v)
    return scores


def _apply_importance_floor(
    gains: Dict[str, float], max_features: int, floor_ratio: float = 0.01,
) -> set:
    """Return the top-K features above ``floor_ratio * max(gain)``.

    Implements the §3.3 absolute importance floor — keeps the rescue from
    importing "least-bad noise" when the drop pool has no real signal.
    """
    if not gains:
        return set()
    max_gain = max(gains.values())
    if max_gain <= 0.0:
        return set()
    floor = max_gain * floor_ratio
    above_floor = [f for f, g in gains.items() if g > floor]
    above_floor.sort(key=lambda f: gains[f], reverse=True)
    return set(above_floor[:max_features])


def _apply_shadow_floor(
    gains: Dict[str, float], shadow_gains: Dict[str, float],
) -> set:
    """Boruta-style floor: keep features whose gain exceeds the best shadow."""
    if not shadow_gains:
        return set(gains.keys())
    shadow_max = max(shadow_gains.values()) if shadow_gains else 0.0
    return {f for f, g in gains.items() if g > shadow_max}


def _rescue_from_chi_drops(
    df: Any, target_column: str, drop_pool: List[str],
    *,
    l1_enabled: bool = False, gbdt_enabled: bool = True,
    l1_max: int = 50, l1_C: float = 1.0,
    gbdt_max: int = 100, gbdt_n_estimators: int = 200, gbdt_max_depth: int = 6,
    shadow_floor_enabled: bool = False, shadow_floor_fraction: float = 0.1,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> _RescueResult:
    """Run L1 and GBDT rescue selectors on the chi-squared drop pool.

    Both selectors are confined to the drop pool and trained on the slice
    that ``df`` represents.  Spark path stays distributed throughout.
    """
    from customer_retention.core.compat import FeatureSelectionError

    if not drop_pool and (l1_enabled or gbdt_enabled):
        raise FeatureSelectionError(
            "_rescue_from_chi_drops: drop_pool is empty but rescue selectors are enabled"
        )

    l1_keep: set = set()
    gbdt_keep: set = set()
    l1_coefs: Dict[str, float] = {f: 0.0 for f in drop_pool}
    gbdt_gains: Dict[str, float] = {f: 0.0 for f in drop_pool}

    if l1_enabled and drop_pool:
        l1_keep, l1_coefs = _l1_rescue_on_pool(
            df, target_column, drop_pool,
            max_features=l1_max, C=l1_C, progress_fn=progress_fn,
        )
    if gbdt_enabled and drop_pool:
        gbdt_keep, gbdt_gains = _gbdt_rescue_on_pool(
            df, target_column, drop_pool,
            max_features=gbdt_max, n_estimators=gbdt_n_estimators,
            max_depth=gbdt_max_depth, progress_fn=progress_fn,
        )
        if shadow_floor_enabled and gbdt_keep:
            gbdt_keep = _gbdt_rescue_with_shadow(
                df, target_column, drop_pool,
                base_gains=gbdt_gains,
                fraction=shadow_floor_fraction,
                n_estimators=gbdt_n_estimators, max_depth=gbdt_max_depth,
                progress_fn=progress_fn,
            )

    return _RescueResult(
        l1_keep=l1_keep, gbdt_keep=gbdt_keep,
        l1_coefs=l1_coefs, gbdt_gains=gbdt_gains,
    )


def _gbdt_rescue_with_shadow(
    df: Any, target_column: str, drop_pool: List[str],
    base_gains: Dict[str, float], fraction: float,
    n_estimators: int, max_depth: int,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> set:
    """Permute a fraction of the drop pool as shadow features, retrain,
    keep only real features whose gain exceeds the best shadow gain.

    Spark path stays distributed: shadow columns are built via
    ``Window.orderBy(F.rand())`` + row_number join — no driver collection.
    Pandas path permutes via ``.sample(frac=1.0)``.
    """
    from customer_retention.core.compat import _is_spark_pandas

    n_shadow = max(1, int(round(len(drop_pool) * fraction)))
    rng = np.random.default_rng(0)
    chosen = list(rng.choice(drop_pool, size=min(n_shadow, len(drop_pool)), replace=False))
    shadow_names = [f"__shadow_{i}" for i in range(len(chosen))]

    if _is_spark_pandas(df):
        augmented = _spark_augment_with_shadow(df, target_column, drop_pool, chosen, shadow_names)
        augmented_pool = drop_pool + shadow_names
        from customer_retention.core.compat import as_spark_df
        spark_aug = as_spark_df(augmented)
        _, _, scores = _spark_gbdt_importance_selection(
            spark_aug, target_column, augmented_pool,
            num_top_features=len(augmented_pool),
            n_estimators=n_estimators, max_depth=max_depth,
            progress_fn=progress_fn,
        )
    else:
        local = df[drop_pool + [target_column]]
        permuted = local[chosen].sample(frac=1.0, random_state=42).reset_index(drop=True)
        augmented = local.reset_index(drop=True).assign(
            **{shadow_names[i]: permuted[chosen[i]].to_numpy() for i in range(len(chosen))}
        )
        augmented_pool = drop_pool + shadow_names
        _, _, scores = _local_gbdt_importance_selection(
            augmented, target_column, augmented_pool,
            num_top_features=len(augmented_pool),
            n_estimators=n_estimators, max_depth=max_depth,
            progress_fn=progress_fn,
        )

    shadow_scores = {n: float(scores.get(n, 0.0)) for n in shadow_names}
    return _apply_shadow_floor(base_gains, shadow_scores)


def _spark_augment_with_shadow(
    df: Any, target_column: str, drop_pool: List[str],
    chosen: List[str], shadow_names: List[str],
) -> Any:
    """Distributed shadow append: row_number join between original and
    rand-ordered slice — no driver collect."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    from customer_retention.core.compat import as_spark_df
    from customer_retention.core.compat.spark_backend import _as_pandas_api

    spark_df = as_spark_df(df[drop_pool + [target_column]])
    indexed = spark_df.withColumn(
        "__rid", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
    )
    shuffled = (
        spark_df.select([F.col(c).alias(s) for c, s in zip(chosen, shadow_names)])
                .withColumn("__rid", F.row_number().over(Window.orderBy(F.rand(seed=42))))
    )
    joined = indexed.join(shuffled, on="__rid", how="inner").drop("__rid")
    joined = joined.localCheckpoint(eager=True)
    return _as_pandas_api(joined)


def _build_rescue_consensus_reasons(
    dropped: List[str], chi_stats: Dict[str, Dict[str, float]],
    l1_coefs: Dict[str, float], gbdt_gains: Dict[str, float],
    *,
    slice_date: str, slice_strategy: str, slice_row_count: int,
    l1_considered: bool, gbdt_considered: bool,
) -> Dict[str, Dict[str, Any]]:
    """Build the triple-consensus drop_reason dicts for every dropped feature."""
    reasons: Dict[str, Dict[str, Any]] = {}
    for feat in dropped:
        chi = chi_stats.get(feat, {"score": 0.0, "rank": -1})
        l1_coef = float(l1_coefs.get(feat, 0.0))
        gbdt_gain = float(gbdt_gains.get(feat, 0.0))
        params: Dict[str, Any] = {
            "chi_squared_rank": int(chi["rank"]),
            "chi_squared_score": float(chi["score"]),
            "l1_coefficient": l1_coef,
            "l1_considered": bool(l1_considered),
            "gbdt_total_gain": gbdt_gain,
            "gbdt_considered": bool(gbdt_considered),
            "slice_date": slice_date,
            "slice_strategy": slice_strategy,
            "slice_row_count": int(slice_row_count),
        }
        rationale_parts = [f"chi-squared rank {int(chi['rank'])}"]
        if l1_considered:
            rationale_parts.append(f"L1 coef {l1_coef:.4g}")
        if gbdt_considered:
            rationale_parts.append(f"GBDT gain {gbdt_gain:.4g}")
        rationale = "Dropped by all enabled selectors; " + ", ".join(rationale_parts)
        reasons[feat] = {
            "action": "drop_rescue_consensus",
            "parameters": params,
            "rationale": rationale,
        }
    return reasons


def run_chi_squared_rescue_selection(
    df: Any, target_column: str, max_features: int = 500,
    *,
    entity_column: Optional[str] = None,
    time_column: Optional[str] = None,
    slice_strategy: str = "penultimate",
    min_positive_rate: float = 0.03,
    num_buckets: int = 10,
    l1_rescue_enabled: bool = False,
    l1_rescue_max_features: int = 50,
    l1_C: float = 1.0,
    gbdt_rescue_enabled: bool = True,
    gbdt_rescue_max_features: int = 100,
    gbdt_n_estimators: int = 200,
    gbdt_max_depth: int = 6,
    shadow_feature_floor: bool = False,
    progress_fn: Optional[Callable[[str], None]] = None,
    trace_recorder: Optional[SelectionTraceRecorder] = None,
) -> FeatureSelectionResult:
    """Three-stage chi-squared + rescue feature selection on a snapshot grid.

    See ``docs/chi_squared_rescue_selection_plan.md``.

    The selection runs on a single ``slice_strategy`` slice of ``df`` to
    restore IID assumptions in the snapshot-grid panel data; the resulting
    keep set is then applied to the full ``df`` for downstream modeling.

    Spark dispatch is fully distributed: the time-slice filter, chi-squared
    primary, and rescue selectors all stay in batched Spark jobs (see
    ``_spark_chi_squared_selection``, ``_spark_gbdt_importance_selection``,
    ``_spark_l1_selection``).
    """
    log = progress_fn or (lambda _msg: None)
    t_start = time.monotonic()

    from customer_retention.core.compat import (
        FeatureSelectionError,
        resolve_time_slice,
    )

    if target_column not in df.columns:
        raise FeatureSelectionError(
            f"run_chi_squared_rescue_selection: target_column {target_column!r} "
            f"not in DataFrame columns"
        )
    if time_column is None or time_column not in df.columns:
        raise FeatureSelectionError(
            f"run_chi_squared_rescue_selection: time_column {time_column!r} "
            f"not in DataFrame columns"
        )
    if entity_column is not None and entity_column not in df.columns:
        raise FeatureSelectionError(
            f"run_chi_squared_rescue_selection: entity_column {entity_column!r} "
            f"not in DataFrame columns"
        )

    meta_cols = {target_column, time_column}
    if entity_column:
        meta_cols.add(entity_column)
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        raise FeatureSelectionError(
            "run_chi_squared_rescue_selection: no feature columns after excluding meta"
        )

    log(f"  Resolving training slice (strategy={slice_strategy}, "
        f"min_positive_rate={min_positive_rate})...")
    slice_df, slice_date, slice_count = resolve_time_slice(
        df, time_column=time_column, strategy=slice_strategy,
        target_column=target_column, min_positive_rate=min_positive_rate,
        entity_column=entity_column,
    )
    log(f"    slice_date={slice_date}, rows={slice_count:,}")

    selector_cols = feature_cols + [target_column]
    slice_for_selection = slice_df[selector_cols]

    log(f"  Stage 1/3: chi-squared primary on {len(feature_cols)} features (top_k={max_features})...")
    keep_chi, drop_chi, chi_stats = _chi_squared_primary(
        slice_for_selection, target_column=target_column,
        features=feature_cols, top_k=max_features, num_buckets=num_buckets,
        progress_fn=progress_fn,
    )
    log(f"    chi-squared kept={len(keep_chi)}, dropped={len(drop_chi)}")

    rescue: Optional[_RescueResult] = None
    rescue_l1: set = set()
    rescue_gbdt: set = set()
    if drop_chi and (l1_rescue_enabled or gbdt_rescue_enabled):
        log(f"  Stage 2/3: rescue on {len(drop_chi)} chi-squared drops "
            f"(L1={l1_rescue_enabled}, GBDT={gbdt_rescue_enabled})...")
        rescue = _rescue_from_chi_drops(
            slice_for_selection, target_column=target_column,
            drop_pool=sorted(drop_chi),
            l1_enabled=l1_rescue_enabled, gbdt_enabled=gbdt_rescue_enabled,
            l1_max=l1_rescue_max_features, l1_C=l1_C,
            gbdt_max=gbdt_rescue_max_features,
            gbdt_n_estimators=gbdt_n_estimators, gbdt_max_depth=gbdt_max_depth,
            shadow_floor_enabled=shadow_feature_floor,
            progress_fn=progress_fn,
        )
        rescue_l1 = rescue.l1_keep
        rescue_gbdt = rescue.gbdt_keep
        log(f"    rescued L1={len(rescue_l1)}, rescued GBDT={len(rescue_gbdt)}")

    final_keep = keep_chi | rescue_l1 | rescue_gbdt
    dropped = [f for f in feature_cols if f not in final_keep]

    l1_coefs = rescue.l1_coefs if rescue else {f: 0.0 for f in feature_cols}
    gbdt_gains = rescue.gbdt_gains if rescue else {f: 0.0 for f in feature_cols}

    log("  Stage 3/3: building drop_rescue_consensus reasons...")
    reasons = _build_rescue_consensus_reasons(
        dropped, chi_stats, l1_coefs, gbdt_gains,
        slice_date=slice_date, slice_strategy=slice_strategy,
        slice_row_count=slice_count,
        l1_considered=l1_rescue_enabled, gbdt_considered=gbdt_rescue_enabled,
    )

    selected = [f for f in feature_cols if f in final_keep]
    keep_cols = [c for c in df.columns if c in final_keep or c in meta_cols]
    result_df = df[keep_cols]

    importance_scores: Dict[str, float] = {}
    for f in feature_cols:
        importance_scores[f] = float(chi_stats.get(f, {"score": 0.0})["score"])

    elapsed = time.monotonic() - t_start
    log(f"  Done: {len(feature_cols)} -> {len(selected)} features "
        f"({len(dropped)} dropped) in {_format_elapsed(elapsed)}")

    _record_rescue_trace(
        trace_recorder, feature_cols=feature_cols, keep_chi=keep_chi,
        drop_chi=drop_chi, chi_stats=chi_stats, max_features=max_features,
        rescue_l1=rescue_l1, rescue_gbdt=rescue_gbdt,
        l1_coefs=l1_coefs, gbdt_gains=gbdt_gains,
        l1_rescue_enabled=l1_rescue_enabled, gbdt_rescue_enabled=gbdt_rescue_enabled,
    )

    result = FeatureSelectionResult(
        df=result_df, selected_features=selected, dropped_features=dropped,
        drop_reasons=reasons, method_used=SelectionMethod.CHI_SQUARED,
        importance_scores=importance_scores,
    )
    return result


def _record_rescue_trace(
    recorder: Optional[SelectionTraceRecorder], *, feature_cols: List[str],
    keep_chi: set, drop_chi: set, chi_stats: Dict[str, Dict[str, float]],
    max_features: int,
    rescue_l1: set, rescue_gbdt: set,
    l1_coefs: Dict[str, float], gbdt_gains: Dict[str, float],
    l1_rescue_enabled: bool, gbdt_rescue_enabled: bool,
) -> None:
    if recorder is None:
        return
    chi_scores = {f: float(chi_stats.get(f, {"score": 0.0})["score"]) for f in feature_cols}
    chi_decisions = {f: ("kept" if f in keep_chi else "dropped") for f in feature_cols}
    chi_reasons = {f: (None if f in keep_chi else "chi_squared_drop") for f in feature_cols}
    recorder.record_stage(
        stage="chi_squared", score_name="chi2_stat", scores=chi_scores,
        threshold=float(max_features), decisions=chi_decisions, reasons=chi_reasons,
        stage_input_count=len(feature_cols), stage_output_count=len(keep_chi),
    )
    if l1_rescue_enabled:
        l1_pool = sorted(drop_chi)
        l1_scores = {f: float(l1_coefs.get(f, 0.0)) for f in l1_pool}
        l1_decisions = {f: ("kept" if f in rescue_l1 else "dropped") for f in l1_pool}
        l1_reasons = {f: ("l1_rescue" if f in rescue_l1 else "l1_below_floor") for f in l1_pool}
        recorder.record_stage(
            stage="l1_rescue", score_name="l1_abs_coef", scores=l1_scores,
            threshold=None, decisions=l1_decisions, reasons=l1_reasons,
            stage_input_count=len(l1_pool), stage_output_count=len(rescue_l1),
        )
    if gbdt_rescue_enabled:
        gbdt_pool = sorted(drop_chi)
        gbdt_scores = {f: float(gbdt_gains.get(f, 0.0)) for f in gbdt_pool}
        gbdt_decisions = {f: ("kept" if f in rescue_gbdt else "dropped") for f in gbdt_pool}
        gbdt_reasons = {f: ("gbdt_rescue" if f in rescue_gbdt else "gbdt_below_floor") for f in gbdt_pool}
        recorder.record_stage(
            stage="gbdt_rescue", score_name="gbdt_gain", scores=gbdt_scores,
            threshold=None, decisions=gbdt_decisions, reasons=gbdt_reasons,
            stage_input_count=len(gbdt_pool), stage_output_count=len(rescue_gbdt),
        )


_GBDT_SELECTION_MODE_DISPATCH: Dict[str, Callable[..., FeatureSelectionResult]] = {
    "standalone": run_gbdt_importance_selection,
    "chain": run_gbdt_importance_selection,
    "chi_squared_rescue": run_chi_squared_rescue_selection,
}


def resolve_gbdt_selection_mode(mode: Optional[str]) -> Callable[..., FeatureSelectionResult]:
    selector = _GBDT_SELECTION_MODE_DISPATCH.get(mode) if mode else None
    if selector is None:
        valid = sorted(_GBDT_SELECTION_MODE_DISPATCH)
        raise ValueError(
            f"GBDT_SELECTION_MODE={mode!r} is not recognized. Valid modes: {valid}"
        )
    return selector
