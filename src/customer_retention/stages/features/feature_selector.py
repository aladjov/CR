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
    LGBM_IMPORTANCE = "LGBM_IMPORTANCE"


@dataclass
class FeatureSelectionResult:
    df: DataFrame
    selected_features: List[str]
    dropped_features: List[str]
    drop_reasons: Dict[str, str]
    method_used: SelectionMethod
    importance_scores: Optional[Dict[str, float]] = None


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
    def __init__(self, method: SelectionMethod = SelectionMethod.VARIANCE, variance_threshold: float = 0.01, correlation_threshold: float = 0.95, target_column: Optional[str] = None, preserve_features: Optional[List[str]] = None, max_features: Optional[int] = None, apply_correlation_filter: bool = False, precomputed_corr_matrix: Optional[Any] = None, l1_C: float = 1.0, l1_ratio: float = 1.0, progress_fn: Optional[Callable[[str], None]] = None, precomputed_variances: Optional[Any] = None, precomputed_medians: Optional[Dict[str, float]] = None, precomputed_non_null: Optional[Dict[str, int]] = None, correlation_candidates: Optional[List[str]] = None, chi_squared_num_buckets: int = 10, lgbm_num_iterations: int = 200, lgbm_num_leaves: int = 63):
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
        self.lgbm_num_iterations = lgbm_num_iterations
        self.lgbm_num_leaves = lgbm_num_leaves

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
        elif self.method == SelectionMethod.LGBM_IMPORTANCE:
            self._apply_lgbm_importance_selection(df, feature_cols, numeric_set)

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
        if not candidates:
            return
        variances = self._get_variances(df, candidates, numeric_set)
        for feature in candidates:
            variance = variances[feature]
            if isna(variance) or variance < self.variance_threshold:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = f"low variance ({variance:.6f})"

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

        to_drop = set()
        for column in upper.columns:
            correlated = upper.index[upper[column] > self.correlation_threshold].tolist()
            for corr_feature in correlated:
                if corr_feature in self.preserve_features:
                    if column not in self.preserve_features:
                        to_drop.add(column)
                elif column in self.preserve_features:
                    to_drop.add(corr_feature)
                else:
                    if variances[column] >= variances[corr_feature]:
                        to_drop.add(corr_feature)
                    else:
                        to_drop.add(column)

        for feature in to_drop:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = f"high correlation (> {self.correlation_threshold})"

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
            for feature in dropped:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = reasons[feature]
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
        zero_features = [numeric_features[i] for i in range(len(numeric_features)) if max_coefs[i] == 0.0]
        if len(zero_features) == len(numeric_features):
            return
        for feature in zero_features:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = "L1 zero coefficient"

    def _apply_chi_squared_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for CHI_SQUARED")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        candidates = [f for f in features if f not in self.preserve_features and f in numeric_set]
        if not candidates or (self.max_features and self.max_features >= len(candidates)):
            return

        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            from customer_retention.core.compat import as_spark_df
            spark_df = as_spark_df(df[candidates + [self.target_column]])
            dropped, reasons, scores = _spark_chi_squared_selection(
                spark_df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                num_buckets=self.chi_squared_num_buckets,
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

    def _apply_lgbm_importance_selection(self, df: DataFrame, features: List[str], numeric_set: set) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for LGBM_IMPORTANCE")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        candidates = [f for f in features if f not in self.preserve_features and f in numeric_set]
        if not candidates or (self.max_features and self.max_features >= len(candidates)):
            return

        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            from customer_retention.core.compat import as_spark_df
            spark_df = as_spark_df(df[candidates + [self.target_column]])
            dropped, reasons, scores = _spark_lgbm_importance_selection(
                spark_df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                num_iterations=self.lgbm_num_iterations,
                num_leaves=self.lgbm_num_leaves,
            )
        else:
            dropped, reasons, scores = _local_lgbm_importance_selection(
                df, self.target_column, candidates,
                num_top_features=self.max_features or len(candidates),
                num_iterations=self.lgbm_num_iterations,
                num_leaves=self.lgbm_num_leaves,
            )
        self.importance_scores = scores
        for feature in dropped:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = reasons[feature]

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

def _import_spark_chi_squared_ml():
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.feature import Bucketizer, ChiSqSelector, VectorAssembler
    return ChiSqSelector, VectorAssembler, Bucketizer, F


_CHI_BUCKET_BATCH = 200


def _spark_chi_squared_selection(
    spark_df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 1000, num_buckets: int = 10,
) -> tuple:
    ChiSqSelector, VectorAssembler, Bucketizer, F = _import_spark_chi_squared_ml()

    n_features = len(feature_columns)
    if num_top_features >= n_features:
        return [], {}, None

    work_df = spark_df.select(
        [F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c) for c in feature_columns]
        + [F.col(target_column).cast("double").alias(target_column)]
    )
    work_df = work_df.na.fill(0.0, subset=feature_columns)
    if n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)

    quantiles = [i / num_buckets for i in range(1, num_buckets)]
    all_splits: Dict[str, list] = {}
    for start in range(0, n_features, _CHI_BUCKET_BATCH):
        batch = feature_columns[start:start + _CHI_BUCKET_BATCH]
        exprs = [F.percentile_approx(c, quantiles).alias(f"__p_{i}")
                 for i, c in enumerate(batch)]
        row = work_df.agg(*exprs).head()
        for i, c in enumerate(batch):
            all_splits[c] = row[f"__p_{i}"]

    bucketed_names = []
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
        if start + _CHI_BUCKET_BATCH < n_features:
            work_df = work_df.localCheckpoint(eager=True)

    assembler = VectorAssembler(inputCols=bucketed_names, outputCol="__chi_vec__", handleInvalid="keep")
    assembled = assembler.transform(work_df).select("__chi_vec__", target_column)

    selector = ChiSqSelector(numTopFeatures=num_top_features, featuresCol="__chi_vec__",
                              outputCol="__chi_sel__", labelCol=target_column)
    model = selector.fit(assembled)
    selected_idx = set(model.selectedFeatures)
    del work_df, assembled

    dropped = [feature_columns[i] for i in range(n_features) if i not in selected_idx]
    reasons = {c: "chi_squared below threshold" for c in dropped}
    return dropped, reasons, None


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
# LightGBM importance-based selection
# ---------------------------------------------------------------------------

def _import_spark_lgbm_ml():
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.feature import VectorAssembler
    from synapse.ml.lightgbm import LightGBMClassifier
    return LightGBMClassifier, VectorAssembler, F


def _spark_lgbm_importance_selection(
    spark_df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 300, num_iterations: int = 200, num_leaves: int = 63,
) -> tuple:
    LGBMClassifier, VectorAssembler, F = _import_spark_lgbm_ml()

    n_features = len(feature_columns)
    if num_top_features >= n_features:
        scores = {c: 0.0 for c in feature_columns}
        return [], {}, scores

    work_df = spark_df.select(
        [F.col(c).cast("double").alias(c) for c in feature_columns]
        + [F.col(target_column).cast("double").alias(target_column)]
    )
    work_df = work_df.na.fill(0.0, subset=feature_columns)
    if n_features > _CHI_BUCKET_BATCH:
        work_df = work_df.localCheckpoint(eager=True)

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="__lgbm_vec__", handleInvalid="keep")
    assembled = assembler.transform(work_df).select("__lgbm_vec__", target_column)

    lgbm = LGBMClassifier(featuresCol="__lgbm_vec__", labelCol=target_column,
                           numLeaves=num_leaves, numIterations=num_iterations, learningRate=0.1)
    model = lgbm.fit(assembled)
    importances = model.getFeatureImportances("gain")
    del work_df, assembled

    scores: Dict[str, float] = {feature_columns[i]: float(importances[i]) for i in range(n_features)}
    top_indices = set(np.argsort(importances)[-num_top_features:])
    dropped = [feature_columns[i] for i in range(n_features) if i not in top_indices]
    reasons = {c: f"lgbm_importance below top-{num_top_features}" for c in dropped}
    return dropped, reasons, scores


def _local_lgbm_importance_selection(
    df: Any, target_column: str, feature_columns: List[str],
    num_top_features: int = 300, num_iterations: int = 200, num_leaves: int = 63,
) -> tuple:
    import lightgbm as lgb

    n_features = len(feature_columns)
    if num_top_features >= n_features:
        return [], {}, {c: 0.0 for c in feature_columns}

    X = df[feature_columns].fillna(0).to_numpy()
    y = df[target_column].to_numpy()

    model = lgb.LGBMClassifier(n_estimators=num_iterations, num_leaves=num_leaves,
                                learning_rate=0.1, importance_type="gain", n_jobs=-1, verbose=-1)
    model.fit(X, y)
    importances = model.feature_importances_

    scores: Dict[str, float] = {feature_columns[i]: float(importances[i]) for i in range(n_features)}
    top_indices = set(np.argsort(importances)[-num_top_features:])
    dropped = [feature_columns[i] for i in range(n_features) if i not in top_indices]
    reasons = {c: f"lgbm_importance below top-{num_top_features}" for c in dropped}
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
) -> FeatureSelectionResult:
    log = progress_fn or (lambda msg: print(msg))
    t0 = time.monotonic()

    exclude = {target_column}
    if temporal_column:
        exclude.add(temporal_column)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in exclude]
    work_df = df[feature_columns + [target_column]]
    if temporal_column and temporal_column in df.columns:
        work_df = work_df.drop(columns=[temporal_column], errors="ignore")

    n_initial = len(feature_columns)
    log(f"  Chi-squared selection: {n_initial} features, selecting top {max_features}...")

    selector = FeatureSelector(
        method=SelectionMethod.CHI_SQUARED, target_column=target_column,
        max_features=max_features, chi_squared_num_buckets=num_buckets,
    )
    result = selector.fit_transform(work_df)
    elapsed = time.monotonic() - t0
    log(f"    {_format_elapsed(elapsed)} — dropped {len(result.dropped_features)}, "
        f"remaining {len(result.selected_features)}")
    return result


def run_lgbm_importance_selection(
    df: DataFrame, target_column: str, max_features: int = 300,
    num_iterations: int = 200, num_leaves: int = 63,
    feature_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> FeatureSelectionResult:
    log = progress_fn or (lambda msg: print(msg))
    t0 = time.monotonic()

    exclude = {target_column}
    if temporal_column:
        exclude.add(temporal_column)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in exclude]
    work_df = df[feature_columns + [target_column]]
    if temporal_column and temporal_column in df.columns:
        work_df = work_df.drop(columns=[temporal_column], errors="ignore")

    n_initial = len(feature_columns)
    log(f"  LightGBM importance selection: {n_initial} features, selecting top {max_features}...")

    selector = FeatureSelector(
        method=SelectionMethod.LGBM_IMPORTANCE, target_column=target_column,
        max_features=max_features, lgbm_num_iterations=num_iterations,
        lgbm_num_leaves=num_leaves,
    )
    result = selector.fit_transform(work_df)
    elapsed = time.monotonic() - t0
    log(f"    {_format_elapsed(elapsed)} — dropped {len(result.dropped_features)}, "
        f"remaining {len(result.selected_features)}")
    return result
