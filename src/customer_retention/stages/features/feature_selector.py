from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, batched_corr_matrix, is_numeric_dtype, isna

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata


class SelectionMethod(Enum):
    VARIANCE = "VARIANCE"
    CORRELATION = "CORRELATION"
    MUTUAL_INFO = "MUTUAL_INFO"
    IMPORTANCE = "IMPORTANCE"
    RECURSIVE = "RECURSIVE"
    L1_SELECTION = "L1_SELECTION"


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
    def __init__(self, method: SelectionMethod = SelectionMethod.VARIANCE, variance_threshold: float = 0.01, correlation_threshold: float = 0.95, target_column: Optional[str] = None, preserve_features: Optional[List[str]] = None, max_features: Optional[int] = None, apply_correlation_filter: bool = False):
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_column = target_column
        self.preserve_features = preserve_features or []
        self.max_features = max_features
        self.apply_correlation_filter = apply_correlation_filter

        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.drop_reasons: Dict[str, str] = {}
        self.importance_scores: Optional[Dict[str, float]] = None
        self._is_fitted = False

    def fit(self, df: DataFrame) -> "FeatureSelector":
        feature_cols = [c for c in df.columns if c != self.target_column]

        self.selected_features = feature_cols.copy()
        self.dropped_features = []
        self.drop_reasons = {}

        if self.method == SelectionMethod.VARIANCE:
            self._apply_variance_selection(df, feature_cols)
        elif self.method == SelectionMethod.CORRELATION:
            self._apply_correlation_selection(df, feature_cols)
        elif self.method == SelectionMethod.L1_SELECTION:
            self._apply_l1_selection(df, feature_cols)

        if self.apply_correlation_filter and self.method != SelectionMethod.CORRELATION:
            self._apply_correlation_selection(df, self.selected_features.copy())

        if self.max_features and len(self.selected_features) > self.max_features:
            feature_df = df[self.selected_features]
            variances = feature_df.var().sort_values(ascending=False)
            to_keep = variances.head(self.max_features).index.tolist()
            to_drop = [f for f in self.selected_features if f not in to_keep]
            for feature in to_drop:
                if feature not in self.preserve_features:
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

    def _apply_variance_selection(self, df: DataFrame, features: List[str]) -> None:
        numeric_features = [
            f for f in features
            if f not in self.preserve_features and is_numeric_dtype(df[f])
        ]
        if not numeric_features:
            return

        variances = df[numeric_features].var()
        for feature in numeric_features:
            variance = variances[feature]
            if isna(variance) or variance < self.variance_threshold:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = f"low variance ({variance:.6f})"

    def _apply_correlation_selection(self, df: DataFrame, features: List[str]) -> None:
        numeric_features = [f for f in features if f in df.columns and is_numeric_dtype(df[f]) and f in self.selected_features]

        if len(numeric_features) < 2:
            return

        corr_matrix = batched_corr_matrix(df, numeric_features).abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

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
                    var1 = df[column].var()
                    var2 = df[corr_feature].var()
                    if var1 >= var2:
                        to_drop.add(corr_feature)
                    else:
                        to_drop.add(column)

        for feature in to_drop:
            if feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = f"high correlation (> {self.correlation_threshold})"

    def _apply_l1_selection(self, df: DataFrame, features: List[str]) -> None:
        if not self.target_column:
            raise ValueError("target_column is required for L1_SELECTION")
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not in DataFrame")

        numeric_features = [
            f for f in features
            if f not in self.preserve_features and f in df.columns and is_numeric_dtype(df[f])
        ]
        if not numeric_features:
            return

        from customer_retention.core.compat import _is_spark_pandas
        work_df = df[numeric_features + [self.target_column]]
        if _is_spark_pandas(work_df):
            work_df = work_df.to_pandas()

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
        n_classes = len(np.unique(y))
        if n_classes > 2:
            model = LogisticRegression(solver="saga", C=1.0, max_iter=2000, l1_ratio=1.0, penalty="elasticnet")
        else:
            model = LogisticRegression(solver="saga", C=1.0, max_iter=2000, l1_ratio=1.0)
        model.fit(X_scaled, y)

        coefs = np.abs(model.coef_)
        if coefs.ndim == 2:
            max_coefs = coefs.max(axis=0)
        else:
            max_coefs = coefs

        self.importance_scores = {numeric_features[i]: float(max_coefs[i]) for i in range(len(numeric_features))}
        for i, feature in enumerate(numeric_features):
            if max_coefs[i] == 0.0 and feature in self.selected_features:
                self.selected_features.remove(feature)
                self.dropped_features.append(feature)
                self.drop_reasons[feature] = "L1 zero coefficient"

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


def run_selection_pipeline(
    df: DataFrame, target_column: str,
    variance_threshold: float = 0.01, correlation_threshold: float = 0.95,
    l1_enabled: bool = False, max_features: Optional[int] = None,
    preserve_features: Optional[List[str]] = None,
) -> FeatureSelectionResult:
    all_dropped: List[str] = []
    all_reasons: Dict[str, str] = {}
    importance_scores: Optional[Dict[str, float]] = None
    current_df = df
    last_method = SelectionMethod.VARIANCE

    result_var = FeatureSelector(
        method=SelectionMethod.VARIANCE, variance_threshold=variance_threshold,
        target_column=target_column, preserve_features=preserve_features,
    ).fit_transform(current_df)
    current_df = result_var.df
    all_dropped.extend(result_var.dropped_features)
    all_reasons.update(result_var.drop_reasons)

    last_method = SelectionMethod.CORRELATION
    result_corr = FeatureSelector(
        method=SelectionMethod.CORRELATION, correlation_threshold=correlation_threshold,
        target_column=target_column, preserve_features=preserve_features,
    ).fit_transform(current_df)
    current_df = result_corr.df
    all_dropped.extend(result_corr.dropped_features)
    all_reasons.update(result_corr.drop_reasons)

    if l1_enabled:
        last_method = SelectionMethod.L1_SELECTION
        result_l1 = FeatureSelector(
            method=SelectionMethod.L1_SELECTION, target_column=target_column,
            preserve_features=preserve_features, max_features=max_features,
        ).fit_transform(current_df)
        current_df = result_l1.df
        all_dropped.extend(result_l1.dropped_features)
        all_reasons.update(result_l1.drop_reasons)
        importance_scores = result_l1.importance_scores

    selected = [c for c in current_df.columns if c != target_column]
    return FeatureSelectionResult(
        df=current_df, selected_features=selected, dropped_features=all_dropped,
        drop_reasons=all_reasons, method_used=last_method, importance_scores=importance_scores,
    )
