from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, is_numeric_dtype, pd

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
        )

    def fit_transform(self, df: DataFrame) -> FeatureSelectionResult:
        self.fit(df)
        return self.transform(df)

    def _apply_variance_selection(self, df: DataFrame, features: List[str]) -> None:
        for feature in features:
            if feature in self.preserve_features:
                continue

            series = df[feature]
            if not is_numeric_dtype(series):
                continue

            variance = series.var()
            if pd.isna(variance) or variance < self.variance_threshold:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = f"low variance ({variance:.6f})"

    def _apply_correlation_selection(self, df: DataFrame, features: List[str]) -> None:
        numeric_features = [f for f in features if f in df.columns and is_numeric_dtype(df[f]) and f in self.selected_features]

        if len(numeric_features) < 2:
            return

        corr_matrix = df[numeric_features].corr().abs()

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
