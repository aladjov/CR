"""
Feature selection for customer retention analysis.

This module provides feature selection methods including variance,
correlation, and other filtering techniques.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series, is_numeric_dtype


class SelectionMethod(Enum):
    """Feature selection methods."""
    VARIANCE = "VARIANCE"
    CORRELATION = "CORRELATION"
    MUTUAL_INFO = "MUTUAL_INFO"
    IMPORTANCE = "IMPORTANCE"
    RECURSIVE = "RECURSIVE"
    L1_SELECTION = "L1_SELECTION"


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    df: DataFrame
    selected_features: List[str]
    dropped_features: List[str]
    drop_reasons: Dict[str, str]
    method_used: SelectionMethod
    importance_scores: Optional[Dict[str, float]] = None


class FeatureSelector:
    """
    Feature selection based on various statistical criteria.

    Parameters
    ----------
    method : SelectionMethod
        Primary selection method to use.
    variance_threshold : float, default 0.01
        Minimum variance to keep a feature (for VARIANCE method).
    correlation_threshold : float, default 0.95
        Maximum correlation allowed between features.
    target_column : str, optional
        Name of target column (excluded from selection).
    preserve_features : List[str], optional
        Features to never drop.
    max_features : int, optional
        Maximum number of features to select.
    apply_correlation_filter : bool, default False
        Whether to also apply correlation filtering after primary method.

    Attributes
    ----------
    selected_features : List[str]
        Features selected after fitting.
    dropped_features : List[str]
        Features dropped during selection.
    """

    def __init__(
        self,
        method: SelectionMethod = SelectionMethod.VARIANCE,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        target_column: Optional[str] = None,
        preserve_features: Optional[List[str]] = None,
        max_features: Optional[int] = None,
        apply_correlation_filter: bool = False,
    ):
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
        """
        Fit the selector by determining which features to keep.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        self
        """
        # Get feature columns (exclude target)
        feature_cols = [c for c in df.columns if c != self.target_column]

        # Start with all features selected
        self.selected_features = feature_cols.copy()
        self.dropped_features = []
        self.drop_reasons = {}

        # Apply primary selection method
        if self.method == SelectionMethod.VARIANCE:
            self._apply_variance_selection(df, feature_cols)
        elif self.method == SelectionMethod.CORRELATION:
            self._apply_correlation_selection(df, feature_cols)

        # Apply correlation filter if requested
        if self.apply_correlation_filter and self.method != SelectionMethod.CORRELATION:
            remaining_features = [f for f in self.selected_features]
            self._apply_correlation_selection(df, remaining_features)

        # Apply max_features limit
        if self.max_features and len(self.selected_features) > self.max_features:
            # Sort by variance and keep top max_features
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
        """
        Apply feature selection to the DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        FeatureSelectionResult
            Result containing selected features.
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        # Select only the chosen features (and target if present)
        cols_to_keep = self.selected_features.copy()
        if self.target_column and self.target_column in df.columns:
            cols_to_keep.append(self.target_column)

        # Only keep columns that exist in df
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
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        FeatureSelectionResult
            Result containing selected features.
        """
        self.fit(df)
        return self.transform(df)

    def _apply_variance_selection(
        self,
        df: DataFrame,
        features: List[str]
    ) -> None:
        """Apply variance-based feature selection."""
        for feature in features:
            if feature in self.preserve_features:
                continue

            series = df[feature]
            # Skip non-numeric columns
            if not is_numeric_dtype(series):
                continue

            variance = series.var()
            if pd.isna(variance) or variance < self.variance_threshold:
                if feature in self.selected_features:
                    self.selected_features.remove(feature)
                    self.dropped_features.append(feature)
                    self.drop_reasons[feature] = f"low variance ({variance:.6f})"

    def _apply_correlation_selection(
        self,
        df: DataFrame,
        features: List[str]
    ) -> None:
        """Apply correlation-based feature selection."""
        # Get numeric features only
        numeric_features = [
            f for f in features
            if f in df.columns and is_numeric_dtype(df[f])
            and f in self.selected_features
        ]

        if len(numeric_features) < 2:
            return

        # Compute correlation matrix
        corr_matrix = df[numeric_features].corr().abs()

        # Find highly correlated pairs
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # For each highly correlated pair, drop one
        to_drop = set()
        for column in upper.columns:
            correlated = upper.index[upper[column] > self.correlation_threshold].tolist()
            for corr_feature in correlated:
                # Drop the one with lower variance (unless preserved)
                if corr_feature in self.preserve_features:
                    if column not in self.preserve_features:
                        to_drop.add(column)
                elif column in self.preserve_features:
                    to_drop.add(corr_feature)
                else:
                    # Drop the one with lower variance
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
