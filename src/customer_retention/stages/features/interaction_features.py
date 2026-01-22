"""
Interaction feature generation for customer retention analysis.

This module provides feature combinations and ratio calculations
from existing features.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series


@dataclass
class InteractionFeatureResult:
    """Result of interaction feature generation."""
    df: DataFrame
    generated_features: List[str]
    skipped_combinations: List[str] = field(default_factory=list)


class InteractionFeatureGenerator:
    """
    Generates interaction features from combinations of existing features.

    Interaction features are derived by combining two or more features
    using mathematical operations (multiply, divide, add, subtract).

    Parameters
    ----------
    combinations : List[Tuple[str, str, str, str]], optional
        List of feature combinations to create.
        Each tuple contains (col1, col2, output_name, operation).
        Supported operations: "multiply", "divide", "add", "subtract"
    ratios : List[Tuple[str, str, str]], optional
        List of ratio features to create.
        Each tuple contains (numerator, denominator, output_name).

    Attributes
    ----------
    generated_features : List[str]
        Names of features generated during last transform.
    """

    def __init__(
        self,
        combinations: Optional[List[Tuple[str, str, str, str]]] = None,
        ratios: Optional[List[Tuple[str, str, str]]] = None,
    ):
        self.combinations = combinations or []
        self.ratios = ratios or []
        self.generated_features: List[str] = []
        self._is_fitted = False

    def fit(self, df: DataFrame) -> "InteractionFeatureGenerator":
        """
        Fit the generator (validates columns exist).

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        self
        """
        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Generate interaction features for the input DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with interaction features added.
        """
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        result = df.copy()
        self.generated_features = []

        # Process combinations
        for combo in self.combinations:
            col1, col2, output_name, operation = combo
            if col1 in df.columns and col2 in df.columns:
                result[output_name] = self._apply_operation(
                    df[col1], df[col2], operation
                )
                self.generated_features.append(output_name)

        # Process ratios
        for ratio in self.ratios:
            numerator, denominator, output_name = ratio
            if numerator in df.columns and denominator in df.columns:
                result[output_name] = self._safe_divide(
                    df[numerator], df[denominator]
                )
                self.generated_features.append(output_name)

        return result

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with interaction features added.
        """
        self.fit(df)
        return self.transform(df)

    def _apply_operation(
        self,
        col1: Series,
        col2: Series,
        operation: str
    ) -> Series:
        """Apply the specified operation to two columns."""
        if operation == "multiply":
            return col1 * col2
        elif operation == "divide":
            return self._safe_divide(col1, col2)
        elif operation == "add":
            return col1 + col2
        elif operation == "subtract":
            return col1 - col2
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _safe_divide(
        self,
        numerator: Series,
        denominator: Series
    ) -> Series:
        """
        Safely divide two series, handling division by zero.

        Returns NaN where denominator is zero or null.
        """
        # Replace zeros with NaN to avoid inf
        safe_denominator = denominator.replace(0, np.nan)
        return numerator / safe_denominator
