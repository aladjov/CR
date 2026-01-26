"""Class imbalance handling strategies for model training."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np

from customer_retention.core.compat import DataFrame, Series


class ImbalanceStrategy(Enum):
    CLASS_WEIGHT = "class_weight"
    SMOTE = "smote"
    RANDOM_OVERSAMPLE = "random_oversample"
    RANDOM_UNDERSAMPLE = "random_undersample"
    SMOTEENN = "smoteenn"
    ADASYN = "adasyn"
    NONE = "none"


class ClassWeightMethod(Enum):
    BALANCED = "balanced"
    CUSTOM = "custom"
    INVERSE = "inverse"


@dataclass
class ImbalanceResult:
    X_resampled: Optional[DataFrame]
    y_resampled: Optional[Series]
    strategy_used: ImbalanceStrategy
    original_class_counts: Dict[int, int]
    resampled_class_counts: Optional[Dict[int, int]] = None
    class_weights: Optional[Dict[int, float]] = None
    imbalance_ratio: Optional[float] = None


class ImbalanceHandler:
    def __init__(
        self,
        strategy: ImbalanceStrategy = ImbalanceStrategy.CLASS_WEIGHT,
        weight_method: ClassWeightMethod = ClassWeightMethod.BALANCED,
        custom_weights: Optional[Dict[int, float]] = None,
        sampling_strategy: Union[str, float] = "auto",
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.weight_method = weight_method
        self.custom_weights = custom_weights
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self._class_weights = None

    def fit(self, X: DataFrame, y: Series) -> ImbalanceResult:
        original_counts = y.value_counts().to_dict()
        imbalance_ratio = max(original_counts.values()) / min(original_counts.values())

        class_weights = None
        if self.strategy == ImbalanceStrategy.CLASS_WEIGHT:
            class_weights = self._compute_class_weights(y)

        return ImbalanceResult(
            X_resampled=None,
            y_resampled=None,
            strategy_used=self.strategy,
            original_class_counts=original_counts,
            resampled_class_counts=None,
            class_weights=class_weights,
            imbalance_ratio=imbalance_ratio,
        )

    def fit_transform(self, X: DataFrame, y: Series) -> ImbalanceResult:
        original_counts = y.value_counts().to_dict()
        imbalance_ratio = max(original_counts.values()) / min(original_counts.values())

        if self.strategy == ImbalanceStrategy.NONE:
            return ImbalanceResult(
                X_resampled=X,
                y_resampled=y,
                strategy_used=self.strategy,
                original_class_counts=original_counts,
                resampled_class_counts=original_counts,
                imbalance_ratio=imbalance_ratio,
            )

        if self.strategy == ImbalanceStrategy.CLASS_WEIGHT:
            return ImbalanceResult(
                X_resampled=X,
                y_resampled=y,
                strategy_used=self.strategy,
                original_class_counts=original_counts,
                resampled_class_counts=original_counts,
                class_weights=self._compute_class_weights(y),
                imbalance_ratio=imbalance_ratio,
            )

        X_res, y_res = self._resample(X, y)
        resampled_counts = Series(y_res).value_counts().to_dict()

        return ImbalanceResult(
            X_resampled=DataFrame(X_res, columns=X.columns),
            y_resampled=Series(y_res),
            strategy_used=self.strategy,
            original_class_counts=original_counts,
            resampled_class_counts=resampled_counts,
            imbalance_ratio=imbalance_ratio,
        )

    def _compute_class_weights(self, y: Series) -> Dict[int, float]:
        if self.weight_method == ClassWeightMethod.CUSTOM:
            return self.custom_weights

        classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(classes)

        if self.weight_method == ClassWeightMethod.BALANCED:
            weights = {}
            for cls in classes:
                n_cls = (y == cls).sum()
                weights[cls] = n_samples / (n_classes * n_cls)
            return weights

        if self.weight_method == ClassWeightMethod.INVERSE:
            weights = {}
            for cls in classes:
                proportion = (y == cls).sum() / n_samples
                weights[cls] = 1.0 / proportion
            return weights

        return {cls: 1.0 for cls in classes}

    def _resample(self, X: DataFrame, y: Series) -> tuple:
        if self.strategy == ImbalanceStrategy.SMOTE:
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return sampler.fit_resample(X, y)

        if self.strategy == ImbalanceStrategy.RANDOM_OVERSAMPLE:
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return sampler.fit_resample(X, y)

        if self.strategy == ImbalanceStrategy.RANDOM_UNDERSAMPLE:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return sampler.fit_resample(X, y)

        if self.strategy == ImbalanceStrategy.SMOTEENN:
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return sampler.fit_resample(X, y)

        if self.strategy == ImbalanceStrategy.ADASYN:
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return sampler.fit_resample(X, y)

        return X.values, y.values


@dataclass
class ImbalanceRecommendation:
    """Recommendation for handling class imbalance."""
    severity: str  # "low", "moderate", "high", "severe"
    ratio: float
    strategies: list
    primary_strategy: ImbalanceStrategy
    explanation: str

    def print_recommendation(self):
        icons = {"low": "🟢", "moderate": "🟡", "high": "🟠", "severe": "🔴"}
        print(f"\n{icons.get(self.severity, '⚪')} Class Imbalance: {self.severity.upper()} ({self.ratio:.1f}:1)")
        print(f"\n{self.explanation}")
        print("\nRecommended strategies (in order of preference):")
        for i, (strategy, desc) in enumerate(self.strategies, 1):
            marker = "→" if strategy == self.primary_strategy else " "
            print(f"  {marker} {i}. {strategy.value}: {desc}")


class ImbalanceRecommender:
    """Recommends imbalance handling strategies based on data characteristics."""

    THRESHOLDS = {"low": 3, "moderate": 10, "high": 20, "severe": float("inf")}

    STRATEGY_DESCRIPTIONS = {
        ImbalanceStrategy.CLASS_WEIGHT: "Adjust loss function weights (no data modification)",
        ImbalanceStrategy.SMOTE: "Generate synthetic minority samples using k-NN interpolation",
        ImbalanceStrategy.RANDOM_UNDERSAMPLE: "Randomly remove majority samples",
        ImbalanceStrategy.RANDOM_OVERSAMPLE: "Duplicate minority samples (risk of overfitting)",
        ImbalanceStrategy.SMOTEENN: "SMOTE + ENN cleaning (removes noisy samples)",
        ImbalanceStrategy.ADASYN: "Adaptive synthetic sampling (focuses on harder examples)",
    }

    def recommend(self, y: Series, n_samples: Optional[int] = None) -> ImbalanceRecommendation:
        """Recommend imbalance handling strategy based on class distribution."""
        counts = y.value_counts().to_dict()
        ratio = max(counts.values()) / min(counts.values())
        n_minority = min(counts.values())
        n_total = n_samples or len(y)

        severity = self._get_severity(ratio)
        strategies, primary, explanation = self._get_strategies(severity, ratio, n_minority, n_total)

        return ImbalanceRecommendation(
            severity=severity, ratio=ratio, strategies=strategies,
            primary_strategy=primary, explanation=explanation
        )

    def _get_severity(self, ratio: float) -> str:
        if ratio < self.THRESHOLDS["low"]:
            return "low"
        elif ratio < self.THRESHOLDS["moderate"]:
            return "moderate"
        elif ratio < self.THRESHOLDS["high"]:
            return "high"
        return "severe"

    def _get_strategies(self, severity: str, ratio: float, n_minority: int, n_total: int):
        strategies = []
        primary = ImbalanceStrategy.CLASS_WEIGHT
        explanation = ""

        if severity == "low":
            explanation = f"Ratio {ratio:.1f}:1 is manageable. Class weights are usually sufficient."
            strategies = [
                (ImbalanceStrategy.CLASS_WEIGHT, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.CLASS_WEIGHT]),
            ]
            primary = ImbalanceStrategy.CLASS_WEIGHT

        elif severity == "moderate":
            explanation = f"Ratio {ratio:.1f}:1 may affect model performance. Consider resampling if class weights aren't enough."
            strategies = [
                (ImbalanceStrategy.CLASS_WEIGHT, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.CLASS_WEIGHT]),
                (ImbalanceStrategy.SMOTE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTE]),
            ]
            primary = ImbalanceStrategy.CLASS_WEIGHT

        elif severity == "high":
            explanation = f"Ratio {ratio:.1f}:1 is significant. SMOTE recommended to create synthetic minority samples."
            if n_minority < 6:
                explanation += f"\n⚠️ Only {n_minority} minority samples - SMOTE needs k=5 neighbors minimum."
                strategies = [
                    (ImbalanceStrategy.RANDOM_OVERSAMPLE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.RANDOM_OVERSAMPLE]),
                    (ImbalanceStrategy.CLASS_WEIGHT, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.CLASS_WEIGHT]),
                ]
                primary = ImbalanceStrategy.RANDOM_OVERSAMPLE
            else:
                strategies = [
                    (ImbalanceStrategy.SMOTE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTE]),
                    (ImbalanceStrategy.SMOTEENN, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTEENN]),
                    (ImbalanceStrategy.CLASS_WEIGHT, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.CLASS_WEIGHT]),
                ]
                primary = ImbalanceStrategy.SMOTE

        else:  # severe
            explanation = f"Ratio {ratio:.1f}:1 is severe. Combination of techniques recommended."
            if n_total > 100000:
                explanation += f"\nDataset is large ({n_total:,} rows) - undersampling majority is viable."
                strategies = [
                    (ImbalanceStrategy.RANDOM_UNDERSAMPLE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.RANDOM_UNDERSAMPLE]),
                    (ImbalanceStrategy.SMOTE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTE]),
                    (ImbalanceStrategy.SMOTEENN, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTEENN]),
                ]
                primary = ImbalanceStrategy.RANDOM_UNDERSAMPLE
            elif n_minority < 6:
                explanation += f"\n⚠️ Only {n_minority} minority samples - limited options."
                strategies = [
                    (ImbalanceStrategy.RANDOM_OVERSAMPLE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.RANDOM_OVERSAMPLE]),
                    (ImbalanceStrategy.CLASS_WEIGHT, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.CLASS_WEIGHT]),
                ]
                primary = ImbalanceStrategy.RANDOM_OVERSAMPLE
            else:
                strategies = [
                    (ImbalanceStrategy.SMOTE, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTE]),
                    (ImbalanceStrategy.ADASYN, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.ADASYN]),
                    (ImbalanceStrategy.SMOTEENN, self.STRATEGY_DESCRIPTIONS[ImbalanceStrategy.SMOTEENN]),
                ]
                primary = ImbalanceStrategy.SMOTE

        return strategies, primary, explanation
