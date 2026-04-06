from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from customer_retention.core.components.enums import ModelType

from .baseline_trainer import BaselineTrainer, TrainedModel
from .spark_classifier_wrapper import SparkClassifierWrapper

_LR_PARAMS: Dict[str, Any] = {
    # maxIter caps the worst case; tol gives L-BFGS an early-termination
    # criterion so it stops as soon as the parameter delta drops below 1e-4.
    # The default tol=1e-6 is too strict for class_weight='balanced' on
    # imbalanced data — L-BFGS oscillates near optimum and runs all 1000
    # iterations, paying per-task scheduling overhead × N partitions × 1000
    # on shared/Spark-Connect clusters.
    "maxIter": 1000,
    "tol": 1e-4,
}

_SPARK_MODEL_MAP: Dict[ModelType, tuple[str, Dict[str, Any]]] = {
    ModelType.LOGISTIC_REGRESSION: (
        "LogisticRegression",
        dict(_LR_PARAMS),
    ),
    ModelType.RANDOM_FOREST: (
        "RandomForestClassifier",
        {"numTrees": 100, "maxDepth": 10, "minInstancesPerNode": 2},
    ),
    ModelType.XGBOOST: (
        "GBTClassifier",
        {"maxIter": 100, "maxDepth": 6},
    ),
}


class SparkBaselineTrainer(BaselineTrainer):

    def __init__(
        self,
        model_type: ModelType,
        feature_names: List[str],
        model_params: Optional[Dict[str, Any]] = None,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            model_type=model_type,
            model_params=model_params,
            class_weight=class_weight,
            random_state=random_state,
            verbose=verbose,
        )
        self.feature_names = list(feature_names)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> TrainedModel:
        start_time = time.time()
        model = self._create_distributed_model()
        model.fit(X, y)
        training_time = time.time() - start_time

        return TrainedModel(
            model=model,
            model_type=self.model_type,
            hyperparameters=model.spark_model_params,
            training_time=training_time,
            feature_names=self.feature_names,
            class_weight=self.class_weight,
        )

    def _get_spark_model_config(self) -> tuple[str, Dict[str, Any]]:
        entry = _SPARK_MODEL_MAP.get(self.model_type)
        if entry is None:
            raise ValueError(
                f"No spark.ml mapping for {self.model_type}. "
                f"Supported: {list(_SPARK_MODEL_MAP.keys())}"
            )
        spark_class, default_params = entry
        params = dict(default_params)
        if self.model_params:
            params.update(self.model_params)
        return spark_class, params

    def _create_distributed_model(self) -> SparkClassifierWrapper:
        spark_class, params = self._get_spark_model_config()
        return SparkClassifierWrapper(
            spark_model_class=spark_class,
            spark_model_params=params,
            feature_names=self.feature_names,
            class_weight=self.class_weight,
        )


def create_distributed_models(
    feature_names: List[str],
    class_weight: Optional[str] = "balanced",
) -> Dict[str, SparkClassifierWrapper]:
    return {
        "Logistic Regression": SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params=dict(_LR_PARAMS),
            feature_names=feature_names,
            class_weight=class_weight,
        ),
        "Random Forest": SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={"numTrees": 100, "maxDepth": 10, "minInstancesPerNode": 2},
            feature_names=feature_names,
            class_weight=class_weight,
        ),
        "Gradient Boosting": SparkClassifierWrapper(
            spark_model_class="GBTClassifier",
            spark_model_params={"maxIter": 100, "maxDepth": 6},
            feature_names=feature_names,
            class_weight=class_weight,
        ),
    }
