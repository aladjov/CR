"""Baseline model training for customer retention prediction."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import ModelType
import numpy as np


@dataclass
class TrainingConfig:
    random_state: int = 42
    verbose: bool = False
    n_jobs: int = -1


@dataclass
class TrainedModel:
    model: Any
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    training_time: float
    feature_names: List[str]
    class_weight: Optional[Any] = None


class BaselineTrainer:
    DEFAULT_PARAMS = {
        ModelType.LOGISTIC_REGRESSION: {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        ModelType.RANDOM_FOREST: {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "n_jobs": -1,
        },
        ModelType.XGBOOST: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        },
        ModelType.LIGHTGBM: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
        },
    }

    def __init__(
        self,
        model_type: ModelType,
        model_params: Optional[Dict[str, Any]] = None,
        class_weight: Optional[Any] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: DataFrame,
        y: Series,
        X_val: Optional[DataFrame] = None,
        y_val: Optional[Series] = None,
    ) -> TrainedModel:
        start_time = time.time()
        params = self._build_params()
        model = self._create_model(params)

        if self.model_type == ModelType.XGBOOST and X_val is not None:
            early_stopping = params.pop("early_stopping_rounds", None)
            if early_stopping:
                model.set_params(early_stopping_rounds=early_stopping)
                model.fit(X, y, eval_set=[(X_val, y_val)], verbose=self.verbose)
            else:
                model.fit(X, y)
        else:
            model.fit(X, y)

        training_time = time.time() - start_time

        return TrainedModel(
            model=model,
            model_type=self.model_type,
            hyperparameters=self._get_final_params(model),
            training_time=training_time,
            feature_names=list(X.columns),
            class_weight=self.class_weight,
        )

    def _build_params(self) -> Dict[str, Any]:
        defaults = self.DEFAULT_PARAMS.get(self.model_type, {}).copy()
        defaults.update(self.model_params)
        defaults["random_state"] = self.random_state
        return defaults

    def _create_model(self, params: Dict[str, Any]):
        if self.model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            if self.class_weight:
                params["class_weight"] = self.class_weight
            return LogisticRegression(**params)

        if self.model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            if self.class_weight:
                params["class_weight"] = self.class_weight
            return RandomForestClassifier(**params)

        if self.model_type == ModelType.XGBOOST:
            from xgboost import XGBClassifier
            params.pop("class_weight", None)
            return XGBClassifier(**params, verbosity=0 if not self.verbose else 1)

        if self.model_type == ModelType.LIGHTGBM:
            from lightgbm import LGBMClassifier
            if self.class_weight:
                params["class_weight"] = self.class_weight
            return LGBMClassifier(**params, verbosity=-1 if not self.verbose else 1)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _get_final_params(self, model) -> Dict[str, Any]:
        if hasattr(model, "get_params"):
            return model.get_params()
        return self.model_params
