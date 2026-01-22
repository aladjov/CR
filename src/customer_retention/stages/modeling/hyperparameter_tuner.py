"""Hyperparameter tuning strategies for model optimization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from customer_retention.core.compat import pd, DataFrame, Series
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class SearchStrategy(Enum):
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    HALVING = "halving"


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    cv_results: List[Dict[str, Any]]
    scoring: str


class HyperparameterTuner:
    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.RANDOM_SEARCH,
        param_space: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "average_precision",
        n_jobs: int = -1,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.param_space = param_space or {}
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def tune(self, model, X: DataFrame, y: Series) -> TuningResult:
        search = self._create_search(model)
        search.fit(X, y)

        cv_results = self._extract_cv_results(search)

        return TuningResult(
            best_params=search.best_params_,
            best_score=search.best_score_,
            best_model=search.best_estimator_,
            cv_results=cv_results,
            scoring=self.scoring,
        )

    def _create_search(self, model):
        if self.strategy == SearchStrategy.GRID_SEARCH:
            return GridSearchCV(
                model,
                param_grid=self.param_space,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

        if self.strategy == SearchStrategy.HALVING:
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingRandomSearchCV
            return HalvingRandomSearchCV(
                model,
                param_distributions=self.param_space,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
            )

        return RandomizedSearchCV(
            model,
            param_distributions=self.param_space,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _extract_cv_results(self, search) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(search.cv_results_["mean_test_score"])):
            result = {
                "params": search.cv_results_["params"][i],
                "mean_score": search.cv_results_["mean_test_score"][i],
                "std_score": search.cv_results_["std_test_score"][i],
                "rank": search.cv_results_["rank_test_score"][i],
            }
            results.append(result)
        return results
