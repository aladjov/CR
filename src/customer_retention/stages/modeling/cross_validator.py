"""Cross-validation strategies for model evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from customer_retention.core.compat import DataFrame, Series


class CVStrategy(Enum):
    STRATIFIED_KFOLD = "stratified_kfold"
    REPEATED_STRATIFIED = "repeated_stratified"
    TIME_SERIES = "time_series"
    GROUP_KFOLD = "group_kfold"
    TEMPORAL_ENTITY = "temporal_entity"


class TemporalEntitySplit:
    def __init__(self, n_splits: int = 5, temporal_values: Optional[Series] = None, purge_gap_days: int = 0):
        self.n_splits = n_splits
        self.temporal_values = temporal_values
        self.purge_gap_days = purge_gap_days
        self._inner = GroupKFold(n_splits=n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        import pandas as pd

        for train_idx, test_idx in self._inner.split(X, y, groups):
            if self.purge_gap_days > 0 and self.temporal_values is not None:
                test_dates = self.temporal_values.iloc[test_idx]
                test_min = pd.Timestamp(test_dates.min())
                purge_cutoff = test_min - pd.Timedelta(days=self.purge_gap_days)
                train_dates = self.temporal_values.iloc[train_idx]
                keep_mask = train_dates < purge_cutoff
                purged_train = train_idx[keep_mask.values]
                if len(purged_train) > len(train_idx) * 0.1:
                    train_idx = purged_train
            yield train_idx, test_idx


@dataclass
class CVResult:
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    fold_details: List[Dict[str, Any]]
    scoring: str
    is_stable: bool


class CrossValidator:
    def __init__(
        self,
        strategy: CVStrategy = CVStrategy.STRATIFIED_KFOLD,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = True,
        random_state: int = 42,
        scoring: str = "average_precision",
        stability_threshold: float = 0.10,
        purge_gap_days: int = 0,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.shuffle = shuffle
        self.random_state = random_state
        self.scoring = scoring
        self.stability_threshold = stability_threshold
        self.purge_gap_days = purge_gap_days

    def run(
        self,
        model,
        X: DataFrame,
        y: Series,
        groups: Optional[Series] = None,
        temporal_values: Optional[Series] = None,
    ) -> CVResult:
        cv_splitter = self._create_cv_splitter(groups, temporal_values)
        fold_details = []

        if self.strategy == CVStrategy.TEMPORAL_ENTITY:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=self.scoring, groups=groups)
            fold_details = self._collect_temporal_entity_fold_details(X, y, groups, cv_splitter)
        elif self.strategy == CVStrategy.GROUP_KFOLD:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=self.scoring, groups=groups)
            fold_details = self._collect_fold_details_with_groups(X, y, groups, cv_splitter)
        else:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=self.scoring)
            fold_details = self._collect_fold_details(X, y, cv_splitter)

        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        is_stable = bool(cv_std <= self.stability_threshold)

        return CVResult(
            cv_scores=scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            fold_details=fold_details,
            scoring=self.scoring,
            is_stable=is_stable,
        )

    def _create_cv_splitter(self, groups: Optional[Series] = None, temporal_values: Optional[Series] = None):
        if self.strategy == CVStrategy.STRATIFIED_KFOLD:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        if self.strategy == CVStrategy.REPEATED_STRATIFIED:
            return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)

        if self.strategy == CVStrategy.GROUP_KFOLD:
            return GroupKFold(n_splits=self.n_splits)

        if self.strategy == CVStrategy.TEMPORAL_ENTITY:
            return TemporalEntitySplit(
                n_splits=self.n_splits,
                temporal_values=temporal_values,
                purge_gap_days=self.purge_gap_days,
            )

        if self.strategy == CVStrategy.TIME_SERIES:
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=self.n_splits)

        return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def _collect_fold_details(self, X: DataFrame, y: Series, cv_splitter) -> List[Dict[str, Any]]:
        fold_details = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            y_train = y.iloc[train_idx]
            fold_details.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": y_train.mean(),
                "score": None,
            })
        return fold_details

    def _collect_fold_details_with_groups(
        self,
        X: DataFrame,
        y: Series,
        groups: Series,
        cv_splitter,
    ) -> List[Dict[str, Any]]:
        fold_details = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            y_train = y.iloc[train_idx]
            fold_details.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": y_train.mean(),
                "score": None,
            })
        return fold_details

    def _collect_temporal_entity_fold_details(
        self,
        X: DataFrame,
        y: Series,
        groups: Series,
        cv_splitter,
    ) -> List[Dict[str, Any]]:
        fold_details = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            y_train = y.iloc[train_idx]
            train_entities = groups.iloc[train_idx].nunique()
            test_entities = groups.iloc[test_idx].nunique()
            fold_details.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": y_train.mean(),
                "train_entities": train_entities,
                "test_entities": test_entities,
                "score": None,
            })
        return fold_details
