"""Cross-validation strategies for model evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from customer_retention.core.compat import DataFrame, Series, _is_spark_pandas, to_pandas


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
                purged_train = train_idx[keep_mask.to_numpy()]
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
        if hasattr(model, "clone") and self.strategy == CVStrategy.TEMPORAL_ENTITY:
            return self._run_distributed(model, X, y, groups=groups, temporal_values=temporal_values)

        X, y = to_pandas(X), to_pandas(y)
        if groups is not None:
            groups = to_pandas(groups)
        if temporal_values is not None:
            temporal_values = to_pandas(temporal_values)
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

    def _run_distributed(
        self,
        model,
        X: DataFrame,
        y: Series,
        groups: Optional[Series] = None,
        temporal_values: Optional[Series] = None,
    ) -> CVResult:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if _is_spark_pandas(X):
            X = X.reset_index(drop=True)
        if _is_spark_pandas(y):
            y = y.reset_index(drop=True)

        groups_pd = to_pandas(groups) if groups is not None else None
        temporal_pd = to_pandas(temporal_values) if temporal_values is not None else None

        splitter = TemporalEntitySplit(
            n_splits=self.n_splits,
            temporal_values=temporal_pd,
            purge_gap_days=self.purge_gap_days,
        )

        dummy_X = np.zeros(len(X))
        scores = []
        fold_details = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy_X, groups=groups_pd)):
            fold_model = model.clone()
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_test = X.iloc[test_idx]

            fold_model.fit(X_fold_train, y_fold_train)
            y_proba = fold_model.predict_proba(X_fold_test)

            y_fold_test = to_pandas(y.iloc[test_idx])

            if self.scoring == "roc_auc":
                score = roc_auc_score(y_fold_test, y_proba[:, 1])
            elif self.scoring in ("average_precision", "pr_auc"):
                score = average_precision_score(y_fold_test, y_proba[:, 1])
            else:
                score = roc_auc_score(y_fold_test, y_proba[:, 1])

            scores.append(score)

            detail: Dict[str, Any] = {
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": float(y.iloc[train_idx].mean()),
                "score": score,
            }
            if groups_pd is not None:
                detail["train_entities"] = int(groups_pd.iloc[train_idx].nunique())
                detail["test_entities"] = int(groups_pd.iloc[test_idx].nunique())
            fold_details.append(detail)

        cv_scores = np.array(scores)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        return CVResult(
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            fold_details=fold_details,
            scoring=self.scoring,
            is_stable=bool(cv_std <= self.stability_threshold),
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
