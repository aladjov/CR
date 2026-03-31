"""Cross-validation strategies for model evaluation."""

import gc
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from customer_retention.core.compat import DataFrame, Series, collect_for_sklearn, native_pd

_CV_ENTITY_COL = "__cv_entity__"
_CV_DATE_COL = "__cv_date__"


def _squeeze_to_series(obj: Any) -> Any:
    if isinstance(obj, native_pd.DataFrame) and obj.shape[1] == 1:
        return obj.iloc[:, 0]
    return obj


def _has_cv_metadata(X: Any) -> bool:
    return _CV_ENTITY_COL in X.columns and _CV_DATE_COL in X.columns


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
        for train_idx, test_idx in self._inner.split(X, y, groups):
            if self.purge_gap_days > 0 and self.temporal_values is not None:
                test_dates = self.temporal_values.iloc[test_idx]
                test_min = native_pd.Timestamp(test_dates.min())
                purge_cutoff = test_min - native_pd.Timedelta(days=self.purge_gap_days)
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

    @staticmethod
    def _extract_importance(model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_.tolist()))
        if hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_).flatten().tolist()))
        return None

    def run(
        self,
        model,
        X: DataFrame,
        y: Series,
        groups: Optional[Series] = None,
        temporal_values: Optional[Series] = None,
        on_fold_complete: Optional[Callable] = None,
        capture_fold_importance: bool = False,
        feature_names: Optional[List[str]] = None,
    ) -> CVResult:
        _capture = capture_fold_importance and feature_names is not None
        if hasattr(model, "clone") and self.strategy == CVStrategy.TEMPORAL_ENTITY:
            return self._run_distributed(
                model, X, y, groups=groups, temporal_values=temporal_values,
                on_fold_complete=on_fold_complete,
                capture_fold_importance=_capture, feature_names=feature_names,
            )

        X, y = collect_for_sklearn(X), collect_for_sklearn(y)
        if groups is not None:
            groups = _squeeze_to_series(collect_for_sklearn(groups))
        if temporal_values is not None:
            temporal_values = _squeeze_to_series(collect_for_sklearn(temporal_values))
        cv_splitter = self._create_cv_splitter(groups, temporal_values)
        fold_details = []

        if self.strategy == CVStrategy.TEMPORAL_ENTITY or on_fold_complete is not None or _capture:
            scores, fold_details = self._run_manual_cv(
                model, X, y, cv_splitter, groups=groups, on_fold_complete=on_fold_complete,
                capture_fold_importance=_capture, feature_names=feature_names,
            )
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
        on_fold_complete: Optional[Callable] = None,
        capture_fold_importance: bool = False,
        feature_names: Optional[List[str]] = None,
    ) -> CVResult:
        from customer_retention.core.compat import _is_spark_pandas

        groups_pd = _squeeze_to_series(collect_for_sklearn(groups)) if groups is not None else None
        temporal_pd = _squeeze_to_series(collect_for_sklearn(temporal_values)) if temporal_values is not None else None

        n_rows = len(groups_pd) if groups_pd is not None else len(y)
        splitter = TemporalEntitySplit(
            n_splits=self.n_splits,
            temporal_values=temporal_pd,
            purge_gap_days=self.purge_gap_days,
        )
        all_folds = list(splitter.split(np.zeros(n_rows), groups=groups_pd))

        if _is_spark_pandas(X) and _has_cv_metadata(X):
            return self._score_folds_spark(
                model, X, y, groups_pd, temporal_pd, all_folds,
                on_fold_complete=on_fold_complete,
                capture_fold_importance=capture_fold_importance,
                feature_names=feature_names,
            )

        X_pd = collect_for_sklearn(X)
        y_pd = _squeeze_to_series(collect_for_sklearn(y))
        return self._score_folds_pandas(
            model, X_pd, y_pd, groups_pd, all_folds,
            on_fold_complete=on_fold_complete,
            capture_fold_importance=capture_fold_importance,
            feature_names=feature_names,
        )

    def _score_fold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        if self.scoring == "roc_auc":
            return roc_auc_score(y_true, y_proba[:, 1])
        if self.scoring in ("average_precision", "pr_auc"):
            return average_precision_score(y_true, y_proba[:, 1])
        if self.scoring == "f1_weighted":
            y_pred = np.argmax(y_proba, axis=1)
            return f1_score(y_true, y_pred, average="weighted")
        return roc_auc_score(y_true, y_proba[:, 1])

    def _run_manual_cv(
        self,
        model,
        X,
        y,
        cv_splitter,
        groups: Optional[Series] = None,
        on_fold_complete: Optional[Callable] = None,
        capture_fold_importance: bool = False,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        from sklearn.base import clone

        scores = []
        fold_details = []
        total_folds = cv_splitter.get_n_splits(X, y, groups)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            t0 = time.monotonic()
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_proba = fold_model.predict_proba(X.iloc[test_idx])
            y_fold_test = y.iloc[test_idx].to_numpy()
            score = self._score_fold(y_fold_test, y_proba)
            elapsed = time.monotonic() - t0

            scores.append(score)
            detail: Dict[str, Any] = {
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": float(y.iloc[train_idx].mean()),
                "score": score,
                "elapsed_seconds": elapsed,
            }
            if groups is not None:
                detail["train_entities"] = int(groups.iloc[train_idx].nunique())
                detail["test_entities"] = int(groups.iloc[test_idx].nunique())
            if capture_fold_importance and feature_names:
                imp = self._extract_importance(fold_model, feature_names)
                if imp is not None:
                    detail["feature_importance"] = imp
            fold_details.append(detail)
            del fold_model, y_proba, y_fold_test
            gc.collect()

            if on_fold_complete is not None:
                on_fold_complete(detail, fold_idx + 1, total_folds)

        return np.array(scores), fold_details

    def _score_folds_pandas(self, model, X_pd, y_pd, groups_pd, all_folds,
                            on_fold_complete=None, capture_fold_importance=False,
                            feature_names=None) -> CVResult:
        scores = []
        fold_details = []
        total_folds = len(all_folds)

        for fold_idx, (train_idx, test_idx) in enumerate(all_folds):
            t0 = time.monotonic()
            fold_model = model.clone()
            fold_model.fit(X_pd.iloc[train_idx], y_pd.iloc[train_idx])
            y_proba = fold_model.predict_proba(X_pd.iloc[test_idx])
            y_fold_test = y_pd.iloc[test_idx].to_numpy()

            score = self._score_fold(y_fold_test, y_proba)
            elapsed = time.monotonic() - t0
            scores.append(score)

            detail: Dict[str, Any] = {
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_ratio": float(y_pd.iloc[train_idx].mean()),
                "score": score,
                "elapsed_seconds": elapsed,
            }
            if groups_pd is not None:
                detail["train_entities"] = int(groups_pd.iloc[train_idx].nunique())
                detail["test_entities"] = int(groups_pd.iloc[test_idx].nunique())
            if capture_fold_importance and feature_names:
                imp = self._extract_importance(fold_model, feature_names)
                if imp is not None:
                    detail["feature_importance"] = imp
            fold_details.append(detail)
            del fold_model, y_proba, y_fold_test
            gc.collect()

            if on_fold_complete is not None:
                on_fold_complete(detail, fold_idx + 1, total_folds)

        return self._build_cv_result(scores, fold_details)

    def _score_folds_spark(self, model, X, y, groups_pd, temporal_pd, all_folds,
                           on_fold_complete=None, capture_fold_importance=False,
                           feature_names=None) -> CVResult:
        from customer_retention.core.compat import concat as compat_concat

        combined = compat_concat([X, y.rename("__y__")], axis=1)
        feature_cols = [c for c in X.columns if not c.startswith("__cv_")]

        scores = []
        fold_details = []
        total_folds = len(all_folds)

        for fold_idx, (train_idx, test_idx) in enumerate(all_folds):
            t0 = time.monotonic()
            fold_model = model.clone()

            train_entities = groups_pd.iloc[train_idx].unique().tolist()
            test_entities = groups_pd.iloc[test_idx].unique().tolist()

            train_fold = combined[combined[_CV_ENTITY_COL].isin(train_entities)]
            test_fold = combined[combined[_CV_ENTITY_COL].isin(test_entities)]

            if self.purge_gap_days > 0 and temporal_pd is not None:
                test_dates = temporal_pd.iloc[test_idx]
                test_min = native_pd.Timestamp(test_dates.min())
                purge_cutoff = test_min - native_pd.Timedelta(days=self.purge_gap_days)
                train_purged = train_fold[train_fold[_CV_DATE_COL] < purge_cutoff]
                if len(train_purged) > len(train_fold) * 0.1:
                    train_fold = train_purged

            fold_model.fit(train_fold[feature_cols], train_fold["__y__"])
            y_proba = fold_model.predict_proba(test_fold[feature_cols])
            y_fold_test = collect_for_sklearn(test_fold["__y__"]).to_numpy()

            score = self._score_fold(y_fold_test, y_proba)
            elapsed = time.monotonic() - t0
            scores.append(score)

            detail: Dict[str, Any] = {
                "fold": fold_idx + 1,
                "train_size": int(len(train_fold)),
                "test_size": int(len(test_fold)),
                "train_class_ratio": float(train_fold["__y__"].mean()),
                "score": score,
                "elapsed_seconds": elapsed,
            }
            if groups_pd is not None:
                detail["train_entities"] = int(groups_pd.iloc[train_idx].nunique())
                detail["test_entities"] = int(groups_pd.iloc[test_idx].nunique())
            if capture_fold_importance:
                _fn = feature_names or feature_cols
                imp = self._extract_importance(fold_model, _fn)
                if imp is not None:
                    detail["feature_importance"] = imp
            fold_details.append(detail)

            del fold_model, train_fold, test_fold, y_proba, y_fold_test
            gc.collect()

            if on_fold_complete is not None:
                on_fold_complete(detail, fold_idx + 1, total_folds)

        return self._build_cv_result(scores, fold_details)

    def _build_cv_result(self, scores: list, fold_details: list) -> CVResult:
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
