from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Optional

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    Series,
    _is_spark_pandas,
    bulk_label_encode,
    bulk_median_impute,
    bulk_zero_variance_cols,
    collect_for_sklearn,
    concat,
    lazy_fillna,
    native_pd,
    safe_sample,
    spark_checkpoint,
)
from customer_retention.core.compat.timing import TimingEntry, log_timing, start_collecting, stop_collecting

from .cross_validator import _CV_DATE_COL, _CV_ENTITY_COL
from .data_splitter import DataSplitter, SplitStrategy
from .feature_scaler import FeatureScaler, ScalerType

ProgressCallback = Optional["Callable[[str, float], None]"]


def print_preparation_progress(label: str, elapsed: float) -> None:
    print(f"  {label}: {elapsed:.1f}s")


@dataclass
class TrainingPreparationResult:
    X_train: Any
    X_test: Any
    X_train_scaled: Any
    X_test_scaled: Any
    y_train: Any
    y_test: Any
    y_test_np: np.ndarray
    feature_names: list[str]
    train_entities: Any
    train_dates: Any
    split_info: dict[str, Any]
    class_distribution: dict[int, int]
    zero_variance_dropped: list[str]
    timing_entries: list[TimingEntry] = field(default_factory=list)


class TrainingPreparator:
    def __init__(
        self,
        target_column: str,
        feature_columns: list[str],
        purge_gap_days: int = 104,
        test_size: float = 0.2,
        scaler_type: ScalerType = ScalerType.STANDARD,
        max_rows: Optional[int] = None,
        use_float32: bool = True,
        on_progress: ProgressCallback = None,
    ):
        self._target = target_column
        self._feature_columns = list(feature_columns)
        self._purge_gap_days = purge_gap_days
        self._test_size = test_size
        self._scaler_type = scaler_type
        self._max_rows = max_rows
        self._use_float32 = use_float32
        self._on_progress = on_progress

    def _report(self, label: str, elapsed: float) -> None:
        if self._on_progress:
            self._on_progress(label, elapsed)

    def prepare(self, df: DataFrame) -> TrainingPreparationResult:
        start_collecting()

        with log_timing("filter_datetime_features") as _t:
            feature_cols = self._filter_datetime_features(self._feature_columns, df.dtypes)
        self._report(_t.label, _t.elapsed)

        with log_timing("drop_missing_target") as _t:
            df, _nan_count = self._drop_missing_target(df)
        self._report(_t.label, _t.elapsed)

        with log_timing("encode_object_columns") as _t:
            df = self._encode_object_columns(df, feature_cols)
        self._report(_t.label, _t.elapsed)

        with log_timing("impute_and_checkpoint") as _t:
            df = self._impute_and_checkpoint(df, feature_cols)
        self._report(_t.label, _t.elapsed)

        with log_timing("sample_entities") as _t:
            df = self._sample_entities(df)
        self._report(_t.label, _t.elapsed)

        with log_timing("temporal_split") as _t:
            split_result = self._temporal_split(df, feature_cols)
        self._report(_t.label, _t.elapsed)

        X_train, X_test = split_result.X_train, split_result.X_test
        y_train, y_test = split_result.y_train, split_result.y_test
        train_entities, train_dates = self._extract_train_metadata(split_result, df)

        with log_timing("fillna_and_drop_zero_variance") as _t:
            X_train, X_test, zero_var = self._fillna_and_drop_zero_variance(X_train, X_test)
        self._report(_t.label, _t.elapsed)

        feature_cols = [c for c in X_train.columns if c not in {_CV_ENTITY_COL, _CV_DATE_COL}]

        distributed = _is_spark_pandas(X_train)
        with log_timing("scale_features") as _t:
            if distributed:
                result = self._finalize_distributed(
                    X_train, X_test, y_train, y_test,
                    train_entities, train_dates, feature_cols,
                )
            else:
                result = self._finalize_local(
                    X_train, X_test, y_train, y_test,
                    train_entities, train_dates, feature_cols,
                )
        self._report(_t.label, _t.elapsed)

        class_dist = self._class_distribution(result.y_train)
        timing_entries = stop_collecting()

        return TrainingPreparationResult(
            X_train=result.X_train,
            X_test=result.X_test,
            X_train_scaled=result.X_train_scaled,
            X_test_scaled=result.X_test_scaled,
            y_train=result.y_train,
            y_test=result.y_test,
            y_test_np=result.y_test_np,
            feature_names=result.feature_names,
            train_entities=result.train_entities,
            train_dates=result.train_dates,
            split_info=split_result.split_info,
            class_distribution=class_dist,
            zero_variance_dropped=zero_var,
            timing_entries=timing_entries,
        )

    def _filter_datetime_features(self, feature_cols: list[str], dtypes: Any) -> list[str]:
        return [c for c in feature_cols if not str(dtypes.get(c, "")).startswith(("datetime", "timedelta"))]

    def _drop_missing_target(self, df: DataFrame) -> tuple[DataFrame, int]:
        mask = df[self._target].isna()
        nan_count = int(mask.sum())
        if nan_count == len(df):
            raise ValueError(f"Cannot proceed: all target values are NaN in column '{self._target}'")
        if nan_count > 0:
            df = df[~mask]
        return df, nan_count

    def _encode_object_columns(self, df: DataFrame, feature_cols: list[str]) -> DataFrame:
        obj_cols = [c for c in feature_cols if str(df.dtypes.get(c, "")).startswith("object")]
        if not obj_cols:
            return df
        return bulk_label_encode(df, obj_cols)

    def _impute_and_checkpoint(self, df: DataFrame, feature_cols: list[str]) -> DataFrame:
        df = spark_checkpoint(df)
        return bulk_median_impute(df, columns=feature_cols)

    def _sample_entities(self, df: DataFrame) -> DataFrame:
        if self._max_rows is None or len(df) <= self._max_rows:
            return df
        n_entities = df["entity_id"].nunique()
        rows_per_entity = len(df) / max(1, n_entities)
        target_entities = max(100, int(self._max_rows / rows_per_entity))
        entity_df = df[["entity_id"]].drop_duplicates()
        sampled_entity_df = safe_sample(entity_df, n=target_entities)
        return df.merge(sampled_entity_df, on="entity_id", how="inner")

    def _temporal_split(self, df: DataFrame, feature_cols: list[str]) -> Any:
        exclude = ["as_of_date", "entity_id"]
        split_df = df[feature_cols + [self._target, "as_of_date", "entity_id"]]
        split_df = spark_checkpoint(split_df)
        splitter = DataSplitter(
            target_column=self._target, strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=self._test_size,
            purge_gap_days=self._purge_gap_days, exclude_columns=exclude,
        )
        return splitter.split(split_df)

    def _extract_train_metadata(self, split_result: Any, df: DataFrame) -> tuple[Any, Any]:
        if split_result.train_metadata:
            return split_result.train_metadata["entity_id"], split_result.train_metadata["as_of_date"]
        cutoff = native_pd.Timestamp(split_result.split_info["cutoff_date"])
        purge_start = cutoff - timedelta(days=self._purge_gap_days)
        train_rows = df[df["as_of_date"] < purge_start]
        return train_rows["entity_id"], train_rows["as_of_date"]

    def _fillna_and_drop_zero_variance(
        self, X_train: DataFrame, X_test: DataFrame,
    ) -> tuple[DataFrame, DataFrame, list[str]]:
        X_train = lazy_fillna(X_train, 0)
        X_test = lazy_fillna(X_test, 0)
        zero_var = bulk_zero_variance_cols(X_train)
        if zero_var:
            X_train = X_train.drop(columns=zero_var)
            X_test = X_test.drop(columns=zero_var)
        return X_train, X_test, zero_var

    def _finalize_distributed(
        self,
        X_train: DataFrame, X_test: DataFrame,
        y_train: Series, y_test: Series,
        train_entities: Series, train_dates: Series,
        feature_cols: list[str],
    ) -> TrainingPreparationResult:
        from .spark_feature_scaler import SparkFeatureScaler

        train_bundle = concat([
            X_train, y_train.rename("__y__"),
            train_entities.rename(_CV_ENTITY_COL),
            train_dates.rename(_CV_DATE_COL),
        ], axis=1)
        train_bundle = spark_checkpoint(train_bundle)
        X_train = train_bundle.drop(columns=["__y__", _CV_ENTITY_COL, _CV_DATE_COL])
        y_train = train_bundle["__y__"]
        cv_entity = train_bundle[_CV_ENTITY_COL]
        cv_date = train_bundle[_CV_DATE_COL]

        test_bundle = concat([X_test, y_test.rename("__y__")], axis=1)
        test_bundle = spark_checkpoint(test_bundle)
        X_test = test_bundle.drop(columns=["__y__"])
        y_test = test_bundle["__y__"]

        scaler = SparkFeatureScaler(scaler_type=self._scaler_type)
        scaling_result = scaler.fit_transform(X_train[feature_cols], X_test[feature_cols])
        X_train_scaled = scaling_result.X_train_scaled
        X_test_scaled = scaling_result.X_test_scaled

        X_train[_CV_ENTITY_COL] = cv_entity
        X_train[_CV_DATE_COL] = cv_date
        X_train_scaled[_CV_ENTITY_COL] = cv_entity
        X_train_scaled[_CV_DATE_COL] = cv_date

        y_test_np = collect_for_sklearn(y_test).to_numpy()
        train_entities = collect_for_sklearn(train_entities)
        train_dates = collect_for_sklearn(train_dates)

        return TrainingPreparationResult(
            X_train=X_train, X_test=X_test,
            X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
            y_train=y_train, y_test=y_test,
            y_test_np=y_test_np, feature_names=feature_cols,
            train_entities=train_entities, train_dates=train_dates,
            split_info={}, class_distribution={},
            zero_variance_dropped=[],
        )

    def _finalize_local(
        self,
        X_train: DataFrame, X_test: DataFrame,
        y_train: Series, y_test: Series,
        train_entities: Series, train_dates: Series,
        feature_cols: list[str],
    ) -> TrainingPreparationResult:
        X_train = spark_checkpoint(X_train)
        X_test = spark_checkpoint(X_test)

        X_train = collect_for_sklearn(X_train)
        X_test = collect_for_sklearn(X_test)
        y_train = collect_for_sklearn(y_train)
        y_test = collect_for_sklearn(y_test)
        train_entities = collect_for_sklearn(train_entities)
        train_dates = collect_for_sklearn(train_dates)

        if self._use_float32:
            X_train = X_train.astype("float32")
            X_test = X_test.astype("float32")

        scaler = FeatureScaler(scaler_type=self._scaler_type)
        scaling_result = scaler.fit_transform(X_train, X_test)
        X_train_scaled = scaling_result.X_train_scaled
        X_test_scaled = scaling_result.X_test_scaled

        if self._use_float32:
            X_train_scaled = X_train_scaled.astype("float32")
            X_test_scaled = X_test_scaled.astype("float32")
        X_train_scaled = X_train_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)

        return TrainingPreparationResult(
            X_train=X_train, X_test=X_test,
            X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
            y_train=y_train, y_test=y_test,
            y_test_np=y_test.to_numpy(), feature_names=feature_cols,
            train_entities=train_entities, train_dates=train_dates,
            split_info={}, class_distribution={},
            zero_variance_dropped=[],
        )

    def _class_distribution(self, y: Series) -> dict[int, int]:
        vc = collect_for_sklearn(y.value_counts())
        return {int(k): int(v) for k, v in vc.items()}
