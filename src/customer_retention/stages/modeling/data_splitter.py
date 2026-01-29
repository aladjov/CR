import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from customer_retention.core.compat import DataFrame, Series

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata


class SplitStrategy(Enum):
    RANDOM_STRATIFIED = "random_stratified"
    TEMPORAL = "temporal"
    GROUP = "group"
    CUSTOM = "custom"


@dataclass
class SplitConfig:
    test_size: float = 0.11
    validation_size: float = 0.10
    stratify: bool = True
    random_state: int = 42
    temporal_column: Optional[str] = None
    group_column: Optional[str] = None


@dataclass
class SplitResult:
    X_train: DataFrame
    X_test: DataFrame
    y_train: Series
    y_test: Series
    X_val: Optional[DataFrame] = None
    y_val: Optional[Series] = None
    split_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitWarning:
    column: str
    issue: str
    severity: str
    recommendation: str

    def to_dict(self) -> Dict[str, str]:
        return {"column": self.column, "issue": self.issue, "severity": self.severity, "recommendation": self.recommendation}


class DataSplitter:
    def __init__(self, target_column: str, strategy: SplitStrategy = SplitStrategy.RANDOM_STRATIFIED, test_size: float = 0.11, validation_size: float = 0.10, stratify: bool = True, random_state: int = 42, temporal_column: Optional[str] = None, group_column: Optional[str] = None, exclude_columns: Optional[List[str]] = None, include_validation: bool = False):
        self.target_column = target_column
        self.strategy = strategy
        self.test_size = test_size
        self.validation_size = validation_size
        self.stratify = stratify
        self.random_state = random_state
        self.temporal_column = temporal_column
        self.group_column = group_column
        self.exclude_columns = exclude_columns or []
        self.include_validation = include_validation

    def split(self, df: DataFrame, feature_availability: Optional["FeatureAvailabilityMetadata"] = None) -> SplitResult:
        self._validate_minority_samples(df)
        availability_warnings = self.validate_feature_availability(df, feature_availability)

        if self.strategy == SplitStrategy.TEMPORAL:
            result = self._temporal_split(df)
        elif self.strategy == SplitStrategy.GROUP:
            result = self._group_split(df)
        else:
            result = self._stratified_split(df)

        if availability_warnings:
            result.split_info["availability_warnings"] = [w.to_dict() for w in availability_warnings]
        return result

    def validate_feature_availability(self, df: DataFrame, availability: Optional["FeatureAvailabilityMetadata"]) -> List[SplitWarning]:
        if availability is None:
            return []
        if self.strategy != SplitStrategy.TEMPORAL:
            return []
        warnings_list: List[SplitWarning] = []
        for col in availability.new_tracking:
            if col in df.columns:
                feat_info = availability.features.get(col)
                first_date = feat_info.first_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="new_tracking", severity="warning",
                    recommendation=f"Feature '{col}' only available from {first_date}. Training data before this date will have missing values.",
                ))
        for col in availability.retired_tracking:
            if col in df.columns:
                feat_info = availability.features.get(col)
                last_date = feat_info.last_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="retired", severity="warning",
                    recommendation=f"Feature '{col}' retired at {last_date}. Test data after this date will have missing values.",
                ))
        for col in availability.partial_window:
            if col in df.columns:
                feat_info = availability.features.get(col)
                first_date = feat_info.first_valid_date if feat_info else "unknown"
                last_date = feat_info.last_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="partial_window", severity="warning",
                    recommendation=f"Feature '{col}' only available {first_date} to {last_date}. Both train and test may have gaps.",
                ))
        return warnings_list

    def _stratified_split(self, df: DataFrame) -> SplitResult:
        X, y = self._prepare_features_target(df)
        stratify_col = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_col)

        X_val, y_val = None, None
        if self.include_validation:
            val_ratio = self.validation_size / (1 - self.test_size)
            stratify_train = y_train if self.stratify else None
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=self.random_state, stratify=stratify_train)

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=self._build_split_info(X_train, X_test, X_val)
        )

    def _temporal_split(self, df: DataFrame) -> SplitResult:
        df_sorted = df.sort_values(self.temporal_column).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - self.test_size))

        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

        X_train, y_train = self._prepare_features_target(train_df)
        X_test, y_test = self._prepare_features_target(test_df)

        X_val, y_val = None, None
        if self.include_validation:
            val_split = int(len(X_train) * (1 - self.validation_size / (1 - self.test_size)))
            X_val, y_val = X_train.iloc[val_split:], y_train.iloc[val_split:]
            X_train, y_train = X_train.iloc[:val_split], y_train.iloc[:val_split]

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=self._build_split_info(X_train, X_test, X_val)
        )

    def _group_split(self, df: DataFrame) -> SplitResult:
        X, y = self._prepare_features_target(df)
        groups = df[self.group_column]

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_val, y_val = None, None
        if self.include_validation:
            val_ratio = self.validation_size / (1 - self.test_size)
            train_groups = groups.iloc[train_idx]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=self.random_state)
            train_idx2, val_idx2 = next(gss_val.split(X_train, y_train, train_groups))
            X_val, y_val = X_train.iloc[val_idx2], y_train.iloc[val_idx2]
            X_train, y_train = X_train.iloc[train_idx2], y_train.iloc[train_idx2]

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=self._build_split_info(X_train, X_test, X_val)
        )

    def _prepare_features_target(self, df: DataFrame) -> tuple[DataFrame, Series]:
        exclude = [self.target_column] + self.exclude_columns
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols], df[self.target_column]

    def _validate_minority_samples(self, df: DataFrame):
        class_counts = df[self.target_column].value_counts()
        minority_count = class_counts.min()
        expected_minority_test = minority_count * self.test_size

        if expected_minority_test < 50:
            warnings.warn(
                f"Insufficient minority samples: expected ~{expected_minority_test:.0f} in test set. "
                "Consider using a smaller test_size or collecting more data.",
                UserWarning
            )

    def _build_split_info(self, X_train, X_test, X_val) -> Dict[str, Any]:
        info = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "strategy": self.strategy.value,
            "random_state": self.random_state,
        }
        if X_val is not None:
            info["validation_size"] = len(X_val)
        return info
