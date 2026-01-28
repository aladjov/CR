from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class ColumnType(str, Enum):
    IDENTIFIER = "identifier"
    TARGET = "target"
    FEATURE_TIMESTAMP = "feature_timestamp"
    LABEL_TIMESTAMP = "label_timestamp"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    CATEGORICAL_CYCLICAL = "categorical_cyclical"
    DATETIME = "datetime"
    BINARY = "binary"
    TEXT = "text"
    UNKNOWN = "unknown"


# Column types that should NEVER be used as features (leakage risk)
NON_FEATURE_COLUMN_TYPES = frozenset({
    ColumnType.IDENTIFIER,
    ColumnType.TARGET,
    ColumnType.FEATURE_TIMESTAMP,
    ColumnType.LABEL_TIMESTAMP,
})


class DatasetGranularity(str, Enum):
    """Describes the grain/granularity of a dataset.

    ENTITY_LEVEL: One row per entity (e.g., one row per customer)
    EVENT_LEVEL: Multiple rows per entity over time (e.g., transactions, emails)
    UNKNOWN: Cannot determine granularity
    """
    ENTITY_LEVEL = "entity_level"
    EVENT_LEVEL = "event_level"
    UNKNOWN = "unknown"


class ColumnConfig(BaseModel):
    name: str
    column_type: ColumnType
    nullable: bool = True

    encoding_strategy: Optional[str] = None
    scaling_strategy: Optional[str] = None
    missing_strategy: Optional[str] = None
    ordinal_order: Optional[list[str]] = None
    cyclical_max: Optional[int] = None

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list[str]] = None
    regex_pattern: Optional[str] = None

    description: Optional[str] = None
    business_name: Optional[str] = None
    is_feature: Optional[bool] = None
    exclude_from_model: bool = False

    @model_validator(mode='after')
    def validate_cyclical_and_ordinal(self):
        if self.column_type == ColumnType.CATEGORICAL_CYCLICAL and self.cyclical_max is None:
            raise ValueError("cyclical_max required for CATEGORICAL_CYCLICAL columns")
        if self.column_type == ColumnType.CATEGORICAL_ORDINAL and self.ordinal_order is None:
            raise ValueError("ordinal_order required for CATEGORICAL_ORDINAL columns")
        return self

    def should_be_used_as_feature(self) -> bool:
        if self.exclude_from_model:
            return False
        if self.is_feature is not None:
            return self.is_feature
        return self.column_type not in NON_FEATURE_COLUMN_TYPES

    def is_categorical(self) -> bool:
        return self.column_type in [
            ColumnType.CATEGORICAL_NOMINAL,
            ColumnType.CATEGORICAL_ORDINAL,
            ColumnType.CATEGORICAL_CYCLICAL,
            ColumnType.BINARY
        ]

    def is_numeric(self) -> bool:
        return self.column_type in [
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE
        ]

    def is_temporal(self) -> bool:
        return self.column_type == ColumnType.DATETIME
