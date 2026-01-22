from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel
from customer_retention.core.config.column_config import ColumnType, DatasetGranularity


class TypeConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TypeInference(BaseModel):
    inferred_type: ColumnType
    confidence: TypeConfidence
    evidence: list[str] = []
    alternatives: list[ColumnType] = []


class UniversalMetrics(BaseModel):
    total_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    distinct_percentage: float
    most_common_value: Optional[Any] = None
    most_common_frequency: Optional[int] = None
    memory_size_bytes: Optional[int] = None


class IdentifierMetrics(BaseModel):
    is_unique: bool
    duplicate_count: int
    duplicate_values: list[Any] = []
    format_pattern: Optional[str] = None
    format_consistency: Optional[float] = None
    length_min: Optional[int] = None
    length_max: Optional[int] = None
    length_mode: Optional[int] = None


class TargetMetrics(BaseModel):
    class_distribution: dict[str, int]
    class_percentages: dict[str, float]
    imbalance_ratio: float
    minority_class: Any
    minority_percentage: float
    n_classes: int


class NumericMetrics(BaseModel):
    mean: float
    std: float
    min_value: float
    max_value: float
    range_value: float
    median: float
    q1: float
    q3: float
    iqr: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zero_count: int
    zero_percentage: float
    negative_count: int
    negative_percentage: float
    inf_count: int
    inf_percentage: float
    outlier_count_iqr: int
    outlier_count_zscore: int
    outlier_percentage: float
    histogram_bins: Optional[list[tuple[float, float, int]]] = None


class CategoricalMetrics(BaseModel):
    cardinality: int
    cardinality_ratio: float
    value_counts: dict[str, int]
    top_categories: list[tuple[str, int]] = []
    rare_categories: list[str] = []
    rare_category_count: int
    rare_category_percentage: float
    contains_unknown: bool
    case_variations: list[str] = []
    whitespace_issues: list[str] = []
    encoding_recommendation: Optional[str] = None


class DatetimeMetrics(BaseModel):
    min_date: str
    max_date: str
    date_range_days: int
    format_detected: Optional[str] = None
    format_consistency: Optional[float] = None
    future_date_count: int
    placeholder_count: int
    timezone_info: Optional[str] = None
    timezone_consistent: bool
    weekend_percentage: Optional[float] = None


class BinaryMetrics(BaseModel):
    true_count: int
    false_count: int
    true_percentage: float
    balance_ratio: float
    values_found: list[Any]
    is_boolean: bool


class TextMetrics(BaseModel):
    length_min: int
    length_max: int
    length_mean: float
    length_median: float
    empty_count: int
    empty_percentage: float
    word_count_mean: float
    contains_digits_pct: float
    contains_special_pct: float
    pii_detected: bool
    pii_types: list[str] = []
    language_detected: Optional[str] = None


class ColumnProfile(BaseModel):
    column_name: str
    configured_type: Optional[ColumnType] = None
    inferred_type: Optional[TypeInference] = None
    universal_metrics: UniversalMetrics
    identifier_metrics: Optional[IdentifierMetrics] = None
    target_metrics: Optional[TargetMetrics] = None
    numeric_metrics: Optional[NumericMetrics] = None
    categorical_metrics: Optional[CategoricalMetrics] = None
    datetime_metrics: Optional[DatetimeMetrics] = None
    binary_metrics: Optional[BinaryMetrics] = None
    text_metrics: Optional[TextMetrics] = None
    quality_issues: list[dict[str, Any]] = []
    recommendations: list[str] = []

    def get_effective_type(self) -> ColumnType:
        if self.configured_type:
            return self.configured_type
        if self.inferred_type:
            return self.inferred_type.inferred_type
        return ColumnType.UNKNOWN

    def has_critical_issues(self) -> bool:
        return any(issue.get("severity") == "critical" for issue in self.quality_issues)

    def has_high_issues(self) -> bool:
        return any(issue.get("severity") == "high" for issue in self.quality_issues)

    def get_issues_by_severity(self, severity: str) -> list[dict[str, Any]]:
        return [issue for issue in self.quality_issues if issue.get("severity") == severity]


class ProfileResult(BaseModel):
    dataset_name: str
    total_rows: int
    total_columns: int
    column_profiles: dict[str, ColumnProfile]
    profiling_timestamp: str
    profiling_duration_seconds: float
    sample_size: Optional[int] = None
    quality_score: Optional[float] = None

    def get_profile(self, column_name: str) -> Optional[ColumnProfile]:
        return self.column_profiles.get(column_name)

    def get_columns_by_type(self, column_type: ColumnType) -> list[str]:
        return [
            name for name, profile in self.column_profiles.items()
            if profile.get_effective_type() == column_type
        ]

    def get_columns_with_critical_issues(self) -> list[str]:
        return [
            name for name, profile in self.column_profiles.items()
            if profile.has_critical_issues()
        ]

    def get_columns_with_high_issues(self) -> list[str]:
        return [
            name for name, profile in self.column_profiles.items()
            if profile.has_high_issues()
        ]

    def calculate_quality_score(self) -> float:
        if not self.column_profiles:
            return 0.0

        total_issues = sum(len(p.quality_issues) for p in self.column_profiles.values())
        critical_issues = len(self.get_columns_with_critical_issues())
        high_issues = len(self.get_columns_with_high_issues())

        penalty = (critical_issues * 20) + (high_issues * 10) + (total_issues * 2)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)


class GranularityResult(BaseModel):
    """Result of dataset granularity detection."""
    granularity: DatasetGranularity
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    avg_events_per_entity: Optional[float] = None
    unique_entities: Optional[int] = None
    total_rows: Optional[int] = None
    evidence: list[str] = []
