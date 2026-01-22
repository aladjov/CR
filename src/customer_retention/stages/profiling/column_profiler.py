from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from datetime import datetime
from scipy import stats

from customer_retention.core.compat import pd, Series, is_datetime64_any_dtype, is_bool_dtype, Timestamp
from customer_retention.core.config.column_config import ColumnType
from .profile_result import (
    UniversalMetrics, IdentifierMetrics, TargetMetrics,
    NumericMetrics, CategoricalMetrics, DatetimeMetrics, BinaryMetrics
)


class ColumnProfiler(ABC):
    def compute_universal_metrics(self, series: pd.Series) -> UniversalMetrics:
        total_count = len(series)
        null_count = int(series.isna().sum())
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

        distinct_count = int(series.nunique())
        distinct_percentage = (distinct_count / total_count * 100) if total_count > 0 else 0

        value_counts = series.value_counts()
        most_common_value = value_counts.index[0] if len(value_counts) > 0 else None
        most_common_frequency = int(value_counts.iloc[0]) if len(value_counts) > 0 else None

        memory_size = series.memory_usage(deep=True)

        return UniversalMetrics(
            total_count=total_count,
            null_count=null_count,
            null_percentage=round(null_percentage, 2),
            distinct_count=distinct_count,
            distinct_percentage=round(distinct_percentage, 2),
            most_common_value=most_common_value,
            most_common_frequency=most_common_frequency,
            memory_size_bytes=int(memory_size)
        )

    @abstractmethod
    def profile(self, series: pd.Series) -> dict:
        pass


class IdentifierProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        is_unique = series.nunique() == len(series.dropna())
        duplicates = series[series.duplicated(keep=False)]
        duplicate_count = len(duplicates.unique())
        duplicate_values = duplicates.unique().tolist()[:10]

        str_series = series.dropna().astype(str)
        lengths = str_series.str.len()

        format_pattern, format_consistency = self.detect_format_pattern(str_series)

        return {
            "identifier_metrics": IdentifierMetrics(
                is_unique=is_unique,
                duplicate_count=duplicate_count,
                duplicate_values=duplicate_values,
                format_pattern=format_pattern,
                format_consistency=format_consistency,
                length_min=int(lengths.min()) if len(lengths) > 0 else None,
                length_max=int(lengths.max()) if len(lengths) > 0 else None,
                length_mode=int(lengths.mode().iloc[0]) if len(lengths.mode()) > 0 else None
            )
        }

    def detect_format_pattern(self, str_series: pd.Series) -> tuple[Optional[str], Optional[float]]:
        import re
        if len(str_series) == 0:
            return None, None

        sample = str_series.head(min(100, len(str_series)))
        pattern_map = {
            r'^[A-Z]{3}-\d{5}$': 'AAA-99999',
            r'^\d{3}-\d{3}-\d{4}$': '999-999-9999',
            r'^[A-Z]{2}\d{6}$': 'AA999999',
            r'^\d+$': 'numeric_only',
            r'^[A-Za-z]+$': 'alpha_only',
            r'^[A-Z][0-9]{4,}$': 'A9999+',
            r'^\w+-\d+$': 'text-digits',
            r'^[A-Z0-9]+$': 'alphanumeric'
        }

        for pattern, desc in pattern_map.items():
            matches = str_series.str.match(pattern, na=False)
            match_pct = (matches.sum() / len(str_series)) * 100
            if match_pct > 80:
                return desc, round(match_pct, 2)

        return 'mixed', 0.0


class TargetProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        value_counts = series.value_counts()
        class_distribution = {str(k): int(v) for k, v in value_counts.items()}

        total = len(series.dropna())
        class_percentages = {str(k): round((v / total * 100), 2) for k, v in value_counts.items()}

        minority_class = value_counts.idxmin()
        minority_count = value_counts.min()
        majority_count = value_counts.max()
        minority_percentage = round((minority_count / total * 100), 2) if total > 0 else 0
        imbalance_ratio = round((majority_count / minority_count), 2) if minority_count > 0 else float('inf')

        return {
            "target_metrics": TargetMetrics(
                class_distribution=class_distribution,
                class_percentages=class_percentages,
                imbalance_ratio=imbalance_ratio,
                minority_class=minority_class,
                minority_percentage=minority_percentage,
                n_classes=len(value_counts)
            )
        }


class NumericProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {"numeric_metrics": None}

        mean_val = float(clean_series.mean())
        std_val = float(clean_series.std())
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        range_val = max_val - min_val

        median_val = float(clean_series.median())
        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = q3 - q1

        try:
            skewness_val = float(clean_series.skew())
            kurtosis_val = float(clean_series.kurtosis())
        except:
            skewness_val = None
            kurtosis_val = None

        zero_count = int((clean_series == 0).sum())
        zero_percentage = round((zero_count / len(clean_series) * 100), 2)

        negative_count = int((clean_series < 0).sum())
        negative_percentage = round((negative_count / len(clean_series) * 100), 2)

        inf_count = int(np.isinf(clean_series).sum())
        inf_percentage = round((inf_count / len(clean_series) * 100), 2)

        outliers_iqr = ((clean_series < (q1 - 1.5 * iqr)) | (clean_series > (q3 + 1.5 * iqr)))
        outlier_count_iqr = int(outliers_iqr.sum())

        if std_val > 0:
            z_scores = np.abs((clean_series - mean_val) / std_val)
            outlier_count_zscore = int((z_scores > 3).sum())
        else:
            outlier_count_zscore = 0

        outlier_percentage = round((outlier_count_iqr / len(clean_series) * 100), 2)

        # Filter out infinite values for histogram calculation
        finite_series = clean_series[np.isfinite(clean_series)]
        if len(finite_series) > 0:
            histogram, bin_edges = np.histogram(finite_series, bins=10)
            histogram_bins = [
                (round(float(bin_edges[i]), 4), round(float(bin_edges[i + 1]), 4), int(histogram[i]))
                for i in range(len(histogram))
            ]
        else:
            histogram_bins = []

        return {
            "numeric_metrics": NumericMetrics(
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                min_value=round(min_val, 4),
                max_value=round(max_val, 4),
                range_value=round(range_val, 4),
                median=round(median_val, 4),
                q1=round(q1, 4),
                q3=round(q3, 4),
                iqr=round(iqr, 4),
                skewness=round(skewness_val, 4) if skewness_val is not None else None,
                kurtosis=round(kurtosis_val, 4) if kurtosis_val is not None else None,
                zero_count=zero_count,
                zero_percentage=zero_percentage,
                negative_count=negative_count,
                negative_percentage=negative_percentage,
                inf_count=inf_count,
                inf_percentage=inf_percentage,
                outlier_count_iqr=outlier_count_iqr,
                outlier_count_zscore=outlier_count_zscore,
                outlier_percentage=outlier_percentage,
                histogram_bins=histogram_bins
            )
        }


class CategoricalProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {"categorical_metrics": None}

        cardinality = int(series.nunique())
        cardinality_ratio = round((cardinality / len(clean_series)), 4)

        value_counts = clean_series.value_counts()
        value_counts_dict = {str(k): int(v) for k, v in value_counts.items()}

        top_categories = [(str(k), int(v)) for k, v in value_counts.head(10).items()]

        rare_threshold = len(clean_series) * 0.01
        rare_categories = [str(k) for k, v in value_counts.items() if v < rare_threshold]
        rare_category_count = len(rare_categories)

        rare_rows = sum(v for k, v in value_counts.items() if v < rare_threshold)
        rare_category_percentage = round((rare_rows / len(clean_series) * 100), 2)

        unknown_values = {"unknown", "other", "n/a", "na", "none", "null", "missing"}
        contains_unknown = any(str(v).lower() in unknown_values for v in clean_series.unique()[:100])

        case_variations = self.detect_case_variations(clean_series)
        whitespace_issues = self.detect_whitespace_issues(clean_series)

        encoding_recommendation = self.recommend_encoding(cardinality, rare_category_percentage)

        return {
            "categorical_metrics": CategoricalMetrics(
                cardinality=cardinality,
                cardinality_ratio=cardinality_ratio,
                value_counts=value_counts_dict,
                top_categories=top_categories,
                rare_categories=rare_categories[:20],
                rare_category_count=rare_category_count,
                rare_category_percentage=rare_category_percentage,
                contains_unknown=contains_unknown,
                case_variations=case_variations,
                whitespace_issues=whitespace_issues,
                encoding_recommendation=encoding_recommendation
            )
        }

    def detect_case_variations(self, clean_series: pd.Series) -> list[str]:
        str_series = clean_series.astype(str)
        lower_to_originals = {}

        for value in str_series.unique():
            lower_val = value.lower()
            if lower_val not in lower_to_originals:
                lower_to_originals[lower_val] = []
            lower_to_originals[lower_val].append(value)

        variations = []
        for lower_val, originals in lower_to_originals.items():
            if len(originals) > 1:
                variations.append(f"{originals[0]} vs {originals[1]}")

        return variations[:10]

    def detect_whitespace_issues(self, clean_series: pd.Series) -> list[str]:
        str_series = clean_series.astype(str)
        issues = []

        for value in str_series.unique()[:100]:
            if value != value.strip():
                issues.append(value)

        return issues[:10]

    def recommend_encoding(self, cardinality: int, rare_pct: float) -> str:
        if cardinality <= 5:
            return "one_hot"
        elif cardinality <= 15:
            return "one_hot_or_target"
        elif cardinality <= 50:
            return "target_or_embedding"
        else:
            return "hashing_or_embedding"


class DatetimeProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {"datetime_metrics": None}

        format_detected, format_consistency = self.detect_datetime_format(series)

        if not is_datetime64_any_dtype(clean_series):
            sample = clean_series.head(10)
            if len(sample) > 0 and all(isinstance(v, (Timestamp, datetime)) for v in sample):
                pass
            else:
                try:
                    clean_series = pd.to_datetime(clean_series, errors='coerce', format='mixed')
                except:
                    return {"datetime_metrics": None}

        min_date = clean_series.min()
        max_date = clean_series.max()
        date_range_days = (max_date - min_date).days

        now = Timestamp.now()
        future_date_count = int((clean_series > now).sum())

        placeholder_dates = [
            Timestamp('1970-01-01'),
            Timestamp('1900-01-01'),
            Timestamp('9999-12-31')
        ]
        placeholder_count = int(sum((clean_series == pd_date).sum() for pd_date in placeholder_dates))

        if is_datetime64_any_dtype(clean_series):
            weekend_count = int(clean_series.dt.dayofweek.isin([5, 6]).sum())
        else:
            weekend_count = int(sum(1 for v in clean_series if isinstance(v, Timestamp) and v.dayofweek in [5, 6]))
        weekend_percentage = round((weekend_count / len(clean_series) * 100), 2)

        return {
            "datetime_metrics": DatetimeMetrics(
                min_date=str(min_date),
                max_date=str(max_date),
                date_range_days=date_range_days,
                format_detected=format_detected,
                format_consistency=format_consistency,
                future_date_count=future_date_count,
                placeholder_count=placeholder_count,
                timezone_consistent=True,
                weekend_percentage=weekend_percentage
            )
        }

    def detect_datetime_format(self, series: pd.Series) -> tuple[Optional[str], Optional[float]]:
        if is_datetime64_any_dtype(series):
            return 'datetime64', 100.0

        sample = series.dropna().astype(str).head(min(100, len(series)))
        if len(sample) == 0:
            return None, None

        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%m/%d/%Y',
            '%m-%d-%Y',
        ]

        best_format = None
        best_match_pct = 0.0

        for fmt in formats:
            matches = 0
            for val in sample:
                try:
                    datetime.strptime(val, fmt)
                    matches += 1
                except:
                    pass

            match_pct = (matches / len(sample)) * 100
            if match_pct > best_match_pct:
                best_match_pct = match_pct
                best_format = fmt

        if best_format and best_match_pct > 80:
            return best_format, round(best_match_pct, 2)

        return 'mixed', 0.0


class BinaryProfiler(ColumnProfiler):
    def profile(self, series: pd.Series) -> dict:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {"binary_metrics": None}

        value_counts = clean_series.value_counts()
        values_found = value_counts.index.tolist()

        true_values = {1, 1.0, True, "1", "yes", "Yes", "YES", "true", "True", "TRUE", "y", "Y"}
        false_values = {0, 0.0, False, "0", "no", "No", "NO", "false", "False", "FALSE", "n", "N"}

        true_count = int(sum(value_counts.get(v, 0) for v in values_found if v in true_values))
        false_count = int(sum(value_counts.get(v, 0) for v in values_found if v in false_values))

        if true_count == 0 and false_count == 0:
            true_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            false_count = int(value_counts.iloc[1]) if len(value_counts) > 1 else 0

        total = true_count + false_count
        true_percentage = round((true_count / total * 100), 2) if total > 0 else 0

        balance_ratio = round((max(true_count, false_count) / min(true_count, false_count)), 2) \
            if min(true_count, false_count) > 0 else float('inf')

        is_boolean = is_bool_dtype(series)

        return {
            "binary_metrics": BinaryMetrics(
                true_count=true_count,
                false_count=false_count,
                true_percentage=true_percentage,
                balance_ratio=balance_ratio,
                values_found=values_found,
                is_boolean=is_boolean
            )
        }


class TextProfiler(ColumnProfiler):
    """Profile text columns with PII detection."""

    def profile(self, series: pd.Series) -> dict:
        """Profile text column."""
        import re

        clean_series = series.dropna()

        # Calculate text lengths
        lengths = clean_series.astype(str).str.len()
        length_min = int(lengths.min()) if len(lengths) > 0 else 0
        length_max = int(lengths.max()) if len(lengths) > 0 else 0
        length_mean = float(lengths.mean()) if len(lengths) > 0 else 0.0
        length_median = float(lengths.median()) if len(lengths) > 0 else 0.0

        # Empty text detection
        empty_count = int((clean_series.astype(str) == "").sum())
        empty_percentage = (empty_count / len(series) * 100) if len(series) > 0 else 0.0

        # Word count
        word_counts = clean_series.astype(str).str.split().str.len()
        word_count_mean = float(word_counts.mean()) if len(word_counts) > 0 else 0.0

        # Contains digits
        contains_digits = clean_series.astype(str).str.contains(r'\d', regex=True, na=False)
        contains_digits_pct = float(contains_digits.sum() / len(clean_series) * 100) if len(clean_series) > 0 else 0.0

        # Contains special characters
        contains_special = clean_series.astype(str).str.contains(r'[!@#$%^&*(),.?":{}|<>]', regex=True, na=False)
        contains_special_pct = float(contains_special.sum() / len(clean_series) * 100) if len(clean_series) > 0 else 0.0

        # PII Detection
        pii_detected = False
        pii_types = []

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if clean_series.astype(str).str.contains(email_pattern, regex=True, na=False).any():
            pii_detected = True
            pii_types.append("email")

        # Phone pattern (US format)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        if clean_series.astype(str).str.contains(phone_pattern, regex=True, na=False).any():
            pii_detected = True
            pii_types.append("phone")

        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if clean_series.astype(str).str.contains(ssn_pattern, regex=True, na=False).any():
            pii_detected = True
            pii_types.append("ssn")

        # Credit card pattern (basic)
        cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        if clean_series.astype(str).str.contains(cc_pattern, regex=True, na=False).any():
            pii_detected = True
            pii_types.append("credit_card")

        from .profile_result import TextMetrics

        return {
            "text_metrics": TextMetrics(
                length_min=length_min,
                length_max=length_max,
                length_mean=length_mean,
                length_median=length_median,
                empty_count=empty_count,
                empty_percentage=round(empty_percentage, 2),
                word_count_mean=round(word_count_mean, 2),
                contains_digits_pct=round(contains_digits_pct, 2),
                contains_special_pct=round(contains_special_pct, 2),
                pii_detected=pii_detected,
                pii_types=pii_types,
                language_detected=None  # TODO: Can add language detection later
            )
        }


class ProfilerFactory:
    _profilers = {
        ColumnType.IDENTIFIER: IdentifierProfiler,
        ColumnType.TARGET: TargetProfiler,
        ColumnType.NUMERIC_CONTINUOUS: NumericProfiler,
        ColumnType.NUMERIC_DISCRETE: NumericProfiler,
        ColumnType.CATEGORICAL_NOMINAL: CategoricalProfiler,
        ColumnType.CATEGORICAL_ORDINAL: CategoricalProfiler,
        ColumnType.CATEGORICAL_CYCLICAL: CategoricalProfiler,
        ColumnType.DATETIME: DatetimeProfiler,
        ColumnType.BINARY: BinaryProfiler,
        ColumnType.TEXT: TextProfiler,
    }

    @classmethod
    def get_profiler(cls, column_type: ColumnType) -> Optional[ColumnProfiler]:
        profiler_class = cls._profilers.get(column_type)
        return profiler_class() if profiler_class else None
