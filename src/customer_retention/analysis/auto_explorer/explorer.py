import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

from customer_retention.core.compat import (
    DataFrame,
    Series,
    _is_spark_pandas,
    as_spark_df,
    head_as_list,
    is_dataframe,
    pd,
    safe_memory_usage_bytes,
)
from customer_retention.core.compat.bulk_profiling import (
    BinaryColumnStats,
    CategoricalColumnStats,
    DatetimeColumnStats,
    IdentifierColumnStats,
    NumericColumnStats,
    PerColumnStats,
    TextColumnStats,
    compute_bulk_stats,
    compute_typed_bulk_stats,
)
from customer_retention.core.compat.timing import log_timing
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory, TypeDetector
from customer_retention.stages.temporal import TEMPORAL_METADATA_COLS

from .findings import ColumnFinding, ExplorationFindings


def _try_strptime(value: str, fmt: str) -> bool:
    from datetime import datetime as _dt

    try:
        _dt.strptime(value, fmt)
        return True
    except Exception:
        return False


class DataExplorer:
    def __init__(self, visualize: bool = True, save_findings: bool = True, output_dir: str = "../explorations"):
        self.visualize = visualize
        self.save_findings = save_findings
        self.output_dir = Path(output_dir)
        self.type_detector = TypeDetector()
        self.last_findings_path: Optional[str] = None

    def explore(
        self, source: Union[str, DataFrame], target_hint: Optional[str] = None, name: Optional[str] = None
    ) -> ExplorationFindings:
        with log_timing("explore() total", logger):
            df, source_path, source_format = self._load_source(source)
            cached_spark_df = None
            if hasattr(df, "to_spark"):
                cached_spark_df = as_spark_df(df)
                cached_spark_df.cache()
            try:
                with log_timing("_create_findings", logger):
                    findings = self._create_findings(df, source_path, source_format)

                with log_timing("_explore_all_columns", logger):
                    self._explore_all_columns(df, findings, target_hint)

                self._calculate_overall_metrics(findings)
                self._check_modeling_readiness(findings)
                if self.visualize:
                    self._display_summary(findings)
                self.last_findings_path = self._compute_findings_path(findings, name)
                if self.save_findings:
                    self._save_findings(findings)
                return findings
            finally:
                if cached_spark_df is not None:
                    cached_spark_df.unpersist()

    def _load_source(self, source: Union[str, DataFrame]) -> tuple:
        if is_dataframe(source):
            return source, "<DataFrame>", "dataframe"
        path = Path(source) if isinstance(source, str) else source
        if path.is_dir() and (path / "_delta_log").is_dir():
            try:
                from customer_retention.integrations.adapters.factory import get_delta

                return get_delta(force_local=True).read(str(path)), str(source), "delta"
            except ImportError:
                pass
        source_str = str(source)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(source_str), source_str, "csv"
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(source_str), source_str, "parquet"
        return pd.read_csv(source_str), source_str, "csv"

    @staticmethod
    def _detect_format(path_str: str) -> str:
        if path_str.endswith(".csv"):
            return "csv"
        if path_str.endswith((".parquet", ".pq")):
            return "parquet"
        return "csv"

    def _create_findings(self, df: DataFrame, source_path: str, source_format: str) -> ExplorationFindings:
        return ExplorationFindings(
            source_path=source_path,
            source_format=source_format,
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage_mb=safe_memory_usage_bytes(df) / (1024 * 1024),
        )

    def _explore_all_columns(self, df: DataFrame, findings: ExplorationFindings, target_hint: Optional[str]):
        if len(df.columns) == 0:
            return

        total_cols = len(df.columns)

        with log_timing("compute_bulk_stats", logger, cols=total_cols):
            bulk = compute_bulk_stats(df)

        with log_timing("df.head(200) sample", logger):
            sample_df = df.head(200)
            if _is_spark_pandas(sample_df):
                sample_df = sample_df.to_pandas()

        # --- Pass 1: detect types using sample + bulk distinct_counts (0 extra Spark jobs) ---
        active_cols = [c for c in df.columns if c not in TEMPORAL_METADATA_COLS]
        with log_timing("type detection", logger, cols=len(active_cols)):
            col_types: dict[str, ColumnType] = {}
            type_inferences: dict = {}
            for column_name in active_cols:
                col_stats = bulk.columns.get(column_name)
                distinct_count = col_stats.distinct_count if col_stats else None
                type_inf = self.type_detector.detect_type(
                    sample_df[column_name],
                    column_name,
                    distinct_count=distinct_count,
                    total_count=bulk.total_count,
                )
                if target_hint and column_name.lower() == target_hint.lower():
                    type_inf.inferred_type = ColumnType.TARGET
                    type_inf.evidence.append(f"Matched target hint: {target_hint}")
                col_types[column_name] = type_inf.inferred_type
                type_inferences[column_name] = type_inf

        # --- Classify columns by type for bulk pass 2 ---
        datetime_cols = [
            c for c, t in col_types.items()
            if t in (ColumnType.DATETIME, ColumnType.FEATURE_TIMESTAMP, ColumnType.LABEL_TIMESTAMP)
        ]
        categorical_cols = [
            c for c, t in col_types.items()
            if t in (ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.CATEGORICAL_CYCLICAL)
        ]
        identifier_cols = [c for c, t in col_types.items() if t == ColumnType.IDENTIFIER]
        binary_cols = [c for c, t in col_types.items() if t == ColumnType.BINARY]
        text_cols = [c for c, t in col_types.items() if t == ColumnType.TEXT]
        numeric_cols = [c for c, t in col_types.items() if t in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE)]
        logger.info(
            "Column classification: dt=%d cat=%d id=%d bin=%d txt=%d num=%d",
            len(datetime_cols), len(categorical_cols), len(identifier_cols),
            len(binary_cols), len(text_cols), len(numeric_cols),
        )

        # --- Pass 2: compute typed bulk stats (3-5 Spark jobs) ---
        with log_timing("compute_typed_bulk_stats", logger, dt=len(datetime_cols), cat=len(categorical_cols), id=len(identifier_cols), bin=len(binary_cols), txt=len(text_cols)):
            typed_bulk = compute_typed_bulk_stats(
                df, bulk,
                datetime_cols=datetime_cols,
                categorical_cols=categorical_cols,
                identifier_cols=identifier_cols,
                binary_cols=binary_cols,
                text_cols=text_cols,
            )

        # --- Per-column loop: use sample series + bulk metrics, avoid profiler.profile() ---
        with log_timing("per-column loop", logger, cols=len(active_cols)):
            processed = 0
            log_interval = max(1, total_cols // 5)
            for column_name in active_cols:
                col_stats = bulk.columns.get(column_name)
                numeric_stats = bulk.numeric.get(column_name)
                column_finding = self._explore_column(
                    sample_df[column_name],
                    column_name,
                    target_hint,
                    bulk_total=bulk.total_count,
                    col_stats=col_stats,
                    numeric_stats=numeric_stats,
                    typed_bulk=typed_bulk,
                    pre_detected_type=type_inferences.get(column_name),
                )
                findings.columns[column_name] = column_finding
                self._track_special_columns(findings, column_finding, col_stats=col_stats)
                processed += 1
                if processed % log_interval == 0:
                    logger.info("Profiled %d/%d columns", processed, total_cols)

    def _explore_column(
        self,
        series: Series,
        column_name: str,
        target_hint: Optional[str],
        bulk_total: Optional[int] = None,
        col_stats: Optional[PerColumnStats] = None,
        numeric_stats: Optional[NumericColumnStats] = None,
        typed_bulk=None,
        pre_detected_type=None,
    ) -> ColumnFinding:
        if pre_detected_type is not None:
            type_inference = pre_detected_type
        else:
            distinct_count = col_stats.distinct_count if col_stats else None
            type_inference = self.type_detector.detect_type(series, column_name, distinct_count=distinct_count)
            if target_hint and column_name.lower() == target_hint.lower():
                type_inference.inferred_type = ColumnType.TARGET
                type_inference.evidence.append(f"Matched target hint: {target_hint}")
        universal_metrics = self._compute_universal_metrics(
            series,
            type_inference.inferred_type,
            bulk_total=bulk_total,
            col_stats=col_stats,
        )
        type_metrics = self._compute_type_metrics(
            series, type_inference.inferred_type,
            numeric_stats=numeric_stats,
            typed_bulk=typed_bulk,
            column_name=column_name,
            bulk_total=bulk_total,
            col_stats=col_stats,
        )
        quality_issues = self._identify_quality_issues(universal_metrics, type_metrics)
        quality_score = self._calculate_column_quality(universal_metrics, quality_issues)
        cleaning_recommendations = self._generate_cleaning_recommendations(universal_metrics, quality_issues)
        transformation_recommendations = self._generate_transformation_recommendations(
            type_inference.inferred_type, type_metrics
        )
        return ColumnFinding(
            name=column_name,
            inferred_type=type_inference.inferred_type,
            confidence=self._confidence_to_float(type_inference.confidence),
            evidence=type_inference.evidence,
            alternatives=type_inference.alternatives or [],
            universal_metrics=universal_metrics,
            type_metrics=type_metrics,
            quality_issues=quality_issues,
            quality_score=quality_score,
            cleaning_needed=len(cleaning_recommendations) > 0,
            cleaning_recommendations=cleaning_recommendations,
            transformation_recommendations=transformation_recommendations,
        )

    def _compute_universal_metrics(
        self,
        series: Series,
        col_type: ColumnType,
        bulk_total: Optional[int] = None,
        col_stats: Optional[PerColumnStats] = None,
    ) -> dict:
        profiler = ProfilerFactory.get_profiler(col_type)
        if not profiler:
            return {}

        if col_stats is not None and bulk_total is not None:
            total_count = bulk_total
            null_count = col_stats.null_count
            null_percentage = round((null_count / total_count * 100), 2) if total_count > 0 else 0
            distinct_count = col_stats.distinct_count
            distinct_percentage = round((distinct_count / total_count * 100), 2) if total_count > 0 else 0

            return {
                "total_count": total_count,
                "null_count": null_count,
                "null_percentage": null_percentage,
                "distinct_count": distinct_count,
                "distinct_percentage": distinct_percentage,
                "most_common_value": col_stats.most_common_value,
                "most_common_frequency": col_stats.most_common_frequency,
                "memory_size_bytes": 0,
            }

        universal = profiler.compute_universal_metrics(series)
        return {
            "total_count": universal.total_count,
            "null_count": universal.null_count,
            "null_percentage": universal.null_percentage,
            "distinct_count": universal.distinct_count,
            "distinct_percentage": universal.distinct_percentage,
            "most_common_value": str(universal.most_common_value) if universal.most_common_value is not None else None,
            "most_common_frequency": universal.most_common_frequency,
            "memory_size_bytes": universal.memory_size_bytes,
        }

    def _compute_type_metrics(
        self,
        series: Series,
        col_type: ColumnType,
        numeric_stats: Optional[NumericColumnStats] = None,
        typed_bulk=None,
        column_name: Optional[str] = None,
        bulk_total: Optional[int] = None,
        col_stats: Optional[PerColumnStats] = None,
    ) -> dict:
        if numeric_stats is not None and col_type in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE):
            return self._numeric_metrics_from_bulk(series, numeric_stats)

        if typed_bulk is not None and column_name is not None:
            non_null = (bulk_total - col_stats.null_count) if (bulk_total and col_stats) else len(series)

            if col_type in (ColumnType.DATETIME, ColumnType.FEATURE_TIMESTAMP, ColumnType.LABEL_TIMESTAMP):
                dt_stats = typed_bulk.datetime.get(column_name)
                if dt_stats is not None:
                    return self._datetime_metrics_from_bulk(series, dt_stats, non_null_count=non_null)

            if col_type in (ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.CATEGORICAL_CYCLICAL):
                cat_stats = typed_bulk.categorical.get(column_name)
                if cat_stats is not None:
                    return self._categorical_metrics_from_bulk(series, cat_stats)

            if col_type == ColumnType.IDENTIFIER:
                id_stats = typed_bulk.identifier.get(column_name)
                if id_stats is not None:
                    return self._identifier_metrics_from_bulk(series, id_stats)

            if col_type == ColumnType.BINARY:
                bin_stats = typed_bulk.binary.get(column_name)
                if bin_stats is not None:
                    return self._binary_metrics_from_bulk(series, bin_stats)

            if col_type == ColumnType.TEXT:
                txt_stats = typed_bulk.text.get(column_name)
                if txt_stats is not None:
                    return self._text_metrics_from_bulk(txt_stats, bulk_total or len(series))

        profiler = ProfilerFactory.get_profiler(col_type)
        if not profiler:
            return {}
        profile_result = profiler.profile(series)
        for value in profile_result.values():
            if value is not None and hasattr(value, "__dict__"):
                return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        return {}

    @staticmethod
    def _numeric_metrics_from_bulk(series: Series, ns: NumericColumnStats) -> dict:
        if ns.mean is None:
            return {}

        q1 = ns.q1 if ns.q1 is not None else 0.0
        q3 = ns.q3 if ns.q3 is not None else 0.0
        iqr = q3 - q1
        min_val = ns.min_val if ns.min_val is not None else 0.0
        max_val = ns.max_val if ns.max_val is not None else 0.0
        range_val = max_val - min_val

        non_null_count = max(ns.non_null_count, 1)

        zero_pct = round((ns.zero_count / non_null_count * 100), 2)
        neg_pct = round((ns.negative_count / non_null_count * 100), 2)
        inf_pct = round((ns.inf_count / non_null_count * 100), 2)
        outlier_pct = round((ns.outlier_count_iqr / non_null_count * 100), 2)

        histogram_bins = ns.histogram_bins if ns.histogram_bins else []

        return {
            "mean": round(ns.mean, 4),
            "std": round(ns.std, 4) if ns.std is not None else 0.0,
            "min_value": round(min_val, 4),
            "max_value": round(max_val, 4),
            "range_value": round(range_val, 4),
            "median": round(ns.median, 4) if ns.median is not None else round(ns.mean, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "skewness": round(ns.skewness, 4) if ns.skewness is not None else None,
            "kurtosis": round(ns.kurtosis, 4) if ns.kurtosis is not None else None,
            "zero_count": ns.zero_count,
            "zero_percentage": zero_pct,
            "negative_count": ns.negative_count,
            "negative_percentage": neg_pct,
            "inf_count": ns.inf_count,
            "inf_percentage": inf_pct,
            "outlier_count_iqr": ns.outlier_count_iqr,
            "outlier_count_zscore": ns.outlier_count_zscore,
            "outlier_percentage": outlier_pct,
            "histogram_bins": histogram_bins,
        }

    @staticmethod
    def _datetime_metrics_from_bulk(
        series: Series, ds: DatetimeColumnStats, non_null_count: Optional[int] = None,
    ) -> dict:
        sample = head_as_list(series.dropna(), 10)
        format_detected = None
        format_consistency = None
        if sample:
            from datetime import datetime as _dt

            if all(isinstance(v, _dt) for v in sample):
                format_detected = "datetime64"
                format_consistency = 100.0
            else:
                str_sample = [str(v) for v in sample]
                formats = [
                    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y",
                ]
                for fmt in formats:
                    matches = sum(1 for s in str_sample if _try_strptime(s, fmt))
                    pct = matches / len(str_sample) * 100
                    if pct > 80:
                        format_detected = fmt
                        format_consistency = round(pct, 2)
                        break
                if format_detected is None:
                    format_detected = "mixed"
                    format_consistency = 0.0

        if non_null_count is None:
            non_null_count = len(series) - int(series.isna().sum()) if len(series) > 0 else 0
        weekend_pct = round(ds.weekend_count / non_null_count * 100, 2) if non_null_count > 0 else None

        return {
            "min_date": ds.min_date or "",
            "max_date": ds.max_date or "",
            "date_range_days": ds.date_range_days or 0,
            "format_detected": format_detected,
            "format_consistency": format_consistency,
            "future_date_count": ds.future_date_count,
            "placeholder_count": ds.placeholder_count,
            "timezone_consistent": True,
            "weekend_percentage": weekend_pct,
        }

    @staticmethod
    def _categorical_metrics_from_bulk(series: Series, cs: CategoricalColumnStats) -> dict:
        non_null_count = max(1, sum(v for v in cs.value_counts.values())) if cs.value_counts else 1
        cardinality_ratio = cs.cardinality_ratio

        unknown_values = {"unknown", "other", "n/a", "na", "none", "null", "missing"}
        unique_sample = head_as_list(series.dropna().drop_duplicates(), 100)
        contains_unknown = any(str(v).lower() in unknown_values for v in unique_sample)

        whitespace_issues = []
        for value in head_as_list(series.dropna().drop_duplicates(), 100):
            s = str(value)
            if s != s.strip():
                whitespace_issues.append(s)
                if len(whitespace_issues) >= 10:
                    break

        cardinality = cs.cardinality
        if cardinality <= 5:
            encoding_rec = "one_hot"
        elif cardinality <= 15:
            encoding_rec = "one_hot_or_target"
        elif cardinality <= 50:
            encoding_rec = "target_or_embedding"
        else:
            encoding_rec = "hashing_or_embedding"

        return {
            "cardinality": cardinality,
            "cardinality_ratio": cardinality_ratio,
            "value_counts": cs.value_counts,
            "top_categories": cs.top_categories,
            "rare_categories": [
                k for k, v in cs.value_counts.items()
                if v < non_null_count * 0.01
            ][:20],
            "rare_category_count": cs.rare_category_count,
            "rare_category_percentage": cs.rare_category_percentage,
            "contains_unknown": contains_unknown,
            "case_variations": cs.case_variations,
            "whitespace_issues": whitespace_issues,
            "encoding_recommendation": encoding_rec,
        }

    @staticmethod
    def _identifier_metrics_from_bulk(series: Series, ids: IdentifierColumnStats) -> dict:
        duplicate_values: list = []
        if not ids.is_unique:
            duplicated = series[series.duplicated(keep=False)]
            duplicate_values = head_as_list(duplicated.drop_duplicates(), 10)

        str_sample = head_as_list(series.dropna().astype(str), 100)
        format_pattern = None
        format_consistency = None
        if str_sample:
            import re
            pattern_map = {
                r'^[A-Z]{3}-\d{5}$': 'AAA-99999',
                r'^\d{3}-\d{3}-\d{4}$': '999-999-9999',
                r'^[A-Z]{2}\d{6}$': 'AA999999',
                r'^\d+$': 'numeric_only',
                r'^[A-Za-z]+$': 'alpha_only',
                r'^[A-Z][0-9]{4,}$': 'A9999+',
                r'^\w+-\d+$': 'text-digits',
                r'^[A-Z0-9]+$': 'alphanumeric',
            }
            for pat, desc in pattern_map.items():
                matches = sum(1 for s in str_sample if re.match(pat, s))
                pct = matches / len(str_sample) * 100
                if pct > 80:
                    format_pattern = desc
                    format_consistency = round(pct, 2)
                    break
            if format_pattern is None:
                format_pattern = "mixed"
                format_consistency = 0.0

        return {
            "is_unique": ids.is_unique,
            "duplicate_count": ids.duplicate_count,
            "duplicate_values": duplicate_values,
            "format_pattern": format_pattern,
            "format_consistency": format_consistency,
            "length_min": ids.length_min,
            "length_max": ids.length_max,
            "length_mode": ids.length_mode,
        }

    @staticmethod
    def _binary_metrics_from_bulk(series: Series, bs: BinaryColumnStats) -> dict:
        values_found = head_as_list(series.dropna().drop_duplicates(), 2)
        return {
            "true_count": bs.true_count,
            "false_count": bs.false_count,
            "true_percentage": bs.true_percentage,
            "balance_ratio": bs.balance_ratio,
            "values_found": values_found,
            "is_boolean": bs.is_boolean,
        }

    @staticmethod
    def _text_metrics_from_bulk(ts: TextColumnStats, total_count: int) -> dict:
        pii_detected = (
            ts.pii_email_count > 0
            or ts.pii_phone_count > 0
            or ts.pii_ssn_count > 0
            or ts.pii_cc_count > 0
        )
        pii_types: list[str] = []
        if ts.pii_email_count > 0:
            pii_types.append("email")
        if ts.pii_phone_count > 0:
            pii_types.append("phone")
        if ts.pii_ssn_count > 0:
            pii_types.append("ssn")
        if ts.pii_cc_count > 0:
            pii_types.append("credit_card")

        return {
            "length_min": ts.length_min,
            "length_max": ts.length_max,
            "length_mean": ts.length_mean,
            "length_median": ts.length_median,
            "empty_count": ts.empty_count,
            "empty_percentage": ts.empty_percentage,
            "word_count_mean": ts.word_count_mean,
            "contains_digits_pct": ts.contains_digits_pct,
            "contains_special_pct": ts.contains_special_pct,
            "pii_detected": pii_detected,
            "pii_types": pii_types,
            "language_detected": None,
        }

    def _track_special_columns(
        self, findings: ExplorationFindings, column_finding: ColumnFinding,
        col_stats: Optional[PerColumnStats] = None,
    ):
        if column_finding.inferred_type == ColumnType.TARGET:
            findings.target_column = column_finding.name
            distinct = col_stats.distinct_count if col_stats else 0
            findings.target_type = "binary" if distinct == 2 else "multiclass"
        elif column_finding.inferred_type == ColumnType.IDENTIFIER:
            findings.identifier_columns.append(column_finding.name)
        elif column_finding.inferred_type == ColumnType.DATETIME:
            findings.datetime_columns.append(column_finding.name)

    def _confidence_to_float(self, confidence) -> float:
        mapping = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}
        return mapping.get(confidence.name if hasattr(confidence, "name") else str(confidence), 0.5)

    def _identify_quality_issues(self, universal: dict, type_specific: dict) -> List[str]:
        issues = []
        null_pct = universal.get("null_percentage", 0)
        if null_pct > 50:
            issues.append(f"CRITICAL: {null_pct:.1f}% missing values")
        elif null_pct > 20:
            issues.append(f"WARNING: {null_pct:.1f}% missing values")
        elif null_pct > 5:
            issues.append(f"INFO: {null_pct:.1f}% missing values")
        if type_specific.get("cardinality", 0) > 100:
            issues.append(f"High cardinality: {type_specific['cardinality']} unique values")
        if type_specific.get("outlier_percentage", 0) > 10:
            issues.append(f"WARNING: {type_specific['outlier_percentage']:.1f}% outliers detected")
        if type_specific.get("pii_detected"):
            issues.append(f"CRITICAL: PII detected ({', '.join(type_specific.get('pii_types', []))})")
        if type_specific.get("case_variations"):
            issues.append("Case inconsistency in values")
        if type_specific.get("future_date_count", 0) > 0:
            issues.append(f"Future dates found: {type_specific['future_date_count']}")
        return issues

    def _calculate_column_quality(self, universal: dict, issues: List[str]) -> float:
        score = 100.0
        score -= min(30, universal.get("null_percentage", 0) * 0.5)
        score -= sum(1 for i in issues if "CRITICAL" in i) * 15
        score -= sum(1 for i in issues if "WARNING" in i) * 5
        return max(0, score)

    def _generate_cleaning_recommendations(self, universal: dict, issues: List[str]) -> List[str]:
        recs = []
        null_pct = universal.get("null_percentage", 0)
        if null_pct > 50:
            recs.append("Consider dropping column (>50% missing)")
        elif null_pct > 20:
            recs.append("Impute missing values (mean/median/mode)")
        elif null_pct > 0:
            recs.append("Handle missing values")
        if any("Case inconsistency" in i for i in issues):
            recs.append("Standardize case (lowercase/uppercase)")
        if any("PII detected" in i for i in issues):
            recs.append("REQUIRED: Anonymize or remove PII")
        return recs

    def _generate_transformation_recommendations(self, col_type: ColumnType, metrics: dict) -> List[str]:
        recs = []
        if col_type == ColumnType.NUMERIC_CONTINUOUS:
            if abs(metrics.get("skewness", 0) or 0) > 1:
                recs.append("Apply log transform (high skewness)")
            if metrics.get("outlier_percentage", 0) > 5:
                recs.append("Consider robust scaling")
            else:
                recs.append("Apply standard scaling")
        elif col_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
            recs.append(f"Encoding: {metrics.get('encoding_recommendation', 'one_hot')}")
            if metrics.get("rare_category_count", 0) > 5:
                recs.append("Consider grouping rare categories")
        elif col_type == ColumnType.DATETIME:
            recs.append("Extract temporal features (year, month, day, weekday)")
            recs.append("Calculate days since reference date")
        elif col_type == ColumnType.CATEGORICAL_CYCLICAL:
            recs.append("Apply cyclical encoding (sin/cos)")
        return recs

    def _calculate_overall_metrics(self, findings: ExplorationFindings):
        if not findings.columns:
            return
        scores = [col.quality_score for col in findings.columns.values()]
        findings.overall_quality_score = sum(scores) / len(scores)

    def _check_modeling_readiness(self, findings: ExplorationFindings):
        findings.blocking_issues = []
        if not findings.target_column:
            findings.blocking_issues.append("No target column detected")
        critical_quality = [
            col.name for col in findings.columns.values() if any("CRITICAL" in issue for issue in col.quality_issues)
        ]
        if critical_quality:
            findings.blocking_issues.append(f"Critical issues in: {', '.join(critical_quality)}")
        findings.modeling_ready = len(findings.blocking_issues) == 0

    def _display_summary(self, findings: ExplorationFindings):
        try:
            from customer_retention.analysis.visualization import ChartBuilder, display_summary

            display_summary(findings, ChartBuilder())
        except ImportError:
            self._print_text_summary(findings)

    def _print_text_summary(self, findings: ExplorationFindings):
        print(f"\n{'=' * 60}")
        print(f"EXPLORATION SUMMARY: {findings.source_path}")
        print(f"{'=' * 60}")
        print(f"Rows: {findings.row_count:,} | Columns: {findings.column_count}")
        print(f"Memory: {findings.memory_usage_mb:.2f} MB")
        print(f"Overall Quality Score: {findings.overall_quality_score:.1f}/100")
        print()
        if findings.target_column:
            print(f"Target Column: {findings.target_column} ({findings.target_type})")
        else:
            print("WARNING: No target column detected!")
        print()
        print("Column Types Detected:")
        print("-" * 40)
        for name, col in findings.columns.items():
            conf = "HIGH" if col.confidence > 0.8 else "MED" if col.confidence > 0.5 else "LOW"
            issues = len(col.quality_issues)
            print(f"  {name}: {col.inferred_type.value} [{conf}] {f'({issues} issues)' if issues else ''}")
        if findings.blocking_issues:
            print()
            print("BLOCKING ISSUES:")
            for issue in findings.blocking_issues:
                print(f"  - {issue}")
        print()
        print(f"Modeling Ready: {'YES' if findings.modeling_ready else 'NO'}")
        print(f"{'=' * 60}\n")

    def _compute_findings_path(self, findings: ExplorationFindings, name: Optional[str]) -> str:
        if name is None:
            name = Path(findings.source_path).stem if findings.source_path != "<DataFrame>" else "exploration"
        return str(self.output_dir / f"{name}_findings.yaml")

    def _save_findings(self, findings: ExplorationFindings):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        findings.save(self.last_findings_path)
        print(f"Findings saved to: {self.last_findings_path}")
