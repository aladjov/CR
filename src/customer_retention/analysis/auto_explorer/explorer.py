import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

from customer_retention.core.compat import DataFrame, Series, is_dataframe, pd, safe_isfinite, safe_memory_usage_bytes
from customer_retention.core.compat.bulk_profiling import (
    NumericColumnStats,
    PerColumnStats,
    compute_bulk_stats,
)
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory, TypeDetector
from customer_retention.stages.temporal import TEMPORAL_METADATA_COLS

from .findings import ColumnFinding, ExplorationFindings


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
        df, source_path, source_format = self._load_source(source)
        findings = self._create_findings(df, source_path, source_format)
        self._explore_all_columns(df, findings, target_hint)
        self._calculate_overall_metrics(findings)
        self._check_modeling_readiness(findings)
        if self.visualize:
            self._display_summary(findings)
        self.last_findings_path = self._compute_findings_path(findings, name)
        if self.save_findings:
            self._save_findings(findings)
        return findings

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
        logger.info("Computing bulk statistics for %d columns", total_cols)
        bulk = compute_bulk_stats(df)
        logger.info("Bulk statistics complete (%d numeric columns)", len(bulk.numeric))

        processed = 0
        log_interval = max(1, total_cols // 5)
        for column_name in df.columns:
            if column_name in TEMPORAL_METADATA_COLS:
                continue
            col_stats = bulk.columns.get(column_name)
            numeric_stats = bulk.numeric.get(column_name)
            column_finding = self._explore_column(
                df[column_name],
                column_name,
                target_hint,
                bulk_total=bulk.total_count,
                col_stats=col_stats,
                numeric_stats=numeric_stats,
            )
            findings.columns[column_name] = column_finding
            self._track_special_columns(findings, column_finding, df[column_name])
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
    ) -> ColumnFinding:
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
        type_metrics = self._compute_type_metrics(series, type_inference.inferred_type, numeric_stats=numeric_stats)
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

            value_counts = series.value_counts()
            if len(value_counts) > 0:
                most_common_value = value_counts.idxmax()
                most_common_frequency = int(value_counts.iloc[0])
            else:
                most_common_value = None
                most_common_frequency = None

            return {
                "total_count": total_count,
                "null_count": null_count,
                "null_percentage": null_percentage,
                "distinct_count": distinct_count,
                "distinct_percentage": distinct_percentage,
                "most_common_value": str(most_common_value) if most_common_value is not None else None,
                "most_common_frequency": most_common_frequency,
                "memory_size_bytes": safe_memory_usage_bytes(series),
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
        self, series: Series, col_type: ColumnType, numeric_stats: Optional[NumericColumnStats] = None
    ) -> dict:
        if numeric_stats is not None and col_type in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE):
            return self._numeric_metrics_from_bulk(series, numeric_stats)

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

        clean_series = series.dropna()
        non_null_count = len(clean_series) if len(clean_series) > 0 else 1

        zero_pct = round((ns.zero_count / non_null_count * 100), 2)
        neg_pct = round((ns.negative_count / non_null_count * 100), 2)
        inf_pct = round((ns.inf_count / non_null_count * 100), 2)
        outlier_pct = round((ns.outlier_count_iqr / non_null_count * 100), 2)

        finite_series = clean_series[safe_isfinite(clean_series)]
        if len(finite_series) > 0:
            arr = finite_series.to_numpy() if hasattr(finite_series, "to_numpy") else np.asarray(finite_series)
            histogram, bin_edges = np.histogram(arr, bins=10)
            histogram_bins = [
                (round(float(bin_edges[i]), 4), round(float(bin_edges[i + 1]), 4), int(histogram[i]))
                for i in range(len(histogram))
            ]
        else:
            histogram_bins = []

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

    def _track_special_columns(self, findings: ExplorationFindings, column_finding: ColumnFinding, series: Series):
        if column_finding.inferred_type == ColumnType.TARGET:
            findings.target_column = column_finding.name
            findings.target_type = "binary" if series.nunique() == 2 else "multiclass"
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
