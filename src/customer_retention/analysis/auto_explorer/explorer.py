from pathlib import Path
from typing import Union, Optional, List, Any
import hashlib

from customer_retention.core.compat import pd, DataFrame, Series, to_pandas
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.profiling import TypeDetector, ProfilerFactory
from customer_retention.stages.temporal import TEMPORAL_METADATA_COLS
from .findings import ExplorationFindings, ColumnFinding


class DataExplorer:
    def __init__(self, visualize: bool = True, save_findings: bool = True, output_dir: str = "../explorations"):
        self.visualize = visualize
        self.save_findings = save_findings
        self.output_dir = Path(output_dir)
        self.type_detector = TypeDetector()
        self.last_findings_path: Optional[str] = None

    def explore(self, source: Union[str, DataFrame], target_hint: Optional[str] = None,
                name: Optional[str] = None) -> ExplorationFindings:
        df, source_path, source_format = self._load_source(source)
        findings = self._create_findings(df, source_path, source_format)
        self._explore_all_columns(df, findings, target_hint)
        self._calculate_overall_metrics(findings)
        self._check_modeling_readiness(findings)
        if self.visualize:
            self._display_summary(findings)
        if self.save_findings:
            self._save_findings(findings, name)
        return findings

    def _load_source(self, source: Union[str, DataFrame]) -> tuple:
        if hasattr(source, 'columns'):
            return to_pandas(source), "<DataFrame>", "dataframe"
        path = Path(source)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(source), source, "csv"
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(source), source, "parquet"
        return pd.read_csv(source), source, "csv"

    def _create_findings(self, df: DataFrame, source_path: str, source_format: str) -> ExplorationFindings:
        return ExplorationFindings(
            source_path=source_path,
            source_format=source_format,
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024)
        )

    def _explore_all_columns(self, df: DataFrame, findings: ExplorationFindings, target_hint: Optional[str]):
        for column_name in df.columns:
            # Skip temporal metadata columns added by the snapshot framework
            # These are system columns, not features for analysis
            if column_name in TEMPORAL_METADATA_COLS:
                continue
            column_finding = self._explore_column(df[column_name], column_name, target_hint)
            findings.columns[column_name] = column_finding
            self._track_special_columns(findings, column_finding, df[column_name])

    def _explore_column(self, series: Series, column_name: str, target_hint: Optional[str]) -> ColumnFinding:
        type_inference = self.type_detector.detect_type(series, column_name)
        if target_hint and column_name.lower() == target_hint.lower():
            type_inference.inferred_type = ColumnType.TARGET
            type_inference.evidence.append(f"Matched target hint: {target_hint}")
        universal_metrics = self._compute_universal_metrics(series, type_inference.inferred_type)
        type_metrics = self._compute_type_metrics(series, type_inference.inferred_type)
        quality_issues = self._identify_quality_issues(universal_metrics, type_metrics)
        quality_score = self._calculate_column_quality(universal_metrics, quality_issues)
        cleaning_recommendations = self._generate_cleaning_recommendations(universal_metrics, quality_issues)
        transformation_recommendations = self._generate_transformation_recommendations(type_inference.inferred_type, type_metrics)
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
            transformation_recommendations=transformation_recommendations
        )

    def _compute_universal_metrics(self, series: Series, col_type: ColumnType) -> dict:
        profiler = ProfilerFactory.get_profiler(col_type)
        if not profiler:
            return {}
        universal = profiler.compute_universal_metrics(series)
        return {
            "total_count": universal.total_count,
            "null_count": universal.null_count,
            "null_percentage": universal.null_percentage,
            "distinct_count": universal.distinct_count,
            "distinct_percentage": universal.distinct_percentage,
            "most_common_value": str(universal.most_common_value) if universal.most_common_value is not None else None,
            "most_common_frequency": universal.most_common_frequency,
            "memory_size_bytes": universal.memory_size_bytes
        }

    def _compute_type_metrics(self, series: Series, col_type: ColumnType) -> dict:
        profiler = ProfilerFactory.get_profiler(col_type)
        if not profiler:
            return {}
        profile_result = profiler.profile(series)
        for value in profile_result.values():
            if value is not None and hasattr(value, "__dict__"):
                return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        return {}

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
            col.name for col in findings.columns.values()
            if any("CRITICAL" in issue for issue in col.quality_issues)
        ]
        if critical_quality:
            findings.blocking_issues.append(f"Critical issues in: {', '.join(critical_quality)}")
        findings.modeling_ready = len(findings.blocking_issues) == 0

    def _display_summary(self, findings: ExplorationFindings):
        try:
            from customer_retention.analysis.visualization import display_summary, ChartBuilder
            display_summary(findings, ChartBuilder())
        except ImportError:
            self._print_text_summary(findings)

    def _print_text_summary(self, findings: ExplorationFindings):
        print(f"\n{'='*60}")
        print(f"EXPLORATION SUMMARY: {findings.source_path}")
        print(f"{'='*60}")
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
        print(f"{'='*60}\n")

    def _save_findings(self, findings: ExplorationFindings, name: Optional[str]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            name = Path(findings.source_path).stem if findings.source_path != "<DataFrame>" else "exploration"
        path_hash = hashlib.md5(findings.source_path.encode()).hexdigest()[:6]
        path = self.output_dir / f"{name}_{path_hash}_findings.yaml"
        findings.save(str(path))
        self.last_findings_path = str(path)
        print(f"Findings saved to: {path}")
