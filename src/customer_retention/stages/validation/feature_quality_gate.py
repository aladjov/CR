from datetime import datetime

from customer_retention.core.compat import DataFrame
from customer_retention.core.config import ColumnType, DataSourceConfig
from customer_retention.stages.profiling import ColumnProfile, ProfilerFactory, TypeDetector
from customer_retention.stages.profiling.profile_result import ProfileResult, UniversalMetrics
from customer_retention.stages.profiling.quality_checks import QualityCheckRegistry, QualityCheckResult

from .gates import GateResult, Severity, ValidationGate


class FeatureQualityGate(ValidationGate):
    def __init__(self, fail_on_critical: bool = True, fail_on_high: bool = False):
        super().__init__("Feature Quality Gate (Checkpoint 2)")
        self.fail_on_critical = fail_on_critical
        self.fail_on_high = fail_on_high
        self.type_detector = TypeDetector()

    def run(self, df: DataFrame, config: DataSourceConfig) -> GateResult:
        issues = []
        start_time = datetime.now()

        for column_config in config.columns:
            column_name = column_config.name

            if column_name not in df.columns:
                issues.append(self.create_issue(
                    "FQ000",
                    f"Configured column '{column_name}' not found in dataframe",
                    Severity.CRITICAL,
                    column_name,
                    len(df),
                    len(df)
                ))
                continue

            series = df[column_name]

            profiler = ProfilerFactory.get_profiler(column_config.column_type)
            if profiler is None:
                continue

            universal_metrics = profiler.compute_universal_metrics(series)
            specific_metrics = profiler.profile(series)

            check_results = self.run_quality_checks(
                column_name,
                column_config.column_type,
                universal_metrics,
                specific_metrics,
                column_config.should_be_used_as_feature()
            )

            for check_result in check_results:
                if not check_result.passed:
                    issues.append(self.create_issue(
                        check_result.check_id,
                        check_result.message,
                        check_result.severity,
                        column_name,
                        None,
                        len(df),
                        recommendation=check_result.recommendation
                    ))

        duration = (datetime.now() - start_time).total_seconds()

        return self.create_result(
            issues,
            duration,
            fail_on_critical=self.fail_on_critical,
            fail_on_high=self.fail_on_high,
            metadata={
                "total_columns": len(config.columns),
                "duration_seconds": round(duration, 3)
            }
        )

    def run_quality_checks(self, column_name: str, column_type: ColumnType,
                          universal_metrics: UniversalMetrics, specific_metrics: dict,
                          should_use_as_feature: bool) -> list[QualityCheckResult]:
        results = []
        checks = QualityCheckRegistry.get_checks_for_column_type(column_type)

        for check in checks:
            result = None

            if check.check_id == "FQ001":
                result = check.run(column_name, universal_metrics)
            elif check.check_id == "FQ003":
                result = check.run(column_name, universal_metrics, column_type)
            elif check.check_id in ["CAT001", "FQ009"]:
                result = check.run(column_name, specific_metrics.get("categorical_metrics"))
            elif check.check_id == "CAT002":
                result = check.run(column_name, specific_metrics.get("target_metrics"))
            elif check.check_id in ["NUM002", "NUM003", "NUM004"]:
                result = check.run(column_name, specific_metrics.get("numeric_metrics"))
            elif check.check_id == "LEAK001":
                result = check.run(column_name, column_type, should_use_as_feature)
            elif check.check_id == "DT001":
                result = check.run(column_name, specific_metrics.get("datetime_metrics"))
            elif check.check_id == "DT002":
                result = check.run(column_name, specific_metrics.get("datetime_metrics"), universal_metrics.total_count)
            elif check.check_id in ["CAT003", "CAT004"]:
                result = check.run(column_name, specific_metrics.get("categorical_metrics"))
            elif check.check_id == "NUM001":
                result = check.run(column_name, universal_metrics, column_type)
            elif check.check_id.startswith("TG"):
                if check.check_id == "TG001":
                    result = check.run(column_name, universal_metrics)
                else:
                    result = check.run(column_name, specific_metrics.get("target_metrics"))
            elif check.check_id.startswith("NC"):
                result = check.run(column_name, specific_metrics.get("numeric_metrics"))
            elif check.check_id.startswith("TX"):
                result = check.run(column_name, specific_metrics.get("text_metrics") if check.check_id != "TX004" else universal_metrics)
            elif check.check_id.startswith("ID"):
                if check.check_id == "ID003":
                    result = check.run(column_name, universal_metrics)
                else:
                    result = check.run(column_name, specific_metrics.get("identifier_metrics"))
            elif check.check_id.startswith("CN"):
                result = check.run(column_name, specific_metrics.get("categorical_metrics"))
            elif check.check_id.startswith("DT") and int(check.check_id[2:]) > 2:
                result = check.run(column_name, specific_metrics.get("datetime_metrics"))
            elif check.check_id.startswith("BN"):
                if check.check_id in ["BN001", "BN003"]:
                    result = check.run(column_name, universal_metrics)
                else:
                    result = check.run(column_name, specific_metrics.get("binary_metrics"))
            elif check.check_id in ["FQ005", "FQ008", "FQ011", "FQ012"]:
                pass

            if result:
                results.append(result)

        return results

    def profile_and_validate(self, df: DataFrame, config: DataSourceConfig) -> tuple[ProfileResult, GateResult]:
        start_time = datetime.now()
        column_profiles = {}

        for column_config in config.columns:
            column_name = column_config.name

            if column_name not in df.columns:
                continue

            series = df[column_name]

            type_inference = self.type_detector.detect_type(series, column_name)

            profiler = ProfilerFactory.get_profiler(column_config.column_type)
            if profiler is None:
                continue

            universal_metrics = profiler.compute_universal_metrics(series)
            specific_metrics = profiler.profile(series)

            column_profile = ColumnProfile(
                column_name=column_name,
                configured_type=column_config.column_type,
                inferred_type=type_inference,
                universal_metrics=universal_metrics,
                **specific_metrics
            )

            column_profiles[column_name] = column_profile

        duration = (datetime.now() - start_time).total_seconds()

        profile_result = ProfileResult(
            dataset_name=config.name,
            total_rows=len(df),
            total_columns=len(df.columns),
            column_profiles=column_profiles,
            profiling_timestamp=datetime.now().isoformat(),
            profiling_duration_seconds=round(duration, 3)
        )

        gate_result = self.run(df, config)

        return profile_result, gate_result
