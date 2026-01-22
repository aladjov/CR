import time

from customer_retention.core.compat import DataFrame, Timestamp, is_datetime64_any_dtype, pd
from customer_retention.core.config.column_config import ColumnType
from customer_retention.core.config.pipeline_config import BronzeConfig, PipelineConfig

from .gates import GateResult, Severity, ValidationGate, ValidationIssue


class DataQualityGate(ValidationGate):
    def __init__(self):
        super().__init__("DataQualityGate")

    def run(self, df: DataFrame, config: PipelineConfig) -> GateResult:
        start_time = time.time()
        issues = []

        issues.extend(self.check_missing_values(df))
        issues.extend(self.check_duplicates(df, config.bronze))
        issues.extend(self.check_target_column(df, config))
        issues.extend(self.check_class_imbalance(df, config))
        issues.extend(self.check_temporal_validity(df, config))
        issues.extend(self.check_type_mismatches(df, config))

        duration = time.time() - start_time
        return self.create_result(
            issues,
            duration,
            fail_on_critical=config.validation.fail_on_critical,
            fail_on_high=config.validation.fail_on_high,
            metadata={"row_count": len(df), "column_count": len(df.columns)}
        )

    def check_missing_values(self, df: DataFrame) -> list[ValidationIssue]:
        issues = []
        total_rows = len(df)

        for column in df.columns:
            missing_count = df[column].isna().sum()
            if missing_count == 0:
                continue

            missing_pct = missing_count / total_rows

            if missing_pct > 0.5:
                issues.append(self.create_issue(
                    "DQ001", "Critical missing values in column",
                    Severity.CRITICAL, column, missing_count, total_rows,
                    "Consider dropping this column or investigating data source"
                ))
            elif missing_pct > 0.3:
                issues.append(self.create_issue(
                    "DQ002", "High missing values in column",
                    Severity.HIGH, column, missing_count, total_rows,
                    "Review imputation strategy or consider feature engineering"
                ))

        return issues

    def check_duplicates(self, df: DataFrame, bronze_config: BronzeConfig) -> list[ValidationIssue]:
        issues = []
        total_rows = len(df)

        missing_keys = [key for key in bronze_config.dedup_keys if key not in df.columns]
        if missing_keys:
            issues.append(self.create_issue(
                "DQ010", f"Deduplication keys not found: {', '.join(missing_keys)}",
                Severity.HIGH, None, None, None,
                "Update dedup_keys configuration or check data source"
            ))
            return issues

        duplicate_count = df.duplicated(subset=bronze_config.dedup_keys).sum()
        if duplicate_count == 0:
            return issues

        dup_pct = duplicate_count / total_rows

        if dup_pct > 0.1:
            issues.append(self.create_issue(
                "DQ011", f"High duplicate rate on keys {bronze_config.dedup_keys}",
                Severity.HIGH, None, duplicate_count, total_rows,
                f"Review data source or adjust dedup strategy to {bronze_config.dedup_strategy.value}",
                auto_fixable=True
            ))
        else:
            issues.append(self.create_issue(
                "DQ012", f"Duplicates present on keys {bronze_config.dedup_keys}",
                Severity.MEDIUM, None, duplicate_count, total_rows,
                f"Will be handled by deduplication with {bronze_config.dedup_strategy.value}",
                auto_fixable=True
            ))

        return issues

    def check_target_column(self, df: DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
        issues = []
        target_column = config.modeling.target_column

        if target_column not in df.columns:
            issues.append(self.create_issue(
                "DQ020", f"Target column '{target_column}' not found in dataset",
                Severity.CRITICAL, target_column, None, None,
                "Check column name in configuration or data source"
            ))
            return issues

        null_count = df[target_column].isna().sum()
        if null_count > 0:
            issues.append(self.create_issue(
                "DQ021", "Target column has null values",
                Severity.CRITICAL, target_column, null_count, len(df),
                "Drop rows with null target or investigate data quality"
            ))

        return issues

    def check_class_imbalance(self, df: DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
        issues = []
        target_column = config.modeling.target_column

        if target_column not in df.columns:
            return issues

        value_counts = df[target_column].value_counts()
        if len(value_counts) < 2:
            return issues

        total_rows = len(df)
        minority_count = value_counts.min()
        minority_pct = minority_count / total_rows

        if minority_pct < 0.01:
            issues.append(self.create_issue(
                "DQ022", "Severe class imbalance in target",
                Severity.HIGH, target_column, minority_count, total_rows,
                "Consider oversampling, undersampling, or SMOTE"
            ))
        elif minority_pct < 0.1:
            issues.append(self.create_issue(
                "DQ023", "Moderate class imbalance in target",
                Severity.MEDIUM, target_column, minority_count, total_rows,
                "Monitor model performance on minority class"
            ))

        return issues

    def check_temporal_validity(self, df: DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
        issues = []
        date_columns = self.get_date_columns(df, config)

        for column in date_columns:
            if column not in df.columns:
                continue

            try:
                df_temp = df[column].dropna()
                if len(df_temp) == 0:
                    continue

                if not is_datetime64_any_dtype(df_temp):
                    df_temp = pd.to_datetime(df_temp, errors='coerce', format='mixed')

                future_dates = df_temp > Timestamp.now()
                future_count = future_dates.sum()

                if future_count > 0:
                    issues.append(self.create_issue(
                        "DQ030", "Future dates detected",
                        Severity.HIGH, column, future_count, len(df),
                        "Verify data source or timezone handling"
                    ))
            except Exception:
                pass

        issues.extend(self.check_temporal_logic(df, config))
        return issues

    def check_temporal_logic(self, df: DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
        issues = []

        if 'created' in df.columns and 'firstorder' in df.columns:
            try:
                df_temp = df[['created', 'firstorder']].dropna()
                if len(df_temp) == 0:
                    return issues

                created = pd.to_datetime(df_temp['created'], errors='coerce', format='mixed')
                firstorder = pd.to_datetime(df_temp['firstorder'], errors='coerce', format='mixed')

                violations = created > firstorder
                violation_count = violations.sum()

                if violation_count > 0:
                    issues.append(self.create_issue(
                        "DQ031", "Temporal logic violation: created > firstorder",
                        Severity.HIGH, "created,firstorder", violation_count, len(df),
                        "Review date logic in data source"
                    ))
            except Exception:
                pass

        return issues

    def check_type_mismatches(self, df: DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
        issues = []

        for source in config.data_sources:
            for col_config in source.columns:
                if col_config.name not in df.columns:
                    continue

                column_data = df[col_config.name]

                if col_config.is_numeric() and column_data.dtype == 'object':
                    try:
                        pd.to_numeric(column_data.dropna(), errors='raise')
                        issues.append(self.create_issue(
                            "DQ040", "Numeric column stored as string",
                            Severity.MEDIUM, col_config.name, len(df), len(df),
                            "Convert to numeric type during processing",
                            auto_fixable=True
                        ))
                    except Exception:
                        pass

        return issues

    def get_date_columns(self, df: DataFrame, config: PipelineConfig) -> list[str]:
        date_columns = []
        for source in config.data_sources:
            for col_config in source.columns:
                if col_config.column_type == ColumnType.DATETIME:
                    date_columns.append(col_config.name)
        return date_columns
