from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from customer_retention.core.compat import pd, DataFrame

from customer_retention.core.config import ColumnType
from customer_retention.stages.cleaning import MissingValueHandler, OutlierHandler, OutlierTreatmentStrategy
from .numeric_transformer import NumericTransformer, ScalingStrategy
from .categorical_encoder import CategoricalEncoder, EncodingStrategy
from .datetime_transformer import DatetimeTransformer
from .binary_handler import BinaryHandler


@dataclass
class TransformationManifest:
    timestamp: str = ""
    version: str = "1.0"
    input_rows: int = 0
    input_columns: int = 0
    output_rows: int = 0
    output_columns: int = 0
    columns_dropped: dict = field(default_factory=dict)
    missing_value_handling: dict = field(default_factory=dict)
    outlier_treatment: dict = field(default_factory=dict)
    numeric_transformations: dict = field(default_factory=dict)
    categorical_encodings: dict = field(default_factory=dict)
    datetime_transformations: dict = field(default_factory=dict)
    binary_mappings: dict = field(default_factory=dict)
    column_mapping: dict = field(default_factory=dict)
    final_schema: dict = field(default_factory=dict)
    execution_order: list = field(default_factory=list)


@dataclass
class PipelineResult:
    df: DataFrame
    manifest: TransformationManifest
    validation_passed: bool = True
    validation_errors: list = field(default_factory=list)


class TransformationPipeline:
    EXECUTION_ORDER = [
        "drop_columns", "handle_missing", "treat_outliers",
        "transform_datetime", "transform_numeric",
        "encode_categorical", "standardize_binary", "validate"
    ]

    def __init__(
        self,
        column_types: Optional[dict[str, ColumnType]] = None,
        auto_from_profile: bool = True,
        column_configs: Optional[dict] = None,
        drop_constant_columns: bool = False,
        drop_high_missing: bool = True,
        create_missing_indicators: bool = False,
        validate_output: bool = True
    ):
        self.column_types = column_types or {}
        self.auto_from_profile = auto_from_profile
        self.column_configs = column_configs or {}
        self.drop_constant_columns = drop_constant_columns
        self.drop_high_missing = drop_high_missing
        self.create_missing_indicators = create_missing_indicators
        self.validate_output = validate_output

        self._missing_handlers: dict[str, MissingValueHandler] = {}
        self._outlier_handlers: dict[str, OutlierHandler] = {}
        self._numeric_transformers: dict[str, NumericTransformer] = {}
        self._categorical_encoders: dict[str, CategoricalEncoder] = {}
        self._datetime_transformers: dict[str, DatetimeTransformer] = {}
        self._binary_handlers: dict[str, BinaryHandler] = {}
        self._columns_to_drop: list[str] = []
        self._is_fitted = False

    def fit(self, df: DataFrame) -> "TransformationPipeline":
        self._identify_columns_to_drop(df)
        working_df = df.drop(columns=self._columns_to_drop, errors='ignore')

        for col, col_type in self.column_types.items():
            if col in self._columns_to_drop or col not in working_df.columns:
                continue
            self._fit_column(working_df, col, col_type)

        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> PipelineResult:
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() or fit_transform() first.")
        return self._apply_transformations(df)

    def fit_transform(self, df: DataFrame) -> PipelineResult:
        self.fit(df)
        return self._apply_transformations(df)

    def _identify_columns_to_drop(self, df: DataFrame):
        self._columns_to_drop = []
        for col, col_type in self.column_types.items():
            if col not in df.columns:
                continue
            if col_type == ColumnType.IDENTIFIER:
                self._columns_to_drop.append(col)
            if self.drop_high_missing and df[col].isna().mean() > 0.95:
                self._columns_to_drop.append(col)
            if self.drop_constant_columns and df[col].nunique() <= 1:
                self._columns_to_drop.append(col)

    def _fit_column(self, df: DataFrame, col: str, col_type: ColumnType):
        if col_type == ColumnType.TARGET:
            return

        series = df[col]
        config = self.column_configs.get(col, {})

        if series.isna().any():
            handler = MissingValueHandler.from_column_type(col_type)
            if "missing_strategy" in config:
                from customer_retention.stages.cleaning import ImputationStrategy
                handler.strategy = ImputationStrategy(config["missing_strategy"])
            handler.fit(series)
            self._missing_handlers[col] = handler

        if col_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            self._outlier_handlers[col] = OutlierHandler(
                treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
            )
            self._outlier_handlers[col].fit(series.dropna())

            # Fit numeric transformer on CAPPED data to ensure proper scaling
            outlier_result = self._outlier_handlers[col].transform(series.dropna())
            self._numeric_transformers[col] = NumericTransformer(scaling=ScalingStrategy.STANDARD)
            self._numeric_transformers[col].fit(outlier_result.series)

        elif col_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
            self._categorical_encoders[col] = CategoricalEncoder(
                strategy=EncodingStrategy.ONE_HOT, drop_first=True
            )
            self._categorical_encoders[col].fit(series)

        elif col_type == ColumnType.DATETIME:
            self._datetime_transformers[col] = DatetimeTransformer()
            self._datetime_transformers[col].fit(series)

        elif col_type == ColumnType.BINARY:
            self._binary_handlers[col] = BinaryHandler()
            self._binary_handlers[col].fit(series)

    def _apply_transformations(self, df: DataFrame) -> PipelineResult:
        manifest = TransformationManifest(
            timestamp=datetime.now().isoformat(),
            input_rows=len(df), input_columns=len(df.columns),
            execution_order=self.EXECUTION_ORDER
        )

        working_df = df.copy()

        manifest.columns_dropped = {col: "identifier/high_missing/constant" for col in self._columns_to_drop}
        working_df = working_df.drop(columns=self._columns_to_drop, errors='ignore')

        for col, handler in self._missing_handlers.items():
            if col in working_df.columns:
                result = handler.transform(working_df[col])
                working_df[col] = result.series
                manifest.missing_value_handling[col] = {
                    "strategy": str(result.strategy_used), "values_imputed": result.values_imputed
                }

        for col, handler in self._outlier_handlers.items():
            if col in working_df.columns:
                result = handler.transform(working_df[col])
                working_df[col] = result.series
                manifest.outlier_treatment[col] = {
                    "method": str(result.method_used),
                    "outliers_detected": result.outliers_detected
                }

        datetime_cols_to_drop = []
        datetime_extracted_cols = []
        for col, transformer in self._datetime_transformers.items():
            if col in working_df.columns:
                result = transformer.transform(working_df[col])
                for new_col in result.df.columns:
                    working_df[new_col] = result.df[new_col].values
                    datetime_extracted_cols.append(new_col)
                datetime_cols_to_drop.append(col)
                manifest.datetime_transformations[col] = {
                    "extracted": result.extracted_features
                }
                manifest.column_mapping[col] = list(result.df.columns)
        working_df = working_df.drop(columns=datetime_cols_to_drop, errors='ignore')

        # Handle NaN values from invalid datetime parsing (e.g., '1/0/00')
        for col in datetime_extracted_cols:
            if col in working_df.columns and working_df[col].isna().any():
                # Fill with median for extracted datetime features
                median_val = working_df[col].median()
                if pd.notna(median_val):
                    working_df[col] = working_df[col].fillna(median_val)

        for col, transformer in self._numeric_transformers.items():
            if col in working_df.columns:
                result = transformer.transform(working_df[col])
                working_df[col] = result.series
                manifest.numeric_transformations[col] = {
                    "transformations": [str(t) for t in result.transformations_applied]
                }

        categorical_cols_to_drop = []
        for col, encoder in self._categorical_encoders.items():
            if col in working_df.columns:
                result = encoder.transform(working_df[col])
                if result.df is not None:
                    for new_col in result.df.columns:
                        working_df[new_col] = result.df[new_col].values
                    categorical_cols_to_drop.append(col)
                    manifest.column_mapping[col] = list(result.df.columns)
                manifest.categorical_encodings[col] = {
                    "strategy": str(result.strategy), "columns_created": result.columns_created
                }
        working_df = working_df.drop(columns=categorical_cols_to_drop, errors='ignore')

        for col, handler in self._binary_handlers.items():
            if col in working_df.columns:
                result = handler.transform(working_df[col])
                working_df[col] = result.series
                manifest.binary_mappings[col] = {"mapping": result.mapping}

        validation_passed, validation_errors = self._validate_output(working_df)

        manifest.output_rows = len(working_df)
        manifest.output_columns = len(working_df.columns)
        manifest.final_schema = {col: str(working_df[col].dtype) for col in working_df.columns}

        return PipelineResult(
            df=working_df, manifest=manifest,
            validation_passed=validation_passed, validation_errors=validation_errors
        )

    def _validate_output(self, df: DataFrame) -> tuple[bool, list[str]]:
        errors = []

        target_cols = [c for c, t in self.column_types.items() if t == ColumnType.TARGET and c in df.columns]
        non_target = df.drop(columns=target_cols, errors='ignore')

        if non_target.isna().any().any():
            null_cols = non_target.columns[non_target.isna().any()].tolist()
            errors.append(f"TQ001: Null values in columns: {null_cols}")

        numeric_df = non_target.select_dtypes(include=[np.number])
        if np.isinf(numeric_df.values).any():
            errors.append("TQ002: Infinite values found")

        return len(errors) == 0, errors
