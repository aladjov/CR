from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation, RecommendationRegistry
from customer_retention.core.compat.detection import is_databricks


class DataMaterializer:
    def __init__(self, registry: RecommendationRegistry, output_dir: Optional[str] = None):
        self.registry = registry
        self.output_dir = output_dir or "./prepared_data"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_bronze(df)
        df = self.apply_silver(df)
        df = self.apply_gold(df)
        return df

    def materialize(self, df: pd.DataFrame, output_name: str) -> Tuple[pd.DataFrame, str]:
        result_df = self.transform(df)
        output_path = self._save(result_df, output_name)
        return result_df, output_path

    def apply_bronze(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.registry.bronze:
            return df
        df = df.copy()
        for rec in self.registry.bronze.null_handling:
            df = self._apply_null_handling(df, rec)
        for rec in self.registry.bronze.outlier_handling:
            df = self._apply_outlier_handling(df, rec)
        for rec in self.registry.bronze.type_conversions:
            df = self._apply_type_conversion(df, rec)
        for rec in self.registry.bronze.filtering:
            df = self._apply_filtering(df, rec)
        return df

    def apply_silver(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.registry.silver:
            return df
        df = df.copy()
        for rec in self.registry.silver.aggregations:
            df = self._apply_aggregation(df, rec)
        for rec in self.registry.silver.derived_columns:
            df = self._apply_derived(df, rec)
        return df

    def apply_gold(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.registry.gold:
            return df
        df = df.copy()
        for rec in self.registry.gold.transformations:
            df = self._apply_transformation(df, rec)
        for rec in self.registry.gold.encoding:
            df = self._apply_encoding(df, rec)
        for rec in self.registry.gold.scaling:
            df = self._apply_scaling(df, rec)
        return df

    def _apply_null_handling(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        strategy = rec.parameters.get("strategy", "median")
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else df[col])
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        return df

    def _apply_outlier_handling(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        method = rec.parameters.get("method", "iqr")
        if method == "iqr":
            factor = rec.parameters.get("factor", 1.5)
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    def _apply_type_conversion(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        target_type = rec.parameters.get("target_type", "str")
        if target_type == "datetime":
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(target_type)
        return df

    def _apply_filtering(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if rec.action == "drop" and col in df.columns:
            df = df.drop(columns=[col])
        return df

    def _apply_aggregation(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns or not self.registry.silver:
            return df
        entity_col = self.registry.silver.entity_column
        agg_func = rec.parameters.get("aggregation", "sum")
        feature_name = f"{col}_{agg_func}"
        df[feature_name] = df.groupby(entity_col)[col].transform(agg_func)
        return df

    def _apply_derived(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        return df

    def _apply_transformation(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        method = rec.parameters.get("method", "log")
        if method == "log":
            df[col] = np.log1p(df[col])
        elif method == "sqrt":
            df[col] = np.sqrt(df[col])
        return df

    def _apply_encoding(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        method = rec.parameters.get("method", "one_hot")
        if method == "one_hot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _apply_scaling(self, df: pd.DataFrame, rec: LayeredRecommendation) -> pd.DataFrame:
        col = rec.target_column
        if col not in df.columns:
            return df
        method = rec.parameters.get("method", "standard")
        scaler = StandardScaler() if method == "standard" else RobustScaler()
        df[col] = scaler.fit_transform(df[[col]])
        return df

    def _save(self, df: pd.DataFrame, output_name: str) -> str:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if is_databricks():
            output_path = f"{self.output_dir}/{output_name}"
            self._save_delta(df, output_path)
        else:
            output_path = f"{self.output_dir}/{output_name}.parquet"
            df.to_parquet(output_path, index=False)
        return output_path

    def _save_delta(self, df: pd.DataFrame, path: str) -> None:
        try:
            from customer_retention.integrations.adapters.storage.local import LocalDelta
            storage = LocalDelta()
            storage.write(df, path)
        except ImportError:
            df.to_parquet(f"{path}.parquet", index=False)
