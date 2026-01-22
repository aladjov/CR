from typing import List

from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation, RecommendationRegistry


class PipelineCodeGenerator:
    def __init__(self, registry: RecommendationRegistry):
        self.registry = registry

    def generate_bronze_code(self) -> str:
        lines = [
            "from customer_retention.stages.cleaning import MissingValueHandler, OutlierHandler",
            "",
            "",
            "def bronze_transform(df):",
        ]
        if not self.registry.bronze or not self.registry.bronze.all_recommendations:
            lines.append("    return df")
            return "\n".join(lines)

        for rec in self.registry.bronze.null_handling:
            lines.extend(self._generate_null_handling(rec))
        for rec in self.registry.bronze.outlier_handling:
            lines.extend(self._generate_outlier_handling(rec))
        for rec in self.registry.bronze.type_conversions:
            lines.extend(self._generate_type_conversion(rec))
        for rec in self.registry.bronze.filtering:
            lines.extend(self._generate_filtering(rec))

        lines.append("    return df")
        return "\n".join(lines)

    def generate_silver_code(self) -> str:
        lines = ["", "", "def silver_transform(df):"]
        if not self.registry.silver or not self.registry.silver.all_recommendations:
            lines.append("    return df")
            return "\n".join(lines)

        entity_col = self.registry.silver.entity_column
        time_col = self.registry.silver.time_column

        for rec in self.registry.silver.aggregations:
            lines.extend(self._generate_aggregation(rec, entity_col, time_col))
        for rec in self.registry.silver.derived_columns:
            lines.extend(self._generate_derived(rec))
        for rec in self.registry.silver.joins:
            lines.extend(self._generate_join(rec))

        lines.append("    return df")
        return "\n".join(lines)

    def generate_gold_code(self) -> str:
        lines = [
            "from sklearn.preprocessing import StandardScaler, RobustScaler",
            "import numpy as np",
            "",
            "",
            "def gold_transform(df):",
        ]
        if not self.registry.gold or not self.registry.gold.all_recommendations:
            lines.append("    return df")
            return "\n".join(lines)

        for rec in self.registry.gold.transformations:
            lines.extend(self._generate_transformation(rec))
        for rec in self.registry.gold.encoding:
            lines.extend(self._generate_encoding(rec))
        for rec in self.registry.gold.scaling:
            lines.extend(self._generate_scaling(rec))

        lines.append("    return df")
        return "\n".join(lines)

    def generate_full_pipeline(self) -> str:
        bronze = self.generate_bronze_code()
        silver = self.generate_silver_code()
        gold = self.generate_gold_code()
        main = self._generate_main_function()
        return f"{bronze}\n{silver}\n{gold}\n{main}"

    def _generate_null_handling(self, rec: LayeredRecommendation) -> List[str]:
        strategy = rec.parameters.get("strategy", "median")
        return [
            f"    # {rec.rationale}",
            f"    handler = MissingValueHandler(strategy='{strategy}')",
            f"    df = handler.fit_transform(df, columns=['{rec.target_column}'])",
            "",
        ]

    def _generate_outlier_handling(self, rec: LayeredRecommendation) -> List[str]:
        method = rec.parameters.get("method", "iqr")
        factor = rec.parameters.get("factor", 1.5)
        return [
            f"    # {rec.rationale}",
            f"    outlier_handler = OutlierHandler(method='{method}', factor={factor})",
            f"    df['{rec.target_column}'] = outlier_handler.fit_transform(df[['{rec.target_column}']])",
            "",
        ]

    def _generate_type_conversion(self, rec: LayeredRecommendation) -> List[str]:
        target_type = rec.parameters.get("target_type", "str")
        return [
            f"    # {rec.rationale}",
            f"    df['{rec.target_column}'] = df['{rec.target_column}'].astype('{target_type}')",
            "",
        ]

    def _generate_filtering(self, rec: LayeredRecommendation) -> List[str]:
        if rec.action == "drop":
            return [
                f"    # {rec.rationale}",
                f"    df = df.drop(columns=['{rec.target_column}'])",
                "",
            ]
        return []

    def _generate_aggregation(self, rec: LayeredRecommendation, entity_col: str, time_col: str) -> List[str]:
        agg = rec.parameters.get("aggregation", "sum")
        windows = rec.parameters.get("windows", ["7d"])
        col = rec.target_column
        lines = [f"    # {rec.rationale}"]
        for window in windows:
            feature_name = f"{col}_{agg}_{window}"
            lines.append(f"    df['{feature_name}'] = df.groupby('{entity_col}')['{col}'].transform('{agg}')")
        lines.append("")
        return lines

    def _generate_derived(self, rec: LayeredRecommendation) -> List[str]:
        formula = rec.parameters.get("formula", "")
        return [
            f"    # {rec.rationale}",
            f"    df['{rec.target_column}'] = {formula}  # TODO: adapt formula",
            "",
        ]

    def _generate_join(self, rec: LayeredRecommendation) -> List[str]:
        dataset = rec.parameters.get("dataset", "")
        join_type = rec.parameters.get("join_type", "left")
        return [
            f"    # {rec.rationale}",
            f"    # df = df.merge(load('{dataset}'), on='{rec.target_column}', how='{join_type}')",
            "",
        ]

    def _generate_encoding(self, rec: LayeredRecommendation) -> List[str]:
        method = rec.parameters.get("method", "one_hot")
        col = rec.target_column
        if method == "one_hot":
            drop_first = rec.parameters.get("drop_first", False)
            return [
                f"    # {rec.rationale}",
                f"    df = pd.get_dummies(df, columns=['{col}'], drop_first={drop_first})",
                "",
            ]
        elif method == "target":
            return [
                f"    # {rec.rationale} - target encoding",
                f"    # Use TargetEncoder for '{col}'",
                "",
            ]
        return []

    def _generate_scaling(self, rec: LayeredRecommendation) -> List[str]:
        method = rec.parameters.get("method", "standard")
        col = rec.target_column
        scaler = "StandardScaler" if method == "standard" else "RobustScaler"
        return [
            f"    # {rec.rationale}",
            f"    scaler = {scaler}()",
            f"    df['{col}'] = scaler.fit_transform(df[['{col}']])",
            "",
        ]

    def _generate_transformation(self, rec: LayeredRecommendation) -> List[str]:
        method = rec.parameters.get("method", "log")
        col = rec.target_column
        if method == "log":
            return [
                f"    # {rec.rationale}",
                f"    df['{col}'] = np.log1p(df['{col}'])",
                "",
            ]
        return []

    def _generate_main_function(self) -> str:
        return """

def run_pipeline(df):
    df = bronze_transform(df)
    df = silver_transform(df)
    df = gold_transform(df)
    return df
"""
