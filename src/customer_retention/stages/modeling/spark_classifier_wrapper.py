from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import _is_spark_pandas, as_spark_df, enable_arrow_optimization, native_pd
from customer_retention.core.compat.detection import get_default_parallelism

_MODEL_REGISTRY: Dict[str, str] = {
    "LogisticRegression": "pyspark.ml.classification.LogisticRegression",
    "RandomForestClassifier": "pyspark.ml.classification.RandomForestClassifier",
    "GBTClassifier": "pyspark.ml.classification.GBTClassifier",
}

_WEIGHT_COL = "__weight__"
_LABEL_COL = "__label__"
_FEATURES_COL = "__features__"


def _import_class(fully_qualified: str) -> type:
    module_path, cls_name = fully_qualified.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _get_spark_session() -> Any:
    from pyspark.sql import SparkSession
    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


def _make_assembler(input_cols: List[str]) -> Any:
    from pyspark.ml.feature import VectorAssembler
    return VectorAssembler(
        inputCols=input_cols,
        outputCol=_FEATURES_COL,
        handleInvalid="keep",
    )


class SparkClassifierWrapper:

    def __init__(
        self,
        spark_model_class: str,
        spark_model_params: Dict[str, Any],
        feature_names: List[str],
        class_weight: Optional[str] = None,
    ):
        if not feature_names:
            raise ValueError("SparkClassifierWrapper requires at least one feature column")
        self.spark_model_class = spark_model_class
        self.spark_model_params = dict(spark_model_params)
        self.feature_names = list(feature_names)
        self.class_weight = class_weight
        self._fitted_model: Any = None
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any, prepared_df: Any = None) -> SparkClassifierWrapper:
        if prepared_df is not None:
            self._validate_prepared_df(prepared_df)
            spark_df = prepared_df
            owns_cache = False
        else:
            spark_df = self._to_spark_df(X, y)
            spark_df = self._repartition_for_training(spark_df)
            owns_cache = True
        spark_model = self._create_spark_model()
        self._fitted_model = spark_model.fit(spark_df)
        if owns_cache:
            spark_df.unpersist()
        self._classes = np.array([0, 1])
        return self

    def _validate_prepared_df(self, spark_df: Any) -> None:
        cols = set(spark_df.columns)
        missing = {_FEATURES_COL, _LABEL_COL} - cols
        if missing:
            raise ValueError(f"prepared_df missing required columns: {missing}")
        if self.class_weight == "balanced" and self.spark_model_class != "GBTClassifier" and _WEIGHT_COL not in cols:
            raise ValueError(f"prepared_df missing {_WEIGHT_COL} but class_weight='balanced'")

    def prepare_training_data(self, X: Any, y: Any) -> Any:
        """Prepare, repartition, and cache training data for reuse across models.

        Call once per data variant (e.g. scaled vs unscaled), then pass
        the result to ``fit(prepared_df=...)`` for each model.  Call
        ``unpersist()`` on the returned DataFrame when done with all models.
        """
        spark_df = self._to_spark_df(X, y)
        return self._repartition_for_training(spark_df)

    def evaluate_distributed(self, prepared_df: Any) -> Dict[str, float]:
        """Compute AUC and PR-AUC using Spark evaluators — data stays on cluster."""
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        predictions = self._fitted_model.transform(prepared_df)
        return {
            "roc_auc": BinaryClassificationEvaluator(
                rawPredictionCol="rawPrediction", labelCol=_LABEL_COL, metricName="areaUnderROC",
            ).evaluate(predictions),
            "pr_auc": BinaryClassificationEvaluator(
                rawPredictionCol="rawPrediction", labelCol=_LABEL_COL, metricName="areaUnderPR",
            ).evaluate(predictions),
        }

    def predict(self, X: Any) -> np.ndarray:
        assembled = self._assemble_features(X)
        predictions = self._fitted_model.transform(assembled)
        enable_arrow_optimization()
        pdf = predictions.select("prediction").toPandas()
        return pdf["prediction"].to_numpy().astype(float)

    def predict_proba(self, X: Any) -> np.ndarray:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.ml.functions import vector_to_array

        assembled = self._assemble_features(X)
        predictions = self._fitted_model.transform(assembled)
        enable_arrow_optimization()
        prob_array = vector_to_array(F.col("probability"))
        pdf = predictions.select(
            prob_array[0].alias("p0"), prob_array[1].alias("p1"),
        ).toPandas()
        return pdf.to_numpy()

    def clone(self) -> SparkClassifierWrapper:
        return SparkClassifierWrapper(
            spark_model_class=self.spark_model_class,
            spark_model_params=self.spark_model_params,
            feature_names=self.feature_names,
            class_weight=self.class_weight,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        model = self._fitted_model
        if hasattr(model, "featureImportances"):
            return model.featureImportances.toArray()
        if hasattr(model, "coefficients"):
            return np.abs(model.coefficients.toArray())
        return np.zeros(len(self.feature_names))

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is not None:
            return self._classes
        return np.array([0, 1])

    def _to_spark_df(self, X: Any, y: Any) -> Any:
        spark = _get_spark_session()

        if _is_spark_pandas(X):
            import pyspark.pandas as ps
            X_sel = X[self.feature_names].reset_index(drop=True)
            y_sel = y.rename(_LABEL_COL).reset_index(drop=True)
            combined = ps.concat([X_sel, y_sel], axis=1)
            spark_df = as_spark_df(combined)
        else:
            combined = native_pd.concat(
                [X[self.feature_names].reset_index(drop=True), y.reset_index(drop=True).rename(_LABEL_COL)],
                axis=1,
            )
            spark_df = spark.createDataFrame(combined)

        spark_df = spark_df.withColumn(_LABEL_COL, spark_df[_LABEL_COL].cast("double"))

        if self.class_weight == "balanced":
            spark_df = self._add_weight_column(spark_df)

        return _make_assembler(self.feature_names).transform(spark_df)

    @staticmethod
    def _repartition_for_training(spark_df: Any) -> Any:
        """Repartition + cache before iterative model fit for full core utilization.

        Uses 2x cores to mitigate straggler tasks — if one partition
        finishes early, the core picks up the next pending partition
        instead of sitting idle.  The extra scheduler overhead is
        negligible vs thousands of L-BFGS iterations.
        """
        target = get_default_parallelism() * 2
        if target > 2:
            spark_df = spark_df.repartition(target)
        spark_df.cache()
        spark_df.count()
        return spark_df

    def _add_weight_column(self, spark_df: Any) -> Any:
        import pyspark.sql.functions as F  # noqa: N812

        counts = spark_df.groupBy(_LABEL_COL).count().collect()
        class_counts = {int(row[_LABEL_COL]): row["count"] for row in counts}
        n_total = sum(class_counts.values())
        n_classes = len(class_counts)

        weight_expr = F.lit(1.0)
        for cls, cnt in class_counts.items():
            w = n_total / (n_classes * cnt)
            weight_expr = F.when(F.col(_LABEL_COL) == float(cls), F.lit(w)).otherwise(weight_expr)

        return spark_df.withColumn(_WEIGHT_COL, weight_expr)

    def _assemble_features(self, X: Any) -> Any:
        spark = _get_spark_session()

        if _is_spark_pandas(X):
            spark_df = as_spark_df(X[self.feature_names])
        else:
            spark_df = spark.createDataFrame(X[self.feature_names])

        return _make_assembler(self.feature_names).transform(spark_df)

    def as_pipeline_model(self) -> Any:
        from pyspark.ml import PipelineModel
        if self._fitted_model is None:
            raise ValueError("Cannot create PipelineModel from unfitted wrapper")
        return PipelineModel(stages=[_make_assembler(self.feature_names), self._fitted_model])

    @staticmethod
    def from_pipeline_model(pipeline_model: Any, spark_model_class: str,
                            spark_model_params: Dict[str, Any], feature_names: List[str],
                            class_weight: Optional[str] = None) -> SparkClassifierWrapper:
        wrapper = SparkClassifierWrapper(spark_model_class, spark_model_params, feature_names, class_weight)
        wrapper._fitted_model = pipeline_model.stages[-1]
        wrapper._classes = np.array([0, 1])
        return wrapper

    def _create_spark_model(self) -> Any:
        fqn = _MODEL_REGISTRY.get(self.spark_model_class)
        if fqn is None:
            raise ValueError(f"Unknown spark model class: {self.spark_model_class}")

        model_cls = _import_class(fqn)
        params = dict(self.spark_model_params)
        params["featuresCol"] = _FEATURES_COL
        params["labelCol"] = _LABEL_COL

        if self.class_weight == "balanced" and self.spark_model_class != "GBTClassifier":
            params["weightCol"] = _WEIGHT_COL

        if "aggregationDepth" not in params and hasattr(model_cls, "aggregationDepth"):
            n_partitions = get_default_parallelism() * 2
            if n_partitions > 2:
                import math
                params["aggregationDepth"] = max(2, int(math.ceil(math.log2(n_partitions) / math.log2(4))))

        return model_cls(**params)
