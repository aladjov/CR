from typing import List
import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class FeatureSelectionStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.FEATURE_SELECTION

    @property
    def title(self) -> str:
        return "06 - Feature Selection"

    @property
    def description(self) -> str:
        return "Select best features using variance, correlation, and importance filters."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        target = self.get_target_column()
        var_thresh = self.config.variance_threshold
        corr_thresh = self.config.correlation_threshold
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.features": ["FeatureSelector"],
                "pandas": ["pd"],
                "numpy": ["np"],
            }),
            self.cb.section("Load Gold Data"),
            self.cb.code('''df = pd.read_parquet("./experiments/data/gold/customers_features.parquet")
print(f"Input shape: {df.shape}")'''),
            self.cb.section("Identify Feature Columns"),
            self.cb.code(f'''target_col = "{target}"
id_cols = {self.get_identifier_columns()}
feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
X = df[feature_cols]
y = df[target_col] if target_col in df.columns else None
print(f"Feature columns: {{len(feature_cols)}}")'''),
            self.cb.section("Variance Filter"),
            self.cb.code(f'''variance_threshold = {var_thresh}
variances = X.var()
low_variance = variances[variances < variance_threshold].index.tolist()
print(f"Low variance features ({{len(low_variance)}}): {{low_variance[:5]}}")
X = X.drop(columns=low_variance)'''),
            self.cb.section("Correlation Filter"),
            self.cb.code(f'''correlation_threshold = {corr_thresh}
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if any(upper[c] > correlation_threshold)]
print(f"High correlation features ({{len(high_corr)}}): {{high_corr[:5]}}")
X = X.drop(columns=high_corr)'''),
            self.cb.section("Save Selected Features"),
            self.cb.code(f'''selected_df = df[[*id_cols, *X.columns, target_col]].dropna(subset=[target_col])
selected_df.to_parquet("./experiments/data/gold/customers_selected.parquet", index=False)
print(f"Selected {{len(X.columns)}} features, saved {{len(selected_df)}} rows")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        target = self.get_target_column()
        return self.header_cells() + [
            self.cb.section("Load Gold Data"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.gold_customers")'''),
            self.cb.section("Compute Feature Correlations"),
            self.cb.code(f'''from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

target_col = "{target}"
numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ["IntegerType()", "DoubleType()", "FloatType()"] and f.name != target_col]

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
df_vec = assembler.transform(df)
corr_matrix = Correlation.corr(df_vec, "features").head()[0].toArray()
print(f"Correlation matrix shape: {{corr_matrix.shape}}")'''),
            self.cb.section("Remove Highly Correlated Features"),
            self.cb.code(f'''import numpy as np

correlation_threshold = {self.config.correlation_threshold}
to_drop = set()
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if abs(corr_matrix[i,j]) > correlation_threshold:
            to_drop.add(numeric_cols[j])

selected_cols = [c for c in numeric_cols if c not in to_drop]
print(f"Dropped {{len(to_drop)}} highly correlated features, keeping {{len(selected_cols)}}")'''),
            self.cb.section("Save Selected Features"),
            self.cb.code(f'''final_cols = {self.get_identifier_columns()} + selected_cols + [target_col]
df_selected = df.select(final_cols)
df_selected.write.format("delta").mode("overwrite").saveAsTable("{catalog}.{schema}.gold_selected")
print("Selected features saved")'''),
        ]
