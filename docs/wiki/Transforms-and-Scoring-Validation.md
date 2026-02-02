# Transforms & Scoring Validation

## Why This Matters

Train/serve skew — where the scoring pipeline applies transformations differently from training — is the **#1 cause of silent production failures** in ML systems. A scaler fitted on training data must use the exact same mean and standard deviation during scoring. An encoder must map the same categories to the same integers. If any of these drift, model predictions become unreliable without any obvious error.

This framework addresses train/serve skew at two levels:

1. **Transforms framework** — fit transformers once during training, persist them, replay identically during scoring
2. **Validation gates** — automatically detect when training and scoring outputs diverge

## Transforms Architecture

```
TransformationStep (from exploration recommendations)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│               TransformExecutor                          │
│                                                          │
│   _DISPATCH table maps TransformationType → handler      │
│                                                          │
│   ┌───────────────────┐   ┌────────────────────────┐    │
│   │  ops.py            │   │  fitted.py              │    │
│   │  (stateless)       │   │  (stateful)             │    │
│   │                    │   │                          │    │
│   │  apply_impute_null │   │  FittedScaler            │    │
│   │  apply_cap_outlier │   │  FittedEncoder           │    │
│   │  apply_log_transform│   │  FittedPowerTransform    │    │
│   │  apply_one_hot_encode│  │                          │    │
│   │  apply_derived_*   │   │  fit_transform() → save  │    │
│   │  ...               │   │  transform()   → load    │    │
│   └───────────────────┘   └────────────┬───────────┘    │
│                                         │                │
│                                         ▼                │
│                              ┌──────────────────┐        │
│                              │  ArtifactStore    │        │
│                              │  manifest.yaml    │        │
│                              │  *.joblib files   │        │
│                              └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### Stateless Operations (`ops.py`)

Functions that don't retain state between training and scoring. They apply the same logic regardless of `fit_mode`:

| Function | What It Does |
|----------|-------------|
| `apply_impute_null(df, column, value=0)` | Fill nulls with a fixed value or median |
| `apply_cap_outlier(df, column, lower, upper)` | Clip values to bounds |
| `apply_log_transform(df, column)` | `log1p` transform |
| `apply_sqrt_transform(df, column)` | Square root transform |
| `apply_winsorize(df, column, lower_bound, upper_bound)` | Clip to explicit bounds |
| `apply_segment_aware_cap(df, column, n_segments)` | KMeans clustering then IQR capping |
| `apply_cap_then_log(df, column)` | Cap at 99th percentile, then `log1p` |
| `apply_one_hot_encode(df, column)` | `pd.get_dummies` encoding |
| `apply_zero_inflation_handling(df, column)` | Create zero indicator + log non-zero values |
| `apply_derived_ratio(df, column, numerator, denominator)` | Create ratio column |
| `apply_derived_interaction(df, column, col_a, col_b)` | Multiply two columns |
| `apply_derived_composite(df, column, columns)` | Mean of multiple columns |

### Stateful Wrappers (`fitted.py`)

Classes that **fit** on training data and **replay** on scoring data using persisted parameters:

| Class | Wraps | fit_transform() | transform() |
|-------|-------|-----------------|-------------|
| `FittedScaler` | `StandardScaler` or `MinMaxScaler` | Fits scaler, saves to ArtifactStore | Loads scaler, applies |
| `FittedEncoder` | `LabelEncoder` | Fits encoder, saves to ArtifactStore | Loads encoder, maps (unknown → 0) |
| `FittedPowerTransform` | `PowerTransformer` (Yeo-Johnson) | Fits transformer, saves to ArtifactStore | Loads transformer, applies |

### Dispatch Table (`executor.py`)

`TransformExecutor` routes each `TransformationStep` to the correct handler:

```python
from customer_retention.transforms.executor import TransformExecutor
from customer_retention.transforms.artifact_store import ArtifactStore

executor = TransformExecutor()
store = ArtifactStore("./artifacts")

# Training: fit and persist
df = executor.apply_all(df, steps, fit_mode=True, artifact_store=store)
store.save_manifest()

# Scoring: load and replay
store = ArtifactStore.from_manifest("./artifacts/manifest.yaml")
df = executor.apply_all(df, steps, fit_mode=False, artifact_store=store)
```

The dispatch table maps `PipelineTransformationType` enums to handler methods. Stateless types (impute, cap, log, etc.) ignore `fit_mode`. Stateful types (scale, encode, yeo-johnson) route through `_apply_fitted()` which calls `fit_transform()` or `transform()` depending on the mode.

## Fit Artifact Registry

### ArtifactStore

The `ArtifactStore` class (`transforms/artifact_store.py`) manages persistence of fitted transformers:

```python
from customer_retention.transforms.artifact_store import ArtifactStore

# During training
store = ArtifactStore("./artifacts")
store.register("scale", "revenue", fitted_scaler)     # Saves revenue_scale.joblib
store.register("encode", "contract", fitted_encoder)   # Saves contract_encode.joblib
store.save_manifest()                                   # Writes manifest.yaml

# During scoring
store = ArtifactStore.from_manifest("./artifacts/manifest.yaml")
scaler = store.load("revenue_scale")
encoder = store.load("contract_encode")
```

### manifest.yaml Structure

```yaml
revenue_scale:
  type: scale
  column: revenue
  path: ./artifacts/revenue_scale.joblib
contract_encode:
  type: encode
  column: contract
  path: ./artifacts/contract_encode.joblib
monthly_charges_yeo_johnson:
  type: yeo_johnson
  column: monthly_charges
  path: ./artifacts/monthly_charges_yeo_johnson.joblib
```

### Artifact Directory Layout

```
artifacts/
├── manifest.yaml
├── revenue_scale.joblib
├── contract_encode.joblib
└── monthly_charges_yeo_johnson.joblib
```

Each `.joblib` file contains the fitted sklearn transformer object, serialized with joblib for efficient loading.

## Scoring Pipeline Validation

### Three Validators

| Validator | Input | What It Checks |
|-----------|-------|----------------|
| **ScoringPipelineValidator** | Training features + scoring features (DataFrames or Delta paths) | Feature distribution comparison (numeric: mean, std, max diff; categorical: value mismatches) |
| **AdversarialScoringValidator** | Gold features with holdout column | Re-runs transforms on holdout entities and compares to training-produced features |
| **PipelineValidationRunner** | Paths to artifacts + optional model | Orchestrates full validation: load data, split holdout, run validators, produce report |

### Severity Levels

Both `ScoringPipelineValidator` and `AdversarialScoringValidator` classify issues by severity:

| Level | `MismatchSeverity` / `DriftSeverity` | Relative Threshold |
|-------|--------------------------------------|-------------------|
| **LOW** | 1 | < 1% |
| **MEDIUM** | 2 | 1–5% |
| **HIGH** | 3 | 5–10% |
| **CRITICAL** | 4 | > 25% |

### ValidationReport

The `ValidationReport` dataclass aggregates all findings:

```python
report = validator.validate()

# Quick summary
print(report.summary())
# {'passed': False, 'total_feature_mismatches': 3, 'high_severity_features': 1, ...}

# Human-readable text
print(report.to_text())

# DataFrame for analysis
df = report.to_dataframe()

# Persist
report.save("validation_report.yaml")
```

### ScoringPipelineValidator Example

```python
from customer_retention.stages.validation.scoring_pipeline_validator import (
    ScoringPipelineValidator,
    ValidationConfig,
)

validator = ScoringPipelineValidator(
    training_features="./data/gold/training",       # Path or DataFrame
    scoring_features="./data/gold/scoring",         # Auto-detects Delta/parquet/CSV
    entity_column="customer_id",
    target_column="target",
    config=ValidationConfig(
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    ),
)

report = validator.validate()
```

### AdversarialScoringValidator Example

```python
from customer_retention.stages.validation.adversarial_scoring_validator import (
    AdversarialScoringValidator,
)

validator = AdversarialScoringValidator(
    gold_features=gold_df,
    entity_column="customer_id",
    target_column="target",
)

# Option 1: Compare pre-computed features
result = validator.validate_features(recomputed_scoring_features)

# Option 2: Re-run a transform function on holdout silver data
result = validator.validate_with_transform(silver_df, transform_fn=my_gold_transform)

print(result.summary)
print(result.to_dataframe())
```

## Notebook 11: Scoring Validation

Notebook 11 (`11_scoring_validation.ipynb`) runs the full validation workflow:

1. **Load gold features** — reads the gold layer output that includes holdout entities (where `target` is NaN but `original_target` is populated)
2. **Identify holdout entities** — `AdversarialScoringValidator.get_holdout_entity_ids()` finds entities whose target was masked during training
3. **Re-run scoring pipeline** — applies the same transformations with `fit_mode=False`, loading artifacts from `manifest.yaml`
4. **Compare features** — `AdversarialScoringValidator.validate_features()` checks that recomputed features match training-produced features within tolerance
5. **Compare predictions** — if a trained model is available, `ScoringPipelineValidator.validate_with_model()` generates predictions from both feature sets and compares them
6. **Generate report** — saves `ValidationReport` as YAML with severity-classified mismatches

```python
# Simplified notebook 11 workflow
from customer_retention.stages.validation.pipeline_validation_runner import run_pipeline_validation

report = run_pipeline_validation(
    gold_features_path="./experiments/gold_features",
    entity_column="customer_id",
    target_column="target",
    model=trained_model,
    feature_columns=feature_cols,
)
# Prints full validation report with feature/prediction mismatches
```

## Delta Time Travel for Validation

`compare_pipeline_outputs()` supports comparing different versions of the same Delta table — useful for validating that a new pipeline run produces consistent results with a previous one:

```python
from customer_retention.stages.validation.pipeline_validation_runner import compare_pipeline_outputs

# Compare version 3 (training run) against version 5 (scoring run) of the same table
report = compare_pipeline_outputs(
    training_output_path="./data/gold",
    version_a=3,
    version_b=5,
    entity_column="customer_id",
    target_column="target",
    output_report_path="./validation_report.yaml",
)

# Or compare two different paths
report = compare_pipeline_outputs(
    training_output_path="./data/gold/training",
    scoring_output_path="./data/gold/scoring",
    entity_column="customer_id",
)
```

The `load_artifact()` helper automatically detects Delta tables (by checking for `_delta_log/` directory) and supports version-pinned reads:

```python
from customer_retention.stages.validation.pipeline_validation_runner import load_artifact

# Read latest version
df = load_artifact("./data/gold")

# Read specific version
df_v3 = load_artifact("./data/gold", version=3)

# Falls back to parquet if not a Delta table
df = load_artifact("./data/gold/features.parquet")
```

## Next Steps

- [[Architecture]] - System design and medallion layers
- [[Local Track]] - Generate and run pipelines locally
- [[Databricks Track]] - Deploy to Databricks
- [[Temporal Framework]] - Leakage-safe data preparation
