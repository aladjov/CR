# Dead Code Report

**Generated**: 2026-03-12
**Method**: Traced all call chains from every entry point:
- **Exploration notebooks** 00–12 (including 01a, 01a_a, 01b, 01c, 01d)
- **Scripts** in `scripts/` (run_exploration.py, data migration, Databricks deploy, notebook management — 21 scripts total)
- **CLI entry points** (`churnkit-init`, `churnkit-sync`, `churnkit-merge`)
- **Generated pipeline code** produced by `generators/notebook_generator/stages/s01–s11` and `generators/pipeline_generator/renderer.py` + `databricks_renderer.py`
- **Databricks-conditional paths** gated behind `is_databricks()`, `is_remote_spark()`, `is_spark_available()`
- **Internal cross-references** between source modules

Items marked "dead" have **no callers** across all of the above. Items only used by generated code or conditional paths are marked accordingly.

---

## 1. Entry Point Summary

| Entry Point | Type | Key Modules Reached |
|---|---|---|
| Exploration notebooks 00–12 | Interactive | `auto_explorer`, `profiling`, `temporal`, `modeling`, `features`, `validation`, `scoring`, `visualization`, `diagnostics`, `compat`, `transforms`, `adapters` |
| `scripts/notebooks/run_exploration.py` | Script | `auto_explorer.skip_logic`, `run_namespace`, `ProjectContext`, `compat.detection` |
| `scripts/data/create_snapshot.py` | Script | `stages.temporal.SnapshotManager`, `adapters.factory.get_delta` |
| `scripts/data/migrate_to_temporal.py` | Script | `stages.temporal.ScenarioDetector`, `stages.temporal.UnifiedDataPreparer`, `get_delta` |
| `scripts/data/migrate_parquet_to_delta.py` | Script | `adapters.factory.get_delta` |
| `scripts/notebooks/sync_notebooks.py` | Script | `generators.notebook_sync` |
| `scripts/notebooks/migrate_notebook_cell_ids.py` | Script | `generators.notebook_sync.cell_id_standardizer`, `cell_types` |
| `scripts/notebooks/tag_framework_cells.py` | Script | `generators.notebook_sync.cell_types` |
| `scripts/notebooks/export_tutorial_html.py` | Script | `analysis.plotly_preprocessor` |
| `churnkit-init` CLI | CLI | `generators.notebook_generator.ProjectInitializer` |
| `churnkit-sync` CLI | CLI | `generators.notebook_sync` |
| `churnkit-merge` CLI | CLI | `generators.notebook_merge` |
| Generated pipeline notebooks (s01–s11) | Generated | `ingestion`, `cleaning`, `transformation`, `preprocessing`, `features`, `modeling`, `deployment`, `monitoring`, `feature_store`, `orchestration` |
| Generated local pipeline (renderer.py) | Generated | `transforms`, `temporal`, `profiling`, `modeling`, `compat`, `adapters`, `auto_explorer` |
| Generated Databricks pipeline (databricks_renderer.py) | Generated | `profiling.spark_*`, `temporal.spark_*`, `modeling.feature_profile`, `compat` |

---

## 2. Modules Used ONLY by Generated Pipeline Code

These modules are **not dead** but are only reachable when users run `churnkit-init` to generate project notebooks and then execute the generated pipeline. They are NOT used by the exploration notebooks or scripts.

| Subpackage | Used By | Specific Exports Used |
|---|---|---|
| `stages/ingestion/` | Generated s01_ingestion.py | `DataSourceRegistry` |
| `stages/preprocessing/` | Generated s04_transformation.py | `TransformerManager` |
| `stages/transformation/` | Generated s04_transformation.py | `NumericTransformer`, `CategoricalEncoder` |
| `stages/deployment/` | Generated s08_deployment.py | `ModelRegistry`, `ModelStage` |
| `stages/monitoring/` | Generated s09_monitoring.py | `PerformanceMonitor`, `DriftDetector` |
| `generators/orchestration/` | Generated s01–s11 | `setup_notebook_context`, `PipelineContext` |

### Partially-used exports within these modules

Even though the modules above are used by generated code, many of their exports are **never referenced** even in that context:

| Module | Used Exports | Unused Exports |
|---|---|---|
| `stages/ingestion/` (8 exports) | `DataSourceRegistry` | `DataLoader`, `CSVLoader`, `ParquetLoader`, `DeltaLoader`, `LoaderFactory`, `LoadResult` |
| `stages/preprocessing/` (3 exports) | `TransformerManager` | `TransformerBundle`, `TransformerManifest` |
| `stages/transformation/` (12 exports) | `NumericTransformer`, `CategoricalEncoder` | `DatetimeTransformer`, `BinaryHandler`, `TransformationPipeline`, `TransformationManifest`, + 6 result types |
| `stages/deployment/` (18 exports) | `ModelRegistry`, `ModelStage` | `BatchScorer`, `RetrainingTrigger`, `ChampionChallenger`, `RollbackManager`, `ModelMetadata`, + 11 more |
| `stages/monitoring/` (16 exports) | `PerformanceMonitor`, `DriftDetector` | `AlertManager`, `Alert`, `AlertLevel`, `AlertChannel`, `AlertConfig`, + 9 more |

---

## 3. Entirely Dead Subpackages

These subpackages have **zero callers** from any entry point — not from notebooks, scripts, CLI, generated code, or Databricks-conditional paths.

| Subpackage | Files | Exports | Notes |
|---|---|---|---|
| `analysis/discovery/` | `type_inferencer.py`, `config_generator.py`, `discovery_flow.py` | 8 | Discovery is handled by `auto_explorer` in practice. |
| `analysis/interpretability/` | `shap_explainer.py`, `pdp_generator.py`, `cohort_analyzer.py`, `individual_explainer.py`, `counterfactual.py` | 12 | NB08 does SHAP directly via the `shap` library, not through this wrapper. |
| `analysis/business/` | `risk_profile.py`, `intervention_matcher.py`, `roi_analyzer.py`, `fairness_analyzer.py`, `report_generator.py`, `ab_test_designer.py` | 15 | Only internal ref is `core.components.enums` import (for type hint). |
| `integrations/streaming/` | `event_schema.py`, `window_aggregator.py`, `online_store_writer.py`, `early_warning_model.py`, `trigger_engine.py`, `realtime_scorer.py`, `batch_integration.py` | 58 | Internal refs only to `stages.monitoring` which is itself only used by generated code. No caller triggers this chain. |
| `integrations/iteration/` | `context.py`, `recommendation_tracker.py`, `feedback_collector.py`, `signals.py`, `orchestrator.py` | 14 | Only internal ref is `signals.py` → `stages.monitoring` (TYPE_CHECKING). |
| `integrations/llm_context/` | `context_builder.py`, `prompts.py` | 2 | Zero references anywhere. |

**Total truly dead exports: ~109 across ~30 files**

---

## 4. Effectively Dead: `generators/spec_generator/`

| Subpackage | Files | Exports | Notes |
|---|---|---|---|
| `generators/spec_generator/` | `mlflow_pipeline_generator.py`, `recommendation_parser.py` | 5 (`MLflowPipelineGenerator`, `MLflowConfig`, `RecommendationParser`, `CleanAction`, `TransformAction`) | NB10 imports `MLflowPipelineGenerator` and `MLflowConfig` but the code path is gated behind a condition that is never true in any tested flow. `RecommendationParser` has zero callers anywhere. |

---

## 5. Dead Classes/Functions Within Active Subpackages

These are exported in `__all__` but never called from any entry point (notebooks, scripts, CLI, generated code, or conditional Databricks paths).

### `stages/profiling/`

| Export | File | Notes |
|---|---|---|
| `ReportGenerator` | `report_generator.py` | Distinct from `analysis.business.ReportGenerator`. Never instantiated anywhere. |
| `SCDAnalyzer`, `SCDResult` | `scd_analyzer.py` | Slowly Changing Dimension analyzer. Never used. |
| `TargetLevelAnalyzer`, `TargetLevelResult`, `TargetLevel`, `AggregationMethod`, `TargetDistribution` | `target_level_analyzer.py` | Never instantiated. |
| `TargetColumnDetector` | `target_level_analyzer.py` | Never instantiated. |

### `stages/modeling/`

| Export | File | Notes |
|---|---|---|
| `HyperparameterTuner`, `SearchStrategy`, `TuningResult` | `hyperparameter_tuner.py` | Entire module is dead. Only tests. |
| `ThresholdOptimizer`, `OptimizationObjective`, `ThresholdResult` | `threshold_optimizer.py` | Entire module is dead. Only tests. |
| `ModelComparator`, `ComparisonResult`, `ModelMetrics` | `model_comparator.py` | Entire module is dead. Only tests. |
| `SparkBaselineTrainer` | `spark_baseline_trainer.py` | Never instantiated. Not used in Databricks-generated training template (which uses raw PySpark MLlib directly). |

### `stages/validation/`

| Export | File | Notes |
|---|---|---|
| `DataQualityGate` | `data_quality_gate.py` | Never instantiated. Not in any Databricks conditional path. |
| `FeatureQualityGate` | `feature_quality_gate.py` | Never instantiated. |
| `LeakageGate`, `LeakageCheckResult` | `leakage_gate.py` | Never instantiated. NB07 uses `LeakageDetector` (from `analysis.diagnostics`) instead. |
| `ModelValidityGate`, `ModelValidityResult` | `model_validity_gate.py` | Never instantiated. |
| `BusinessSenseGate`, `BusinessSenseResult`, `BusinessCheck` | `business_sense_gate.py` | Never instantiated. |
| `QualityScorer`, `QualityScoreResult`, `QualityLevel` | `quality_scorer.py` | Never instantiated. |
| `AdversarialScoringValidator`, `AdversarialValidationResult`, `FeatureDrift`, `DriftSeverity` | `adversarial_scoring_validator.py` | Never instantiated. |
| `PipelineValidationRunner`, `PipelineValidationConfig`, `run_pipeline_validation`, `compare_pipeline_outputs` | `pipeline_validation_runner.py` | Never called. |
| `TimeSeriesValidator`, `TimeSeriesValidationResult` | `timeseries_detector.py` | Only appears in docstring example. |

### `stages/features/`

| Export | File | Notes |
|---|---|---|
| `FeatureManifest`, `FeatureSet`, `FeatureSetRegistry` | `feature_manifest.py` | Entire module is dead. Never instantiated. |

### `core/compat/`

| Export | File | Notes |
|---|---|---|
| `register_temp_view` | `__init__.py` | Defined but never called from any path. |
| `configure_spark_pandas` | `detection.py` | Defined but never called. |
| `api_types` | `__init__.py` | Re-exported but never referenced. |

---

## 6. Transitively Dead Chains

```
DEAD (no entry point reaches these):
├── analysis/discovery/                          ← standalone dead
├── analysis/interpretability/                   ← standalone dead
├── analysis/business/                           ← standalone dead
├── integrations/llm_context/                    ← standalone dead
├── integrations/streaming/ → stages/monitoring  ← monitoring is alive via generated code,
│                                                   but streaming never calls into it from any entry point
└── integrations/iteration/ → stages/monitoring  ← same: iteration never triggered

ALIVE only via generated pipeline (churnkit-init):
├── stages/ingestion/      ← generated s01
├── stages/preprocessing/  ← generated s04
├── stages/transformation/ ← generated s04
├── stages/deployment/     ← generated s08
├── stages/monitoring/     ← generated s09
└── generators/orchestration/ ← generated s01-s11

ALIVE via exploration notebooks + scripts + CLI:
├── stages/profiling/      ✅ (partial dead classes above)
├── stages/temporal/       ✅
├── stages/modeling/       ✅ (partial dead classes above)
├── stages/features/       ✅ (partial: FeatureManifest dead)
├── stages/validation/     ✅ (partial: Gates/Scorer dead)
├── stages/scoring/        ✅
├── stages/cleaning/       ✅ (via profiling + generated code)
├── analysis/auto_explorer ✅
├── analysis/visualization ✅
├── analysis/diagnostics   ✅
├── analysis/recommendations ✅ (via MLflow adapter)
├── core/compat            ✅ (3 dead functions)
├── core/config            ✅
├── core/naming            ✅
├── core/components        ✅ (Severity enum widely used; orchestration only via generated)
├── generators/pipeline_generator  ✅
├── generators/notebook_generator  ✅ (via CLI)
├── generators/notebook_sync       ✅ (via CLI + scripts)
├── generators/notebook_merge      ✅ (via CLI)
├── transforms             ✅
├── integrations/adapters  ✅
├── integrations/feature_store ✅ (via generated s10/s11 + features/feature_engineer)
└── artifacts              ✅ (via profiling/text_processor)
```

---

## 7. Summary

| Category | Dead Exports | Dead Files (approx) |
|---|---|---|
| Entirely dead subpackages | ~109 | ~30 |
| Effectively dead (spec_generator) | 5 | 2 |
| Dead classes in active packages | ~45 | ~12 |
| Dead compat functions | 3 | 0 (inline) |
| **Total confirmed dead** | **~162** | **~44** |
| Unused exports in generated-code-only modules | ~40 | 0 (partial files) |

### Key Distinctions

- **Truly dead (~162 exports, ~44 files)**: No caller from any entry point. Safe to remove (along with their tests).
- **Generated-code-only (~6 modules)**: Alive but only through `churnkit-init` generated pipelines. If the notebook generator is considered a supported workflow, these are alive. If only exploration notebooks matter, they're dead.
- **Unused exports within alive modules (~40)**: The module is used but specific exports like `BatchScorer`, `AlertManager`, `DatetimeTransformer` are never referenced. These are lower-priority candidates — removing individual exports is more surgical.

### Notes

- **Tests exist for most dead code.** Removing dead modules would also require removing ~500–800 corresponding tests.
- **`stages/cleaning/`** is NOT dead — used by `profiling/segment_aware_outlier.py`, `core/components/`, and generated cleaning code.
- **`core/components/`**: `Severity` enum is widely consumed. The component orchestration classes (`Ingester`, `Profiler`, `Trainer`, etc.) are only invoked from generated notebook code, not exploration notebooks.
- **Databricks paths** don't activate any additional dead modules — all Databricks-conditional code (`DatabricksDelta`, `DatabricksFeatureStore`, `DatabricksMLflow`, `RemotePath`) is in modules that are already alive via exploration notebooks.
- **`SparkBaselineTrainer`** remains dead even on Databricks — the Databricks training template in `databricks_renderer.py` uses raw PySpark MLlib directly, not this wrapper.
