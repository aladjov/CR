# Notebook Section Analysis: Recommendations, Pipeline Parity & Gaps

> Regenerated 2026-02-21 from current codebase state.

## Legend
- **Rec. Made**: What recommendation/finding the section produces
- **Applied in NB**: Whether the recommendation is consumed within the notebook flow
- **Local Pipeline**: Whether the local `PipelineGenerator` consumes this finding (`YES`/`NO`/`PARTIAL`)
- **DBX Pipeline**: Whether `DatabricksPipelineGenerator` consumes this finding (`YES`/`NO`/`PARTIAL`)
- **Note**: Purpose if informational-only; untapped potential; future relevance

---

## NB00: Start Here (Prerequisites & Multi-Dataset Setup)

NB00 has been restructured into an environment-setup and multi-dataset registration notebook. Fingerprinting, semantics, and intent configuration have moved downstream (primarily into NB01 and the RunNamespace/ProjectContext system).

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 0.1 Verify Environment | Diagnostic: package installed | YES (gates notebook) | NO | NO | Standard prerequisite check (`customer_retention` importable). |
| 0.2 Available Datasets | Dataset inventory (entity vs event, included vs download) | YES (informational) | NO | NO | Displays 6 datasets (3 entity, 3 event) with availability status. |
| 0.3 Kaggle API Setup | Instructions for `~/.kaggle/kaggle.json` | YES (gates 0.4-0.5) | NO | NO | Prerequisite for optional downloads. |
| 0.4-0.5 Download Kaggle Datasets | CSV file artifacts (bank churn, Netflix churn) | YES (section 0.6 verifies) | NO | NO | Uses `kaggle datasets download` CLI. |
| 0.6 Verify Downloads | Per-dataset shape report (rows, cols, preview) | NO | NO | NO | Informational validation. |
| 0.7 Generate EDI Ticketing 3-Set (Optional) | 3 synthetic CSVs (profiles, transactions, tickets) | YES (section 0.8) | NO | NO | Disabled by default (`GENERATE_EDI_DATASET=False`). Configurable N_CUSTOMERS, CHURN_RATE, N_YEARS, SEED. |
| 0.8 Define Your Datasets | `dataset_context.yaml`: target dataset, target column, entity column, per-dataset join keys & relationship types | YES (downstream NB01+) | YES (source paths, relationships) | YES (table names, relationships) | **Key persisted output.** `DatasetContextScanner` auto-detects targets, join keys, and relationship metadata. Drives config.py source definitions and silver join scaffold. |
| 0.9-0.10 Temporal Framework Overview | None (documentation only) | NO | NO | NO | Code examples for `SnapshotManager`, `ScenarioDetector`, `TimestampConfig`. No code is executed. |
| 0.11-0.12 Next Steps / Save | None | NO | NO | NO | Transition guidance to NB01. |

**Notable vs. Original Report:** The original NB00 contained sections for auto-fingerprinting, semantics confirmation, target dataset selection, prediction objective detection, objective priority review, join scaffold, temporal posture, intent configuration, and snapshot grid initialization. These have been redistributed: fingerprinting and type detection are now in NB01 (1.2); intent/posture configuration is handled by RunNamespace/ProjectContext; join scaffold is within the DatasetContextScanner (0.8); snapshot grid initialization happens in NB01 (1.8b).

---

## NB01: Data Discovery (Per-Dataset)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1.1 Configuration | DATA_PATH, TARGET_COLUMN, ENTITY_COLUMN, DROP_COLUMNS, AUTO_DROP_TEXT_COLUMNS, RECENT_DAYS, ALLOW_FUTURE_COLUMNS | YES | YES (source_path, columns) | YES | Core user inputs. RunNamespace initialized (from_env → from_latest → create). Session-based active dataset via `set_active_dataset()`. RECENT_DAYS read from project intent if configured. |
| 1.2 Load Data + Explore | Column types, target detection, granularity, fingerprint, timestamp column | YES (findings) | YES (column types drive bronze/gold) | YES | `DatasetFingerprinter` generates structural snapshot. `DataExplorer` profiles columns. `inferred_type` per column is the backbone of all generated code. Auto-drops text columns if configured. |
| 1.3 Column Summary Table | Visual table of types/confidence/nulls/distinct | NO | NO | NO | Informational. User reviews auto-detection accuracy before type overrides. |
| 1.4 Type Override | Manual type corrections (inferred_type updates, confidence=1.0) | YES (findings save) | YES (overridden types used) | YES | Flows through FindingsParser to pipeline. Low-confidence (<80%) detections flagged. |
| 1.5 Dataset Structure Detection | `TimeSeriesDetector` output: dataset_type, temporal pattern, time span, is_event_level | YES (routing) | NO | NO | **Informational only.** `dataset_type` detected but not consumed by generators. Directs event datasets to 01a-01d track, entity datasets to 02+. **Potential**: Longitudinal datasets could get different aggregation strategies; snapshot datasets could skip windowed aggregation. |
| 1.6 Active Dataset Creation | Feature timestamp derivation, future-value detection, datetime feature derivation (days_since_X, milestones), entity sampling, Delta Lake save | YES (all downstream) | YES (landing datetime_derivation, mask_future_columns) | YES | **Key output.** `derive_extra_datetime_features()` with per-column leakage guards. `EntityFeatureTimestampDeriver` creates `feature_timestamp`. Cross-dataset entity sampling for consistency. `findings.datetime_derivation_sources` and `findings.datetime_allow_future_columns` saved. |
| 1.7 Structural Stability | `dataset_stability_score`, volume/entity/distribution/cadence/target drift metrics, ObjectiveSupport signals (ImmediateRisk, Disengagement, Renewal) | YES (findings metadata) | NO | NO | **Informational only.** Stability score computed and stored but generators ignore it. Signals fed to ExplorationManager. **Potential**: (1) Low stability triggers warning comment in generated code. (2) Drive recommendation for temporal cross-validation. (3) Regime detection could inform `data_regime` feature. (4) Post-model stability report for top features. |
| 1.7 Objective Support Signals | Per-objective signal levels with why/positives/negatives/gaps | YES (ExplorationManager) | NO | NO | **Informational only.** Signals guide exploration but don't flow to generated code. |
| 1.8 Save Findings | ExplorationFindings YAML (column types, target, metadata, skip flags) | YES (all downstream) | YES (FindingsParser reads) | YES | Critical persistence point. 3-tuple `load_notebook_findings()` return. `publish_skip_flags()` routes downstream notebooks. |
| 1.8b Snapshot Grid Vote | Dataset temporal coverage registered on grid | YES (01d readiness) | NO | NO | Coordination mechanism for multi-dataset. `DatasetGridVote` with granularity and data span. |
| 1.9 Summary | Routing guidance | NO | NO | NO | Informational. |

---

## NB01a: Temporal Deep Dive (Event Bronze Track)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1a.1-1a.2 Load | Boilerplate + column configuration | - | - | - | Loads findings + data. Uses `load_active_dataset_distributed` for Spark support. |
| 1a.3 Time Series Profile | Events/entity stats, time span, avg events per entity, inter-event timing | YES (downstream sections) | NO | NO | **Informational.** Profile stats inform window recommendations but are not directly consumed by generators. |
| 1a.4 Events per Entity Distribution | Activity segments (One-time, Low, Medium, High) with Q25/Q75 thresholds, segment recommendations | YES (window collector) | NO | NO | **Informational only.** Segment counts/thresholds displayed but not persisted as pipeline config. **Potential**: (1) `is_one_time_entity` binary feature in bronze. (2) Activity segment as categorical feature. |
| 1a.5 Entity Lifecycle Analysis | Lifecycle quadrants (steady_loyal, occasional_loyal, intense_brief, one_shot), tenure/intensity thresholds | YES (findings metadata) | YES | YES | `lifecycle_quadrant` feature generated in bronze via `include_lifecycle_quadrant` flag in `metadata["temporal_patterns"]["feature_flags"]` or `metadata["aggregation"]`. FindingsParser reads this flag. |
| 1a.6 Temporal Coverage & Drift | Volume trend, gap detection, drift_risk_level, regime_count, recommended_training_start, population_stability, inter-event skew | YES (findings metadata) | YES | YES | `recommended_training_start` now consumed by FindingsParser → `config.training.recommended_training_start`. `drift_risk_level` and `regime_count` stored but NOT consumed by generators. **Remaining gap**: Volume trend could generate a `volume_regime` feature. |
| 1a.7 Temporal Aggregation Perspective | Within-CV vs between-CV per column, aggregation guidance | NO | NO | NO | **Informational only.** Per-column guidance printed but not saved. **Potential**: Columns marked "all_time mean sufficient" could use only `all_time` window, reducing feature count. |
| 1a.8 Update Findings | `time_series_metadata` updated: suggested_aggregations, window_coverage_threshold, heterogeneity_level, eta_squared, drift_risk, recommended_training_start, segmentation_advisory | YES (saved to YAML) | PARTIAL | PARTIAL | `suggested_aggregations` → consumed as `aggregation.windows`. `heterogeneity_level`/`segmentation_advisory` → NOT consumed. **Gap**: `segmentation_advisory="consider_separate_models"` could generate two training scripts. |

---

## NB01a_a: Temporal Text Deep Dive

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1a.a.1-1a.a.2 Load | Boilerplate | - | - | - | Gates on EVENT_LEVEL; exits if entity-level. Checks LIGHT_RUN. |
| 1a.a.3 Configuration | Embedding model preset (minilm/qwen3 variants), variance threshold, component bounds | YES | YES | YES | `TextProcessingConfig` created. FindingsParser now reads `text_processing` → `TextFeatureConfig` with `embedding_model`, `n_components`, `component_columns`. |
| 1a.a.4 Text Column Analysis | Text length stats, sample texts, per-entity distribution | NO | NO | NO | Informational. |
| 1a.a.5 Process Text Columns | PC features (e.g., ticket_text_pc1..pcN), explained variance, component count | YES (df updated) | YES | YES | **Gap closed vs. original report.** Text→embedding→PCA results now flow through `findings.text_processing` which FindingsParser reads into `TextFeatureConfig`. Bronze templates generate text embedding steps. |
| 1a.a.6 Plan Aggregation | Feature count, aggregation plan (windows x funcs x PCs) | YES (section 1a.a.8) | NO | NO | Plan is informational; aggregation itself happens in 01d. |
| 1a.a.7 Visualize PCs | PC distribution histograms, PC1 vs PC2 scatter | NO | NO | NO | Informational. Uses `safe_sample()`. |
| 1a.a.8 Update Findings | `findings.text_processing[col]` = TextProcessingMetadata (model, dim, n_components, component_columns, variance_threshold, processing_approach) | YES (saved to YAML) | YES | YES | **Previously orphaned; now consumed.** FindingsParser reads `text_processing` and generates `TextFeatureConfig` for bronze templates. |

---

## NB01b: Temporal Quality

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1b.1-1b.2 Load/Config | Boilerplate + REFERENCE_DATE, EXPECTED_FREQUENCY, MAX_GAP_MULTIPLE | YES | NO | NO | Quality check params are not in generated pipeline. |
| 1b.3 Quality Checks | TQ001 (duplicates), TQ002 (gaps), TQ003 (future dates), TQ004 (ordering) | YES (quality score) | YES | YES | **TQ003 now consumed.** FindingsParser reads `metadata["temporal_quality"]["checks"]` with `code="TQ003"` to set `config.training.filter_future_dates`. TQ001/TQ002/TQ004 remain informational. |
| 1b.4 Quality Score | Score (0-100), grade (A-D), passed boolean | YES (findings metadata) | NO | NO | **Informational.** Score stored but generators don't use it. **Potential**: Grade D could generate a warning header. Quality score could become MLflow tag. |
| 1b.5 Event Volume Analysis | Volume over time visualization | NO | NO | NO | Informational. Automatic frequency selection (D/W/M). |
| 1b.6 Outlier Analysis | Segment-aware vs global outlier comparison, segmentation_recommended flag | YES (recommendation) | NO | NO | **Informational within 01b.** Outlier recommendations are made in NB02/04 and those ARE consumed. |
| 1b.7 Data Validation | Binary field checks, string consistency issues | NO | NO | NO | Informational. |
| 1b.8 Recommendations | Column-level cleaning recommendations from RecommendationEngine | NO | NO | NO | Printed only. Re-derived in NB02/04 where they ARE persisted. |
| 1b.9 Save Results | `findings.metadata["temporal_quality"]` with score/grade/check results | YES (saved) | PARTIAL | PARTIAL | TQ003 check consumed for filter_future_dates. Score/grade not consumed. |

---

## NB01c: Temporal Patterns

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1c.1-1c.2 Load/Config | Boilerplate + target detection, TARGET_COLUMN_OVERRIDE | YES | NO | NO | Windows loaded from 01a findings. |
| 1c.3 Aggregation Window Config | Load suggested_aggregations from 01a (e.g., ["7d", "30d", "90d", "all_time"]) | YES | YES (via aggregation config) | YES | Critical handoff from 01a. |
| 1c.4 Value Column Selection | Primary VALUE_COLUMN for analysis | YES | NO | NO | Selects non-temporal numeric column. |
| 1c.5 Trend Detection | Trend direction (increasing/decreasing/stable), slope, R^2, confidence | YES (feature flags) | NO | NO | **Informational.** Trend direction/strength stored but generators don't produce trend-adjusted features. **Potential**: Generate `recent_vs_overall_ratio` or detrending logic. |
| 1c.6 Seasonality Detection | Periodicities (weekly/monthly/quarterly), autocorrelation strengths, FFT analysis | YES (feature flags) | PARTIAL | PARTIAL | `include_cyclical_features` flag read by FindingsParser → drives cyclical encoding in bronze lifecycle config. But monthly/quarterly seasonality does NOT generate additional cyclical features beyond DOW. **Gap**: `month_sin/cos`, `quarter_sin/cos` not generated. |
| 1c.7 Cohort Analysis | Cohort distribution, retention by cohort, acquisition timing | YES (feature flags) | NO | NO | **Informational.** `include_cohort_features` flag saved but NOT consumed by FindingsParser. **Potential**: Generate `cohort_year`/`cohort_quarter` features. |
| 1c.8 Correlation Matrix | Pairwise correlations between event features | NO | NO | NO | Informational. Uses `batched_corr_matrix()` for Spark compatibility. |
| 1c.9 Sparklines | Per-feature temporal patterns by cohort, trend lines | NO | NO | NO | Informational visualization. |
| 1c.10 Effect Sizes | Cohen's d, Cramer's V per feature (retained vs churned) | YES | NO | NO | Informational ranking. |
| 1c.11 Recency Analysis | Days since last event distribution, predictive power, quantile thresholds, bucket boundaries | YES (feature flags) | YES | YES | `include_recency_bucket` flag and `recency_analysis.bucket_boundaries` now consumed by FindingsParser → `bronze.lifecycle.recency_bucket_edges` and `recency_bucket_labels`. **Gap closed vs. original report** (was hardcoded). |
| 1c.12 Velocity Analysis | Rate of change metrics, divergent velocity columns | YES (01d config) | NO | NO | **Informational for 01c, consumed by 01d.** Velocity features NOT generated in pipeline. **Potential**: `{col}_velocity_{window}` features. |
| 1c.13 Momentum Analysis | Recent vs historical window ratios, momentum pairs | YES (01d config) | YES | YES | Momentum ratio features generated in bronze via `bronze.lifecycle.momentum_pairs` (short_window, long_window). |
| 1c.14 Lag Feature Analysis | Autocorrelation per feature, recommended lag days, lag columns | YES (feature flags) | YES | YES | **Now consumed.** FindingsParser reads `metadata["temporal_patterns"]["lag_features_computed"]`, `lag_columns`, `lag_window_days` → `bronze_event.temporal_features` config. |
| 1c.15 Predictive Power | IV and KS statistics per feature | YES | NO | NO | Informational ranking. |
| 1c.16 Categorical Feature Analysis | Cramer's V, WOE, high-risk categories | YES | NO | NO | Informational. Handled separately in NB04/05. |
| 1c.17 Feature Summary | Feature engineering checklist with window formulas | NO | NO | NO | Informational summary. |
| 1c.18 Save Patterns | `findings.metadata["temporal_patterns"]` with all analysis + `feature_flags` dict | YES (saved) | YES | YES | **Significant change from original report.** FindingsParser NOW reads `temporal_patterns.feature_flags` for: `include_recency_bucket`, `include_lifecycle_quadrant`, `include_cyclical_features`, and lag feature config. Previously reported as fully orphaned. **Remaining gaps**: `include_trend_features`, `include_cohort_features` still NOT consumed. |
| 1c.19 Snapshot Grid Vote | Dataset vote registered on grid | YES (grid) | NO | NO | Coordination mechanism. |

---

## NB01d: Event Aggregation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1d.1 Load + Grid Check | Validates upstream completion, loads all findings from 01a-01c | YES | NO | NO | Boilerplate + validation gate. Checks LIGHT_RUN. |
| 1d.1b Grid Readiness | Verify all event-level datasets submitted grid votes | YES (gates aggregation) | NO | NO | Blocks if votes incomplete (ALLOW_ADJUSTMENTS mode). |
| 1d.2 Configure Aggregation | Windows (from 01a), quality constraints (from 01b), pattern features (from 01c), text PCs (from 01a_a), value columns, agg functions, lifecycle flags, target-proxy exclusion | YES | PARTIAL | PARTIAL | `suggested_aggregations` → pipeline's `aggregation.windows`. Divergent column prioritization NOT in pipeline (all numeric cols aggregated equally). |
| 1d.3 Preview Plan | Feature count preview from `TimeWindowAggregator.generate_plan()` | NO | NO | NO | Informational. |
| 1d.4 Execute Aggregation | Multi-step aggregation: time windows, lifecycle quadrant, entity target, cyclical features, trend features, cohort features, momentum ratios, recency buckets, text PC aggregation | YES (aggregated df) | PARTIAL | PARTIAL | Pipeline generates basic time-window aggregation + lifecycle enrichment. **Still missing from generated pipeline**: (1) Trend features (recent_vs_overall_ratio, entity_trend_slope), (2) Cohort features (cohort_year, cohort_quarter). Cyclical, recency, momentum, lag, and text features now have pipeline parity. |
| 1d.4b Temporal Features | Lag-based velocity/regularity features via TemporalFeatureEngineer | YES (merged to agg) | YES | YES | **Now in pipeline.** Temporal lag features generated via `bronze_event.temporal_features` config from FindingsParser. |
| 1d.5 Quality Check | Null check, entity count validation, constant column detection | NO | NO | NO | Informational validation. |
| 1d.6 Save Aggregated Data | Delta table + auto-profiled aggregated findings YAML + aggregation metadata | YES (downstream NBs) | YES (aggregated findings read by FindingsParser) | YES | Critical. Aggregated findings become input to silver/gold generation. |
| 1d.X Leakage Validation | Target leakage check on aggregated features | NO | NO | NO | Informational validation. |

---

## NB02: Source Integrity

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 2.1 Setup | Boilerplate | - | - | - | Uses `load_notebook_findings()` 3-tuple. |
| 2.2 Duplicate Analysis | Deduplication strategy (drop_exact, keep_first + conflict cols) → registry.bronze | YES (registry) | NO | NO | **Change from original report.** `registry.sources[name].deduplication` is defined but FindingsParser does NOT consume it. Deduplication logic is NOT generated in pipeline code. **Gap**: Bronze templates should include dedup step when flagged. |
| 2.3 Overall Quality Score | Quality score 0-100 | NO | NO | NO | Informational. |
| 2.4 Target Variable Analysis | Imbalance ratio, strategy (stratified/class_weights/SMOTE) → registry.bronze | YES (registry) | PARTIAL | PARTIAL | `registry.bronze.modeling_strategy` is checked but NOT fully consumed. Training uses `class_weight="balanced"`. **Gap**: Explicit class weights or SMOTE step should be generated based on severity. |
| 2.5 Missing Value Analysis | Column null inventory | NO | NO | NO | Informational. Context for 2.6. |
| 2.6 Missing Value Patterns | MCAR/MAR/MNAR classification via correlation heatmap | NO | NO | NO | **Informational only.** Missingness mechanism detected but not used. **Potential**: MAR → conditional imputation, MNAR → `{col}_is_missing` indicator features. |
| 2.7 Segment-Aware Outlier | False outlier rates, segment_aware_cap recommendations → registry.bronze | YES (registry) | YES (SEGMENT_AWARE_CAP step) | YES | Drives bronze transformation. `n_segments` and recommendations in findings.metadata. |
| 2.8 Global Outlier (IQR) | Log transform / winsorize recommendations → registry.bronze | YES (registry) | YES (CAP_OUTLIER/WINSORIZE) | YES | Drives bronze transformation with IQR bounds. Skips if segment-aware already added. |
| 2.9 Date Logic Validation | Date ranges, placeholder detection, sequence violations | NO | NO | NO | **Informational only.** Placeholder dates (pre-2005) detected but not filtered. **Potential**: Generate date validation filter in landing. |
| 2.10 Binary Field Validation | Invalid value detection in binary columns | NO | NO | NO | Informational. |
| 2.11 Data Consistency | Case variants detected → registry.bronze (normalize_lower) | YES (registry) | YES (consistency normalization) | YES | Generates `.str.lower()` in bronze via `columns[name].cleaning_recommendations`. |
| 2.12 Quality Recommendations | Aggregated cleaning recommendations → registry.bronze (null/outlier) | YES (registry) | YES (null/outlier steps) | YES | `registry.bronze.null_handling` and `registry.bronze.outlier_handling` consumed by FindingsParser. |
| 2.13 Save Findings | Updated findings + registry YAML | YES | YES | YES | |

---

## NB03: Dataset Merge

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 3.1-3.2 Setup/Load | Boilerplate + ProjectContext, SnapshotGrid from namespace | - | - | - | Loads entity_id, grid dates for spine building. |
| 3.3 Load Bronze Datasets | Granularity-aware dataset loading, per-dataset shape | YES | NO | NO | Exploration-only merge. Pipeline generates its own silver merge. |
| 3.4 Build Spine | Entity x grid date Cartesian product | YES (merge input) | YES (concept: silver merge) | YES | Uses TemporalMerger (local) or SparkTemporalMerger (distributed). |
| 3.5 Merge | Equi-join / broadcast / as-of join based on granularity | YES (merged df) | YES (silver joins) | YES | `MergeReport.renamed_columns` flows to merged findings. Pipeline generates equivalent join code from `config.silver.joins`. |
| 3.6 Validation | Temporal integrity check, row preservation | NO | NO | NO | Informational. Asserts `len(merged) == report.spine_rows`. |
| 3.7 Save Merged | Silver merged Delta table + merged findings YAML | YES (downstream) | YES (merged findings read by generators) | YES | `renamed_columns` dict affects column name references in silver/gold. |
| 3.8-3.9 Preview/Summary | Informational | NO | NO | NO | Uses `safe_describe()`. |

---

## NB04: Column Deep Dive

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 4.1-4.2 Setup | Boilerplate, load merged findings | - | - | - | Detects event-level data; recommends 01d first. |
| 4.3 Value Range Validation | Bronze filtering rules (cap %, filter negatives, binary validation) → registry.bronze | YES (registry) | NO | NO | **Change from original report.** `registry.sources[name].filtering` is defined but NOT consumed by FindingsParser. Filtering rules NOT generated in pipeline. **Gap**: Validation rules should generate bronze filter steps. |
| 4.4-4.5 Numeric Distribution Analysis | Skewness/kurtosis → transformation recommendations (log/sqrt/box_cox/yeo_johnson) → registry.gold | YES (registry) | YES (GOLD transforms) | YES | Drives `apply_log_transform()`, `apply_sqrt_transform()` etc. via `registry.gold.transformations`. |
| 4.6 Categorical Analysis | Cardinality, imbalance, entropy → encoding recommendations (one-hot/target/frequency/ordinal/sin_cos) → registry.gold | YES (registry) | YES (GOLD encodings) | YES | Drives encoding steps via `registry.gold.encoding`. Detects cyclical columns separately. |
| 4.7 Datetime Analysis | Growth trends, seasonality, feature engineering (days_since_X, cyclical, tenure) → registry.silver/bronze | YES (registry) | PARTIAL | PARTIAL | Silver derived columns (days_since) generated via `registry.silver.derived_columns`. But cyclical encoding (month_sin/cos) and modeling strategies (time-based split) from this section NOT consumed. **Gap**: Temporal modeling strategies should inform training split logic. |
| 4.8 Type Override | Manual type corrections → findings | YES (findings) | YES (corrected types used) | YES | |
| 4.9 Data Segmentation | Natural segments, target variance by segment, single vs multi-model recommendation | NO | NO | NO | **Informational only.** Quality score, segment count, rationale displayed. Uses SegmentAnalyzer/SparkSegmentAnalyzer. **Potential**: (1) `segment_id` as categorical feature. (2) Segment-specific training runs if high target variance ratio. |
| 4.10 Save | Findings + registry YAML | YES | YES | YES | |

---

## NB04a: Text Columns Deep Dive (Entity-Level)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 4a.1-4a.2 Load | Boilerplate | - | - | - | Early exit if no TEXT columns. |
| 4a.3 Configuration | Embedding model, variance threshold, component bounds | YES | YES | YES | FindingsParser reads text_processing → TextFeatureConfig. |
| 4a.4 Text Analysis | Text length stats, character count histograms | NO | NO | NO | Informational. |
| 4a.5 Process Text | Embeddings → PCA → PC features (batch_size=32) | YES (df updated) | YES | YES | **Gap closed.** Text embedding results now flow to pipeline via `findings.text_processing`. Bronze templates generate text processing steps. |
| 4a.6 Visualize | PC variance, PC1 vs PC2 scatter | NO | NO | NO | Informational. |
| 4a.7 Update Findings | `findings.text_processing[col]` = TextProcessingMetadata | YES (saved) | YES | YES | **Previously orphaned; now consumed.** FindingsParser reads `text_processing` and generates `TextFeatureConfig`. |
| 4a.8 Recommendations | Production action summary (embed_reduce per column) | NO | NO | NO | Informational. |

---

## NB05: Relationship Analysis

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 5.1 Setup | Boilerplate | - | - | - | 3-tuple load; creates empty registry if none prior. |
| 5.1b Leakage Exclusion | `excluded_leaking_features` list via `detect_leaking_features()` | YES (columns removed) | NO | NO | Excluded features removed from findings.columns before analysis. Generators don't explicitly check `excluded_leaking_features` field - they just don't see the columns. Works indirectly. |
| 5.2 Correlation Matrix | Pairwise correlation heatmap | NO | NO | NO | Informational. Uses `batched_corr_matrix()`. |
| 5.3 High Correlation Pairs | Pairs with |r| >= 0.7 | YES (5.9 input) | NO | NO | Informational table. Fed to recommender in 5.9. |
| 5.4 Feature Distributions by Target | Cohen's d effect sizes per feature | YES (5.9 input) | NO | NO | Effect sizes inform 5.9 weak/strong classification. |
| 5.5 Feature-Target Correlations | Correlation ranking | NO | NO | NO | Informational. |
| 5.6 Categorical Feature Analysis | Cramer's V, high-risk categories, lift, retention rates per category | NO | NO | NO | **Informational only.** Uses `CategoricalTargetAnalyzer`. **Potential**: High-risk categories could generate binary indicator features. |
| 5.7 Scatter Plot Matrix | Pairwise scatter (top 4 features, sampled 1000 pts) | NO | NO | NO | Informational. |
| 5.8 Datetime Feature Analysis | Yearly/monthly/DOW retention trends, seasonal spread | NO | NO | NO | **Informational only.** Uses `TemporalTargetAnalyzer`. **Potential**: Strong DOW/month effects could generate features. |
| 5.9 Recommendations | Multicollinear pairs → drop_multicollinear, strong predictors → prioritize, weak → drop_weak → registry.gold | YES (registry) | YES (gold feature_selection) | YES | Drives feature selection in gold layer. Stores metadata in `findings.metadata["relationship_analysis"]`. |
| 5.9.2 Stratification | High-risk segments needing representation in splits | NO | NO | NO | **Informational.** Not enforced in generated split logic. |
| 5.9.3 Model Selection | Linear vs tree-based recommendation | NO | NO | NO | **Informational.** Training always trains 3 models (LR, RF, XGB/GBM). |
| 5.9.4 Feature Engineering | Ratio/interaction feature suggestions → registry.silver | YES (registry) | YES (silver derived_columns) | YES | Generates `create_ratio_features()`, `create_interaction_features()` via `registry.silver.derived_columns`. |
| 5.9.5 Save | Findings + registry + merged findings/recommendations | YES | YES | YES | Saves to namespace for multi-dataset visibility. |

---

## NB06: Feature Opportunities

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 6.1 Setup | Boilerplate | - | - | - | `load_notebook_findings(prefer_merged=True)`. |
| 6.2 Automated Feature Recs | Printed feature suggestions from `RecommendationEngine` | NO | NO | NO | Informational. |
| 6.3 Feature Capacity (EPV) | EPV score, capacity status, recommended feature count → registry + findings | YES (registry + metadata) | NO | NO | **Informational for pipeline.** `registry.add_bronze_feature_capacity()` persisted but generators don't enforce feature count limits. **Potential**: Gold layer auto-trim by EPV count. |
| 6.3.1 Model Complexity | Recommended model type (linear/regularized/tree), max features → registry | YES (registry) | NO | NO | **Not consumed by generators.** `registry.add_bronze_model_type()` persisted but training always trains all model types. |
| 6.3.2 Segment Capacity | Per-segment EPV viability, recommended strategy | YES (metadata) | NO | NO | Informational. Stored in `findings.metadata["segment_capacity"]`. |
| 6.3.3 Action Items | Feature budget summary | NO | NO | NO | Informational. |
| 6.3.4 Feature Availability | Unavailable features with coverage %, remediation options | YES (metadata) | NO | NO | Stored in `findings.metadata["unavailable_features"]`. Used by NB08 for exclusion. **Gap**: `{col}_available` indicator not generated in bronze. |
| 6.4 Datetime Opportunities | Potential extractions (year, month, DOW, is_weekend, days_since) | NO | NO | NO | Informational. Already handled in NB01/04 datetime derivation. |
| 6.5 Business-Driven Features | tenure_days, days_since_last_activity, engagement_score, click_to_open_rate, service_adoption_score, value_frequency_product → registry.silver | YES (registry) | YES (silver derived_columns) | YES | Generates ratio/interaction/composite features in silver. |
| 6.6 Customer Segmentation Features | Value-frequency, recency, engagement segments → registry | YES (registry) | YES (silver derived_columns) | YES | `CustomerSegmenter` segment assignments persisted to registry and generated in silver. |
| 6.7 Numeric Transformations | Log/sqrt/standard scaling → registry.gold | YES (registry) | YES (gold transforms/scaling) | YES | May overlap with NB04; registry deduplicates. |
| 6.8 Categorical Encoding | One-hot/target/frequency encoding → registry.gold | YES (registry) | YES (gold encodings) | YES | May overlap with NB04; registry deduplicates. |
| 6.9 Summary | Feature table display | NO | NO | NO | Informational. |

---

## NB07: Modeling Readiness

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 7.1 Setup | Boilerplate | - | - | - | `load_notebook_findings(prefer_merged=True)`. |
| 7.2 Readiness Checklist | Pass/Fail gates (target, features, missing <50%, quality >=70, rows >=100) | NO | NO | NO | Informational validation. |
| 7.3 Class Imbalance | Severity, ratio, sklearn class_weight config, mitigation strategy | NO | NO | NO | **Informational.** Duplicates NB02 analysis. Neither flows to generated training. Uses `ImbalanceRecommender`. |
| 7.4 Leakage Risk | High-correlation (>0.9) flags, suspicious column names | NO | NO | NO | Informational warning. |
| 7.5 Feature Type Summary | Column count by type | NO | NO | NO | Informational inventory. |
| 7.6 Readiness Score | Score 0-100, status tier (READY/MOSTLY READY/NEEDS WORK/NOT READY) | NO | NO | NO | Informational. 25-point scoring. **Potential**: MLflow tag. |
| 7.7 Feature Availability | Unavailable features flagged for exclusion | YES (NB08 exclusion) | NO | NO | References `findings.metadata["unavailable_features"]` from NB06. |
| 7.X Leakage Validation | LeakageDetector checks | YES (blocking gate) | NO | NO | Raises error if critical leakage. Does NOT write to findings. |

---

## NB08: Baseline Experiments

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 8.1 Setup | Boilerplate | - | - | - | sklearn, XGBoost/GBM imports. |
| 8.2 Data Preparation | Feature selection, unavailable feature removal, imputation, scaling, train/test split | YES (modeling) | NO | NO | Exploration-only prep. Handles temporal splits with purge gap. Drops zero-variance columns. Uses `FeatureSelector.get_availability_recommendations()`. |
| 8.3 Baseline Models | LR/RF/GBM training + evaluation (AUC, PR-AUC, F1, Precision, Recall) with balanced class weights | YES (model artifacts) | NO | NO | Exploration-only models. Stratified 5-fold or temporal entity-based CV. |
| 8.4 Feature Importance | Top 15 features from RF | NO | NO | NO | **Informational.** First place where model-based feature importance is available. **Key potential**: Cross-reference with stability (NB01a), patterns (NB01c), effect sizes (NB05). |
| 8.5-8.6 Reports/Grid | Classification reports, confusion matrices, ROC/PR curves | NO | NO | NO | Informational. Binary and multiclass support. |
| 8.7 Key Takeaways | Best model, top features, performance assessment (excellent/strong/moderate/weak) | NO | NO | NO | Informational summary with next steps. |

---

## NB09: Business Alignment

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 9.1 Setup | Boilerplate | - | - | - | |
| 9.2 Business Context | project_name, objective, stakeholders, timeline | YES (findings) | NO | NO | Stored in `findings.metadata["business_context"]`. **Potential**: Pipeline README or report comments. |
| 9.3 Success Metrics | AUC >= 0.80, Precision >= 0.60, Churn Reduction 20%, Latency < 100ms, Fairness >= 0.8 | YES (findings) | NO | NO | Stored in `findings.metadata["success_metrics"]`. **Potential**: Training threshold validation or early stopping. |
| 9.4 Deployment Requirements | Batch/real-time, frequency, latency, Databricks infra, MLflow registry, retraining | YES (findings) | NO | NO | Stored in `findings.metadata["deployment_requirements"]`. **Potential**: workflow.json scheduling, Databricks job cadence. |
| 9.5 Data Constraints | PII, freshness, historical depth, protected attributes | YES (findings) | NO | NO | **Potential**: PII auto-exclusion, fairness validation in scoring. |
| 9.6 Intervention Strategy | Risk tiers (high/medium/low), costs, effectiveness rates | YES (findings) | NO | NO | **Potential**: Scoring output enrichment with intervention tier. |
| 9.7 Save to Findings | All above → findings.metadata | YES (saved) | NO | NO | Persisted but generators don't read business context. |

---

## NB10: Pipeline Generation (Spec Generation)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 10.1 Configuration | Generation target (LOCAL/DATABRICKS/LLM_DOCS), OUTPUT_FORMAT (PY/NOTEBOOK), catalog/schema | YES | YES | YES | Auto-detects Databricks vs local. |
| 10.2 Load Findings/Recs | Loads all findings + recommendations YAMLs | YES | YES | YES | Handles multiple naming patterns; loads multi_dataset findings. |
| 10.3 Review Recommendations | Grouped display of bronze/silver/gold recommendations (first 5 per layer) | NO | NO | NO | Informational review. |
| 10.4A Generate Local Pipeline | `PipelineGenerator.generate_all()` → config, landing, bronze, silver, gold, training, runner, workflow.json, feast, validation, docs | YES | **YES** | N/A | Core local generation. Also generates MLflow pipeline files. |
| 10.4B Generate Databricks Pipeline | `DatabricksPipelineGenerator.generate_all()` → config, bronze, silver, gold, training, runner (Unity Catalog + PySpark) | YES | N/A | **YES** | Core Databricks generation. No landing layer (assumes Databricks source ingestion). No Feast integration. |
| 10.4C Generate LLM Docs | Markdown documentation (overview, bronze per source, silver, gold, training) | YES | PARTIAL | NO | Documentation only. Per-layer + per-source docs. |
| 10.5 Convert to Notebooks | .py → .ipynb conversion | YES | OPTIONAL | N/A | Format choice. |
| 10.6 Run Pipeline | Execute Bronze → Silver → Gold → Training | YES | YES (if RUN_PIPELINE) | YES | dbutils.notebook.run (DBX) or subprocess (local). |
| 10.7 Summary | File tree display with sizes | NO | NO | NO | Informational. |
| 10.8 Recommendations Hash | Unique hash from gold config → MLflow tag, version tag | YES | YES | YES | Critical for experiment tracking and reproducibility. |
| 10.9 Feast Validation | Feature store registry inspection (entities, views, sources) | NO | NO | NO | Post-generation validation. |

---

## NB11: Scoring Validation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 11.1 Run Scoring | Holdout predictions → predictions.parquet, feature alignment, validation metrics | YES | YES (scoring output) | YES | Validates pipeline end-to-end. Loads model from MLflow. Aligns scoring features to training features. |
| 11.2 Summary Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, probability histogram | NO | NO | NO | Informational. |
| 11.3 Model Comparison | All models loaded by model_type + recommendations_hash, comparison table | NO | NO | NO | Informational. Highlights best by ROC-AUC. |
| 11.4 Adversarial Validation | Feature drift between training and scoring (max delta per feature) | NO | NO | NO | Validation with 1e-6 tolerance. **Potential**: Automated drift monitoring. |
| 11.5 Transformation Validation | Encoding/scaling consistency check | NO | NO | NO | Validation. Uses `validate_feature_transformation()`. |
| 11.6 SHAP Explanations | Top 20 features by mean |SHAP|, per-entity waterfall, TreeExplainer/PermutationExplainer | NO | NO | NO | **Key informational output.** Definitive feature ranking. **Potential**: Cross-reference with stability, pattern, and effect size analyses. Low-SHAP features for pipeline v2 trimming. |
| 11.7 Customer Browser | Per-entity prediction + SHAP lookup functions | NO | NO | NO | Diagnostic utility. Merges predictions with scoring features. |
| 11.8 Error Analysis | FP/FN examples with SHAP waterfalls | NO | NO | NO | Informational. **Potential**: Systematic error patterns for feature gap identification. |
| 11.9 Export Results | feature_importance.csv, predictions_with_shap.parquet (top 10 SHAP), Delta table (on DBX) | YES | YES (artifacts) | YES (Delta) | Handles both local and Databricks exports. |

---

## NB12: View Documentation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 12.1-12.3 | HTML export inventory and display | NO | NO | NO | Purely documentation viewing. `track_and_export_previous()`, `check_exported_html()`, `display_html_documentation()`. No pipeline impact. |

---

# Summary: Pipeline Parity Status

## CLOSED GAPS (fixed since original analysis)

| Gap | Original Status | Current Status | How Fixed |
|-----|----------------|----------------|-----------|
| **Text embeddings (event-level)** | Orphaned from generators | YES (both local + DBX) | FindingsParser reads `findings.text_processing` → `TextFeatureConfig`; bronze templates generate text embedding steps |
| **Text embeddings (entity-level)** | Orphaned from generators | YES (both local + DBX) | Same mechanism via NB04a → `findings.text_processing` |
| **Lag features** | Not generated | YES (both local + DBX) | FindingsParser reads `metadata["temporal_patterns"]["lag_features_computed"]`, `lag_columns`, `lag_window_days` → `bronze_event.temporal_features` |
| **Recency bucket boundaries** | Hardcoded buckets | YES (both local + DBX) | FindingsParser reads `metadata["temporal_patterns"]["recency_analysis"]["bucket_boundaries"]` → `bronze.lifecycle.recency_bucket_edges` |
| **Lifecycle quadrant feature_flag** | Not consumed | YES (both local + DBX) | FindingsParser reads `metadata["temporal_patterns"]["feature_flags"]["include_lifecycle_quadrant"]` |
| **Cyclical features flag** | Not consumed | YES (both local + DBX) | FindingsParser reads `include_cyclical_features` flag → bronze lifecycle config |
| **Future date filtering (TQ003)** | Not generated | YES (both local + DBX) | FindingsParser reads `metadata["temporal_quality"]["checks"]` TQ003 → `config.training.filter_future_dates` |
| **recommended_training_start** | Not used | YES (both local + DBX) | FindingsParser reads `metadata["time_series"]["recommended_training_start"]` → `config.training.recommended_training_start` |
| **Momentum pairs** | Partial (multi-dataset only) | YES (both local + DBX) | FindingsParser reads momentum config → `bronze.lifecycle.momentum_pairs` |

## REMAINING HIGH PRIORITY GAPS

| Gap | Exploration Source | What's Missing in Pipeline | Impact |
|-----|-------------------|---------------------------|--------|
| **Trend features** | NB01c `include_trend_features` flag, NB01d trend step | `include_trend_features` flag NOT consumed by FindingsParser. `recent_vs_overall_ratio`, `entity_trend_slope` features not generated. | Trend signal lost between exploration and production. |
| **Cohort features** | NB01c `include_cohort_features` flag, NB01d cohort step | `include_cohort_features` flag NOT consumed by FindingsParser. `cohort_year`, `cohort_quarter` features not generated. | Acquisition-timing signal lost. |
| **Velocity features** | NB01c velocity analysis, NB01d velocity step | Velocity (rate of change per window) computed in exploration but not replicated in generated bronze event template. | Rate-of-change signal lost. |
| **Bronze deduplication** | NB02 section 2.2 | `registry.sources[name].deduplication` is defined but NOT consumed by FindingsParser. No dedup step in generated bronze code. | Duplicate rows may persist in production pipeline. |
| **Bronze filtering rules** | NB04 section 4.3 | `registry.sources[name].filtering` is defined but NOT consumed by FindingsParser. Value range validation not generated. | Invalid values (negatives, out-of-range) not filtered in production. |

## REMAINING MEDIUM PRIORITY GAPS

| Gap | Exploration Source | What's Missing | Impact |
|-----|-------------------|----------------|--------|
| **Imbalance strategy** | NB02, NB07 | `registry.bronze.modeling_strategy` checked but not fully consumed; training uses `class_weight="balanced"`. | Suboptimal handling of severe imbalance (SMOTE, explicit weights). |
| **Split strategy config** | NB00 intent, NB04 | `SPLIT_STRATEGY` (temporal vs cohort) should drive training script's split method. FindingsParser sets `split_strategy` but training template may not fully differentiate. | Potential train/test contamination. |
| **Purge gap** | NB00 intent | `PURGE_GAP_DAYS` defined, FindingsParser reads it into `config.training.purge_gap_days`, but enforcement in generated training template may be incomplete. | Potential leakage between train/test. |
| **Model type recommendation** | NB06 | EPV-based model recommendation (`registry.add_bronze_model_type()`) not consumed; all 3 models always trained. | Potential overfitting with inadequate EPV. |
| **Success metrics** | NB09 | Target AUC/precision thresholds not used as training validation gates. | No automated quality gate. |
| **Deployment frequency** | NB09 | Retraining cadence not used in workflow.json scheduling. | Manual scheduling required. |
| **Monthly/quarterly seasonality** | NB01c section 1c.6 | DOW cyclical features generated, but `month_sin/cos` and `quarter_sin/cos` NOT generated even when detected. | Partial seasonality encoding. |

## LOW PRIORITY (informational analyses with untapped potential)

| Gap | Source | Potential Value | When Relevant |
|-----|--------|-----------------|---------------|
| **Stability per feature** | NB01 stability | Top model features should be temporally stable; could flag risky features | After model training (NB08/11) |
| **Dataset type routing** | NB01 structure detection | Snapshot vs longitudinal vs time_series could drive different aggregation | Multi-pattern datasets |
| **Aggregation guidance per column** | NB01a CV analysis | All-time-only vs short-window-only per column would reduce feature explosion | Large feature counts (>100) |
| **Segmentation features** | NB04, NB06 | Detected segments could become categorical features | High target variance by segment |
| **Missing value indicators** | NB02 MNAR detection | `{col}_is_missing` features for MNAR columns | After model shows missingness matters |
| **High-risk category indicators** | NB05 categorical | Binary `is_high_risk_{category}` features | After model importance confirms |
| **Objective-specific config** | NB01 ObjectiveSupport signals | Multi-objective label logic, objective-aware feature weighting | Multi-objective use cases |
| **Intervention tier enrichment** | NB09 business | Scoring output with recommended action tier | Production deployment |
| **Feature importance feedback** | NB08/11 SHAP | Low-importance features dropped in pipeline v2 | Iterative improvement |
| **Drift monitoring** | NB11 adversarial validation | Automated drift detection in scoring pipeline | Production monitoring |

---

# Appendix: FindingsParser Consumption Map

## Fields Consumed from ExplorationFindings

| Field | Source NB | Pipeline Config Target |
|-------|-----------|----------------------|
| `source_path` | NB01 | `SourceConfig.raw_source_path` |
| `source_format` | NB01 | `SourceConfig.format` |
| `row_count`, `column_count` | NB01 | `DatasetInfo` |
| `target_column` | NB01 | `config.target_column` |
| `identifier_columns[0]` | NB01 | `entity_key` per source |
| `columns[name].inferred_type` | NB01/04 | Determines encoding/scaling strategy |
| `columns[name].cleaning_needed` | NB02/04 | Triggers bronze transformations |
| `columns[name].cleaning_recommendations` | NB02/04 | Drives impute/cap/drop actions |
| `time_series_metadata.*` | NB01a | Time column, entity column, aggregation windows |
| `datetime_ordering` | NB01 | `TimestampCoalesceConfig` |
| `datetime_derivation_sources` | NB01 | Landing datetime derivation |
| `datetime_allow_future_columns` | NB01 | `mask_future_columns` in landing/bronze |
| `label_timestamp_column` | NB01 | Label timestamp config |
| `observation_window_days` | NB01 | Fallback window (default 180) |
| `text_processing[col]` | NB01a_a/04a | `TextFeatureConfig` (model, components) |
| `metadata["temporal_patterns"]["feature_flags"]` | NB01c | Lifecycle toggles (recency, lifecycle quadrant, cyclical) |
| `metadata["temporal_patterns"]["recency_analysis"]["bucket_boundaries"]` | NB01c | Recency bucket edges/labels |
| `metadata["temporal_patterns"]["lag_*"]` | NB01c | Temporal feature config (lag_window, num_lags, columns) |
| `metadata["aggregation"]["include_lifecycle_quadrant"]` | NB01d | Fallback lifecycle detection |
| `metadata["time_series"]["recommended_training_start"]` | NB01a | Training start date |
| `metadata["temporal_quality"]["checks"]` TQ003 | NB01b | `filter_future_dates` flag |
| `metadata["original_target_column"]` | NB01 | Original target before renaming |

## Fields Consumed from RecommendationRegistry

| Registry Path | Source NB | Pipeline Config Target |
|---------------|-----------|----------------------|
| `bronze.null_handling` | NB02/04 | IMPUTE_NULL / DROP_COLUMN steps |
| `bronze.outlier_handling` | NB02/04 | CAP_OUTLIER / WINSORIZE / SEGMENT_AWARE_CAP steps |
| `bronze.text_processing` | NB01a_a/04a | TextFeatureConfig (partial) |
| `silver.derived_columns` | NB05/06 | DERIVED_COLUMN steps (ratio, interaction, composite) |
| `gold.encoding` | NB04/06 | ENCODE steps (one_hot, label, target, frequency) |
| `gold.scaling` | NB06 | SCALE steps (standard, minmax) |
| `gold.transformations` | NB04/06 | Transform steps (log, sqrt, yeo_johnson, cap_then_log) |
| `gold.feature_selection` | NB05 | Drop multicollinear/weak; prioritize strong |

## Fields NOT Consumed (Orphaned)

| Field/Registry Path | Source NB | Status |
|---------------------|-----------|--------|
| `overall_quality_score` | NB01 | Stored, not used |
| `universal_metrics`, `type_metrics`, `confidence`, `evidence` | NB01 | Column-level metadata, not used |
| `feature_availability` | NB06 | Entirely unused by generators |
| `excluded_leaking_features` | NB05 | Works indirectly (columns removed before save) |
| `snapshot_id`, `snapshot_path`, `timestamp_scenario` | NB01 | Snapshot fields unused |
| `metadata["temporal_patterns"]["feature_flags"]["include_trend_features"]` | NB01c | **Not consumed** |
| `metadata["temporal_patterns"]["feature_flags"]["include_cohort_features"]` | NB01c | **Not consumed** |
| `metadata["temporal_patterns"]["feature_flags"]["include_seasonality_features"]` | NB01c | Partially consumed (cyclical only, not month/quarter) |
| `metadata["temporal_quality"]["score"]`, `grade` | NB01b | Stored, not used |
| `metadata["business_context"]` | NB09 | Stored, not used |
| `metadata["success_metrics"]` | NB09 | Stored, not used |
| `metadata["deployment_requirements"]` | NB09 | Stored, not used |
| `registry.sources[name].deduplication` | NB02 | **Defined but not consumed** |
| `registry.sources[name].filtering` | NB04 | **Defined but not consumed** |
| `registry.bronze.modeling_strategy` | NB02 | Checked but not fully consumed |
| `registry.bronze.feature_capacity` | NB06 | Stored, not used |
| `registry.bronze.model_type` | NB06 | Stored, not used |
| `registry.fit_artifacts` | Various | Not used |
| `recommendation.priority`, `dependencies`, `fit_artifact_id` | Various | Metadata only, not used in generation |
