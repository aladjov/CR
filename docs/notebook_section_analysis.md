# Notebook Section Analysis: Recommendations, Pipeline Parity & Gaps

## Legend
- **Rec. Made**: What recommendation/finding the section produces
- **Applied in NB**: Whether the recommendation is consumed within the notebook flow
- **Local Pipeline**: Whether the local `PipelineGenerator` consumes this finding (`YES`/`NO`/`PARTIAL`)
- **DBX Pipeline**: Whether `DatabricksPipelineGenerator` consumes this finding (`YES`/`NO`/`PARTIAL`)
- **Note**: Purpose if informational-only; untapped potential; future relevance

---

## NB00: Start Here (Project Bootstrap)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 0.1 Project Metadata | Run ID, namespace, LIGHT_RUN, sampling config | YES (all downstream) | NO | NO | Boilerplate setup. Sampling params affect data volume but not generated code structure. |
| 0.2 Dataset Registration | Dataset name-to-path mapping | YES (fingerprinting) | YES (source paths) | YES (table names) | Core input. Drives `config.py` source definitions. |
| 0.3 Auto Fingerprinting | Granularity, entity col, time col, target candidates | YES (semantics) | YES (via ProjectContext) | YES | Foundational detection. Determines event vs entity bronze split. |
| 0.4 Confirm Semantics | Entity/time/granularity overrides | YES (context save) | YES (entity_column, time_column) | YES | User review gate. **Gap**: No generated comment in pipeline code showing which overrides were applied - would help auditing. |
| 0.5 Target Dataset Selection | TARGET_DATASET, TARGET_COLUMN, ENTITY_COLUMN | YES (context save) | YES (target_column in training) | YES | **Gap**: If target is in a non-primary dataset, the relationship between target dataset and feature datasets could be surfaced as a lineage note. |
| 0.6 Prediction Objective Detection | Feasibility scores per objective type (IMMEDIATE_RISK, LONG_TERM_RETENTION, etc.) | YES (priority assignment) | NO | NO | **Informational only.** Objectives are detected and scored but do NOT flow into pipeline code. The pipeline always generates a single binary classification regardless. **Potential**: Multi-objective pipelines (e.g., generate separate label logic per objective), or at minimum generate comments/config for the user indicating which objective the pipeline serves. |
| 0.7 Objective Priority Review | PRIMARY_OBJECTIVE, anchor type | YES (intent config) | NO | NO | User can swap priorities. **Gap**: The primary objective's anchor (e.g., PURCHASE_RECENCY) could inform which features get prioritized in gold layer feature selection - currently doesn't. |
| 0.8 Join Scaffold | MergeScaffoldEntry list (join keys, relationship types) | YES (context save) | YES (silver joins) | YES | Drives silver merge logic. `many_to_one` vs `many_to_many` affects join type. |
| 0.9 Temporal Posture | STABLE vs ADAPTIVE enum | YES (intent defaults) | NO | NO | **Informational only.** Affects IntentDefaultsEngine suggestions but posture itself is not in generated config. **Potential**: ADAPTIVE posture could generate shorter default windows or more frequent retraining schedules in workflow.json. |
| 0.10 Intent Configuration | Prediction horizons, observation window, purge gap, label window, cadence, split strategy | YES (context save) | PARTIAL | PARTIAL | `observation_window_days` drives `label_timestamp` offset in landing. Cadence/split strategy NOT used in generated code. **Gap**: `PURGE_GAP_DAYS` is defined but not enforced in generated train/test split logic. `CADENCE_INTERVAL` could drive workflow scheduling. `SPLIT_STRATEGY` (temporal vs cohort) should drive training script's split method. |
| 0.11 Save ProjectContext | ProjectContext YAML (all-in-one) | YES (all notebooks) | YES (indirectly via FindingsParser) | YES | Single source of truth. |
| 0.12 Initialize Snapshot Grid | Grid dates, cadence, observation window, voting status | YES (01d aggregation) | NO | NO | Grid coordinates multi-dataset aggregation timing. **Gap**: Generated pipeline doesn't use snapshot grid for incremental processing - it's one-shot. Future: incremental pipeline could use grid dates for micro-batch boundaries. |

---

## NB01: Data Discovery (Per-Dataset)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1.1 Configuration | DATA_PATH, TARGET_COLUMN, ENTITY_COLUMN, DROP_COLUMNS, ALLOW_FUTURE_COLUMNS | YES | YES (source_path, columns) | YES | Core user inputs. |
| 1.2 Load Data + Explore | Column types, target detection, granularity, fingerprint | YES (findings) | YES (column types drive bronze/gold) | YES | `inferred_type` per column is the backbone of all generated code. |
| 1.3 Column Summary Table | Visual table of types/confidence/nulls | NO | NO | NO | Informational. User reviews auto-detection accuracy. |
| 1.4 Type Override | Manual type corrections (inferred_type updates) | YES (findings save) | YES (overridden types used) | YES | Flows through FindingsParser to pipeline. |
| 1.5 Dataset Structure Detection | Temporal pattern (time_series/snapshot/longitudinal), routing guidance | NO | NO | NO | **Informational only.** `dataset_type` is detected but not consumed by generators. **Potential**: Longitudinal datasets could get different aggregation strategies (e.g., first/last vs rolling windows). Snapshot datasets could skip windowed aggregation entirely. |
| 1.6 Feature Timestamp Derivation | `feature_timestamp` source, `datetime_derivation_sources`, `datetime_allow_future_columns` | YES (active dataset) | YES (landing template) | YES | Drives `derive_datetime_features()` in landing + bronze. `mask_future_columns` prevents leakage. |
| 1.6 Datetime Feature Derivation | Derived temporal columns (days_since_X, milestone pairs) | YES (active dataset) | YES (landing datetime_derivation) | YES | Creates features like `days_since_signup`, `days_between_created_resolved`. |
| 1.6 Entity Sampling | Sampled entity IDs (cross-dataset consistency) | YES (active dataset filter) | NO | NO | **Not in pipeline.** Sampling is exploration-only. **Gap**: For very large production datasets, the pipeline could generate an optional sampling step for development runs. |
| 1.6 Save Active Dataset | Delta table with cleaned/derived data | YES (downstream NBs) | NO (pipeline re-derives from raw) | NO | Active dataset is for exploration only. Pipeline starts from raw CSV. |
| 1.7 Structural Stability | `dataset_stability_score`, volume/entity/distribution/cadence drift metrics | YES (findings metadata) | NO | NO | **Informational only.** Stability score is computed and stored but generators ignore it. **Potential**: (1) Low stability could trigger a warning comment in generated code. (2) Could drive recommendation to use temporal cross-validation instead of random split. (3) Regime detection (`regime_count`) could inform whether to add a `data_regime` feature column. (4) Once model feature importance is known, stability of top features becomes critical - a post-model "feature stability report" could flag risky features. |
| 1.7 Objective Support Signals | Per-objective signal levels (ImmediateRisk, Disengagement, Renewal) | NO | NO | NO | **Informational only.** Signals are displayed but not persisted to findings or consumed anywhere. **Potential**: Could inform objective-specific feature weighting or be logged as MLflow tags for experiment context. |
| 1.8 Save Findings | ExplorationFindings YAML (column types, target, metadata) | YES (all downstream) | YES (FindingsParser reads) | YES | Critical persistence point. |
| 1.8b Snapshot Grid Vote | Dataset temporal coverage registered on grid | YES (01d readiness) | NO | NO | Coordination mechanism for multi-dataset. Not in generated pipeline. |

---

## NB01a: Temporal Deep Dive (Event Bronze Track)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1a.1-1a.2 Load | Boilerplate | - | - | - | Loads findings + data. |
| 1a.3 Time Series Profile | Events/entity stats, time span, inter-event timing | YES (downstream sections) | NO | NO | **Informational.** Profile stats inform window recommendations but are not directly consumed by generators. |
| 1a.4 Events per Entity Distribution | Activity segments (One-time, Low, Medium, High) with thresholds | YES (window collector) | NO | NO | **Informational only.** Segment counts/thresholds displayed but not persisted as pipeline config. **Potential**: (1) One-time entities could be flagged with a binary `is_one_time_entity` feature in bronze. (2) Activity segment could become a categorical feature. (3) After model training, if activity segment correlates with prediction error, it suggests segment-specific models. |
| 1a.5 Entity Lifecycle Analysis | Lifecycle quadrants (steady/occasional/intense/one-shot), tenure/intensity thresholds | YES (findings metadata) | PARTIAL | PARTIAL | `lifecycle_quadrant` feature is generated in bronze if `include_lifecycle_quadrant` flag is set via 01c feature_flags. Quadrant definitions/thresholds themselves are NOT in pipeline config - hardcoded in `classify_lifecycle_quadrants()` function. **Gap**: Thresholds should be configurable in generated code to allow retuning without re-exploration. |
| 1a.6 Temporal Coverage & Drift | Volume trend, gap detection, drift risk level, regime count, recommended_training_start, inter-event skew | YES (findings metadata) | PARTIAL | PARTIAL | `drift_risk_level` and `regime_count` stored but NOT consumed by generators. `recommended_training_start` is stored but NOT used in generated train/test split. **Gap**: `recommended_training_start` should drive the temporal split date in `ml_experiment.py`. Volume trend could generate a `volume_regime` feature. |
| 1a.7 Temporal Aggregation Perspective | Within-CV vs between-CV per column, aggregation guidance | NO | NO | NO | **Informational only.** Per-column guidance (e.g., "all_time mean sufficient" vs "short windows preserve signal") is printed but not saved. **Potential**: Columns marked "all_time mean sufficient" could use only `all_time` window, reducing feature count. Columns with "high temporal dynamics" could get extra short windows. This would directly reduce feature explosion in bronze event aggregation. |
| 1a.8 Update Findings | `time_series_metadata` updated: suggested_aggregations, heterogeneity_level, segmentation_advisory, drift_risk, recommended_training_start | YES (saved to YAML) | PARTIAL | PARTIAL | `suggested_aggregations` → consumed as `aggregation.windows`. `heterogeneity_level`/`segmentation_advisory` → NOT consumed. **Gap**: `segmentation_advisory="consider_separate_models"` could generate two training scripts (one per segment). |

---

## NB01a_a: Temporal Text Deep Dive

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1a.a.1-1a.a.2 Load | Boilerplate | - | - | - | |
| 1a.a.3 Configuration | Embedding model preset, variance threshold, component bounds, aggregation windows/funcs | YES | PARTIAL | PARTIAL | `TextProcessingConfig` created but generators don't currently consume text processing metadata to generate embedding code. **Gap**: Pipeline should generate a bronze step that applies the chosen embedding model + PCA, matching the exploration config. Currently text columns are just dropped. |
| 1a.a.4 Text Column Analysis | Text length stats, sample texts | NO | NO | NO | Informational. |
| 1a.a.5 Process Text Columns | PC features (e.g., ticket_text_pc1..pcN), explained variance, component count | YES (df updated) | NO | NO | **Major gap.** Text→embedding→PCA is done in exploration but NOT replicated in generated pipeline. Generated bronze code does not embed text. PC features exist only in the exploration active dataset. **Potential**: Generate a bronze step that loads the embedding model, embeds text, applies PCA with saved components. |
| 1a.a.6 Plan Aggregation | Feature count, aggregation plan (windows x funcs x PCs) | YES (section 1a.a.8) | NO | NO | Plan is informational; not consumed by generator. |
| 1a.a.7 Visualize PCs | PC distribution histograms, scatter plots | NO | NO | NO | Informational. |
| 1a.a.8 Update Findings | `findings.text_processing[col]` = TextProcessingMetadata (model, dim, n_components, component_columns) | YES (saved to YAML) | NO | NO | Metadata is saved but **FindingsParser does not read `text_processing`**. **Major gap**: This entire notebook's output is orphaned from pipeline generation. |

---

## NB01b: Temporal Quality

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1b.1-1b.2 Load/Config | Boilerplate + REFERENCE_DATE, EXPECTED_FREQUENCY, MAX_GAP_MULTIPLE | YES | NO | NO | Quality check params are not in generated pipeline. |
| 1b.3 Quality Checks | TQ001 (duplicates), TQ002 (gaps), TQ003 (future dates), TQ004 (ordering) | YES (quality score) | PARTIAL | PARTIAL | `deduplicate` flag is set in bronze event config if duplicates detected. Future date filtering is NOT generated. **Gap**: TQ003 (future dates) should generate a `WHERE timestamp <= reference_date` filter in bronze. TQ002 (gaps) could generate a `has_data_gap` indicator feature. |
| 1b.4 Quality Score | Score (0-100), grade (A-D), passed boolean | YES (findings metadata) | NO | NO | **Informational.** Score is stored but generators don't use it. **Potential**: Grade D could generate a warning header in pipeline scripts. Quality score could become an MLflow tag. |
| 1b.5 Event Volume Analysis | Volume over time visualization | NO | NO | NO | Informational. No recommendations. |
| 1b.6 Outlier Analysis | Segment-aware vs global outlier comparison, segmentation_recommended flag | YES (recommendation) | NO | NO | **Informational within 01b.** Outlier recommendations are made in NB02/04 and those ARE consumed. |
| 1b.7 Data Validation | Binary field checks, string consistency issues | NO | NO | NO | Informational. No findings written. **Potential**: String inconsistencies (case variants) detected here could auto-generate `.str.lower()` in bronze, but this is handled in NB02 instead. |
| 1b.8 Recommendations | Column-level cleaning recommendations (impute, transform, group rare) | NO | NO | NO | Printed recommendations only. Not persisted to registry. **Note**: These are re-derived in NB02/04 where they ARE persisted. Redundant analysis. |
| 1b.9 Save Results | `findings.metadata["temporal_quality"]` with score/grade/check results | YES (saved) | NO | NO | Stored but not consumed by generators. |

---

## NB01c: Temporal Patterns

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1c.1-1c.4 Load/Config | Boilerplate + target detection, aggregation window config, value column selection | YES | NO | NO | Windows come from 01a. |
| 1c.5 Trend Detection | Trend direction (increasing/decreasing/stable), R^2, confidence, slope | YES (feature flags) | NO | NO | **Informational.** Trend direction/strength stored but generators don't produce trend-adjusted features. **Potential**: (1) Generate `recent_vs_overall_ratio` feature in bronze. (2) If strong trend detected, generate detrending logic (subtract rolling mean). (3) After model training, trend features' importance could validate whether detrending helps. |
| 1c.6 Seasonality Detection | Periodicities (weekly/monthly/quarterly), autocorrelation strengths | YES (feature flags) | NO | NO | **Informational.** `include_seasonality_features` flag is set but generators don't read it. Cyclical features (dow_sin/cos) are hardcoded in bronze lifecycle enrichment. **Gap**: Monthly/quarterly seasonality should generate additional cyclical features (month_sin/cos, quarter_sin/cos). |
| 1c.7 Cohort Analysis | Cohort distribution, retention by cohort, onboarding concentration | YES (feature flags) | NO | NO | **Informational.** `include_cohort_features` flag set but NOT consumed by generators. **Potential**: Generate `cohort_year`/`cohort_quarter` features in bronze. After model training, cohort feature importance would indicate whether acquisition timing matters. |
| 1c.8 Recency Analysis | Median recency, effect size (Cohen's d), bucket boundaries | YES (feature flags) | PARTIAL | PARTIAL | `include_recency` flag drives `add_recency_tenure()` in bronze. Bucket boundaries from 01c are NOT used - bronze uses hardcoded buckets. **Gap**: Custom bucket boundaries should be configurable in generated code. |
| 1c.9 Velocity Analysis | Divergent velocity columns between retained/churned | YES (01d config) | NO | NO | **Informational for 01c, consumed by 01d.** 01d uses divergent columns to prioritize aggregation. But velocity features themselves are NOT generated in pipeline code. **Potential**: Generate `{col}_velocity_{window}` features (rate of change) in bronze. |
| 1c.10 Momentum Analysis | Divergent momentum columns, acceleration/deceleration patterns | YES (01d config) | PARTIAL | PARTIAL | Momentum ratio features ARE generated in bronze if lifecycle config includes them. **Gap**: Only generated if multi-dataset findings has lifecycle flags. Single-dataset momentum is lost. |
| 1c.11 Sparklines | Per-feature temporal patterns (trend/seasonality/scaling) by cohort | NO | NO | NO | **Informational only.** Visual inspection of temporal dynamics per feature. **Potential**: Features with strong cohort divergence in sparklines could be auto-flagged for interaction terms (feature x cohort). |
| 1c.12 Lag Feature Analysis | Autocorrelation per feature at multiple lags, recommended lag days | YES (feature flags) | NO | NO | **Informational.** Lag recommendations stored but generators don't produce lag features. **Potential**: Generate `{col}_lag_{N}d` features in bronze event aggregation for columns with strong autocorrelation. |
| 1c.13-14 Effect Size & Predictive Power | Feature ranking by Cohen's d and KS statistic | YES (feature flags) | NO | NO | **Informational.** Rankings stored but not used for feature selection in generated code. **Potential**: Low-power features could be auto-dropped in gold layer feature_selection. |
| 1c.15 Categorical Feature Analysis | Cramer's V, high-risk categories, effect strength | YES (feature flags) | NO | NO | **Informational.** Not consumed by generators. Already handled separately in NB04/05. |
| 1c.16-17 Feature Summary | Feature engineering checklist, recommended temporal features | NO | NO | NO | Informational summary for user. |
| 1c.18 Save Patterns | `findings.metadata["temporal_patterns"]` with all analysis + `feature_flags` dict | YES (saved) | NO | NO | **Major gap.** `feature_flags` (include_recency, include_tenure, include_lifecycle_quadrant, include_trend_features, include_seasonality_features, include_cohort_features) are computed and saved but **FindingsParser does NOT read `temporal_patterns.feature_flags`**. These flags are only consumed by NB01d during exploration. Pipeline generation ignores them. |
| 1c.19 Snapshot Grid Vote | Dataset vote registered on grid | YES (grid) | NO | NO | Coordination mechanism. |

---

## NB01d: Event Aggregation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 1d.1 Load + Grid Check | Validates upstream completion, locks snapshot grid | YES | NO | NO | Boilerplate + validation gate. |
| 1d.2 Configure Aggregation | Windows (from 01a), divergent columns (from 01c), value columns, agg functions, lifecycle feature flags, target-proxy exclusion | YES | PARTIAL | PARTIAL | `suggested_aggregations` → pipeline's `aggregation.windows`. Divergent column prioritization NOT in pipeline (all numeric cols aggregated equally). **Gap**: Pipeline could generate priority comments or order columns by signal strength. |
| 1d.3 Preview Plan | Feature count preview | NO | NO | NO | Informational. |
| 1d.4 Execute Aggregation | 9-step aggregation: time windows, lifecycle quadrant, entity target, cyclical features, trend features, cohort features, momentum ratios, recency buckets | YES (aggregated df) | PARTIAL | PARTIAL | Pipeline generates basic time-window aggregation + lifecycle enrichment but **misses**: (1) Cyclical seasonality features from 01c, (2) Trend features (recent_vs_overall_ratio, entity_trend_slope), (3) Cohort features (cohort_year, cohort_quarter), (4) Custom recency buckets (uses hardcoded). **These exist only in exploration aggregation, not in generated pipeline.** |
| 1d.4b Temporal Features | Lag-based velocity/regularity features via TemporalFeatureEngineer | YES (merged to agg) | NO | NO | **Not in pipeline.** Temporal lag features are exploration-only. **Potential**: Generate velocity/regularity computation in bronze event template. |
| 1d.5 Quality Check | Null check, entity count validation, constant column detection, quadrant-target correlation | NO | NO | NO | Informational validation. |
| 1d.6 Save Aggregated Data | Delta table + auto-profiled aggregated findings YAML + aggregation metadata in original findings | YES (downstream NBs) | YES (aggregated findings read by FindingsParser) | YES | Critical. Aggregated findings become input to silver/gold generation. |
| 1d.X Leakage Validation | Target leakage check on aggregated features | NO | NO | NO | Informational validation. |

---

## NB02: Source Integrity

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 2.1 Setup | Boilerplate | - | - | - | |
| 2.2 Duplicate Analysis | Deduplication strategy (drop_exact, keep_first + conflict cols) → registry.bronze | YES (registry) | YES (bronze dedup) | YES | Drives `deduplicate` flag in bronze. |
| 2.3 Overall Quality Score | Quality score 0-100 | NO | NO | NO | Informational. Not consumed. |
| 2.4 Target Variable Analysis | Imbalance ratio, strategy (stratified/class_weights/SMOTE) → registry.bronze | YES (registry) | NO | NO | **Gap**: Imbalance strategy is stored in registry but generators don't produce class_weight logic in training. Training script uses hardcoded `class_weight="balanced"`. **Potential**: Generate explicit class weights or SMOTE step based on severity. |
| 2.5 Missing Value Analysis | Column null inventory | NO | NO | NO | Informational. Context for 2.6. |
| 2.6 Missing Value Patterns | MCAR/MAR/MNAR classification via correlation heatmap | NO | NO | NO | **Informational only.** Missingness mechanism detected but not used. **Potential**: MAR columns could get conditional imputation (impute using correlated column), MNAR columns could get indicator features (`{col}_is_missing`). Currently all nulls get the same strategy. After model training, missing-indicator features' importance would validate whether missingness carries signal. |
| 2.7 Segment-Aware Outlier | False outlier rates, segment_aware_cap recommendations → registry.bronze | YES (registry) | YES (SEGMENT_AWARE_CAP step) | YES | Drives bronze transformation. |
| 2.8 Global Outlier (IQR) | Log transform / winsorize recommendations → registry.bronze | YES (registry) | YES (CAP_OUTLIER/WINSORIZE) | YES | Drives bronze transformation with IQR bounds. |
| 2.9 Date Logic Validation | Date ranges, placeholder detection, sequence violations | NO | NO | NO | **Informational only.** Placeholder dates (pre-2005) detected but not filtered in pipeline. **Potential**: Generate a date validation filter in landing (drop rows with impossible dates). Sequence violations could generate a `has_sequence_violation` flag. |
| 2.10 Binary Field Validation | Invalid value detection in binary columns | NO | NO | NO | Informational. |
| 2.11 Data Consistency | Case variants detected → registry.bronze (normalize_lower) | YES (registry) | YES (consistency normalization) | YES | Generates `.str.lower()` in bronze. |
| 2.12 Quality Recommendations | Aggregated cleaning recommendations → registry.bronze | YES (registry) | YES (null/outlier steps) | YES | Consolidation of prior sections. |
| 2.13 Save Findings | Updated findings + registry YAML | YES | YES | YES | |

---

## NB03: Dataset Merge

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 3.1-3.2 Setup/Load | Boilerplate | - | - | - | |
| 3.3 Load Bronze Datasets | Granularity-aware dataset loading | YES | NO | NO | Exploration-only merge. Pipeline generates its own silver merge. |
| 3.4 Build Spine | Entity x grid date Cartesian product | YES (merge input) | YES (concept: silver merge) | YES | Pipeline silver does equivalent join logic. |
| 3.5 Merge | Equi-join / broadcast / as-of join based on granularity | YES (merged df) | YES (silver joins) | YES | `MergeReport.renamed_columns` flows to merged findings. Pipeline generates equivalent join code. |
| 3.6 Validation | Temporal integrity check, row preservation | NO | NO | NO | Informational. |
| 3.7 Save Merged | Silver merged Delta table + merged findings YAML | YES (downstream) | YES (merged findings read by generators) | YES | `renamed_columns` dict affects column name references in silver/gold. |
| 3.8-3.9 Preview/Summary | Informational | NO | NO | NO | |

---

## NB04: Column Deep Dive

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 4.1-4.2 Setup | Boilerplate | - | - | - | |
| 4.3 Value Range Validation | Bronze filtering rules (cap percentage, filter negatives) → registry.bronze | YES (registry) | YES (IMPUTE/CAP in bronze) | YES | Drives validation transforms. |
| 4.4-4.5 Numeric Distribution Analysis | Skewness/kurtosis → transformation recommendations (log/sqrt/cap) → registry.gold | YES (registry) | YES (GOLD transforms) | YES | Drives `apply_log_transform()`, `apply_sqrt_transform()` etc. |
| 4.6 Categorical Analysis | Cardinality, imbalance, entropy → encoding recommendations (one-hot/target/frequency) → registry.gold | YES (registry) | YES (GOLD encodings) | YES | Drives `apply_onehot_encoding()`, `apply_label_encoding()`. |
| 4.7 Datetime Analysis | Growth trends, seasonality, feature engineering (days_since_X, cyclical) → registry.silver/bronze | YES (registry) | PARTIAL | PARTIAL | Silver derived columns (days_since) are generated. But cyclical encoding (month_sin/cos) and modeling strategies (time-based split) from this section are NOT consumed. **Gap**: Temporal modeling strategies should inform training script split logic. |
| 4.8 Type Override | Manual type corrections → findings | YES (findings) | YES (corrected types used) | YES | |
| 4.9 Data Segmentation | Natural segments, target variance by segment, single vs multi-model recommendation | NO | NO | NO | **Informational only.** Segmentation result (quality score, segment count, target variance ratio) displayed but not persisted to registry. **Potential**: (1) `segment_id` could be added as a categorical feature. (2) If target variance ratio is high, generate separate training runs per segment. (3) After model training, segment-level error analysis would validate whether segmentation improves predictions. |
| 4.10 Save | Findings + registry YAML | YES | YES | YES | |

---

## NB04a: Text Columns Deep Dive (Entity-Level)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 4a.1-4a.2 Load | Boilerplate | - | - | - | |
| 4a.3 Configuration | Embedding model, variance threshold, component bounds | YES | NO | NO | **Not consumed by generators.** Same gap as NB01a_a. |
| 4a.4 Text Analysis | Text length stats | NO | NO | NO | Informational. |
| 4a.5 Process Text | Embeddings → PCA → PC features | YES (df updated) | NO | NO | **Major gap.** Entity-level text processing not replicated in pipeline. PC features exist only in exploration. |
| 4a.6 Visualize | PC distribution plots | NO | NO | NO | Informational. |
| 4a.7 Update Findings | `findings.text_processing[col]` metadata | YES (saved) | NO | NO | **Orphaned from generators.** FindingsParser ignores `text_processing`. |
| 4a.8 Recommendations | Production action summary (embed_reduce per column) | NO | NO | NO | Informational. |

---

## NB05: Relationship Analysis

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 5.1 Setup | Boilerplate | - | - | - | |
| 5.1b Leakage Exclusion | `excluded_leaking_features` list | YES (columns removed) | PARTIAL | PARTIAL | Excluded features are removed from findings.columns before downstream analysis. But generators don't explicitly check this list - they just don't see the columns. Works indirectly. |
| 5.2 Correlation Matrix | Pairwise correlation heatmap | NO | NO | NO | Informational. Visual multicollinearity detection. |
| 5.3 High Correlation Pairs | Pairs with |r| >= 0.7 | YES (5.9 input) | NO | NO | Informational table. Fed to recommender in 5.9. |
| 5.4 Feature Distributions by Target | Cohen's d effect sizes per feature | YES (5.9 input) | NO | NO | Informational. Effect sizes inform 5.9 weak/strong classification. |
| 5.5 Feature-Target Correlations | Correlation ranking | NO | NO | NO | Informational visualization. |
| 5.6 Categorical Feature Analysis | Cramer's V, high-risk categories, lift | NO | NO | NO | **Informational only.** High-risk categories identified but not used in pipeline. **Potential**: High-risk categories could generate binary indicator features (e.g., `is_high_risk_plan_type`). After model training, these indicators' importance would show if categorical risk segments add predictive value beyond the raw category. |
| 5.7 Scatter Plot Matrix | Pairwise scatter (top 4 features) | NO | NO | NO | Informational. |
| 5.8 Datetime Feature Analysis | Yearly/monthly/DOW retention trends, seasonality spread | NO | NO | NO | **Informational only.** Temporal retention patterns visualized but not encoded. **Potential**: If strong DOW effect found, generate `signup_day_of_week` as feature. Monthly patterns could generate `signup_month` feature. |
| 5.9 Recommendations | Multicollinear pairs → drop_multicollinear, strong predictors → prioritize, weak predictors → drop_weak → registry.gold | YES (registry) | YES (gold feature_selection) | YES | Drives feature selection in gold layer: drops weak/multicollinear, preserves strong. |
| 5.9.2 Stratification | High-risk segments needing representation in splits | NO | NO | NO | **Informational.** Not enforced in generated train/test split. **Potential**: Generate stratified split logic ensuring high-risk segments are proportionally represented. |
| 5.9.3 Model Selection | Linear vs tree-based recommendation | NO | NO | NO | **Informational.** Training script hardcodes 3 models (LR, RF, XGB) regardless. **Potential**: Could skip LR if multicollinearity is severe, or add regularization params. |
| 5.9.4 Feature Engineering | Ratio/interaction feature suggestions → registry.silver | YES (registry) | YES (silver derived_columns) | YES | Generates `create_ratio_features()`, `create_interaction_features()` in silver. |
| 5.9.5 Save | Findings + registry + merged findings/recommendations | YES | YES | YES | |

---

## NB06: Feature Opportunities

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 6.1 Setup | Boilerplate | - | - | - | |
| 6.2 Automated Feature Recs | Printed feature suggestions from engine | NO | NO | NO | Informational. |
| 6.3 Feature Capacity (EPV) | EPV score, capacity status (adequate/limited/inadequate), recommended feature count → registry + findings | YES (registry + metadata) | NO | NO | **Informational for pipeline.** EPV and capacity stored but generators don't enforce feature count limits. **Potential**: Gold layer could auto-trim features to EPV-recommended count by dropping lowest-importance ones. After model training, comparing actual feature count vs EPV recommendation validates whether overfitting risk was real. |
| 6.3.1 Model Complexity | Recommended model type (linear/regularized/tree), max features by type → registry | YES (registry) | NO | NO | **Not consumed by generators.** Training script always trains all 3 model types. **Potential**: If EPV is inadequate, could skip complex models or add stronger regularization. |
| 6.3.2 Segment Capacity | Per-segment EPV viability | YES (metadata) | NO | NO | Informational. |
| 6.3.3 Action Items | Feature budget summary | NO | NO | NO | Informational. |
| 6.3.4 Feature Availability | Features with tracking changes, coverage issues, remediation options | YES (metadata) | NO | NO | **Gap**: Available features with <100% coverage could get an `{col}_available` indicator in bronze. Currently just flagged for NB08 exclusion. |
| 6.4 Datetime Opportunities | Potential extractions (year, month, DOW, is_weekend, days_since) | NO | NO | NO | Informational. Already handled in NB01/04 datetime derivation. |
| 6.5 Business-Driven Features | tenure_days, days_since_last_activity, engagement_score, click_to_open_rate, service_adoption_score, value_frequency_product → registry.silver | YES (registry) | YES (silver derived_columns) | YES | Generates ratio/interaction/composite features in silver. |
| 6.6 Customer Segmentation Features | Value-frequency, recency, engagement segments | NO | NO | NO | **Informational only.** Segments created in exploration df but definitions not persisted. **Potential**: Generate segment assignment logic in silver/gold as categorical features. |
| 6.7 Numeric Transformations | Log/sqrt/standard scaling → registry.gold | YES (registry) | YES (gold transforms/scaling) | YES | May overlap with NB04 recommendations. Registry deduplicates. |
| 6.8 Categorical Encoding | One-hot/target/frequency encoding → registry.gold | YES (registry) | YES (gold encodings) | YES | May overlap with NB04 recommendations. |
| 6.9 Summary | Feature table display | NO | NO | NO | Informational. |

---

## NB07: Modeling Readiness

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 7.1 Setup | Boilerplate | - | - | - | |
| 7.2 Readiness Checklist | Pass/Fail gates (target, features, missing, quality, rows) | NO | NO | NO | Informational validation. |
| 7.3 Class Imbalance | Severity, ratio, sklearn class_weight config, mitigation strategy | NO | NO | NO | **Informational.** Duplicates NB02's analysis. Neither flows to generated training code. **Potential**: Generate explicit `class_weight={0: X, 1: Y}` in training script instead of hardcoded `"balanced"`. |
| 7.4 Leakage Risk | High-correlation flags, suspicious column names | NO | NO | NO | Informational warning. |
| 7.5 Feature Type Summary | Column count by type | NO | NO | NO | Informational inventory. |
| 7.6 Readiness Score | Score 0-100, status tier | NO | NO | NO | Informational. **Potential**: Could become an MLflow tag for experiment metadata. |
| 7.7 Feature Availability | Unavailable features flagged for exclusion | YES (NB08 exclusion) | NO | NO | Informational for NB08. Not in pipeline. |
| 7.X Leakage Validation | LeakageDetector checks (LD052, LD053, LD001) | YES (blocking gate) | NO | NO | Raises error if critical leakage. Does NOT write to findings. |

---

## NB08: Baseline Experiments

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 8.1 Setup | Boilerplate | - | - | - | |
| 8.2 Data Preparation | Feature selection (removes unavailable, non-feature types), imputation, scaling, train/test split | YES (modeling) | NO | NO | Exploration-only preparation. Pipeline generates its own feature prep in gold. **Gap**: Split strategy (temporal vs random) should match intent config but may not. |
| 8.3 Baseline Models | LR/RF/XGB training + evaluation (AUC, PR-AUC, F1) | YES (model artifacts) | NO | NO | Exploration-only models. Pipeline re-trains in `ml_experiment.py`. **Note**: These baseline results provide a floor for comparison with pipeline-trained models. |
| 8.4 Feature Importance | Top 15 features from RF | NO | NO | NO | **Informational.** This is the first place where feature importance from an actual model is available. **Key potential**: (1) Cross-reference with NB01a stability - are top features temporally stable? (2) Cross-reference with NB01c patterns - do trend/seasonality features appear? (3) Cross-reference with NB05 effect sizes - do statistical measures agree with model importance? (4) Features NOT important could be candidates for removal to reduce pipeline complexity. |
| 8.5-8.6 Reports/Grid | Classification reports, confusion matrices, ROC/PR curves | NO | NO | NO | Informational model comparison. |
| 8.7 Key Takeaways | Best model, top features, performance assessment | NO | NO | NO | Informational summary. |

---

## NB09: Business Alignment

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 9.1 Setup | Boilerplate | - | - | - | |
| 9.2 Business Context | project_name, objective, stakeholders, timeline | YES (findings) | NO | NO | **Informational metadata.** Not consumed by generators. **Potential**: Could populate pipeline README or exploration report comments. |
| 9.3 Success Metrics | AUC >= 0.80, Precision >= 0.60, Churn Reduction 20%, Latency < 100ms | YES (findings) | NO | NO | **Informational.** Not used as training thresholds. **Potential**: Generate early stopping or threshold validation in training script. |
| 9.4 Deployment Requirements | Batch/real-time, frequency, latency, registry, retraining | YES (findings) | NO | NO | **Informational.** Not used in pipeline generation. **Potential**: Generate workflow.json scheduling based on frequency. Generate Databricks job with retraining cadence. |
| 9.5 Data Constraints | PII, freshness, historical depth, protected attributes | YES (findings) | NO | NO | **Informational.** **Potential**: PII columns could be auto-excluded from feature engineering. Protected attributes could generate fairness validation in scoring. |
| 9.6 Intervention Strategy | Risk tiers, costs, effectiveness rates | YES (findings) | NO | NO | **Informational.** **Potential**: Generate a scoring output enrichment step that adds intervention tier recommendations based on prediction probability thresholds. |
| 9.7 Save to Findings | All above → findings.metadata | YES (saved) | NO | NO | Persisted but generators don't read business context. |

---

## NB10: Pipeline Generation (Spec Generation)

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 10.1 Configuration | Generation target (LOCAL/DATABRICKS), OUTPUT_FORMAT, catalog/schema | YES | YES | YES | Core generation config. |
| 10.2 Load Findings/Recs | Loads all findings + recommendations YAMLs | YES | YES | YES | Inputs for generation. |
| 10.3 Review Recommendations | Grouped display of bronze/silver/gold recommendations | NO | NO | NO | Informational review. |
| 10.4A Generate Local Pipeline | `PipelineGenerator.generate_all()` → landing, bronze, silver, gold, training, runner, feast, validation, docs | YES | **YES** | N/A | Core local generation. |
| 10.4B Generate Databricks Pipeline | `DatabricksPipelineGenerator.generate_all()` → config, bronze, silver, gold, training, runner | YES | N/A | **YES** | Core Databricks generation. |
| 10.4C Generate LLM Docs | Markdown documentation for AI-assisted development | YES | PARTIAL | NO | Documentation only. |
| 10.5 Convert to Notebooks | .py → .ipynb conversion | YES | OPTIONAL | N/A | Format choice. |
| 10.6 Run Pipeline | Execute Bronze → Silver → Gold → Training | YES | YES (if RUN_PIPELINE) | YES | Optional execution. |
| 10.7 Summary | File tree display | NO | NO | NO | Informational. |
| 10.8 Recommendations Hash | Unique hash from gold config → MLflow tag | YES | YES | YES | Versioning/reproducibility. |
| 10.9 Feast Validation | Feature store registry inspection | NO | NO | NO | Post-generation validation. |

---

## NB11: Scoring Validation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 11.1 Run Scoring | Holdout predictions → predictions.parquet | YES | YES (scoring output) | YES | Validates pipeline end-to-end. |
| 11.2 Summary Metrics | Confusion matrix, ROC curve, probability histogram | NO | NO | NO | Informational. |
| 11.3 Model Comparison | 3-model comparison on holdout | NO | NO | NO | Informational. |
| 11.4 Adversarial Validation | Feature drift between train/score | NO | NO | NO | Validation. **Potential**: Generate automated drift monitoring in scoring pipeline. |
| 11.5 Transformation Validation | Encoding/scaling consistency check | NO | NO | NO | Validation. |
| 11.6 SHAP Explanations | Top 20 features by mean |SHAP|, per-entity waterfall | NO | NO | NO | **Key informational output.** SHAP importance is the definitive feature ranking from the trained model. **Potential**: (1) Cross-reference with NB01a stability - are top SHAP features temporally stable? (2) If top SHAP features weren't flagged as "strong predictors" in NB05, the relationship analysis may have missed something. (3) Low-SHAP features could feed back into gold layer feature_selection for pipeline v2. |
| 11.7 Customer Browser | Per-entity prediction + SHAP lookup | NO | NO | NO | Diagnostic utility. |
| 11.8 Error Analysis | FP/FN examples with SHAP waterfalls | NO | NO | NO | Informational. **Potential**: Systematic FP/FN analysis could identify feature gaps (e.g., all FNs have certain categorical value → need new feature). |
| 11.9 Export Results | feature_importance.csv, predictions_with_shap.parquet, Delta table | YES | YES (artifacts) | YES (Delta) | |

---

## NB12: View Documentation

| Section | Rec. Made | Applied in NB | Local Pipeline | DBX Pipeline | Note |
|---------|-----------|---------------|----------------|--------------|------|
| 12.1-12.3 | HTML export inventory and display | NO | NO | NO | Purely documentation viewing. No pipeline impact. |

---

# Summary: Major Pipeline Parity Gaps

## HIGH PRIORITY (features computed in exploration but missing from generated pipeline)

| Gap | Exploration Source | What's Missing in Pipeline | Impact |
|-----|-------------------|---------------------------|--------|
| **Text embeddings** | NB01a_a, NB04a | Text columns are dropped. No embedding/PCA in generated code. `findings.text_processing` is orphaned. | Text features unavailable in production. |
| **Temporal pattern features** | NB01c feature_flags | `include_trend_features`, `include_seasonality_features`, `include_cohort_features` flags ignored by FindingsParser. Trend (recent_vs_overall_ratio), cohort (cohort_year), and extended seasonality (month/quarter cyclical) features not generated. | Feature gap between exploration model and production pipeline. |
| **Lag features** | NB01c lag analysis | Autocorrelation-based lag features (`{col}_lag_{N}d`) computed in exploration but not generated in pipeline. | Temporal memory features missing. |
| **Velocity features** | NB01c velocity, NB01d temporal features | Velocity (rate of change) features computed in 01d exploration but not replicated in pipeline bronze event template. | Rate-of-change signal lost. |
| **Recency bucket boundaries** | NB01c recency analysis | Custom bucket boundaries (from effect size analysis) not used; pipeline uses hardcoded buckets. | Suboptimal discretization. |

## MEDIUM PRIORITY (recommendations made but not consumed by generators)

| Gap | Exploration Source | What's Missing | Impact |
|-----|-------------------|----------------|--------|
| **Imbalance strategy** | NB02, NB07 | Class weight/SMOTE strategy in registry not consumed; training uses `class_weight="balanced"`. | Suboptimal handling of severe imbalance. |
| **Model type recommendation** | NB06 | EPV-based model recommendation ignored; all 3 models always trained. | Potential overfitting with inadequate EPV. |
| **Temporal split strategy** | NB00 intent, NB04 | `SPLIT_STRATEGY` and `recommended_training_start` not enforced in training script. | Train/test contamination risk. |
| **Purge gap** | NB00 intent | `PURGE_GAP_DAYS` defined but not enforced between train cutoff and label start. | Potential leakage. |
| **Future date filtering** | NB01b TQ003 | Future dates detected but no `WHERE timestamp <= reference_date` in pipeline. | Data leakage from future rows. |
| **Success metrics** | NB09 | Target AUC/precision thresholds not used in training as early stopping or validation gates. | No automated quality gate. |
| **Deployment frequency** | NB09 | Retraining cadence not used in workflow.json scheduling. | Manual scheduling required. |

## LOW PRIORITY (informational analyses with untapped potential)

| Gap | Source | Potential Value | When Relevant |
|-----|--------|-----------------|---------------|
| **Stability per feature** | NB01 stability | Top model features should be temporally stable; could flag risky features | After model training (NB08/11) |
| **Dataset type routing** | NB01 structure detection | Snapshot vs longitudinal vs time_series could drive different aggregation | Multi-pattern datasets |
| **Aggregation guidance per column** | NB01a CV analysis | All-time-only vs short-window-only per column would reduce feature explosion | Large feature counts (>100) |
| **Segmentation features** | NB04, NB06 | Detected segments could become categorical features | High target variance by segment |
| **Missing value indicators** | NB02 MNAR detection | `{col}_is_missing` features for MNAR columns | After model shows missingness matters |
| **High-risk category indicators** | NB05 categorical | Binary `is_high_risk_{category}` features | After model importance confirms |
| **Objective-specific config** | NB00 objectives | Multi-objective label logic, objective-aware feature weighting | Multi-objective use cases |
| **Intervention tier enrichment** | NB09 business | Scoring output with recommended action tier | Production deployment |
| **Feature importance feedback** | NB08/11 SHAP | Low-importance features dropped in pipeline v2 | Iterative improvement |
