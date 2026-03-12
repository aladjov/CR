Customer Retention Pipeline — Final Mental Model

Notebook 0 — Intent Contract (One Intent per Run)

Before any exploration begins, Notebook 0 defines the intent of the run.
Each run targets a single modeling goal, and all downstream notebooks become aware of this intent.

Intent includes:

objective:

 - immediate risk
 - renewal risk
 - disengagement


Temporal Posture:

 - reactive
 - stable

prediction horizon(s)

 - anchor preference (calendar, contract-end, inactivity)
 - evaluation defaults (temporal split, purge gap)

This ensures:

event datasets generate only relevant temporal evidence in 01a–01c
entry datasets know which as-of grid they must inherit
aggregation (01d) produces only the snapshots required for that goal
downstream analysis (02, 04) focuses on the correct default snapshot
modeling setup (07–08) uses the correct training grid and split logic


The followup  workflow is built around one core principle:
All event datasets independently produce temporal evidence (01a–01c), then a global, objective-specific as-of grid is agreed. Everything downstream uses that grid deterministically.
The pipeline then progresses in three major phases:
1. Bronze — Evidence & aggregation
2. Silver — Merge & feature discovery
3. Gold — Modeling & productionization
All steps after Bronze operate per objective and optionally per model type (reactive vs stable).

Phase I — Bronze: Evidence → Grid → Aggregation
01a–01c — Temporal Exploration (per event dataset)
Each event dataset is explored independently and produces:
* window feasibility evidence
* density / coverage signals
* recency / velocity / momentum dynamics
* seasonality and trend indicators
* cadence feasibility per objective
* stability vs burstiness signals
Each dataset then casts a vote:
* optimal cadence per objective:
    * immediate risk
    * disengagement
    * renewal risk
* whether reactive vs stable modeling is supported
* anchor feasibility:
    * calendar cadence
    * contract boundary / renewal anchors
These are votes, not decisions.

Global Step — Consensus Grid Derivation
After all datasets complete 01c:
* votes are aggregated
* a common as-of grid per objective is derived
* optionally differentiated for:
    * reactive models (dense, recent)
    * stable models (coarser, longer history)
Output:
* global_asof_grid_per_objective
* global_asof_grid_per_model_type
This grid becomes the deterministic backbone of aggregation.

01d — Aggregation (per dataset)
Each dataset is aggregated using the consensus grid:
Output:
* entity × as_of_date snapshots
* objective-aware features
* recency / velocity / momentum derived
* lifecycle / cohort / trend features
* aggregation metadata
This is the final Bronze representation of each dataset.

02 — Source Integrity Cleanup (per dataset, post-aggregation)
Before merging datasets:
Run a single notebook running the same analysis for each dataset that performs:
* duplicate detection
* extreme missingness checks
* date logic validation
* binary/string consistency checks
* leakage smell checks
* column removal decisions (source-local)
This step exists to:
* avoid provenance complexity later
* drop unusable columns early
* ensure clean source inputs before merge
After this, Bronze is complete.

Phase II — Silver:
03 Silver: Merge & Analytical Exploration
All Bronze outputs are merged using:

(entity_id, as_of_date)
Result:
* unified feature matrix
* aligned across datasets
* objective-specific default snapshot selected for exploration

04 — Column Deep Dive (merged, per objective snapshot)
Run on merged dataset:
* type validation
* value range validation
* skewness / kurtosis
* zero-inflation
* transformation candidates
* encoding hints
* column usability
Origin of column no longer matters — only usability.

05 — Relationship Analysis (merged)
Now that all columns are visible:
* feature-feature correlations
* redundancy detection
* multicollinearity pruning
* feature-target relationships
* categorical-target association
* interaction opportunities
This produces structural understanding of the final feature space.

Phase III — Gold:
06 — Feature Opportunities
Consolidate:
* transformations
* derived features
* segmentation features
* encoding strategies
* dimensionality considerations
Produces the candidate feature set.

07 — Modeling Readiness & Training Setup
Per objective and per model type:
* select training snapshot set from grid:
    * reactive → recent, dense
    * stable → longer, coarser
* define training cadence
* define label horizon
* define entity-grouped train/test split
* define temporal purge gap
* finalize modeling dataset
This stage formalizes:

training_grid
sampling_policy
split_policy

08 — Model Training & Validation
Baseline architectures trained:
* Logistic Regression
* XGBoost
* Random Forest
Validation includes:
* entity holdout
* temporal holdout (purged)
* performance comparison
* feature importance stability
* drift sensitivity
Model candidates selected.

09 — Business Alignment (per objective & model type)
Translate model into operational decision logic:
* reactive vs stable use cases
* intervention timing
* explainability needs
* risk thresholds
* cost/benefit tradeoffs
* SLA considerations
Outputs:
* business-approved model configuration
* deployment readiness signals

10 — Production Pipeline Generation
Generate deterministic pipeline that replicates:
* grid derivation
* aggregation logic
* feature engineering
* transformations
* encoding
* scoring logic
Production pipeline mirrors exploration exactly.

11 — Scoring / Inference
Operational stage:
* periodic scoring on production cadence
* reactive models: frequent updates
* stable models: slower cadence
* monitoring:
    * drift
    * calibration
    * feature stability
    * performance decay

Key Architectural Decisions (Locked)
1) Grid is global; training is objective-specific
Aggregation uses:
* consensus grid
Training uses:
* objective-specific subset of that grid

2) Exploration uses a default snapshot; training uses multiple snapshots
* 02/04 run on one representative snapshot
* training uses full objective slice

3) Source cleanup happens before merge
03 runs per dataset post-aggregation to:
* avoid provenance complexity
* eliminate broken columns early

4) Relationship analysis happens only after merge
Because interactions and redundancy only exist in merged space.

5) Reactive vs Stable is a sampling policy, not a separate pipeline
Same grid, different selection:
* dense recent vs broad historical

6) Train/test split is entity-aware + temporally purged
Never row-random.

Final Flow (Compressed)

Event datasets
   ↓
01a–01c (per dataset)
   ↓
Consensus grid per objective & model type
   ↓
01d Aggregation (per dataset)
   ↓
02 Source cleanup (per dataset)
   ↓
03 Merge
   ↓
04 Column deep dive (merged)
05 Relationship analysis (merged)
   ↓
06 Feature opportunities
   ↓
07 Modeling readiness & split setup
   ↓
08 Train & validate models
   ↓
09 Business alignment
   ↓
10 Production pipeline generation
   ↓
11 Scoring / inference

This is now a coherent, progressive system where:
* each notebook produces something that is not reevaluated later
* temporal evidence drives grid design
* aggregation happens once
* cleanup happens before merge
* modeling happens per objective
* production faithfully replicates exploration
