# SPS Production Configuration Analysis

## 1. Entity Key: ACCOUNT_ID

`ACCOUNT_ID` from the `account` table is the universal entity key. Every table must ultimately resolve to it. Tables fall into three categories based on how they reach ACCOUNT_ID:

### Direct (1 hop) — ACCOUNT_ID present in table

| Table | Join Column | Relationship |
|---|---|---|
| case | ACCOUNT_ID | many_to_one |
| contract | ACCOUNT_ID | many_to_one |
| implementation_project | ACCOUNT_ID | many_to_one |
| opportunity | ACCOUNT_ID | many_to_one |
| request | ACCOUNT_ID | many_to_one |

### Indirect (2 hops) — needs a bridge table

| Table | Bridge Table | Step 1 | Step 2 |
|---|---|---|---|
| case_history | case | case_history.CASE_ID → case.CASE_ID | case.ACCOUNT_ID → account.ACCOUNT_ID |
| opportunity_product | opportunity | opp_product.OPPORTUNITY_ID → opportunity.OPPORTUNITY_ID | opportunity.ACCOUNT_ID → account.ACCOUNT_ID |
| subscription | contract | subscription.CONTRACT_ID → contract.CONTRACT_ID | contract.ACCOUNT_ID → account.ACCOUNT_ID |

### Indirect (3+ hops) — needs external bridge

| Table | Join Path |
|---|---|
| customer_visible_transaction_volume_daily | txn.senderOrgId → sps_identity.account.ACCOUNT_ID → (SALESFORCE_ID) → salesforce.account.ACCOUNT_ID |

The transaction volume table requires a separate identity-mapping table (`sps_identity.account`) not currently in the dataset list. This should either be pre-joined in a view/query upstream, or we add `sps_identity.account` as an explicit bridge dataset.

### Gap in Current Framework

`MergeScaffoldEntry` supports only direct A→B joins. For multi-hop joins we have two options:

**Option A — Pre-materialized bridge columns (recommended):** During landing/bronze, run a simple lookup query against the bridge table and add `ACCOUNT_ID` as a column to the child table before it enters the pipeline. This is essentially what the Databricks SQL snippet in `table_descriptions.md` already does for the transaction table. The pipeline then sees every table with a direct `ACCOUNT_ID` column.

**Option B — Chain of MergeScaffoldEntries:** Register intermediate tables (case, contract, opportunity) as bridges and define sequential join steps. This is more complex, fragile, and wasteful (bringing full bridge tables into the merge just for a key lookup).

**Recommendation:** Option A. For each indirect table, define a `bridge_query` or `key_resolution` step in notebook 00 that pre-joins the ID before the data enters the pipeline. This keeps the rest of the framework (bronze/silver/gold) simple with direct ACCOUNT_ID joins.


## 2. Target Column: Deriving Churn

There is no single binary "churned" column. The target must be derived. Below are the candidate signals, ordered by reliability:

### Signal 1: Case with record_type = "cancellation" (strongest)

From the `case` table description:
- `RECORD_TYPE_NAME = 'cancellation'` → post-cancellation submission
- `RECORD_TYPE_NAME = 'retention'` → pre-cancellation (at-risk but not yet churned)

A cancellation case with `IS_CLOSED = 1` is a strong churn signal. `CLOSED_DATE` provides the churn timestamp.

### Signal 2: Contract non-renewal / termination

- `CONTRACT_STATUS` indicating terminated/expired
- `BILLING_TERMINATION_DATE` set and in the past
- `CONTRACT_END_DATE` passed without a newer contract for the same ACCOUNT_ID
- `NEXT_RENEWAL_DATE` passed with no new contract

### Signal 3: Subscription termination

- `SUBSCRIPTION_STATUS` = cancelled/terminated
- `IS_ACTIVE = 0`
- `TERMINATED_DATE` set

### Signal 4: Request with cancellation

- `CANCELLAION_SUBMITTED` (note: typo in source, the column is misspelled) = some truthy value
- `REQUEST_TYPE` or `REQUEST_STATUS` indicating cancellation

### Recommended Target Derivation Strategy

Build a **derived binary target per account** using a priority waterfall:

```
churned = 1 IF any of:
  (a) account has a cancellation case (RECORD_TYPE_NAME = 'cancellation', IS_CLOSED = 1)
  (b) account has ALL contracts terminated/expired (no active contract remains)
  (c) account has ALL subscriptions inactive (IS_ACTIVE = 0)

churned = 0 IF:
  Account has at least one active contract AND no cancellation case
```

**Churn timestamp** = earliest of:
- cancellation case CLOSED_DATE
- last contract BILLING_TERMINATION_DATE
- last subscription TERMINATED_DATE

This gives us both the label and the label timestamp needed for temporal splitting.

### Target Leakage Warnings

These columns MUST be excluded from features (they are contemporaneous with or post-churn):
- case.CANCELLATION_REASON, CANCELLATION_REASON_CATEGORY, CANCEL_COMMENTS_DSAT, CANCEL_COMMENTS_GENERAL
- case.OUTCOME (if it encodes the churn result)
- case.FUTURE_SERVICE_MANAGEMENT (post-churn decision)
- request.CANCELLAION_SUBMITTED, REQUEST_STATUS (if it encodes final resolution)
- contract.CONTRACT_STATUS (if "Cancelled" is in there after the fact)
- subscription.IS_ACTIVE, TERMINATED_DATE (these ARE the target)
- account.EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR (retrospective label)


## 3. Table Classification: Entity vs Event

| Table | Granularity | Reasoning |
|---|---|---|
| account | **entity_level** | One row per account; describes the entity |
| case | **event_level** | Multiple cases per account over time |
| case_history | **event_level** | Multiple field-change events per case |
| contract | **event_level** | Multiple contracts per account (see distribution: up to 21 active per account) |
| implementation_project | **event_level** | Multiple projects per account |
| opportunity | **event_level** | Multiple opportunities per account |
| opportunity_product | **event_level** | Multiple products per opportunity |
| request | **event_level** | Multiple requests per account |
| subscription | **event_level** | Multiple subscriptions per contract/account |
| transaction_volume_daily | **event_level** | Daily volume aggregates per sender/receiver pair |


## 4. Datetime Columns & Feature Timestamps

### Per-Table Event Timestamp Selection

For each event table, we need a primary `feature_timestamp` (the "when did this event happen" column):

| Table | Primary Timestamp | Reasoning |
|---|---|---|
| case | **CREATED_DATE** | When the case was opened — this is the event occurrence |
| case_history | **CREATED_DATE** | When the field change happened |
| contract | **CONTRACT_START_DATE** | When the contract began; ACTIVATED_DATE as fallback |
| implementation_project | **CREATED_DATE** | When the project record was created |
| opportunity | **CREATED_DATE** | When the opportunity was opened |
| opportunity_product | **CREATED_DATE** | When the line item was created |
| request | **CREATED_DATE** | When the request was filed |
| subscription | **SUBSCRIPTION_START_DATE** | When the subscription period began |
| transaction_volume_daily | **startDay** | The date of the transactions |

### Entity Table Timestamp

| Table | Update Timestamp | Created Timestamp |
|---|---|---|
| account | **LAST_MODIFIED_DATE** | CREATED_DATE |

### ETL Timestamps (exclude from features)

Every Salesforce table has `ETL_CREATED_TIMESTAMP` and `ETL_UPDATED_TIMESTAMP`. These reflect data pipeline timing, NOT business events. They should be excluded from features but can be used for data-quality checks.


## 5. Datetime-Derived Features

These are the high-value duration/elapsed-time features derivable from timestamp pairs within each table.

### case

| Feature Name | Formula | Unit | Business Meaning |
|---|---|---|---|
| case_resolution_time | CLOSED_DATE - CREATED_DATE | days | How long to close the case |
| case_first_response_time | RESPONSE_DATE_TIME - CREATED_DATE | hours | Time to first response (SLA) |
| case_first_assignment_time | FIRST_ASSIGNED_DATE_TIME - CREATED_DATE | hours | Time to assign |
| case_first_close_time | FIRST_CLOSED_DATE_TIME - CREATED_DATE | days | Time to first closure |
| case_reopen_delay | REOPENED_DATE_TIME - FIRST_CLOSED_DATE_TIME | days | How long until reopened |
| case_survey_delay | CSAT_SURVEY_SENT_DATE - CLOSED_DATE | days | Delay before sending survey |

### contract

| Feature Name | Formula | Unit | Business Meaning |
|---|---|---|---|
| contract_duration | CONTRACT_END_DATE - CONTRACT_START_DATE | days | Total contract length |
| contract_remaining | CONTRACT_END_DATE - as_of_date | days | Time until expiry (at prediction time) |
| contract_time_to_next_renewal | NEXT_RENEWAL_DATE - as_of_date | days | Urgency of renewal |
| contract_activation_delay | ACTIVATED_DATE - CONTRACT_START_DATE | days | Delay between signing and activation |
| billing_termination_lead | BILLING_TERMINATION_DATE - as_of_date | days | Time until billing stops |

### implementation_project

| Feature Name | Formula | Unit | Business Meaning |
|---|---|---|---|
| project_duration | PRODUCTION_DATE - PROJECT_START_DATE | days | Total implementation time |
| project_delay_vs_target | PRODUCTION_DATE - TARGETED_IMPLEMENTATION_DATE | days | Positive = late delivery |
| project_delay_vs_initial_target | PRODUCTION_DATE - INITIAL_TARGETED_IMPLEMENTATION_DATE | days | Scope creep indicator |
| project_setup_time | SETUP_COMPLETE_DATE - PROJECT_START_DATE | days | Setup phase duration |
| project_on_hold_duration | (if ON_HOLD_DATE set) as_of_date or resume - ON_HOLD_DATE | days | Stalled project signal |

### opportunity

| Feature Name | Formula | Unit | Business Meaning |
|---|---|---|---|
| opportunity_cycle_time | SALE_DATE - CREATED_DATE | days | Sales cycle length |
| opportunity_submission_delay | SUBMITTED_DATE - CREATED_DATE | days | Time to submit |
| opportunity_close_delay | ESTIMATED_CLOSE_DATE - CREATED_DATE | days | Expected deal duration |
| escalation_resolution_time | ESCALATION_RESOLUTION_DATE - ESCALATION_DATE | days | Escalation handling speed |

### subscription

| Feature Name | Formula | Unit | Business Meaning |
|---|---|---|---|
| subscription_duration | SUBSCRIPTION_END_DATE - SUBSCRIPTION_START_DATE | days | Subscription period length |
| subscription_remaining | SUBSCRIPTION_END_DATE - as_of_date | days | Time until sub ends |
| subscription_tenure | as_of_date - SUBSCRIPTION_START_DATE | days | How long customer has been subscribed |
| time_since_termination | as_of_date - TERMINATED_DATE | days | Recency of termination |

### Cross-table (account-level)

| Feature Name | Formula | Business Meaning |
|---|---|---|
| customer_tenure | as_of_date - account.FIRST_SUBSCRIPTION_DATE | Overall customer lifetime |
| time_since_net_new | as_of_date - account.NET_NEW_CUSTOMER_DATE | Maturity since becoming a customer |
| days_since_last_case | as_of_date - max(case.CREATED_DATE) per account | Recency of support contact |
| days_since_last_opportunity | as_of_date - max(opportunity.CREATED_DATE) per account | Recency of sales activity |


## 6. Aggregation Windows & Feature Types

For each event table, after joining ACCOUNT_ID and establishing the event timestamp, the system will aggregate per standard windows (24h, 7d, 30d, 90d, 180d, 365d, all_time). Key aggregation feature families:

### case (per account, windowed)
- **Counts**: total cases, cases by type (retention/cancellation/optimization/ops), cases by severity
- **Rates**: closure rate, reopen rate, escalation rate
- **Durations (avg/median/max)**: resolution time, first response time
- **Recency**: days since last case, days since last cancellation case
- **Flags**: has_open_case, has_cancellation_case, has_retention_case

### case_history (per account, windowed, via case bridge)
- **Counts**: total field changes, changes by field name (e.g., status changes, priority changes)
- **Velocity**: field changes per case, status transitions per case
- **Patterns**: frequent old→new value transitions (status escalation patterns)

### contract (per account, windowed)
- **Counts**: active contracts, expired contracts, total contracts
- **Financials**: total document allotment, contract term distribution
- **Timing**: avg contract duration, shortest remaining contract, nearest renewal
- **Flags**: has_expiring_contract_30d, all_contracts_expired

### implementation_project (per account, windowed)
- **Counts**: total projects, completed projects, on-hold projects, cancelled projects
- **Durations**: avg project duration, avg delay vs target
- **Rates**: completion rate, on-time delivery rate
- **Flags**: has_stalled_project, has_cancelled_project

### opportunity (per account, windowed)
- **Counts**: total opportunities, won, lost, open
- **Financials**: total bookings MRR, avg MRR per opp, total one-time fees
- **Rates**: win rate, competitive kill rate, first subscription sale rate
- **Timing**: avg sales cycle length, escalation rate
- **Revenue trends**: MRR change (current opp vs prior), lift values

### opportunity_product (per account, windowed, via opportunity bridge)
- **Counts**: distinct products, product diversity
- **Financials**: total MRR across products, avg unit price, total quantity
- **Product mix**: breakdown by product type (requires PRODUCT_ID mapping)

### request (per account, windowed)
- **Counts**: total requests, by type, by status
- **Financials**: total credit requested, total rate reduction, current vs proposed MRR delta
- **Flags**: has_cancellation_request, has_credit_request
- **Recency**: days since last request

### subscription (per account, windowed, via contract bridge)
- **Counts**: active subscriptions, terminated subscriptions, total
- **Financials**: total net price, avg net price, total documents
- **Timing**: avg subscription duration, nearest expiring subscription
- **Rates**: termination rate, active/total ratio
- **Flags**: has_terminating_subscription_30d

### transaction_volume (per account, windowed, via identity bridge)
- **Volume**: total transactions, avg daily volume, peak daily volume
- **Errors**: total errors, error rate, error trend
- **Activity**: active trading days, doc type diversity
- **Trends**: volume trend (recent vs prior window)


## 7. Notebook 00 Configuration Plan

### Notebook structure

| Section | Cell IDs | Purpose |
|---|---|---|
| 0.1 Project Metadata | `846f56cb` (config), `a355da3c` (code) | PROJECT_NAME, LIGHT_RUN, MAX_GRID_DATES |
| 0.2 Dataset Registration | `f1eb641c` (config), `cdc69ccc` (code) | datasets dict, path resolution |
| 0.3 Auto Fingerprinting | `43e89806` | DatasetFingerprinter |
| 0.4 Confirm Semantics | `ca9650f3` | entity/time columns, granularity |
| 0.5 Target Dataset Selection | `1446e49e` (config), `637064ae` (code) | TARGET_DATASET, TARGET_COLUMN, ENTITY_COLUMN |
| 0.6 Prediction Objective Detection | `f13e88da` | PredictionObjectiveDetector |
| 0.7 Objective Priority Review | `85a730b7` | Objective priority overrides |
| 0.8 Join Scaffold | `2a900807` (config), `7f4b562b` (code) | MANUAL_SCAFFOLD, EXCLUDE_DATASETS |
| 0.8.1 Key Resolution | `7b1ec50d` (config), `0866aa83` (code) | KEY_RESOLUTION for multi-hop joins |
| 0.8.2 Dataset Registry | `915bcef8` | Build DatasetRegistryEntry per dataset |
| 0.9 Temporal Posture | `09a0854f` (config), `04cf48a9` (code) | TEMPORAL_POSTURE |
| 0.10 Intent Configuration | `ea85abf2` (config), `a7f7bf52` (code) | Prediction horizons, windows, cadence |
| 0.11 Save Project Context | `c6211778` | Build & save ProjectContext |
| 0.12 Exploration Sampling | `e4b7c9a1` (config), `f5c8d6b2` (code) | SAMPLE_ENTITY_COUNT, SAMPLE_STRATIFY_COLUMNS |
| 0.13 Initialize Snapshot Grid | `2ef9c4c6` (config), `6b061a37` (code) | SnapshotGrid from intent |

### Project settings (0.1)

```python
# @cr:config name='project_settings' id=846f56cb
PROJECT_NAME = "sps_production"
LIGHT_RUN = False
MAX_GRID_DATES = None       # e.g. 10 to cap the snapshot grid to 10 dates
```

Note: `SAMPLE_FRACTION` and `SAMPLE_ENTITY_COUNT` have been moved out of project settings into the dedicated sampling section (0.12).

### datasets dictionary

```python
datasets = {
    # Entity (target bearer)
    "account": "prod_corp_snowflake_provisioning_shared.salesforce.account",

    # Direct-join event tables
    "case": "prod_corp_snowflake_provisioning_shared.salesforce.case",
    "contract": "prod_corp_snowflake_provisioning_shared.salesforce.contract",
    "implementation_project": "prod_corp_snowflake_provisioning_shared.salesforce.implementation_project",
    "opportunity": "prod_corp_snowflake_provisioning_shared.salesforce.opportunity",
    "request": "prod_corp_snowflake_provisioning_shared.salesforce.request",

    # Indirect-join event tables (need key resolution)
    "case_history": "prod_corp_snowflake_provisioning_shared.salesforce.case_history",
    "opportunity_product": "prod_corp_snowflake_provisioning_shared.salesforce.opportunity_product",
    "subscription": "prod_corp_snowflake_provisioning_shared.salesforce.subscription",

    # Transaction volume (needs external bridge — defer to Phase 2)
    # "transaction_volume_daily": "prod_networkdata.reporting_gold.customer_visible_transaction_volume_daily",
}
```

### Semantics overrides

```python
semantics["account"]["entity_column"] = "ACCOUNT_ID"
semantics["account"]["granularity"] = "entity_level"
semantics["account"]["raw_time_column_role"] = RawTimeColumnRole.ENTITY_UPDATE_TIME
semantics["account"]["time_column"] = "LAST_MODIFIED_DATE"

# Direct-join events
for name in ["case", "contract", "implementation_project", "opportunity", "request"]:
    semantics[name]["entity_column"] = "ACCOUNT_ID"

# Indirect events — entity_column is their local PK, resolved to ACCOUNT_ID in bridge step
semantics["case_history"]["entity_column"] = "CASE_ID"           # bridge: case
semantics["opportunity_product"]["entity_column"] = "OPPORTUNITY_ID"  # bridge: opportunity
semantics["subscription"]["entity_column"] = "CONTRACT_ID"       # bridge: contract

# Event timestamps
semantics["case"]["time_column"] = "CREATED_DATE"
semantics["case_history"]["time_column"] = "CREATED_DATE"
semantics["contract"]["time_column"] = "CONTRACT_START_DATE"
semantics["implementation_project"]["time_column"] = "CREATED_DATE"
semantics["opportunity"]["time_column"] = "CREATED_DATE"
semantics["opportunity_product"]["time_column"] = "CREATED_DATE"
semantics["request"]["time_column"] = "CREATED_DATE"
semantics["subscription"]["time_column"] = "SUBSCRIPTION_START_DATE"
```

### Target configuration

```python
TARGET_DATASET = "account"       # target lives at entity level
TARGET_COLUMN  = "churned"       # derived column (see Section 2)
ENTITY_COLUMN  = "ACCOUNT_ID"
```

The `churned` column does not exist yet — it must be derived. This should happen as a pre-processing step in notebook 00 after loading the data, before fingerprinting. See Section 8 below.

### Merge scaffold (direct joins only)

```python
MANUAL_SCAFFOLD = [
    MergeScaffoldEntry(left_dataset="account", right_dataset="case",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="contract",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="implementation_project",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="opportunity",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="request",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    # After key resolution, these also join directly on ACCOUNT_ID:
    MergeScaffoldEntry(left_dataset="account", right_dataset="case_history",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="opportunity_product",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(left_dataset="account", right_dataset="subscription",
                       join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
]
```

### Intent configuration

```python
PREDICTION_HORIZON         = 90
PREDICTION_HORIZONS        = [30, 60, 90]
RECENT_WINDOW_DAYS         = 365
OBSERVATION_WINDOW_DAYS    = 365
PURGE_GAP_DAYS             = 30
LABEL_WINDOW_DAYS          = 90
TEMPORAL_SPLIT             = True
CADENCE_INTERVAL           = CadenceInterval.MONTHLY
SPLIT_STRATEGY             = SplitStrategy.TEMPORAL
```

### Exploration sampling (0.12)

Section 0.12 provides a dedicated sampling configuration with an estimation table showing accuracy trade-offs at different sample sizes. Sampling preserves target class balance, temporal cohort coverage, and optionally user-specified column distributions. All downstream notebooks operate on this sample; production pipelines always use full data.

```python
# @cr:config name='sampling_config' id=e4b7c9a1
SAMPLE_ENTITY_COUNT = None          # set after reviewing estimation table (e.g. 5000)
SAMPLE_STRATIFY_COLUMNS = []        # extra columns to stratify by (e.g. ["ACCOUNT_TYPE"])
SAMPLE_FILTER_COLUMNS = {            # per-dataset segment filters using query syntax
    "account": "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small']",
}                                    # entities with ANY non-matching row are excluded entirely
```

#### Segment filters (`SAMPLE_FILTER_COLUMNS`)

Segment filters restrict the entity population before sampling. Each filter is a pandas `query()` expression applied to a specific dataset. An entity is excluded from the sample if **any single row** in a filtered dataset does not satisfy the criteria. When multiple datasets have filters, only entities that pass **all** filters (intersection) are retained.

```python
SAMPLE_FILTER_COLUMNS = {
    "account": "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small']",
}
```

In this example:
- Only accounts in the Emerging or Small revenue market segments are kept
- Accounts in other segments (e.g., Mid-Market, Enterprise) are excluded entirely

The implementation uses `resolve_segment_entity_ids()` in `sampling.py`, which:
1. For each filtered dataset, compares per-entity row counts before and after applying the query
2. Entities where all rows match (pre_count == post_count) pass; others are excluded
3. Intersects passing entity sets across all filtered datasets
4. The resulting entity set restricts the target dataset before stratified sampling

Downstream in NB01, entity filtering is handled entirely by the `sample_entity_ids.json` file — no per-row filtering is needed since excluded entities are never in the sample.

#### Estimation table

The estimation table (displayed when the code cell runs) shows for each candidate sample size:
- Churn rate 95% CI half-width
- Correlation estimation error
- Expected minority class count
- Whether cohort coverage is sufficient (>=30 entities per cohort)

When `SAMPLE_ENTITY_COUNT` is set, the `stratified_entity_sample` function:
1. Deduplicates to one row per entity
2. Builds a stratification key from target class, cohort quarter, and any `SAMPLE_STRATIFY_COLUMNS`
3. Allocates proportionally per stratum (floor of 1 per stratum)
4. Guarantees all rare-class entities are kept if count <= `min_rare_count`
5. Writes `sample_entity_ids.json` to the run namespace

The `CR_SAMPLE_ENTITY_COUNT` environment variable can override the notebook setting (useful for CI/Databricks workflows).

### Snapshot grid (0.13)

```python
GRID_MODE = GridAdjustmentMode.NO_ADJUSTMENTS
```


## 8. Key Resolution: Multi-Hop Join Implementation

For the three indirect tables, we need to resolve ACCOUNT_ID before they enter the standard pipeline. This should be a pre-processing step that runs after loading the raw data but before fingerprinting.

### case_history → ACCOUNT_ID

```sql
SELECT ch.*, c.ACCOUNT_ID
FROM case_history ch
INNER JOIN case c ON ch.CASE_ID = c.CASE_ID
```

Drop `case_history` rows where the case has no ACCOUNT_ID (orphans). After this, `case_history` has an `ACCOUNT_ID` column and can join directly.

### opportunity_product → ACCOUNT_ID

```sql
SELECT op.*, o.ACCOUNT_ID
FROM opportunity_product op
INNER JOIN opportunity o ON op.OPPORTUNITY_ID = o.OPPORTUNITY_ID
```

### subscription → ACCOUNT_ID

```sql
SELECT s.*, c.ACCOUNT_ID
FROM subscription s
INNER JOIN contract c ON s.CONTRACT_ID = c.CONTRACT_ID
```

### Implementation in Notebook 00

These resolution queries should run as a cell between "load datasets" and "fingerprint". In Databricks, they can be Spark SQL. Locally (if ever needed for testing), they would be pandas merges. The output replaces the original DataFrame in the `datasets` dict so the rest of the pipeline sees a flat table with `ACCOUNT_ID`.

### transaction_volume_daily (Phase 2)

Requires the `sps_identity.account` table which is not yet in our dataset list. Options:
1. Add `snowflake_corp.sps_identity.account` as a dataset and do the double-hop join
2. Create a pre-materialized view in Snowflake/Databricks that already has ACCOUNT_ID
3. Defer to Phase 2 once the identity table is accessible

**Recommendation:** Defer. The Salesforce tables alone provide rich signal. Transaction volume can be added later as an incremental improvement.


## 9. Target Derivation: Step-by-Step

### Step 1: Identify churned accounts from cancellation cases

```python
cancellation_cases = case_df[
    (case_df["RECORD_TYPE_NAME"] == "cancellation") &
    (case_df["IS_CLOSED"] == 1)
]
churned_from_cases = cancellation_cases.groupby("ACCOUNT_ID").agg(
    churn_date=("CLOSED_DATE", "min")  # earliest cancellation
)
```

### Step 2: Identify churned from contracts (all expired, none active)

```python
active_contracts = contract_df[contract_df["CONTRACT_STATUS"] == "Active"]  # check actual values
accounts_with_active = set(active_contracts["ACCOUNT_ID"])
all_contract_accounts = set(contract_df["ACCOUNT_ID"])
churned_from_contracts = all_contract_accounts - accounts_with_active
```

### Step 3: Combine signals

```python
# An account is churned if:
# - It has a closed cancellation case, OR
# - It has zero active contracts (and had at least one contract)
# Priority: cancellation case date > max contract end date > max subscription terminated date
```

### Step 4: Attach to account table

```python
account_df["churned"] = account_df["ACCOUNT_ID"].isin(churned_accounts).astype(int)
account_df["churn_date"] = account_df["ACCOUNT_ID"].map(churn_date_lookup)
```

### Temporal Consideration

For temporal splitting, the `churn_date` becomes the `label_timestamp_column`. An account is "churned" only relative to a point in time — this is critical for the snapshot grid. At each `as_of_date`:
- Look forward `LABEL_WINDOW_DAYS` (90 days)
- If a churn event falls in that window → label = 1
- Otherwise → label = 0

This avoids target leakage from future churn events.


## 10. Columns to Exclude (Leakage & Low-Value)

### Leakage columns (encode the target or are post-churn)

| Table | Columns | Reason |
|---|---|---|
| case | CANCELLATION_REASON, CANCELLATION_REASON_CATEGORY, CANCEL_COMMENTS_DSAT, CANCEL_COMMENTS_GENERAL, OUTCOME, FUTURE_SERVICE_MANAGEMENT, FUTURE_SERVICE_MANAGEMENT_COMMENTS | Post-churn data |
| request | CANCELLAION_SUBMITTED | Direct churn indicator |
| account | EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR | Retrospective label |
| subscription | IS_ACTIVE, TERMINATED_DATE | These ARE the target signal |

### Low-value / identifier columns (exclude from features)

| Table | Columns | Reason |
|---|---|---|
| all | *_EMPLOYEE_ID columns | Internal staff IDs, not customer behavior |
| all | ETL_CREATED_TIMESTAMP, ETL_UPDATED_TIMESTAMP | Pipeline metadata |
| all | *_ID primary keys (CASE_ID, CONTRACT_ID, etc.) | Identifiers, not features |
| account | WEBSITE, BILLING_STREET, BILLING_LATITUDE, BILLING_LONGITUDE, ACCOUNT_NOTES, ACCOUNT_ALSO_KNOWN_AS | Free-text/PII, not useful for modeling |
| case | DESCRIPTION, CASE_SUBJECT, DETAIL_TAGS | Free-text (could use text features later) |
| case | CONTACT_ID | Identifier |
| opportunity | CUSTOMER_ASK, RATIONALE_FOR_APPROVAL | Free-text |
| request | TRADING_PARTNERS_TEXT, CUSTOMER_ASK, RATIONALE_FOR_APPROVAL, RECENT_DISPUTES_ISSUES_COMMENTS | Free-text |


## 11. Phased Rollout

### Phase 1 (Now): Core Salesforce Tables
- account + 5 direct-join event tables (case, contract, implementation_project, opportunity, request)
- 3 indirect tables with key resolution (case_history, opportunity_product, subscription)
- Derived target from cancellation cases + contract status
- Datetime-derived features within each table

### Phase 2 (Later): Network Data
- Add sps_identity.account as bridge table
- Add transaction_volume_daily with 3-hop key resolution
- Volume/error features as additional signal

### Phase 3 (Optional): Text Features
- case.DESCRIPTION, case.CASE_SUBJECT → text embeddings or keyword features
- request.CUSTOMER_ASK → sentiment/topic features
- opportunity.CUSTOMER_ASK → upsell/churn intent signals


## 12. Open Questions

1. **CONTRACT_STATUS values**: What are the actual distinct values? Need to confirm which value means "active" vs "terminated/expired/cancelled" to correctly derive the target.

2. **RECORD_TYPE_NAME values for case**: The description mentions retention, cancellation, optimization, Customer Ops Case. Are there others? Need exact string values.

3. **Cancellation case = account churn?** Does a closed cancellation case always mean the entire account churned, or can it be partial (one product line)? If partial, the contract/subscription signal may be more reliable.

4. **Account hierarchy**: TOP_PARENT_ACCOUNT_ID suggests parent-child relationships. Should we model at the individual account level or roll up to the top parent? Individual is simpler; parent adds hierarchy features but complicates the entity definition.

5. **Multi-contract accounts**: With up to 21 active contracts per account, should the target be "account lost ALL contracts" (full churn) or "account lost ANY contract" (partial churn)? Full churn is cleaner for binary classification.

6. **Transaction volume availability**: The table description says "last 90 days". If this is a rolling 90-day window, historical snapshots may not be available, limiting temporal feature construction. Confirm retention period.

7. **case.RELATED_TO_ACCOUNT_ID vs ACCOUNT_ID**: Some cases have both. Which is the correct one for our entity join? Likely ACCOUNT_ID, but RELATED_TO_ACCOUNT_ID might capture cross-account relationships.
