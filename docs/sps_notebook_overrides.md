# SPS Production Notebook Configuration Overrides

Concrete manual overrides for each notebook configuration cell, derived from
`table_descriptions.md` and the bridge dataset (key resolution) mechanism.

## Locating Cells

Every override below is identified by **two** stable anchors so you can find
the right cell regardless of how many cells have been inserted or deleted:

1. **Section heading** — the `## X.Y` markdown cell that precedes the code cell
2. **Embedded cell ID** — the `id=HEXUUID` in the `# @cr:` tag on line 1 of
   the cell. These 8-character hex IDs are embedded in the source code and
   never change, even when cells are reordered or new cells are inserted.

> Cells also have a human-readable `name='...'` attribute in the tag line.
> Use `name` for quick identification and `id` for precise matching.

### Sync tags

| Tag | Meaning | Survives `churnkit-sync`? |
|-----|---------|--------------------------|
| `# @cr:config name='...' id=...` | User configuration cell | Yes — your values are preserved |
| `# @cr:user_code name='...' id=...` | User-written logic | Yes — your code is preserved |
| `# @cr:code name='...' id=...` | Framework code cell | **No** — overwritten on upgrade |
| **NEW** | Cell you must insert | Yes, if tagged `config` or `user_code` |

---

## Notebook 00 — Start Here

### Cell Map (current template)

Each section has a `@cr:config` cell (user values, survives sync) and a
`@cr:code` cell (framework logic, overwritten on sync). Only config cells
and `@cr:user_code` cells need editing.

| Section | Config cell | Code cell | Key Definitions |
|---------|-------------|-----------|-----------------|
| 0.1 Project Metadata | `846f56cb` project_settings | `a355da3c` core_imports | `PROJECT_NAME`, imports, `_namespace` |
| 0.2 Dataset Registration | `f1eb641c` dataset_paths | `cdc69ccc` resolve_dataset_paths | `datasets`, `_load_source()`, `is_table_name()` |
| 0.3 Auto Fingerprinting | — | `43e89806` fingerprint_datasets | `fingerprints`, `fingerprinter` |
| 0.4 Confirm Semantics | — | `ca9650f3` detect_time_columns | **`semantics`**, **`RawTimeColumnRole`** |
| 0.5 Target Dataset Selection | `1446e49e` target_overrides | `637064ae` auto_detect_target | `TARGET_DATASET`, `TARGET_COLUMN`, `ENTITY_COLUMN` |
| 0.6 Prediction Objective | — | `f13e88da` detect_prediction_objective | `objective_specs` |
| 0.7 Objective Priority | — | `85a730b7` configure_prediction_anchor | `PRIMARY_OBJECTIVE` |
| 0.8 Join Scaffold | `2a900807` merge_scaffold_overrides | `7f4b562b` detect_relationships | `MANUAL_SCAFFOLD`, `merge_scaffold`, `loaded_frames` |
| 0.8.1 Key Resolution | `7b1ec50d` key_resolution_overrides | `0866aa83` apply_key_resolutions | **`KEY_RESOLUTION`**, `key_resolutions` |
| 0.8.2 Dataset Registry | — | `915bcef8` build_dataset_registry | `registry` |
| 0.9 Temporal Posture | `09a0854f` temporal_posture_config | `04cf48a9` display_temporal_posture | `TEMPORAL_POSTURE` |
| 0.10 Intent Configuration | `ea85abf2` intent_config | `a7f7bf52` build_intent | `intent`, `PREDICTION_HORIZON` |
| 0.11 Save Project Context | — | `c6211778` build_exploration_contract | `project_context`, `context_path` |
| 0.12 Exploration Sampling | `e4b7c9a1` sampling_config | `f5c8d6b2` run_sampling | `SAMPLE_ENTITY_COUNT`, `SAMPLE_FILTER_COLUMNS` |
| 0.13 Snapshot Grid | `2ef9c4c6` snapshot_grid_config | `6b061a37` build_snapshot_grid | `snapshot_grid` |

---

### Section 0.1 — `846f56cb` project_settings — Project Metadata

Edit the config variables at the top of the cell.

```python
PROJECT_NAME = "sps_churn"
LIGHT_RUN = False
SAMPLE_FRACTION = None
```

---

### Section 0.2 — `f1eb641c` dataset_paths — Dataset Registration

Edit **only** the `datasets = { ... }` dict at the top of the cell.
The rest of the cell (function definitions and display code) stays untouched.

```python
datasets = {
#Salesforce Account:
    "account": "prod_corp_snowflake_provisioning_shared.salesforce.account",
#Salesforce Support Cases:
    "case":"prod_corp_snowflake_provisioning_shared.salesforce.case",
#Salesforce Contract:
    "contract": "prod_corp_snowflake_provisioning_shared.salesforce.contract",
#Salesforce Onboarding Projects:
    "implementation_project":"prod_corp_snowflake_provisioning_shared.salesforce.implementation_project",
#Salesforce Opportunities
    "opportunity":"prod_corp_snowflake_provisioning_shared.salesforce.opportunity",
    "opportunity_product":"prod_corp_snowflake_provisioning_shared.salesforce.opportunity_product",
#Salesforce Requests:
    "request":"prod_corp_snowflake_provisioning_shared.salesforce.request",
#Salesforce Subscription Renewal:
    "subscription":"prod_corp_snowflake_provisioning_shared.salesforce.subscription",

#Network Usage:
    #"reporting_service_data":"corpdev_snowflake_cs_org_product_catalog.public.reporting_service_data",
    #"customer_visible_transaction_volume_daily":"prod_networkdata.reporting_gold.customer_visible_transaction_volume_daily",
    #"orderexchange_gold":"prod_networkdata.orderexchange_gold",
#Network Errors:
    #"reporting_service_dets_events": "corpdev_snowflake_cs_org_product_catalog.public.reporting_service_dets_events"
}
```

---

### **NEW** `# @cr:user_code` — Target Derivation

**Insert after:** section 0.2 code cell (`cdc69ccc` resolve_dataset_paths)
**Insert before:** section 0.3 markdown cell ("Auto Fingerprinting")

**Available at this point:** `_load_source` (defined in `cdc69ccc`),
`datasets` (defined in `f1eb641c`), `display`, `Markdown` (imported in `a355da3c`).

The `churned` column does not exist in raw data. This cell derives it from
contract status (ground truth) and uses cancellation cases for the label
timestamp, then replaces the `datasets["account"]` entry so fingerprinting
sees the enriched table.

#### Target (`churned`) logic

An account is churned when it has **no remaining active contracts** but has
had at least one contract (i.e., was a customer). This is purely
contract-status-driven — cancellation cases are **not** part of the target
definition, which avoids false positives from partial cancellations where the
account still has active contracts.

#### Label timestamp (`churn_date`) — two strategies

**Strategy A — Decision Date (default):** The date the cancellation process
completed in the CRM, i.e., when the last cancellation case was created
(`MAX(case.CREATED_DATE)` for cancellation-type cases). Falls back to
`MAX(CONTRACT_END_DATE)` then `MAX(BILLING_TERMINATION_DATE)` for accounts
that churned without a cancellation case (contract expiry / non-renewal).
Best for retention intervention models — gives maximum lead time before the
effective date.

**Strategy B — Effective Cancellation Date (commented alternative):** The
contractual date when the cancellation actually takes effect
(`MAX(CONTRACT_END_DATE)`, fallback `MAX(BILLING_TERMINATION_DATE)`). The
case creation date is ignored for labelling. Best for revenue / financial
modelling — aligns with when MRR impact materialises. Note that this can
place `churn_date` in the future relative to the case; the temporal framework
handles this correctly (label is "not yet known" at earlier snapshots).

> **Databricks:** Uses `load_spark_table` and `register_temp_view` from
> `core.compat` — framework utilities that keep data distributed.
> The enriched account is saved as a global temp view
> (`global_temp.sps_enriched_account`). `is_table_name` recognizes the dot,
> so `_load_source` and `DatasetFingerprinter._load_table` route through
> Spark automatically. No `.toPandas()` at any point.

```python
# @cr:user_code name='derive_churn_target' id=<generate-uuid>
from customer_retention.core.compat import (
    clamp_spark_timestamps,
    load_spark_table,
    register_temp_view,
    strip_spark_timestamp_tz,
)
from customer_retention.core.compat.detection import is_databricks

from pyspark.sql import functions as F

_account_sdf = load_spark_table(datasets["account"])
_case_sdf = load_spark_table(datasets["case"])
_contract_sdf = load_spark_table(datasets["contract"])

# ── Target: contract-status-based ────────────────────────────────────────
# An account is churned iff it has contracts but none are active.
# This correctly excludes partial cancellations (some lines cancelled,
# account still active) and accounts that never had contracts (prospects).
_has_contracts = _contract_sdf.select("ACCOUNT_ID").distinct()
_has_active_contracts = (
    _contract_sdf
    .filter(F.col("CONTRACT_STATUS") == "Active")
    .select("ACCOUNT_ID").distinct()
)
_churned_ids = _has_contracts.subtract(_has_active_contracts)

# ── Label timestamp components ───────────────────────────────────────────
# Case-based: latest cancellation case creation date.
# No IS_CLOSED filter — an open cancellation case whose contracts are already
# non-active is still a valid churn signal; the backoffice just hasn't
# closed the case yet.
_case_dates = (
    _case_sdf
    .filter(F.col("RECORD_TYPE_NAME") == "cancellation")
    .groupBy("ACCOUNT_ID")
    .agg(F.max("CREATED_DATE").alias("_cd_case"))
)
# Contract-based: latest contract end / billing termination date.
_contract_end_dates = (
    _contract_sdf
    .groupBy("ACCOUNT_ID")
    .agg(
        F.max("CONTRACT_END_DATE").alias("_cd_contract_end"),
        F.max("BILLING_TERMINATION_DATE").alias("_cd_billing_term"),
    )
)

# ── Assemble enriched account ────────────────────────────────────────────
_account_enriched = (
    _account_sdf
    .join(_churned_ids.withColumn("_flag", F.lit(1)), "ACCOUNT_ID", "left")
    .withColumn("churned", F.coalesce(F.col("_flag"), F.lit(0)).cast("int"))
    .drop("_flag")
    .join(_case_dates, "ACCOUNT_ID", "left")
    .join(_contract_end_dates, "ACCOUNT_ID", "left")
    # ── Strategy A (default): Decision Date ──────────────────────────────
    # When the cancellation process completed in the CRM (case creation),
    # falling back to contract end date for non-case churn.
    .withColumn(
        "churn_date",
        F.coalesce("_cd_case", "_cd_contract_end", "_cd_billing_term"),
    )
    # ── Strategy B (alternative): Effective Cancellation Date ────────────
    # When services are actually deactivated (contract end date).
    # Uncomment the line below and comment out Strategy A above to switch.
    # .withColumn(
    #     "churn_date",
    #     F.coalesce("_cd_contract_end", "_cd_billing_term"),
    # )
    .drop("_cd_case", "_cd_contract_end", "_cd_billing_term")
)
_account_enriched = clamp_spark_timestamps(strip_spark_timestamp_tz(_account_enriched))
datasets["account"] = register_temp_view(_account_enriched, "sps_enriched_account")

_stats = _account_enriched.agg(
    F.count("*").alias("total"), F.sum("churned").alias("churned")
).first()
display(Markdown(
    f"**Target Derived:** {_stats['churned']:,} churned "
    f"/ {_stats['total']:,} accounts ({_stats['churned'] / _stats['total']:.1%})"
))
```

---

### **NEW** `# @cr:user_code` — Semantics Overrides

**Insert after:** section 0.4 code cell (`ca9650f3` detect_time_columns)
**Insert before:** section 0.5 markdown cell ("Target Dataset Selection")

**Available at this point:** `semantics` (dict built in `ca9650f3`),
`RawTimeColumnRole` (imported in `ca9650f3`).
**Not yet available:** `DatasetGranularity` — import it in this cell.

Cell `ca9650f3` (detect_time_columns) builds the `semantics` dict from auto-fingerprinting.
This cell overrides the detected semantics for all 9 SPS tables.

```python
# @cr:user_code name='semantics_overrides' id=<generate-uuid>
from customer_retention.core.config.column_config import DatasetGranularity

# --- Entity table ---
semantics["account"]["entity_column"] = "ACCOUNT_ID"
semantics["account"]["granularity"] = DatasetGranularity.ENTITY_LEVEL
semantics["account"]["raw_time_column_role"] = RawTimeColumnRole.ENTITY_UPDATE_TIME
semantics["account"]["time_column"] = "LAST_MODIFIED_DATE"

# --- Direct-join event tables ---
for name in ["case", "contract", "implementation_project", "opportunity", "request"]:
    semantics[name]["entity_column"] = "ACCOUNT_ID"
    semantics[name]["granularity"] = DatasetGranularity.EVENT_LEVEL
    semantics[name]["raw_time_column_role"] = RawTimeColumnRole.EVENT_TIME

semantics["case"]["time_column"] = "CREATED_DATE"
semantics["contract"]["time_column"] = "CONTRACT_START_DATE"
semantics["implementation_project"]["time_column"] = "CREATED_DATE"
semantics["opportunity"]["time_column"] = "CREATED_DATE"
semantics["request"]["time_column"] = "CREATED_DATE"

# --- Indirect event tables (ACCOUNT_ID resolved via bridge in NB03) ---
for name in ["opportunity_product", "subscription"]:
    semantics[name]["entity_column"] = "ACCOUNT_ID"
    semantics[name]["granularity"] = DatasetGranularity.EVENT_LEVEL
    semantics[name]["raw_time_column_role"] = RawTimeColumnRole.EVENT_TIME

semantics["opportunity_product"]["time_column"] = "CREATED_DATE"
semantics["subscription"]["time_column"] = "SUBSCRIPTION_START_DATE"
```

---

### Section 0.5 — `1446e49e` target_overrides — Target Configuration

Edit the config variables at the top of the cell:

```python
TARGET_DATASET = "account"
TARGET_COLUMN = "churned"
ENTITY_COLUMN = "ACCOUNT_ID"
```

---

### Section 0.8 — `2a900807` merge_scaffold_overrides — Join Scaffold

Edit the config cell. Replace `MANUAL_SCAFFOLD = None` with
the full scaffold. When `MANUAL_SCAFFOLD` is set, auto-detection is
skipped but `loaded_frames` are still built (all datasets are loaded as
distributed pyspark.pandas DataFrames on Databricks — no OOM).

```python
MANUAL_SCAFFOLD = [
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="case",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="contract",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="implementation_project",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="opportunity",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="request",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="opportunity_product",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
    MergeScaffoldEntry(
        left_dataset="account", right_dataset="subscription",
        join_keys=["ACCOUNT_ID"], relationship="one_to_many"),
]
EXCLUDE_DATASETS = []
```

> `MergeScaffoldEntry` is already imported in `2a900807`.

---

### Section 0.8.1 — `7b1ec50d` key_resolution_overrides — Key Resolution

Edit the config cell. Replace `KEY_RESOLUTION = None` with
the three bridge paths for the indirect tables (case_history,
opportunity_product, subscription).

> `loaded_frames` is always populated (distributed on Databricks), so
> `suggest_key_resolutions(loaded_frames, ...)` works. Setting
> `KEY_RESOLUTION` explicitly bypasses auto-detection entirely.

```python
KEY_RESOLUTION = {
    "opportunity_product": [
        KeyResolutionStep(bridge_dataset="opportunity", source_key="OPPORTUNITY_ID",
                          bridge_key="OPPORTUNITY_ID", resolve_column="ACCOUNT_ID"),
    ],
    "subscription": [
        KeyResolutionStep(bridge_dataset="contract", source_key="CONTRACT_ID",
                          bridge_key="CONTRACT_ID", resolve_column="ACCOUNT_ID"),
    ],
}
```

> `KeyResolutionStep` is already imported in `7b1ec50d`.

Expected output for SPS:

```
Key Resolution (Bridge Datasets)
- case_history: CASE_ID -> case.CASE_ID -> ACCOUNT_ID
- opportunity_product: OPPORTUNITY_ID -> opportunity.OPPORTUNITY_ID -> ACCOUNT_ID
- subscription: CONTRACT_ID -> contract.CONTRACT_ID -> ACCOUNT_ID
```

Downstream flow: `key_resolution` -> NB03 `resolve_entity_keys()` inner-merge ->
generated pipeline code (both local and Databricks renderers).

---

### Section 0.10 — `ea85abf2` intent_config — Intent Configuration

Edit the config variables at the top of the cell:

```python
PREDICTION_HORIZON = 90
```

And after `engine.suggest()`, override the suggested defaults:

```python
PREDICTION_HORIZONS = [30, 60, 90]
RECENT_WINDOW_DAYS = 365
OBSERVATION_WINDOW_DAYS = 365
PURGE_GAP_DAYS = 30
LABEL_WINDOW_DAYS = 90
TEMPORAL_SPLIT = True
CADENCE_INTERVAL = CadenceInterval.MONTHLY
SPLIT_STRATEGY = SplitStrategy.TEMPORAL
```

---

### Section 0.12 — `e4b7c9a1` sampling_config — Exploration Sampling

Edit the config variables at the top of the cell:

```python
SAMPLE_ENTITY_COUNT = 5000
SAMPLE_STRATIFY_COLUMNS = ["ACCOUNT_TYPE"]
SAMPLE_FILTER_COLUMNS = {
    "account": "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small']"
}

```

`SAMPLE_FILTER_COLUMNS` is a **segment gate**: for each dataset listed, the
query is evaluated against every row. If an entity has **any** row that does not
satisfy the filter, that entity is excluded from the sample entirely. When
multiple datasets have filters, only entities passing **all** filters
(intersection) are retained.

The implementation (`resolve_segment_entity_ids` in `sampling.py`) compares
per-entity row counts before and after the query — entities where
pre_count == post_count pass; all others are excluded. The resulting entity set
restricts the target dataset before stratified sampling. Downstream in NB01,
entity filtering is handled by `sample_entity_ids.json` — no per-row filtering
is needed.

---

## Notebook 01 — Data Discovery

### Cell Map (key cells)

| Cell ID | Name | Section | Key Definitions |
|---------|------|---------|-----------------|
| `250a4a48` | init_progress | 1.1 | imports |
| `e5b9a57b` | discovery_config | 1.1 | config: `DATA_PATH`, `TARGET_COLUMN`, etc. |
| `5a76bd95` | load_and_explore | 1.2 | load data, fingerprint |
| `4a1c9d40` | type_override_review | 1.4 | `TYPE_OVERRIDES` |
| `e0e5e3a8` | milestone_pairs_config | 1.6 | config: `MILESTONE_PAIRS` |
| `a4e98584` | save_dataset_findings | 1.6 | active dataset creation code |

---

### Data Loading — Table Name Support

Cell `5a76bd95` (load_and_explore, `@cr:code`) recognizes Spark table names automatically via
`is_table_name()`. When `DATA_PATH` is a dotted table name (e.g.,
`global_temp.sps_enriched_account` or a Unity Catalog three-part name),
the cell loads via `load_spark_table` + `as_pandas_api` — data stays
distributed as pyspark.pandas throughout.

In the normal runner flow `DATA_PATH` is resolved from the project context
(set in NB00), so no manual override is needed. For standalone manual runs
after NB00 in the same Spark session, you can set:

```python
DATA_PATH = "global_temp.sps_enriched_account"
```

---

### Section 1.1 — `e5b9a57b` discovery_config — Per-Dataset Configuration

This is the main config cell. All variables accept a scalar (applies to every
dataset) or a dict keyed by dataset name. Set values here — they survive sync.

#### TARGET_COLUMN

Only the target dataset ("account") has a target column. All other datasets
should have `TARGET_COLUMN = None` so their columns remain as potential
features and no auto-detection fires on ambiguous names like `IS_CLOSED`
or `CONTRACT_STATUS`.

```python
TARGET_COLUMN = {"account": "churned"}
```

> **Why a dict?** `resolve_config({"account": "churned"}, "case")` returns
> `None`. The framework code in cell `adb992a8` (validate_entity_column)
> also suppresses target auto-detection for non-target datasets when
> `ProjectContext.target_dataset` is set, but setting the dict explicitly
> makes the intent clear and works even without a ProjectContext.

> **Leakage note:** Columns in non-target datasets that contribute to the
> target derivation (e.g., `CONTRACT_STATUS` → `churned`,
> `IS_ACTIVE` / `TERMINATED_DATE` in subscription) are already handled via
> `DROP_COLUMNS` below. They must remain dropped to prevent leakage.

#### DROP_COLUMNS

```python
DROP_COLUMNS = {
    "account": [
        # Leakage: retrospective label
        "EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR",
        "EXISTING_CUSTOMER_PREVIOUS_FISCAL_YEAR",
        "EXISTING_CUSTOMER_SECOND_PREVIOUS_FISCAL_YEAR",
        # ETL metadata
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        # Employee identifiers
        "CREATED_BY_EMPLOYEE_ID", "LAST_MODIFIED_BY_EMPLOYEE_ID",
        # PII / free-text / low-value
        "WEBSITE", "BILLING_STREET", "BILLING_CITY", "BILLING_STATE",
        "BILLING_POSTAL_CODE", "BILLING_COUNTRY",
        "BILLING_LATITUDE", "BILLING_LONGITUDE",
        "ACCOUNT_NOTES", "ACCOUNT_ALSO_KNOWN_AS", "ACCOUNT_NUMBER",
        # Hierarchy IDs (not features)
        "TOP_PARENT_ACCOUNT_ID", "TOP_PARENT_ACCOUNT_NAME",
        "HUB_ID", "ASSORTMENT_COMPANY_ID",
        # Secondary NAV codes (identifiers)
        "NAV_CUSTOMER_NUMBER", "NAV_CUSTOMER_NUMBER_2",
        "NAV_CUSTOMER_NUMBER_ACQUISITION", "NAV_CUSTOMER_NUMBER_EDIFICE",
        # Derived target columns (not raw features)
        "churn_date",
    ],
    "case": [
        # Leakage: post-churn / post-cancellation data
        "CANCELLATION_REASON", "CANCELLATION_REASON_CATEGORY",
        "CANCEL_COMMENTS_DSAT", "CANCEL_COMMENTS_GENERAL",
        "OUTCOME", "FUTURE_SERVICE_MANAGEMENT", "FUTURE_SERVICE_MANAGEMENT_COMMENTS",
        # ETL metadata
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        # Identifiers
        "CASE_ID", "CASE_NUMBER", "CASE_REFERENCE", "SUB_REFERENCE",
        "RECORD_TYPE_ID", "CONTACT_ID", "IMPLEMENTATION_PROJECT_ID",
        "RELATED_TO_ACCOUNT_ID",
        # Employee IDs
        "CREATED_BY_EMPLOYEE_ID", "LAST_MODIFIED_BY_EMPLOYEE_ID",
        "OWNER_EMPLOYEE_ID", "SPS_REQUESTOR_EMPLOYEE_ID",
        # Free-text
        "DESCRIPTION", "CASE_SUBJECT", "DETAIL_TAGS",
        "PRIORITY_CHANGE_COMMENTS",
        "ESCALATED_ISSUE_NUMBER",
        # Trading partners (free text)
        "TRADING_PARTNERS",
    ],
    "contract": [
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        "CONTRACT_ID", "CONTRACT_NUMBER", "CONTRACT_NAME",
        "DESCRIPTION",
        # Leakage: status encodes churn after the fact
        "CONTRACT_STATUS",
    ],
    "implementation_project": [
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        "IMPLEMENTATION_PROJECT_ID", "IMPLEMENTATION_PROJECT_NAME",
        "RECORD_TYPE_ID",
        "OPPORTUNITY_ID", "RELATED_TO_ACCOUNT_ID",
        "PARENT_RELEASE_MANAGEMENT_PROJECT", "PARENT_IMPLEMENTATION_PROJECT_ID",
        # Employee IDs
        "CREATED_BY_EMPLOYEE_ID", "LAST_MODIFIED_BY_EMPLOYEE_ID",
        "OWNER_EMPLOYEE_ID",
        "IMPLEMENTATION_ANALYST_EMPLOYEE_ID", "BUSINESS_ANALYST_EMPLOYEE_ID",
        "PROJECT_MANAGER_EMPLOYEE_ID", "SECONDARY_RESOURCE_EMPLOYEE_ID",
        # Free text
        "PROJECT_SCOPE", "DOCUMENT", "EXEMPTED_DOCUMENTS",
        "OPPORTUNITY_CAMPAIGN_SOURCE",
    ],
    "opportunity": [
        "ETL_CREATED_TIMESTAMP",
        "OPPORTUNITY_ID", "OPPORTUNITY_NAME",
        "RECORD_TYPE_ID", "CAMPAIGN_ID", "PRIMARY_CONTACT_ID",
        # Employee IDs
        "CREATED_BY_EMPLOYEE_ID", "LAST_MODIFIED_BY_EMPLOYEE_ID", "OWNER_ID",
        # Free-text
        "CUSTOMER_ASK", "RATIONALE_FOR_APPROVAL",
        "COMP_KILL_MONTHLY_OPTION",
    ],
    "opportunity_product": [
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        "OPPORTUNITY_PRODUCT_ID", "PRODUCT_ID", "RETAILER_ACCOUNT_ID",
        "OPPORTUNITY_PRODUCT_DESCRIPTION",
    ],
    "request": [
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        "REQUEST_ID", "REQUEST_NAME", "RECORD_TYPE_ID",
        # Employee IDs
        "CREATED_BY_EMPLOYEE_ID", "LAST_MODIFIED_BY_EMPLOYEE_ID",
        "ASSIGNED_TO_EMPLOYEE_ID", "VENDOR_CONTACT_ID",
        "CUSTOMER_RELIEF_OWNER_EMPLOYEE_ID",
        "SALES_REPRESENTATIVE_EMPLOYEE_ID", "SALES_DIRECTOR_EMPLOYEE_ID",
        # Leakage
        "CANCELLAION_SUBMITTED",
        # Free text
        "TRADING_PARTNERS_TEXT", "CUSTOMER_ASK", "RATIONALE_FOR_APPROVAL",
        "RECENT_DISPUTES_ISSUES_COMMENTS",
        # Finance identifier codes
        "FINANCE_NAV_CUSTOMER_CODE",
        "FINANCE_CUSTOMER_CODE_1", "FINANCE_CUSTOMER_CODE_2", "FINANCE_CUSTOMER_CODE_3",
        # Identifiers
        "CAMPAIGN_ID", "OPPORTUNITY_ID",
    ],
    "subscription": [
        "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
        "SUBSCRIPTION_ID", "CONNECTION_LOOKUP_ID",
        "TRADING_PARTNER_ACCOUNT_ID", "PRODUCT_ID",
        "SPS_FOR_3PL_LOCATION_ID",
        # Leakage: these ARE the target signal
        "IS_ACTIVE", "TERMINATED_DATE",
    ],
}
```

#### ALLOW_FUTURE_COLUMNS

Planned future dates known at prediction time — do NOT mask to NaN:

```python
ALLOW_FUTURE_COLUMNS = {
    "contract": [
        "CONTRACT_END_DATE", "NEXT_RENEWAL_DATE", "BILLING_TERMINATION_DATE",
    ],
    "subscription": [
        "SUBSCRIPTION_END_DATE",
    ],
    "opportunity": [
        "ESTIMATED_CLOSE_DATE",
    ],
    "request": [
        "REQUEST_EXPIRATION_DATE",
    ],
    "implementation_project": [
        "TARGETED_IMPLEMENTATION_DATE",
        "INITIAL_TARGETED_IMPLEMENTATION_DATE",
        "SOW_TARGETED_IMPLEMENTATION_DATE",
    ],
}
```

#### AUTO_DROP_TEXT_COLUMNS

```python
AUTO_DROP_TEXT_COLUMNS = True
```

---

### Section 1.4 — `4a1c9d40` type_override_review — TYPE_OVERRIDES

Edit `TYPE_OVERRIDES` at the top of the cell:

```python
TYPE_OVERRIDES = {
    "case": {
        "IS_CLOSED": ColumnType.BINARY,
        "CASE_REOPEN_COUNT": ColumnType.NUMERIC_DISCRETE,
        "ECONOMIC_FACTORS_EXCEPTION": ColumnType.BINARY,
        "CSAT_SURVEY_SENT": ColumnType.BINARY,
    },
    "contract": {
        "DOCUMENT_ALLOTMENT": ColumnType.NUMERIC_DISCRETE,
        "CURRENT_TERM": ColumnType.NUMERIC_DISCRETE,
    },
    "opportunity": {
        "CLOSED": ColumnType.BINARY,
        "IS_OPEN": ColumnType.BINARY,
        "SALE_MADE": ColumnType.BINARY,
        "COMPETITIVE_KILL": ColumnType.BINARY,
        "FIRST_SUBSCRIPTION_SALE": ColumnType.BINARY,
    },
    "implementation_project": {
        "IS_HYBRID": ColumnType.BINARY,
        "DIRECT_EDI_MIGRATION_PROJECT": ColumnType.BINARY,
    },
    "subscription": {
        "TOTAL_DOCUMENTS": ColumnType.NUMERIC_DISCRETE,
    },
    "opportunity_product": {
        "IS_FLAT_RATE": ColumnType.BINARY,
    },
}
```

> **Note:** `ColumnType` is imported in `250a4a48` (NB01's init_progress cell).

---

### Section 1.6 — Active Dataset Creation

Cell `e0e5e3a8` (milestone_pairs_config, `@cr:config`) contains `MILESTONE_PAIRS = None`
(auto-detection is used when `None`). Override only if auto-detection picks
wrong pairs:

```python
MILESTONE_PAIRS = {
    "case": [
        ("CREATED_DATE", "CLOSED_DATE"),
        ("CREATED_DATE", "RESPONSE_DATE_TIME"),
        ("CREATED_DATE", "FIRST_ASSIGNED_DATE_TIME"),
        ("CREATED_DATE", "FIRST_CLOSED_DATE_TIME"),
        ("FIRST_CLOSED_DATE_TIME", "REOPENED_DATE_TIME"),
        ("CLOSED_DATE", "CSAT_SURVEY_SENT_DATE"),
    ],
    "contract": [
        ("CONTRACT_START_DATE", "CONTRACT_END_DATE"),
        ("CONTRACT_START_DATE", "ACTIVATED_DATE"),
        ("CONTRACT_START_DATE", "NEXT_RENEWAL_DATE"),
        ("CONTRACT_START_DATE", "BILLING_TERMINATION_DATE"),
    ],
    "implementation_project": [
        ("PROJECT_START_DATE", "PRODUCTION_DATE"),
        ("TARGETED_IMPLEMENTATION_DATE", "PRODUCTION_DATE"),
        ("INITIAL_TARGETED_IMPLEMENTATION_DATE", "PRODUCTION_DATE"),
        ("PROJECT_START_DATE", "SETUP_COMPLETE_DATE"),
    ],
    "opportunity": [
        ("CREATED_DATE", "SALE_DATE"),
        ("CREATED_DATE", "SUBMITTED_DATE"),
        ("CREATED_DATE", "ESTIMATED_CLOSE_DATE"),
        ("ESCALATION_DATE", "ESCALATION_RESOLUTION_DATE"),
    ],
    "subscription": [
        ("SUBSCRIPTION_START_DATE", "SUBSCRIPTION_END_DATE"),
    ],
}
```

Derived numeric features from these pairs (computed automatically by
`derive_extra_datetime_features`):

| Dataset | Derived Feature | Formula | Business Meaning |
|---|---|---|---|
| case | case_resolution_days | CLOSED_DATE - CREATED_DATE | Time to resolve |
| case | case_first_response_hours | RESPONSE_DATE_TIME - CREATED_DATE | SLA metric |
| case | case_first_assign_hours | FIRST_ASSIGNED_DATE_TIME - CREATED_DATE | Assignment speed |
| case | case_first_close_days | FIRST_CLOSED_DATE_TIME - CREATED_DATE | First resolution speed |
| case | case_reopen_delay_days | REOPENED_DATE_TIME - FIRST_CLOSED_DATE_TIME | Quality of resolution |
| case | case_survey_delay_days | CSAT_SURVEY_SENT_DATE - CLOSED_DATE | Survey timing |
| contract | contract_duration_days | CONTRACT_END_DATE - CONTRACT_START_DATE | Contract length |
| contract | contract_activation_delay | ACTIVATED_DATE - CONTRACT_START_DATE | Setup friction |
| contract | contract_renewal_window | NEXT_RENEWAL_DATE - CONTRACT_START_DATE | Renewal cycle |
| contract | contract_billing_span | BILLING_TERMINATION_DATE - CONTRACT_START_DATE | Billing period |
| impl_project | project_duration_days | PRODUCTION_DATE - PROJECT_START_DATE | Delivery time |
| impl_project | project_delay_vs_target | PRODUCTION_DATE - TARGETED_IMPLEMENTATION_DATE | Delivery accuracy |
| impl_project | project_delay_vs_initial | PRODUCTION_DATE - INITIAL_TARGETED_IMPLEMENTATION_DATE | Scope creep |
| impl_project | project_setup_time | SETUP_COMPLETE_DATE - PROJECT_START_DATE | Setup phase |
| opportunity | opp_cycle_days | SALE_DATE - CREATED_DATE | Sales cycle length |
| opportunity | opp_submit_delay | SUBMITTED_DATE - CREATED_DATE | Submit speed |
| opportunity | opp_close_estimate | ESTIMATED_CLOSE_DATE - CREATED_DATE | Expected duration |
| opportunity | escalation_resolution | ESCALATION_RESOLUTION_DATE - ESCALATION_DATE | Escalation speed |
| subscription | sub_duration_days | SUBSCRIPTION_END_DATE - SUBSCRIPTION_START_DATE | Subscription length |

---

## Notebook 01a — Temporal Deep Dive

### Cell Map (config cells)

| Cell ID | Name | Section | Description |
|---------|------|---------|-------------|
| (none) | — | 1a.1 | `DATASET_NAME` override + findings loading |

No manual overrides required beyond the default dataset auto-resolution.
Runs automatically per event dataset.

Optional: to force a specific dataset, override `DATASET_NAME` in the load_findings cell.

```python
DATASET_NAME = None  # auto-resolved; override only to force a specific dataset
```

---

## Notebook 01b — Temporal Quality

### Cell Map (config cells)

| Cell ID | Name | Section | Description |
|---------|------|---------|-------------|
| `dfd0cf93` | dataset_name_config | 1b.1 | `DATASET_NAME` override |
| `346bde89` | quality_thresholds | 1b.2 | Quality check parameters |

### Section 1b.1 — `dfd0cf93` dataset_name_config — Dataset Override

```python
DATASET_NAME = None  # auto-resolved; override only to force a specific dataset
```

### Section 1b.2 — `346bde89` quality_thresholds — Quality Configuration

Most SPS datasets are business-event level, not regularly spaced.
Use liberal gap detection:

```python
REFERENCE_DATE = native_pd.Timestamp.now()
EXPECTED_FREQUENCY = "D"
MAX_GAP_MULTIPLE = 5.0
```

---

## Notebook 01c — Temporal Patterns

### Cell Map (config cells)

| Cell ID | Name | Section | Description |
|---------|------|---------|-------------|
| `814cbef2` | dataset_name_config | 1c.1 | `DATASET_NAME` override |
| `3e4bcf61` | target_config | 1c.2 | `TARGET_COLUMN_OVERRIDE`, `TARGET_AGGREGATION` |
| `3d5529e6` | window_config | 1c.3 | `WINDOW_OVERRIDE` |

### Section 1c.2 — `3e4bcf61` target_config — Target Configuration

```python
TARGET_COLUMN_OVERRIDE = None      # auto-detect from findings
TARGET_AGGREGATION = "max"         # binary churn: max gives "ever churned in window"
```

### Section 1c.3 — `3d5529e6` window_config — Window Override

```python
WINDOW_OVERRIDE = None   # use 01a recommendations
```

### Value Column per Dataset

The `VALUE_COLUMN` is set automatically. If the auto-detection picks the
wrong column, set it manually per dataset run. Use `_event_count` for
relationship/activity tables; use a financial metric where one exists:

| Dataset Run | VALUE_COLUMN |
|---|---|
| case | `"_event_count"` |
| case_history | `"_event_count"` |
| contract | `"_event_count"` |
| implementation_project | `"_event_count"` |
| opportunity | `"USD_BOOKINGS_MONTHLY_RECURRING"` |
| opportunity_product | `"USD_MONTHLY_RECURRING"` |
| request | `"CREDIT_AMOUNT_REQUESTED"` |
| subscription | `"NET_PRICE"` |

---

## Notebook 01d — Event Aggregation

### Cell Map (config cells)

| Cell ID | Name | Section | Description |
|---------|------|---------|-------------|
| `bb0f263e` | dataset_name_config | 1d.1 | `DATASET_NAME` override |
| `f387ea5f` | aggregation_config | 1d.2 | Aggregation config: windows, value columns, features |

### Section 1d.1 — `bb0f263e` dataset_name_config — Dataset Override

```python
DATASET_NAME = None  # auto-resolved; override only to force a specific dataset
```

### Section 1d.2 — `f387ea5f` aggregation_config — Aggregation Configuration

```python
WINDOW_OVERRIDE = None          # use 01a/01c recommendations
```

### Per-Dataset Feature Potential (what 01d will aggregate)

#### case — Support Activity Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Case volume | _event_count | count per window | Support burden |
| Case by type | RECORD_TYPE_NAME | count per category | Retention/cancellation risk |
| Case severity | SEVERITY | count per category | Issue severity profile |
| Case status | CASE_STATUS, IS_CLOSED | rate (closed/total) | Resolution efficiency |
| Case reopens | CASE_REOPEN_COUNT | sum, max | Quality of resolution |
| Resolution time | (derived) CLOSED_DATE - CREATED_DATE | mean, max | SLA performance |
| Response time | (derived) RESPONSE_DATE_TIME - CREATED_DATE | mean, max | Responsiveness |
| Case origin | ORIGIN | count per category | Channel preference |
| Case age | CASE_AGE_HOURS | mean, max | Backlog indicator |
| CSAT survey | CSAT_SURVEY_SENT | sum | Engagement measurement |

#### case_history — Case Change Velocity Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Change volume | _event_count | count per window | Case churn/complexity |
| Field changes | FIELD | count per field name | What is changing most |
| Status changes | FIELD="Status", NEW_VALUE | transition counts | Escalation patterns |
| Value transitions | OLD_VALUE, NEW_VALUE | distinct combos | Volatility indicator |

#### contract — Contract Health Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Active contracts | _event_count | count per window | Contract portfolio size |
| Contract term | CONTRACT_TERM, CURRENT_TERM | mean, max | Commitment depth |
| Document allotment | DOCUMENT_ALLOTMENT | sum, mean | Usage capacity |
| Contract duration | (derived) END - START | mean, min | Tenure stability |
| Time to renewal | (derived) NEXT_RENEWAL_DATE - as_of | min | Renewal urgency |
| Contract type | CONTRACT_TERM_TYPE | count per category | Contract mix |
| Division | CONTRACT_DIVISION | count per category | Business scope |
| Document plan | DOCUMENT_PLAN_TYPE | count per category | Product complexity |

#### implementation_project — Onboarding Health Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Project count | _event_count | count per window | Implementation activity |
| Project status | PROJECT_STATUS | count per status | Completion rate |
| Project stage | PROJECT_STAGE | count per stage | Pipeline maturity |
| Project delays | (derived) PRODUCTION - TARGET | mean, max | Delivery friction |
| Setup time | (derived) SETUP_COMPLETE - START | mean | Onboarding speed |
| On-hold flags | ON_HOLD_DATE (non-null) | count | Stalled projects |
| Cancelled | CANCELLED_DATE (non-null) | count | Failed implementations |
| Project type | PROJECT_TYPE | count per category | Work type mix |
| Work unit type | WORK_UNIT_TYPE | count per category | Complexity indicator |
| Solution status | SOLUTION_STATUS | count per status | Delivery quality |
| Hybrid flag | IS_HYBRID | sum | Integration complexity |

#### opportunity — Sales Activity Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Opportunity count | _event_count | count per window | Sales activity level |
| Win/loss | OPPORTUNITY_STATUS | count per status | Win rate |
| Revenue: MRR | USD_BOOKINGS_MONTHLY_RECURRING | sum, mean | Revenue trajectory |
| Revenue: one-time | USD_BOOKINGS_ONE_TIME_FEES | sum | Implementation spend |
| Revenue: ARR | USD_COMMISSIONABLE_ARR | sum, mean | Contract value |
| Lift | LIFT | sum, mean | Upsell success |
| Estimated MRR | USD_ESTIMATED_NET_MONTHLY_RECURRING | sum, mean | Pipeline value |
| Sales cycle | (derived) SALE_DATE - CREATED_DATE | mean, max | Sales efficiency |
| Competitive kills | COMPETITIVE_KILL | sum | Competitive pressure |
| First subscription | FIRST_SUBSCRIPTION_SALE | sum | New business signal |
| MRR change type | MONTHLY_FEE_CHANGE_TYPE_FROM/TO | category counts | Pricing trajectory |
| Closed decisions | CLOSED_WON_DECISIONS | sum | Decision velocity |
| MRR reduction | ESTIMATED_MRR_REDUCTION_AMOUNT | sum | Revenue risk |

#### opportunity_product — Product Mix Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Product count | _event_count | count per window | Product diversity |
| Revenue: MRR | USD_MONTHLY_RECURRING, USD_ADJUSTED_MONTHLY_RECURRING | sum, mean | Product-level MRR |
| Revenue: ARR | USD_ANNUAL_RECURRING, USD_ADJUSTED_ANNUAL_RECURRING | sum, mean | Annual commitment |
| One-time fees | USD_ONE_TIME_FEES | sum | Setup investment |
| Quantity | QUANTITY | sum, mean | Volume |
| Unit price | UNIT_PRICE | mean, max | Price tier |
| Win/loss | WIN_LOSS | count per value | Product-level success |
| Flat rate flag | IS_FLAT_RATE | sum | Pricing model |
| Billing term | BILLING_TERM | count per category | Billing frequency |

#### request — Customer Relief Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Request count | _event_count | count per window | Complaint/relief volume |
| Request type | REQUEST_TYPE | count per type | Nature of issues |
| Request status | REQUEST_STATUS | count per status | Resolution rate |
| Credit requested | CREDIT_AMOUNT_REQUESTED | sum, mean | Financial concession |
| Rate reduction | USD_TOTAL_RATE_REDUCTION | sum | Revenue erosion |
| Price delta | CURRENT_MONTHLY_PRICE - PROPOSED_MONTHLY_PRICE | sum, mean | Discount pressure |
| Credit type | CREDIT_TYPE | count per type | Credit pattern |
| Credit reason | CREDIT_REQUEST_REASON_CODE | count per code | Root cause |
| Billing status | BILLING_STATUS | count per status | Payment health |
| Record type | RECORD_TYPE_NAME | count per type | Request category |
| Relief bucket | CUSTOMER_RELIEF_BUCKET | count per bucket | Intervention type |

#### subscription — Subscription Health Features

| Feature Family | Columns Used | Aggregations | Business Signal |
|---|---|---|---|
| Subscription count | _event_count | count per window | Portfolio breadth |
| Net price | NET_PRICE | sum, mean | Revenue per sub |
| Quantity | QUANTITY | sum, mean | Usage volume |
| Duration | (derived) END - START | mean, min | Commitment stability |
| Total documents | TOTAL_DOCUMENTS | sum, mean | Usage level |
| Overage rate | OVERAGE_RATE | mean, max | Overage exposure |
| Blended rates | BLENDED_DOC_RATE_* | mean | Pricing efficiency |
| Subscription status | SUBSCRIPTION_STATUS | count per status | Health distribution |

---

## Columns NOT Dropped (Retained as Features)

### account (entity-level features, broadcast in merge)

| Column | Feature Type | Reasoning |
|---|---|---|
| ACCOUNT_NAME | categorical | Company identity (encode or drop) |
| MONTHLY_FEE_TYPE | categorical | Pricing tier |
| ERP_ACCOUNTING_APPLICATION | categorical | Technology stack |
| ADAPTER | categorical | Integration type |
| FI_TEAM | categorical | Financial segment |
| CURRENT_CUSTOMER_SUCCESS_SEGMENT | categorical | CS tier |
| CURRENT_CUSTOMER_SUCCESS_SUBSEGMENT | categorical | CS sub-tier |
| TERRITORY_COUNTRY | categorical | Geography |
| CS_ADAPTER | categorical | CS integration type |
| CS_ERP_ACCOUNTING_APPLICATION | categorical | CS tech stack |
| CUSTOMER_ENGAGEMENT_PLAN | categorical | Engagement level |
| ANNUAL_REVENUE | numeric | Company size |
| ACCOUNT_TYPE | categorical | Account classification |
| ANNUAL_REVENUE_RANGE | ordinal | Revenue bracket |
| INDUSTRY_SPS | categorical | Industry vertical |
| FIRST_SUBSCRIPTION_DATE | datetime->numeric | Customer tenure |
| NET_NEW_CUSTOMER_DATE | datetime->numeric | Time since net-new |
| TERRITORY_RDM | categorical | Regional manager |
| SALES_TEAM | categorical | Sales organization |
| ACCOUNT_LEVEL | ordinal | Account tier |
| ANNIVERSARY_DATE__C | datetime->numeric | Relationship milestone |

---

## Summary of Configuration Flow

```
NB00:
  846f56cb  (0.1)  edit config:     PROJECT_NAME, LIGHT_RUN, SAMPLE_FRACTION
  f1eb641c  (0.2)  edit config:     datasets dict (9 tables)
  NEW cell  (0.2+) insert after:    derive "churned" (contract-based) + "churn_date" (Strategy A/B)
  ca9650f3  (0.4)  no changes:      semantics built by framework
  NEW cell  (0.4+) insert after:    semantics overrides (entity_column, time_column, granularity)
  1446e49e  (0.5)  edit config:     TARGET_DATASET="account", TARGET_COLUMN="churned", ENTITY="ACCOUNT_ID"
  2a900807  (0.8)  edit config:     MANUAL_SCAFFOLD (8 entries, all to account)
  7b1ec50d  (0.8.1)edit config:     KEY_RESOLUTION (3 bridge paths)
  ea85abf2  (0.10) edit config:     90d horizon, monthly cadence, temporal split

NB01:
  e5b9a57b  (1.1)  edit config:     DROP_COLUMNS, ALLOW_FUTURE_COLUMNS, AUTO_DROP_TEXT_COLUMNS
  4a1c9d40  (1.4)  edit config:     TYPE_OVERRIDES
  e0e5e3a8  (1.6)  edit config:     MILESTONE_PAIRS (optional)

NB01b:
  346bde89  (1b.2) edit config:     REFERENCE_DATE, EXPECTED_FREQUENCY, MAX_GAP_MULTIPLE

NB01c:
  3e4bcf61  (1c.2) edit config:     TARGET_COLUMN_OVERRIDE, TARGET_AGGREGATION
  3d5529e6  (1c.3) edit config:     WINDOW_OVERRIDE

NB01d:
  f387ea5f  (1d.2) edit config:     WINDOW_OVERRIDE
```

### New cells to insert (2 total)

| Insert after | Insert before | Tag | Purpose |
|---|---|---|---|
| `cdc69ccc` resolve_dataset_paths | (0.3 Auto Fingerprinting) | `# @cr:user_code` | Derive `churned` (contract-based) + `churn_date` (Strategy A: decision date, Strategy B: effective date commented) |
| `ca9650f3` detect_time_columns | (0.5 Target Dataset Selection) | `# @cr:user_code` | Override semantics for all 9 datasets |

Both inserted cells survive `churnkit-sync` because they are tagged
`@cr:user_code`.

Key Resolution (`7b1ec50d`) is now an existing config cell — edit it, no
insertion needed.

### Variable availability at each insertion point

| Insertion point | Available | Must import |
|---|---|---|
| After `cdc69ccc` resolve_dataset_paths | `datasets`, `_load_source`, `display`, `Markdown` | `load_spark_table`, `register_temp_view`, `is_databricks` |
| After `ca9650f3` detect_time_columns | above + `semantics`, `RawTimeColumnRole`, `fingerprints` | `DatasetGranularity` |

### Framework changes (built into template)

| Cell | Section | Change | Why |
|---|---|---|---|
| `cdc69ccc` resolve_dataset_paths | 0.2 Dataset Registration | `_load_source` returns `as_pandas_api(spark_df)` instead of `.toPandas()` | Data stays distributed on Databricks — no driver OOM |
| `f13e88da` detect_prediction_objective | 0.6 Prediction Objective | DataFrame check uses `isinstance(source, str)` instead of `isinstance(source, native_pd.DataFrame)` | Correctly handles pyspark.pandas DataFrames in `datasets` dict |
| `7f4b562b` detect_relationships | 0.8 Join Scaffold | `loaded_frames` always built (no guard), uses `is_dataframe()` check | All steps work with distributed DataFrames |
| `915bcef8` build_dataset_registry | 0.8.2 Dataset Registry | Path/format detection uses `isinstance(source, str)` | Correctly handles pyspark.pandas DataFrames in `datasets` dict |
| `c6211778` build_exploration_contract | 0.11 Save Context | Entity sampling uses `safe_to_list()` and compat `concat()` | pyspark.pandas `.tolist()` / `native_pd.concat()` incompatibility |
| `core/compat/__init__.py` | — | Added `as_pandas_api()`, `load_spark_table()`, `register_temp_view()` | Reusable Spark-native utilities: table loading, distributed conversion, global temp view registration |
