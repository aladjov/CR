# Production Configuration Guide — SPS Commerce Customer Retention

## 1. Entity Definition

**Entity = ACCOUNT (ACCOUNT_ID)**

The account is the unit we predict retention for. Contracts and subscriptions are child objects — multiple per account (up to 21 active contracts observed). The model predicts at the account level: "will this account churn?"

All tables except `subscription`, `opportunity_product`, and `case_history` have a direct `ACCOUNT_ID` column. The three exceptions join indirectly:
- `subscription` → via `contract.CONTRACT_ID`
- `opportunity_product` → via `opportunity.OPPORTUNITY_ID`
- `case_history` → via `case.CASE_ID`

---

## 2. Table Classification

| Dataset | Granularity | Entity Key | Rationale |
|---------|-------------|------------|-----------|
| **account** | ENTITY | ACCOUNT_ID | One row per account — master record |
| **contract** | EVENT | ACCOUNT_ID | Multiple contracts per account over time |
| **subscription** | EVENT | CONTRACT_ID (→ ACCOUNT_ID via contract) | Multiple subscriptions per contract |
| **implementation_project** | EVENT | ACCOUNT_ID | Multiple projects per account |
| **case** | EVENT | ACCOUNT_ID | Multiple support cases per account |
| **case_history** | EVENT | CASE_ID (→ ACCOUNT_ID via case) | Field-level change log — very granular |
| **opportunity** | EVENT | ACCOUNT_ID | Multiple sales opportunities per account |
| **opportunity_product** | EVENT | OPPORTUNITY_ID (→ ACCOUNT_ID via opportunity) | Line items within opportunities |
| **request** | EVENT | ACCOUNT_ID | Credit/relief requests per account |

### Join Challenges

Three tables need intermediate joins to reach ACCOUNT_ID:

1. **subscription**: `subscription.CONTRACT_ID → contract.CONTRACT_ID → contract.ACCOUNT_ID`
2. **opportunity_product**: `opportunity_product.OPPORTUNITY_ID → opportunity.OPPORTUNITY_ID → opportunity.ACCOUNT_ID`
3. **case_history**: `case_history.CASE_ID → case.CASE_ID → case.ACCOUNT_ID`

**Recommendation**: For first iteration, consider handling these as follows:
- **subscription**: Aggregate to contract level first, or pre-join with contract to bring in ACCOUNT_ID
- **opportunity_product**: Similar — pre-join with opportunity, or just use opportunity table directly (it already has revenue fields)
- **case_history**: Very granular (field-level changes). Consider skipping for v1, or pre-aggregate to case-level metrics (e.g., number of status changes, time between changes)

---

## 3. Timestamp Column Selection

### Entity Table (account)

| Column | Role | Use For |
|--------|------|---------|
| `CREATED_DATE` | FEATURE_TIMESTAMP | When the account was created — use as the entity timestamp |
| `FIRST_SUBSCRIPTION_DATE` | DATETIME (derive feature) | Tenure calculation |
| `NET_NEW_CUSTOMER_DATE` | DATETIME (derive feature) | Alternative tenure marker |
| `LAST_MODIFIED_DATE` | DROP | ETL artifact, not meaningful for prediction |
| `ETL_CREATED_TIMESTAMP` | DROP | ETL metadata |
| `ETL_UPDATED_TIMESTAMP` | DROP | ETL metadata |
| `ANNIVERSARY_DATE__C` | DATETIME (derive feature) | Could derive tenure/seasonality |

**Datetime derivation candidates**: `FIRST_SUBSCRIPTION_DATE`, `NET_NEW_CUSTOMER_DATE` — derive days-since features relative to observation date. `ANNIVERSARY_DATE__C` if populated.

### Event Tables — Primary Timestamp

| Dataset | Event Timestamp | Rationale |
|---------|-----------------|-----------|
| **contract** | `CONTRACT_START_DATE` | When the contract was initiated |
| **subscription** | `SUBSCRIPTION_START_DATE` | When the subscription began |
| **implementation_project** | `CREATED_DATE` | When the project was created |
| **case** | `CREATED_DATE` | When the support case was opened |
| **case_history** | `CREATED_DATE` | When the field change occurred |
| **opportunity** | `CREATED_DATE` | When the opportunity was created |
| **opportunity_product** | `CREATED_DATE` | When the line item was created |
| **request** | `CREATED_DATE` | When the request was filed |

### Future-Value Columns (ALLOW_FUTURE_COLUMNS)

These columns legitimately contain dates in the future and should NOT be masked:

| Dataset | Future-Value Columns | Why |
|---------|----------------------|-----|
| **contract** | `CONTRACT_END_DATE`, `NEXT_RENEWAL_DATE`, `BILLING_TERMINATION_DATE` | Scheduled future events — known at contract creation |
| **subscription** | `SUBSCRIPTION_END_DATE` | Planned end date |
| **implementation_project** | `TARGETED_IMPLEMENTATION_DATE`, `SOW_TARGETED_IMPLEMENTATION_DATE`, `INITIAL_TARGETED_IMPLEMENTATION_DATE` | Target dates set at project start |
| **request** | `REQUEST_EXPIRATION_DATE` | Set when request is created |
| **opportunity** | `ESTIMATED_CLOSE_DATE` | Sales forecast date |

---

## 4. Target Definition

### Recommended Approach: Contract-Based Churn

The target should be derived from the **contract** table. A churned account is one where:
- All active contracts have ended (no renewal), OR
- `CONTRACT_STATUS` indicates cancellation/non-renewal

**Suggested target construction** (done outside the framework, or in NB00 as a pre-processing step):
```
For each ACCOUNT_ID:
  churned = 1 if MAX(CONTRACT_END_DATE) < reference_date AND no active contracts
  churned = 0 otherwise
```

Alternative signals to validate against:
- `case` records with `RECORD_TYPE_NAME = 'cancellation'`
- `request` records with cancellation-related `REQUEST_TYPE`

### NB00 Configuration
```python
TARGET_DATASET = "contract"  # or a derived target table
TARGET_COLUMN = "churned"     # needs to be derived
ENTITY_COLUMN = "ACCOUNT_ID"
```

**Note**: You may need to create a derived target table before NB00, since the target isn't a simple column in any source table. Alternatively, derive it in NB01 during profiling.

---

## 5. Notebook 00 — Project Setup Configuration

```python
PROJECT_NAME = "sps_customer_retention"

datasets = {
    "account": "snowflake_corp.salesforce.account",
    "contract": "snowflake_corp.salesforce.contract",
    "subscription": "snowflake_corp.salesforce.subscription",
    "implementation_project": "snowflake_corp.salesforce.implementation_project",
    "case": "snowflake_corp.salesforce.case",
    "case_history": "snowflake_corp.salesforce.case_history",
    "opportunity": "snowflake_corp.salesforce.opportunity",
    "opportunity_product": "snowflake_corp.salesforce.opportunity_product",
    "request": "snowflake_corp.salesforce.request",
}
```

### Semantics Overrides

```python
semantics = {
    "account": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CREATED_DATE",
        "granularity": "entity",
    },
    "contract": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CONTRACT_START_DATE",
        "granularity": "event",
    },
    "subscription": {
        "entity_column": "CONTRACT_ID",       # indirect — needs join
        "time_column": "SUBSCRIPTION_START_DATE",
        "granularity": "event",
    },
    "implementation_project": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
    "case": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
    "case_history": {
        "entity_column": "CASE_ID",           # indirect — needs join
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
    "opportunity": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
    "opportunity_product": {
        "entity_column": "OPPORTUNITY_ID",    # indirect — needs join
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
    "request": {
        "entity_column": "ACCOUNT_ID",
        "time_column": "CREATED_DATE",
        "granularity": "event",
    },
}
```

### Temporal Configuration

```python
TEMPORAL_POSTURE = "STABLE"            # B2B SaaS — contracts are long-term
PREDICTION_HORIZON = 90                # Predict churn 90 days ahead
RECENT_WINDOW_DAYS = 180               # Recent behavior window
PURGE_GAP_DAYS = 30                    # Gap between features and label
LABEL_WINDOW_DAYS = 90                 # Window to observe churn outcome
SPLIT_STRATEGY = "temporal"            # Always temporal for production
```

### Objective

Primary objective: **renewal_risk** or **immediate_risk** (depending on framework options). The intent is to identify accounts at risk of not renewing.

---

## 6. Notebook 01 — Column Configuration Per Dataset

### 6.1 account (Entity Table)

**DROP_COLUMNS** (identifiers, text, addresses, ETL metadata):
```python
DROP_COLUMNS = [
    # ETL metadata
    "ETL_CREATED_TIMESTAMP",
    "ETL_UPDATED_TIMESTAMP",
    "LAST_MODIFIED_DATE",
    "LAST_MODIFIED_BY_EMPLOYEE_ID",
    "CREATED_BY_EMPLOYEE_ID",
    # Pure identifiers (not useful as features)
    "TOP_PARENT_ACCOUNT_ID",
    "HUB_ID",
    "NAV_CUSTOMER_NUMBER",
    "NAV_CUSTOMER_NUMBER_2",
    "NAV_CUSTOMER_NUMBER_ACQUISITION",
    "NAV_CUSTOMER_NUMBER_EDIFICE",
    "ACCOUNT_NUMBER",
    "ASSORTMENT_COMPANY_ID",
    # Text / free-form
    "ACCOUNT_NAME",
    "TOP_PARENT_ACCOUNT_NAME",
    "ACCOUNT_ALSO_KNOWN_AS",
    "ACCOUNT_NOTES",
    # Address fields (PII / too granular)
    "WEBSITE",
    "BILLING_STREET",
    "BILLING_CITY",
    "BILLING_STATE",
    "BILLING_POSTAL_CODE",
    "BILLING_LATITUDE",
    "BILLING_LONGITUDE",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CREATED_DATE | FEATURE_TIMESTAMP | Entity timestamp |
| MONTHLY_FEE_TYPE | CATEGORICAL_NOMINAL | Fee structure |
| ERP_ACCOUNTING_APPLICATION | CATEGORICAL_NOMINAL | ERP system used |
| ADAPTER | CATEGORICAL_NOMINAL | Integration type |
| FI_TEAM | CATEGORICAL_NOMINAL | Financial team |
| CURRENT_CUSTOMER_SUCCESS_SEGMENT | CATEGORICAL_NOMINAL | CS segment |
| CURRENT_CUSTOMER_SUCCESS_SUBSEGMENT | CATEGORICAL_NOMINAL | CS sub-segment |
| TERRITORY_COUNTRY | CATEGORICAL_NOMINAL | Geographic feature |
| CS_ADAPTER | CATEGORICAL_NOMINAL | CS adapter type |
| CS_ERP_ACCOUNTING_APPLICATION | CATEGORICAL_NOMINAL | CS ERP app |
| CUSTOMER_ENGAGEMENT_PLAN | CATEGORICAL_NOMINAL | Engagement plan |
| EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR | BINARY | Active this year |
| EXISTING_CUSTOMER_PREVIOUS_FISCAL_YEAR | BINARY | Active last year |
| EXISTING_CUSTOMER_SECOND_PREVIOUS_FISCAL_YEAR | BINARY | Active 2 years ago |
| ANNUAL_REVENUE | NUMERIC_CONTINUOUS | Revenue |
| ACCOUNT_TYPE | CATEGORICAL_NOMINAL | Account type |
| ANNUAL_REVENUE_RANGE | CATEGORICAL_ORDINAL | Revenue bucket |
| INDUSTRY_SPS | CATEGORICAL_NOMINAL | Industry |
| FIRST_SUBSCRIPTION_DATE | DATETIME | Derive tenure |
| NET_NEW_CUSTOMER_DATE | DATETIME | Derive tenure |
| TERRITORY_RDM | CATEGORICAL_NOMINAL | Regional manager territory |
| SALES_TEAM | CATEGORICAL_NOMINAL | Sales team |
| BILLING_COUNTRY | CATEGORICAL_NOMINAL | Billing country (keep, drop other address fields) |
| ACCOUNT_LEVEL | CATEGORICAL_NOMINAL | Account tier |
| ANNIVERSARY_DATE__C | DATETIME | Derive tenure features |

**TYPE_OVERRIDES**:
```python
TYPE_OVERRIDES = {
    "EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR": "binary",
    "EXISTING_CUSTOMER_PREVIOUS_FISCAL_YEAR": "binary",
    "EXISTING_CUSTOMER_SECOND_PREVIOUS_FISCAL_YEAR": "binary",
    "ANNUAL_REVENUE_RANGE": "categorical_ordinal",
}
```

**Datetime derivation**: Derive days-since features from `FIRST_SUBSCRIPTION_DATE`, `NET_NEW_CUSTOMER_DATE`, `ANNIVERSARY_DATE__C`.

---

### 6.2 contract (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",
    "ETL_UPDATED_TIMESTAMP",
    "CONTRACT_NAME",
    "CONTRACT_NUMBER",
    "DESCRIPTION",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| CONTRACT_ID | IDENTIFIER | Primary key |
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CONTRACT_START_DATE | FEATURE_TIMESTAMP | Event timestamp |
| CONTRACT_END_DATE | DATETIME (future-allowed) | Derive days-until-end |
| CONTRACT_TERM_TYPE | CATEGORICAL_NOMINAL | Term type |
| CONTRACT_TERM | CATEGORICAL_NOMINAL | Duration category |
| ADVANCE_CANCEL_NOTICE | CATEGORICAL_NOMINAL | Cancellation notice period |
| DOCUMENT_PLAN_TYPE | CATEGORICAL_NOMINAL | Plan type |
| DOCUMENT_PLAN | CATEGORICAL_NOMINAL | Specific plan |
| DOCUMENT_ALLOTMENT | NUMERIC_DISCRETE | Document quota |
| CONTRACT_STATUS | CATEGORICAL_NOMINAL | **Key feature** — active/cancelled/expired |
| CONTRACT_DIVISION | CATEGORICAL_NOMINAL | Division |
| NEXT_RENEWAL_DATE | DATETIME (future-allowed) | Derive days-until-renewal |
| CURRENT_TERM | NUMERIC_DISCRETE | Current term length |
| BILLING_TERMINATION_DATE | DATETIME (future-allowed) | Billing end |
| ACTIVATED_DATE | DATETIME | Activation timing |
| ACTIVE_SUBSCRIPTION_PRODUCT_SELL_GROUP | CATEGORICAL_NOMINAL | Product group |

**ALLOW_FUTURE_COLUMNS**: `["CONTRACT_END_DATE", "NEXT_RENEWAL_DATE", "BILLING_TERMINATION_DATE"]`

**Aggregation windows** (for 01a-01d): 90d, 180d, 365d, all-time
- count of contracts, latest contract status, days until nearest renewal, avg contract term

---

### 6.3 subscription (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",
    "ETL_UPDATED_TIMESTAMP",
    "CONNECTION_LOOKUP_ID",
    "SPS_FOR_3PL_LOCATION_ID",
    "TRADING_PARTNER_ACCOUNT_ID",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| SUBSCRIPTION_ID | IDENTIFIER | Primary key |
| CONTRACT_ID | IDENTIFIER | Join key to contract → account |
| NET_PRICE | NUMERIC_CONTINUOUS | Subscription price |
| QUANTITY | NUMERIC_CONTINUOUS | Quantity |
| PRODUCT_ID | CATEGORICAL_NOMINAL | Product — high cardinality, may need encoding |
| SUBSCRIPTION_START_DATE | FEATURE_TIMESTAMP | Event timestamp |
| SUBSCRIPTION_END_DATE | DATETIME (future-allowed) | Planned end |
| SUBSCRIPTION_STATUS | CATEGORICAL_NOMINAL | Active/cancelled |
| IS_ACTIVE | BINARY | Active flag |
| TERMINATED_DATE | DATETIME | When terminated |
| TOTAL_DOCUMENTS | NUMERIC_DISCRETE | Document count |
| OVERAGE_RATE | NUMERIC_CONTINUOUS | Overage pricing |
| BLENDED_DOC_RATE_DROP_SHIP_DOCUMENTS | NUMERIC_CONTINUOUS | Rate |
| BLENDED_DOC_RATE_STANDARD_DOCUMENT | NUMERIC_CONTINUOUS | Rate |

**Note**: Needs pre-join with `contract` to get `ACCOUNT_ID`. Consider creating a pre-joined view.

---

### 6.4 implementation_project (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",
    "ETL_UPDATED_TIMESTAMP",
    "LAST_MODIFIED_DATE",
    "LAST_MODIFIED_BY_EMPLOYEE_ID",
    "CREATED_BY_EMPLOYEE_ID",
    "OWNER_EMPLOYEE_ID",
    "IMPLEMENTATION_ANALYST_EMPLOYEE_ID",
    "BUSINESS_ANALYST_EMPLOYEE_ID",
    "PROJECT_MANAGER_EMPLOYEE_ID",
    "SECONDARY_RESOURCE_EMPLOYEE_ID",
    # Text fields
    "IMPLEMENTATION_PROJECT_NAME",
    "DOCUMENT",
    "PROJECT_SCOPE",
    "EXEMPTED_DOCUMENTS",
    "UNIQUE_WEBFORMS_SETUP",
    # Pure identifiers
    "IMPLEMENTATION_PROJECT_ID",
    "RELATED_TO_ACCOUNT_ID",
    "RECORD_TYPE_ID",
    "OPPORTUNITY_ID",
    "PARENT_RELEASE_MANAGEMENT_PROJECT",
    "PARENT_IMPLEMENTATION_PROJECT_ID",
    "VENDOR_NUMBER",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CREATED_DATE | FEATURE_TIMESTAMP | Event timestamp |
| PROJECT_STATUS | CATEGORICAL_NOMINAL | Status |
| PROJECT_TYPE | CATEGORICAL_NOMINAL | Type |
| PROJECT_STAGE | CATEGORICAL_NOMINAL | Stage |
| SOLUTION_STATUS | CATEGORICAL_NOMINAL | Solution status |
| IS_HYBRID | BINARY | Hybrid flag |
| LEAD_SOURCE | CATEGORICAL_NOMINAL | Lead source |
| OPPORTUNITY_CAMPAIGN_SOURCE | CATEGORICAL_NOMINAL | Campaign |
| NII_TEAM | CATEGORICAL_NOMINAL | NII team |
| SPECIALTY_TEAM | CATEGORICAL_NOMINAL | Team |
| PORTFOLIO | CATEGORICAL_NOMINAL | Portfolio |
| FI_TEAM_PICK_LIST | CATEGORICAL_NOMINAL | FI team |
| FI_TEAM | CATEGORICAL_NOMINAL | FI team |
| WORK_UNIT_TYPE | CATEGORICAL_NOMINAL | Work type |
| SUPPLIER_VAN | CATEGORICAL_NOMINAL | Supplier |
| DIRECT_EDI_MIGRATION_PROJECT | BINARY | Migration flag |
| PRODUCTION_DATE | DATETIME | Derive: days to production |
| PRODUCTION_READY_DATE | DATETIME | Derive: days to production ready |
| PROJECT_START_DATE | DATETIME | Derive: project duration |
| SETUP_COMPLETE_DATE | DATETIME | Derive: setup duration |
| ON_HOLD_DATE | DATETIME | Project delays |
| CANCELLED_DATE | DATETIME | Cancellation signal |
| TARGETED_IMPLEMENTATION_DATE | DATETIME (future-allowed) | Target date |
| SOW_TARGETED_IMPLEMENTATION_DATE | DATETIME (future-allowed) | SOW target |
| INITIAL_TARGETED_IMPLEMENTATION_DATE | DATETIME (future-allowed) | Initial target |
| REPORTING_PRODUCTION_DATE | DATETIME | Reporting date |

---

### 6.5 case (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",
    "ETL_UPDATED_TIMESTAMP",
    "LAST_MODIFIED_DATE",
    "LAST_MODIFIED_BY_EMPLOYEE_ID",
    "CREATED_BY_EMPLOYEE_ID",
    "OWNER_EMPLOYEE_ID",
    "SPS_REQUESTOR_EMPLOYEE_ID",
    # Text fields
    "CASE_SUBJECT",
    "DESCRIPTION",
    "CANCEL_COMMENTS_DSAT",
    "CANCEL_COMMENTS_GENERAL",
    "FUTURE_SERVICE_MANAGEMENT_COMMENTS",
    "PRIORITY_CHANGE_COMMENTS",
    "TRADING_PARTNERS",
    "ERP_ACCT_APP_POS",
    # Pure identifiers
    "CASE_NUMBER",
    "CASE_REFERENCE",
    "SUB_REFERENCE",
    "RELATED_TO_ACCOUNT_ID",
    "RECORD_TYPE_ID",
    "CONTACT_ID",
    "IMPLEMENTATION_PROJECT_ID",
    "ESCALATED_ISSUE_NUMBER",
    "PROJECT_STATUS_SNAPSHOT",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| CASE_ID | IDENTIFIER | Primary key |
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CREATED_DATE | FEATURE_TIMESTAMP | Event timestamp |
| CLOSED_DATE | DATETIME | Derive: resolution time |
| FIRST_ASSIGNED_DATE_TIME | DATETIME | Derive: assignment delay |
| FIRST_CLOSED_DATE_TIME | DATETIME | Derive: first resolution time |
| RESPONSE_DATE_TIME | DATETIME | Derive: response time |
| REOPENED_DATE_TIME | DATETIME | Reopening signal |
| CSAT_SURVEY_SENT_DATE | DATETIME | Survey timing |
| ORIGIN | CATEGORICAL_NOMINAL | Channel |
| CASE_STATUS | CATEGORICAL_NOMINAL | Status |
| CASE_TYPE | CATEGORICAL_NOMINAL | Type |
| DETAIL_TAGS | CATEGORICAL_NOMINAL | Tags (may need parsing) |
| RECORD_TYPE_NAME | CATEGORICAL_NOMINAL | **Key**: retention/cancellation/optimization/Customer Ops |
| IS_CLOSED | BINARY | Closed flag |
| CASE_AGE_HOURS | NUMERIC_CONTINUOUS | Duration |
| ECONOMIC_FACTORS_EXCEPTION | BINARY | Economic exception |
| CANCELLATION_REASON | CATEGORICAL_NOMINAL | **Key churn signal** |
| CANCELLATION_REASON_CATEGORY | CATEGORICAL_NOMINAL | **Key churn signal** |
| OUTCOME | CATEGORICAL_NOMINAL | Case outcome |
| FUTURE_SERVICE_MANAGEMENT | CATEGORICAL_NOMINAL | Future intent |
| SEVERITY | CATEGORICAL_NOMINAL | Severity |
| CSAT_SURVEY_SENT | BINARY | CSAT sent flag |
| CASE_REOPEN_COUNT | NUMERIC_DISCRETE | Reopening count |
| ACCOUNT_CS_SEGMENT | CATEGORICAL_NOMINAL | CS segment at case time |
| CUSTOMER_PHASE | CATEGORICAL_NOMINAL | Customer lifecycle phase |

**High-value aggregations** (01a-01d):
- Count of cases by RECORD_TYPE_NAME (especially retention + cancellation)
- Avg CASE_AGE_HOURS, count of reopens, avg response time
- Windows: 30d, 90d, 180d, 365d

---

### 6.6 case_history (Event Table)

**Recommendation for v1: SKIP or pre-aggregate.**

This is a very granular field-level change log. If used, pre-aggregate to:
- Number of status changes per case
- Time between changes
- Then join to case → account

If skipping: remove from `datasets` dict in NB00.

---

### 6.7 opportunity (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",  # opportunity has no ETL_ cols but check
    "LAST_MODIFIED_DATE",
    "LAST_MODIFIED_BY_EMPLOYEE_ID",
    "CREATED_BY_EMPLOYEE_ID",
    "OWNER_ID",
    # Text fields
    "OPPORTUNITY_NAME",
    "CUSTOMER_ASK",
    "RATIONALE_FOR_APPROVAL",
    "COMP_KILL_MONTHLY_OPTION",
    # Identifiers
    "RECORD_TYPE_ID",
    "CAMPAIGN_ID",
    "PRIMARY_CONTACT_ID",
    "TASK_LAST_MODIFIED_TIMESTAMP",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| OPPORTUNITY_ID | IDENTIFIER | Primary key |
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CREATED_DATE | FEATURE_TIMESTAMP | Event timestamp |
| OPPORTUNITY_STATUS | CATEGORICAL_NOMINAL | Status |
| MONTHLY_FEE_CHANGE_TYPE_FROM | CATEGORICAL_NOMINAL | Fee change from |
| MONTHLY_FEE_CHANGE_TYPE_TO | CATEGORICAL_NOMINAL | Fee change to |
| ESTIMATED_CLOSE_DATE | DATETIME (future-allowed) | Forecast close |
| ESTIMATED_ONE_TIME_FEES | NUMERIC_CONTINUOUS | Estimated fees |
| ESTIMATED_NET_MONTHLY_RECURRING | NUMERIC_CONTINUOUS | Estimated MRR |
| SALE_MADE | BINARY | Won/lost |
| SALE_DATE | DATETIME | Sale date |
| BOOKINGS_MONTHLY_RECURRING | NUMERIC_CONTINUOUS | Booked MRR |
| BOOKINGS_ONE_TIME_FEES | NUMERIC_CONTINUOUS | Booked fees |
| USD_BOOKINGS_MONTHLY_RECURRING | NUMERIC_CONTINUOUS | USD MRR |
| USD_BOOKINGS_ONE_TIME_FEES | NUMERIC_CONTINUOUS | USD fees |
| LIFT | NUMERIC_CONTINUOUS | Revenue lift |
| CLOSED | BINARY | Closed flag |
| SUBMITTED_DATE | DATETIME | Submission date |
| TESTING_FEE | NUMERIC_CONTINUOUS | Testing fee |
| PRIMARY_PO_QUANTITY | NUMERIC_DISCRETE | PO quantity |
| PRIMARY_PO_VALUE | NUMERIC_CONTINUOUS | PO value |
| PO_VALUE_RANGE | CATEGORICAL_ORDINAL | PO range bucket |
| PRIMARY_SKU_COUNT | NUMERIC_DISCRETE | SKU count |
| ESCALATION_DATE | DATETIME | Escalation signal |
| ESCALATION_RESOLUTION_DATE | DATETIME | Resolution timing |
| COMPETITIVE_KILL | BINARY | Competitive win |
| COMMISSIONABLE_ARR | NUMERIC_CONTINUOUS | ARR |
| USD_COMMISSIONABLE_ARR | NUMERIC_CONTINUOUS | USD ARR |
| NON_COMMISSIONABLE_ARR | NUMERIC_CONTINUOUS | Non-comm ARR |
| USD_NON_COMMISSIONABLE_ARR | NUMERIC_CONTINUOUS | USD non-comm ARR |
| FIRST_SUBSCRIPTION_SALE | BINARY | First sale flag |
| IS_OPEN | BINARY | Open flag |
| CLOSED_WON_DECISIONS | NUMERIC_DISCRETE | Won count |
| CONTRACT_EFFECTIVE_TIMING | CATEGORICAL_NOMINAL | Timing |
| ESTIMATED_MRR_REDUCTION_AMOUNT | NUMERIC_CONTINUOUS | MRR reduction — **key churn signal** |
| CURRENCY_KEY | DROP or CATEGORICAL | Currency indicator |

---

### 6.8 opportunity_product (Event Table)

**Recommendation for v1: SKIP.** The `opportunity` table already captures revenue at the opportunity level. The line-item detail adds complexity (indirect join) with marginal signal gain.

If you want product-level features later, pre-join with opportunity to get ACCOUNT_ID.

---

### 6.9 request (Event Table)

**DROP_COLUMNS**:
```python
DROP_COLUMNS = [
    "ETL_CREATED_TIMESTAMP",  # check if present
    "LAST_MODIFIED_DATE",
    "LAST_MODIFIED_BY_EMPLOYEE_ID",
    "CREATED_BY_EMPLOYEE_ID",
    "ASSIGNED_TO_EMPLOYEE_ID",
    "CUSTOMER_RELIEF_OWNER_EMPLOYEE_ID",
    "SALES_REPRESENTATIVE_EMPLOYEE_ID",
    "SALES_DIRECTOR_EMPLOYEE_ID",
    "VENDOR_CONTACT_ID",
    # Identifiers
    "REQUEST_ID",
    "RECORD_TYPE_ID",
    "CAMPAIGN_ID",
    "OPPORTUNITY_ID",
    "REQUEST_NAME",
    # Finance codes
    "FINANCE_NAV_CUSTOMER_CODE",
    "FINANCE_CUSTOMER_CODE_1",
    "FINANCE_CUSTOMER_CODE_2",
    "FINANCE_CUSTOMER_CODE_3",
    # Text fields
    "TRADING_PARTNERS_TEXT",
    "RECENT_DISPUTES_ISSUES_COMMENTS",
    "CUSTOMER_ASK",
    "RATIONALE_FOR_APPROVAL",
    "RECENT_DISPUTES_ISSUES",
    "PRODUCTS_AFFECTED",
    "SERVICES_PURCHASED",
    "APPROVED_PRODUCTS",
    "PRODUCTS_FOR_SUSPENSION_HOLD",
]
```

**KEEP as features**:
| Column | Expected Type | Notes |
|--------|---------------|-------|
| ACCOUNT_ID | IDENTIFIER | Entity key |
| CREATED_DATE | FEATURE_TIMESTAMP | Event timestamp |
| CREDIT_REQUEST_REASON_CODE | CATEGORICAL_NOMINAL | **Churn signal** |
| CREDIT_TYPE | CATEGORICAL_NOMINAL | Credit type |
| REQUEST_TYPE | CATEGORICAL_NOMINAL | Request type |
| REQUEST_STATUS | CATEGORICAL_NOMINAL | Status |
| CUSTOMER_RELIEF_BUCKET | CATEGORICAL_NOMINAL | Relief category |
| PAYMENT_TERMS | CATEGORICAL_NOMINAL | Payment terms |
| CONTRACT_TERM | CATEGORICAL_NOMINAL | Term |
| APPROVAL_PERIOD | CATEGORICAL_NOMINAL | Approval period |
| PRIOR_PERIOD_INVOICE_UNPAID | CATEGORICAL_NOMINAL | Unpaid invoices — **churn signal** |
| EXTENSION_TO_TERM | CATEGORICAL_NOMINAL | Term extension |
| CANCELLAION_SUBMITTED | CATEGORICAL_NOMINAL | **Direct churn signal** |
| RECORD_TYPE_NAME | CATEGORICAL_NOMINAL | Record type |
| BILLING_STATUS | CATEGORICAL_NOMINAL | Billing status |
| CURRENCY_CODE | CATEGORICAL_NOMINAL | Currency |
| REQUEST_EXPIRATION_DATE | DATETIME (future-allowed) | Expiration |
| TOTAL_RATE_REDUCTION | NUMERIC_CONTINUOUS | **Churn signal** — rate reduction |
| USD_TOTAL_RATE_REDUCTION | NUMERIC_CONTINUOUS | USD rate reduction |
| CURRENT_MONTHLY_PRICE | NUMERIC_CONTINUOUS | Current price |
| PROPOSED_MONTHLY_PRICE | NUMERIC_CONTINUOUS | Proposed price |
| CREDIT_AMOUNT_REQUESTED | NUMERIC_CONTINUOUS | Credit requested |
| OPPORTUNITY_CURRENCY_KEY | DROP | Currency metadata |

---

## 7. Recommended Iteration Strategy

### Phase 1 (Start Here)
Register these datasets in NB00:
1. **account** — entity table
2. **contract** — primary churn signal source
3. **case** — support/retention/cancellation cases
4. **opportunity** — revenue and sales activity
5. **request** — credit requests and cancellation signals

Skip for now: `subscription`, `case_history`, `opportunity_product`

### Phase 2 (Add Depth)
6. **subscription** — pre-join with contract to add ACCOUNT_ID
7. **implementation_project** — onboarding success signals

### Phase 3 (Add Granularity)
8. **case_history** — pre-aggregate to case-level metrics
9. **opportunity_product** — pre-join with opportunity
10. **transaction_volume** — network usage (currently commented out, needs sps_identity join)

---

## 8. Key Churn Signals to Watch For

These columns/patterns are likely the strongest predictors:

| Signal | Source | Why |
|--------|--------|-----|
| Contract approaching end with no renewal | contract | Direct churn indicator |
| Cancellation cases opened | case (RECORD_TYPE_NAME) | Pre-cancellation signal |
| Credit requests filed | request | Financial dissatisfaction |
| Rate reductions requested | request (TOTAL_RATE_REDUCTION) | Price sensitivity |
| CANCELLATION_SUBMITTED flag | request | Direct signal |
| MRR reduction opportunities | opportunity (ESTIMATED_MRR_REDUCTION_AMOUNT) | Revenue at risk |
| Case severity and reopen count | case | Service quality issues |
| Days since last project go-live | implementation_project | Engagement proxy |
| Declining ANNUAL_REVENUE | account | Business health |
| EXISTING_CUSTOMER fiscal year flags | account | Activity trend (1→0 = risk) |

---

## 9. Open Questions

1. **Target derivation**: Is there an existing churn label, or do we need to derive it from contract status? If derived, what's the definition of "churned"?
2. **Subscription join**: Should we pre-join subscription→contract to bring ACCOUNT_ID in, or skip subscription for v1?
3. **case_history**: Skip for v1, or pre-aggregate?
4. **Transaction volume**: The network data is commented out — is the sps_identity join table available? Should we include it?
5. **Multi-currency**: Several tables have both local and USD amounts. Use USD-normalized values only?
6. **High-cardinality categoricals**: PRODUCT_ID (subscription), DETAIL_TAGS (case) — drop, hash-encode, or frequency-encode?
