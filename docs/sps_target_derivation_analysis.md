# SPS Target & Label Timestamp Derivation

## 1. Cancellation Process Overview

From the cancellation journey diagram, churn at SPS follows a multi-stage process:

```
Customer reaches out ──→ Retention Case ──→ Cancel Form ──→ Cancellation Case ──→ ERP Action ──→ Case Closed ──→ Services Off
     (intent)             (pre-cancel)     (submission)     (RECORD_TYPE_NAME     (contract      (backoffice     (after
                                                             = 'cancellation')     status         delay up to    CONTRACT_END_DATE)
                                                                                   updated,       15 days)
                                                                                   future end
                                                                                   date set)
```

Key characteristics:
- **Retention case** (RECORD_TYPE_NAME = 'retention'): Pre-cancellation. Customer is at risk but hasn't formally cancelled. Many accounts are saved here.
- **Cancellation case** (RECORD_TYPE_NAME = 'cancellation'): Formal cancellation submitted. This is the "point of no return" in the CRM.
- **Contract status change**: For full cancels, the entire contract moves to cancel status with a future end date. For partial cancels, individual contract lines get a future end date.
- **Backoffice delay**: ERP processing can take up to 15 days; the queue often backs up.
- **Service deactivation**: Happens after CONTRACT_END_DATE passes.

## 2. Data Model: Cancellations

### Semantic Layer Distinction
- **Cancellations "Expected"**: Cancellation actioned but the official publish date (effective date) hasn't arrived yet. Cancel type is known.
- **Cancellations "Official"**: Effective date has passed. Attributes: Cancel Type (full vs partial), Controlled (in-control vs out-of-control), Cancel Source (manual vs system), Cancel Reason.

### Available Data (Salesforce tables)

| Table | Role in Churn | Key Columns |
|-------|---------------|-------------|
| `case` | Cancellation/retention case lifecycle | RECORD_TYPE_NAME, CREATED_DATE, CLOSED_DATE, IS_CLOSED, CASE_STATUS |
| `contract` | Contract status determines if account is still active | CONTRACT_STATUS, CONTRACT_END_DATE, BILLING_TERMINATION_DATE, CONTRACT_START_DATE |
| `subscription` | Subscription-level termination | IS_ACTIVE, TERMINATED_DATE, SUBSCRIPTION_STATUS |

### Fields Not Available in Raw Salesforce
- Cancel Type (full vs partial) as an explicit field on the case
- Cancel Source (manual vs system)
- SPS Control flag

These attributes exist in the gold/semantic layer (`Cancellations Official`) but we derive from raw Salesforce tables.

## 3. Churn Scenarios

### Scenario A: Full Voluntary Cancellation (Case-Based)
- Customer initiates cancellation
- Retention case may or may not be created first
- **Cancellation case created** (RECORD_TYPE_NAME = 'cancellation')
- All contracts for the account are cancelled/terminated
- Case is eventually closed (IS_CLOSED = 1)
- **Result**: Account has NO active contracts → **churned = 1**

### Scenario B: Partial Cancellation
- Customer cancels some product lines/contracts but retains others
- Cancellation case created and closed
- Some contracts terminated, but at least one remains active
- **Result**: Account still has active contracts → **churned = 0**
- (The partial cancel is a risk signal, not a churn event)

### Scenario C: Non-Renewal (Contract Expiry, No Case)
- Contract reaches end date, customer simply doesn't renew
- No cancellation case in Salesforce (customer never formally initiated)
- All contracts pass their CONTRACT_END_DATE without renewal
- **Result**: Account has NO active contracts → **churned = 1**

### Scenario D: Involuntary Churn (System-Initiated)
- Non-payment, compliance issues, or other system-triggered cancellation
- Cancel Source would be "system" in the semantic layer
- May or may not have a Salesforce cancellation case
- Contracts are terminated
- **Result**: Account has NO active contracts → **churned = 1**

### Scenario E: Renewal Gap (False Positive Risk)
- All contracts have expired but the account is in a renewal negotiation
- Temporarily has no active contracts
- Will eventually get a new contract
- **Result**: Should NOT be classified as churned

## 4. Target Column (`churned`) — Definition

### Principle
The ground truth of churn is **contract status**: an account is churned when it has no remaining active contracts and was previously a customer.

### Definition

```
churned = 1 when:
  (1) Account has at least one contract in contract table (was a customer)
  AND
  (2) Account has ZERO contracts with CONTRACT_STATUS = 'Active'
```

The target is based solely on contract status, not on cancellation cases. This correctly handles all scenarios:
- **Scenario A** (full cancel + case): All contracts non-active → caught by rule (2)
- **Scenario B** (partial cancel): Still has active contracts → churned = 0
- **Scenario C** (non-renewal): All contracts expired → caught by rule (2)
- **Scenario D** (involuntary): All contracts terminated → caught by rule (2)
- **Scenario E** (renewal gap): **False positive risk** — see mitigation below

Cancellation cases are not part of the target definition. An account with a closed partial cancellation case that still has active contracts remains churned = 0. Case data informs the **label timestamp**, not the target.

### Mitigation for Renewal Gaps (Scenario E)

Option 1: **Grace period filter** — only mark as churned if `MAX(CONTRACT_END_DATE)` is more than N days ago (e.g., 90 days). Accounts in active renewal negotiations within the grace period are excluded.

Option 2: **Require supporting evidence** — in addition to no active contracts, require at least ONE of:
  - A closed cancellation case exists
  - `MAX(CONTRACT_END_DATE)` is > 90 days in the past
  - All contracts have a terminal status (e.g., 'Cancelled', 'Terminated', not just 'Expired')

Option 3: **No mitigation** — accept some noise from renewal gaps. In practice, the 90-day prediction horizon and temporal split provide natural buffering.

## 5. Label Timestamp (`churn_date`) — Definition

### What label_timestamp means in the temporal framework

`label_timestamp` = the date when the churn event occurred / became knowable. The temporal framework uses this to:
- Assign labels at each as_of_date: if `churn_date <= as_of_date`, the label is known
- Prevent feature leakage: only features observed before `churn_date` are used

### Candidate Timestamps

| Candidate | Source | When in Process | Pros | Cons |
|-----------|--------|-----------------|------|------|
| Cancellation case `CREATED_DATE` | case table | Cancellation initiated | Earliest reliable CRM signal; actionable for retention | Not available for non-case churn (Scenario C/D) |
| Cancellation case `CLOSED_DATE` | case table | Backoffice processing done | Cancellation fully processed | Up to 15-day processing delay adds noise; not available for non-case churn |
| `CONTRACT_END_DATE` | contract table | Service termination date | Available for all churned accounts; business-meaningful | Set in the future at time of cancellation; mixes contractual timing with churn risk |
| `BILLING_TERMINATION_DATE` | contract table | Billing stops | Financially meaningful | Often NULL; lags behind cancellation decision |
| `TERMINATED_DATE` | subscription table | Subscription terminated | Subscription-level precision | Subscription-level, not account-level; often NULL |

### Two Label Timestamp Strategies

There are two valid interpretations of "when did churn happen," each suited to
a different modelling question. The code implements both columns and selects one
via a toggle.

#### Strategy A — Decision Date (default)

Uses the date when the cancellation was initiated in the CRM — i.e., when the
decision to leave was entered into the system.

```
churn_date =
  (1) If cancellation case(s) exist:
      MAX(case.CREATED_DATE) where RECORD_TYPE_NAME = 'cancellation'

  (2) Else if contracts have end dates:
      MAX(contract.CONTRACT_END_DATE) for the account

  (3) Else fallback:
      MAX(contract.BILLING_TERMINATION_DATE) for the account
```

**Predicts:** "Will this account initiate a cancellation in the next N days?"
Best for retention intervention — the team can act before the effective date.

#### Strategy B — Effective Cancellation Date (commented alternative)

Uses the contractual date when the cancellation actually takes effect — the
future date set during ERP processing when services are deactivated.

```
churn_date =
  (1) MAX(contract.CONTRACT_END_DATE) for churned accounts
      (this is the date set when cancellation is actioned in ERP)

  (2) Else fallback:
      MAX(contract.BILLING_TERMINATION_DATE) for the account
```

**Predicts:** "Will this account's services be deactivated in the next N days?"
Best for revenue/financial modelling — aligns with when revenue actually stops.

#### When to choose which

| Use Case | Strategy | Rationale |
|----------|----------|-----------|
| Retention intervention | A (Decision Date) | Gives the team maximum lead time before effective date |
| Revenue forecasting | B (Effective Date) | Aligns with when MRR/ARR impact materialises |
| Model comparison | Both | Train both, compare lift — Decision Date typically gives a cleaner signal |

Note: Strategy B can place `churn_date` **in the future** relative to the
cancellation case creation. The temporal framework handles this correctly — at
any as_of_date before that future date, the label is "not yet known," so the
account is excluded from the positive class for that snapshot. This means
Strategy B produces fewer positive training examples for recent snapshots
(cancellations that are actioned but not yet effective are invisible).

### Why MAX(CREATED_DATE)

MAX is used instead of MIN because an account may have multiple cancellation cases if they had partial cancels before a full cancel:
- Case 1 (partial cancel, 12 months ago): Cancelled 1 of 3 contracts
- Case 2 (full cancel, 2 months ago): Cancelled remaining 2 contracts

Using MIN would set `churn_date` 12 months ago, even though the account was still active for 10 more months. Features between the partial and full cancel would be incorrectly treated as "post-churn."

MAX gives the date of the **final cancellation** — the one that made the account fully churned. Features before this date are truly pre-churn behavioral data.

### Why CREATED_DATE (not CLOSED_DATE)

| Aspect | CREATED_DATE | CLOSED_DATE |
|--------|-------------|-------------|
| Timing | When customer submitted cancellation | When backoffice finished processing |
| Delay | None (real-time entry into CRM) | Up to 15 days (queue backlog) |
| Business meaning | "Customer decided to leave" | "Paperwork was completed" |
| Prediction target | "Will a cancellation be initiated?" | "Will a cancellation be processed?" |
| Feature purity | Features before this date are pre-decision | Features include up to 15 days of post-decision data |

CREATED_DATE reflects the actual churn decision moment, not backoffice processing. Features prior to this date are truly pre-churn behavioral signals, and there is no noise from variable backoffice processing delays.

### Why CONTRACT_END_DATE is a fallback (not primary)

CONTRACT_END_DATE is set **at the time of cancellation** (or contract creation) and is typically a future date. Using it as the primary label timestamp would mean:
- A cancellation initiated in January with a contract end in June would have `churn_date = June`
- The 5 months between January and June would be treated as "pre-churn" features
- But the customer had already decided to leave in January — those 5 months of features are "in-churn" data

CONTRACT_END_DATE is used as a fallback for accounts that churned without a cancellation case (non-renewal scenario).

## 6. Implementation

```python
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

## 7. Edge Cases & Design Decisions

### IS_CLOSED filter on cancellation cases

The target (churned flag) does not depend on cases at all — it is purely contract-based. For the label timestamp, both open and closed cancellation cases are included:
- A cancellation case that's open but the contract is already non-active → the case CREATED_DATE is still the best label timestamp
- A cancellation case that's open and contracts are still active → account is not churned anyway (churned=0), so churn_date is irrelevant

No IS_CLOSED filter is applied to the case date query. This widens the pool of label timestamps and catches cases where backoffice is slow.

### Retention cases (RECORD_TYPE_NAME = 'retention')

Retention cases are not used for label timestamps. They are pre-cancellation risk signals, and many don't lead to churn. Using them would:
- Create inconsistency (some churners have retention dates, others don't)
- Set the label too early (the account might have been retained for months before the actual cancel)

Retention case data is valuable as a **feature** (e.g., "had a retention case in the last 90 days"), not as a label timestamp.

### Accounts with no contracts

Some accounts in the account table may never have had a contract (prospects, test accounts, etc.). These are **excluded from the churn population** — they are neither churned nor retained.

The rule `_has_contracts.subtract(_has_active_contracts)` handles this: accounts with no contracts aren't in `_has_contracts`, so they won't appear in `_churned_ids`. They'll have `churned = 0` (from the left join + coalesce). If needed, an explicit filter in sampling (NB00 section 0.12) can exclude non-customer accounts.

### Multiple active contracts per account

From the data (table_descriptions.md), accounts commonly have multiple active contracts:
```
active_contract_count | num_accounts
1                     | 36,133
2                     | 5,207
3                     | 1,179
...
```

The contract-based target handles this correctly: `CONTRACT_STATUS = 'Active'` on ANY contract means the account is not churned. Only when ALL contracts are non-active is the account considered churned.

### Interaction with the temporal framework

The temporal framework uses `churn_date` as `label_timestamp_column` in the `TimestampConfig`:

```python
TimestampConfig(
    strategy="production",
    feature_timestamp_column="LAST_MODIFIED_DATE",  # on account
    label_timestamp_column="churn_date",             # derived column
)
```

For each snapshot at as_of_date T:
- If `churn_date <= T`: label is known, account is in the training set with churned=1
- If `churn_date > T` or `churn_date is NULL` (not churned): label is churned=0
- Features are computed using only data where `event_timestamp <= T`

This produces clean point-in-time training examples.
