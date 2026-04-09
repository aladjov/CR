-- Dashboard SQL views for the customer-retention causal track.
--
-- These six views are the read surface that the Databricks AI/BI dashboard
-- consumes. They live as a single .sql file (rather than per-view files) so
-- the c04 notebook can publish them in one execute() pass and the four-way
-- definition anchor stays in lock-step across views.
--
-- All views are CREATE OR REPLACE — re-publishing on every snapshot run is
-- a no-op when the underlying schema hasn't changed.
--
-- Placeholders {catalog} and {schema} are substituted by the publisher
-- before submitting to Spark SQL. The c04 notebook does the substitution
-- with str.format(catalog=CATALOG, schema=SCHEMA) before splitting on the
-- statement separator.

-- ============================================================================
-- 1. v_ranked_at_risk_customers
-- ----------------------------------------------------------------------------
-- Per-customer ranked at-risk view with the four-way anchor and the highest-
-- ranked playbook recommendation. Powers the "ranked at-risk customers"
-- dashboard widget — the primary CSM-facing surface. Filters to the most
-- recent scoring run only and to recommended (not held out, not suppressed)
-- rows.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_ranked_at_risk_customers AS
WITH latest_run AS (
    SELECT scoring_run_id, MAX(as_of_date) AS as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot
    GROUP BY scoring_run_id
    ORDER BY as_of_date DESC
    LIMIT 1
)
SELECT
    s.account_id,
    s.churn_probability,
    s.risk_tier,
    s.value_at_risk,
    s.playbook_id,
    s.playbook_version,
    s.archetype_id,
    s.archetype_version,
    s.eligibility_policy_id,
    s.eligibility_policy_version,
    s.decision_policy_id,
    s.model_name,
    s.model_version,
    s.policy_rank_among_eligible,
    s.priority_rank_within_cohort,
    s.eligible_playbook_count,
    s.eligible_playbooks_set,
    s.eligibility_evidence,
    s.top_shap_features,
    s.scoring_run_id,
    s.as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
WHERE s.recommended = TRUE
  AND s.policy_rank_among_eligible = 1;

-- ============================================================================
-- 2. v_archetype_overview
-- ----------------------------------------------------------------------------
-- Active archetype catalog joined to its top SHAP drivers and cluster sizes.
-- Powers the "archetypes in production" widget that explains what each
-- archetype represents and how many customers it covers.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_archetype_overview AS
SELECT
    a.archetype_id,
    a.archetype_version,
    a.model_name,
    a.model_version,
    a.derivation_run_id,
    a.derivation_method,
    a.cluster_size,
    a.cluster_mean_churn_probability,
    a.top_shap_features,
    a.feature_thresholds,
    a.name AS archetype_name,
    a.description AS archetype_description,
    a.rationale,
    a.llm_model_id,
    a.stability_vs_prior_version,
    a.status,
    a.approved_by,
    a.approved_at,
    a.valid_from,
    a.valid_to
FROM {catalog}.{schema}.archetype_catalog a
WHERE a.status = 'active';

-- ============================================================================
-- 3. v_playbook_eligibility_rules
-- ----------------------------------------------------------------------------
-- Active eligibility policies joined to their playbook descriptions and
-- archetype labels. Powers the "why was this customer surfaced?" drill-in
-- on the dashboard. Renders the eligibility predicate as a SQL string for
-- the inline explanation.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_playbook_eligibility_rules AS
SELECT
    e.eligibility_policy_id,
    e.version AS eligibility_policy_version,
    e.playbook_id,
    e.playbook_version,
    p.name AS playbook_name,
    p.description AS playbook_description,
    e.archetype_ids,
    e.requires_features,
    e.eligibility_rules_sql,
    e.expected_uplift_pct,
    e.rationale,
    e.llm_model_id,
    e.derivation_method,
    e.derivation_run_id,
    e.status,
    e.valid_from,
    e.valid_to
FROM {catalog}.{schema}.eligibility_policy e
LEFT JOIN {catalog}.{schema}.playbook_catalog p
       ON e.playbook_id = p.playbook_id AND e.playbook_version = p.version
WHERE e.status = 'active';

-- ============================================================================
-- 4. v_holdout_assignments
-- ----------------------------------------------------------------------------
-- Holdout audit view: every customer that was eligible but assigned to the
-- experimental holdout for the latest scoring run. Used by analytics to
-- verify holdout fractions match the policy and to support post-hoc uplift
-- estimation once outcomes accumulate.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_holdout_assignments AS
SELECT
    s.account_id,
    s.playbook_id,
    s.playbook_version,
    s.archetype_id,
    s.churn_probability,
    s.risk_tier,
    s.holdout_stratum,
    s.holdout_seed,
    s.eligibility_policy_id,
    s.decision_policy_id,
    s.scoring_run_id,
    s.as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
WHERE s.is_holdout = TRUE;

-- ============================================================================
-- 5. v_capacity_utilization
-- ----------------------------------------------------------------------------
-- Per-playbook capacity-utilization summary for the latest scoring run.
-- Powers the operations widget showing each playbook's eligible cohort
-- size, recommended count, suppressed count, and the mean rank of
-- recommended customers (lower = more headroom under cap).
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_capacity_utilization AS
WITH latest_run AS (
    SELECT scoring_run_id, MAX(as_of_date) AS as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot
    GROUP BY scoring_run_id
    ORDER BY as_of_date DESC
    LIMIT 1
)
SELECT
    s.playbook_id,
    s.playbook_version,
    COUNT(*) AS eligible_count,
    SUM(CASE WHEN s.recommended THEN 1 ELSE 0 END) AS recommended_count,
    SUM(CASE WHEN s.is_holdout THEN 1 ELSE 0 END) AS holdout_count,
    SUM(CASE WHEN s.playbook_suppressed_reason IS NOT NULL THEN 1 ELSE 0 END) AS suppressed_count,
    AVG(CASE WHEN s.recommended THEN s.priority_rank_within_cohort END) AS mean_recommended_rank,
    MAX(s.priority_rank_within_cohort) AS max_rank,
    s.scoring_run_id,
    MAX(s.as_of_date) AS as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
GROUP BY s.playbook_id, s.playbook_version, s.scoring_run_id;

-- ============================================================================
-- 6. v_run_anchor_history
-- ----------------------------------------------------------------------------
-- Historical anchor view: one row per scoring run with the four-way anchor
-- tuple plus aggregate counts. Lets dashboard users compare runs side by
-- side to detect drift in archetype activations or eligible cohort sizes
-- over time.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_run_anchor_history AS
SELECT
    scoring_run_id,
    MAX(as_of_date) AS as_of_date,
    MAX(model_name) AS model_name,
    MAX(model_version) AS model_version,
    MAX(decision_policy_id) AS decision_policy_id,
    COUNT(*) AS total_eligible_rows,
    COUNT(DISTINCT account_id) AS distinct_accounts,
    COUNT(DISTINCT archetype_id) AS distinct_archetypes,
    COUNT(DISTINCT playbook_id) AS distinct_playbooks,
    SUM(CASE WHEN recommended THEN 1 ELSE 0 END) AS recommended_rows,
    SUM(CASE WHEN is_holdout THEN 1 ELSE 0 END) AS holdout_rows,
    SUM(CASE WHEN playbook_suppressed_reason IS NOT NULL THEN 1 ELSE 0 END) AS suppressed_rows,
    AVG(churn_probability) AS mean_churn_probability
FROM {catalog}.{schema}.eligibility_snapshot
GROUP BY scoring_run_id
ORDER BY as_of_date DESC;
