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
    SELECT scoring_run_id, as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
archetype_names AS (
    SELECT archetype_id, MAX(name) AS archetype_name
    FROM {catalog}.{schema}.archetype_catalog
    WHERE status = 'active'
    GROUP BY archetype_id
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
)
SELECT
    s.entity_id,
    s.churn_probability,
    s.risk_tier,
    s.value_at_risk,
    s.playbook_id,
    s.playbook_version,
    COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
    s.archetype_id,
    s.archetype_version,
    COALESCE(an.archetype_name, s.archetype_id) AS archetype_name,
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
LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
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
    COALESCE(e.eligibility_rules_prose, e.eligibility_rules_sql) AS eligibility_rules_prose,
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
WITH archetype_names AS (
    SELECT archetype_id, MAX(name) AS archetype_name
    FROM {catalog}.{schema}.archetype_catalog
    WHERE status = 'active'
    GROUP BY archetype_id
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
)
SELECT
    s.entity_id,
    s.playbook_id,
    s.playbook_version,
    COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
    s.archetype_id,
    COALESCE(an.archetype_name, s.archetype_id) AS archetype_name,
    s.churn_probability,
    s.risk_tier,
    s.holdout_stratum,
    s.holdout_seed,
    s.eligibility_policy_id,
    s.decision_policy_id,
    s.scoring_run_id,
    s.as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
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
    SELECT scoring_run_id, as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
)
SELECT
    s.playbook_id,
    s.playbook_version,
    COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
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
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
GROUP BY s.playbook_id, s.playbook_version, pn.playbook_name, s.scoring_run_id;

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
    COUNT(DISTINCT entity_id) AS distinct_entities,
    COUNT(DISTINCT archetype_id) AS distinct_archetypes,
    COUNT(DISTINCT playbook_id) AS distinct_playbooks,
    SUM(CASE WHEN recommended THEN 1 ELSE 0 END) AS recommended_rows,
    SUM(CASE WHEN is_holdout THEN 1 ELSE 0 END) AS holdout_rows,
    SUM(CASE WHEN playbook_suppressed_reason IS NOT NULL THEN 1 ELSE 0 END) AS suppressed_rows,
    AVG(churn_probability) AS mean_churn_probability
FROM {catalog}.{schema}.eligibility_snapshot
GROUP BY scoring_run_id
ORDER BY as_of_date DESC;

-- ============================================================================
-- Progressive-disclosure dashboard surface (L1 → L4)
-- ----------------------------------------------------------------------------
-- The six views above are the operational read surface. The four views below
-- power the CSM progressive-disclosure dashboard: portfolio heat-map → one
-- playbook's archetype rollup → one (playbook, archetype, risk_tier)
-- account cohort → one account's risk-score interpretation.
--
-- Every view filters to the latest scoring run only (via the `latest_run` CTE)
-- and, where account-grain, honors `is_dashboard_visible` so low-risk rows
-- that a CSM would never act on are hidden without dropping them from the
-- snapshot. `COALESCE(is_dashboard_visible, TRUE)` keeps pre-migration rows
-- visible.
-- ============================================================================

-- ============================================================================
-- 7. v_portfolio_risk_matrix (L1 — portfolio)
-- ----------------------------------------------------------------------------
-- Pre-aggregated (playbook, risk_tier) matrix for the latest scoring run.
-- Powers the landing-page heat-map, the "total value at risk by playbook"
-- bar, and the KPI cards. Computed on the full snapshot (no visibility
-- filter) so portfolio totals stay truthful — account-grain drill-downs apply
-- the visibility flag downstream.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_portfolio_risk_matrix AS
WITH latest_run AS (
    SELECT scoring_run_id
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
)
SELECT
    s.playbook_id,
    COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
    s.risk_tier,
    COUNT(*) AS eligible_count,
    SUM(CASE WHEN s.recommended THEN 1 ELSE 0 END) AS recommended_count,
    SUM(CASE WHEN s.is_holdout THEN 1 ELSE 0 END) AS holdout_count,
    SUM(CASE WHEN s.playbook_suppressed_reason IS NOT NULL THEN 1 ELSE 0 END) AS suppressed_count,
    SUM(COALESCE(s.value_at_risk, 0)) AS total_value_at_risk,
    AVG(s.churn_probability) AS mean_churn_probability,
    s.scoring_run_id
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
GROUP BY s.playbook_id, pn.playbook_name, s.risk_tier, s.scoring_run_id;

-- ============================================================================
-- 8. v_playbook_archetype_rollup (L2 — playbook drill)
-- ----------------------------------------------------------------------------
-- Pre-aggregated (playbook, archetype, risk_tier) rollup for the latest run.
-- Powers the playbook-detail page: once a CSM picks a playbook, this view
-- shows the archetype composition of its cohort, split by risk tier, with
-- counts and total value at risk. Drilling one more level filters this to a
-- chosen archetype + tier and hands off to v_eligible_all_playbooks.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_playbook_archetype_rollup AS
WITH latest_run AS (
    SELECT scoring_run_id
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
archetype_names AS (
    SELECT archetype_id, MAX(name) AS archetype_name
    FROM {catalog}.{schema}.archetype_catalog
    WHERE status = 'active'
    GROUP BY archetype_id
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
)
SELECT
    s.playbook_id,
    COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
    s.archetype_id,
    COALESCE(an.archetype_name, s.archetype_id) AS archetype_name,
    s.risk_tier,
    COUNT(*) AS account_count,
    SUM(CASE WHEN s.recommended THEN 1 ELSE 0 END) AS recommended_count,
    SUM(CASE WHEN s.is_holdout THEN 1 ELSE 0 END) AS holdout_count,
    AVG(s.churn_probability) AS mean_churn_probability,
    SUM(COALESCE(s.value_at_risk, 0)) AS total_value_at_risk,
    s.scoring_run_id
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
GROUP BY
    s.playbook_id,
    pn.playbook_name,
    s.archetype_id,
    an.archetype_name,
    s.risk_tier,
    s.scoring_run_id;

-- ============================================================================
-- 9. v_eligible_all_playbooks (L3 — account cohort)
-- ----------------------------------------------------------------------------
-- Per-account rows for the latest scoring run, one per (entity, playbook)
-- eligibility match — unlike v_ranked_at_risk_customers, NOT filtered to
-- policy_rank_among_eligible = 1. Enables honest playbook-scoped cohorts
-- (an account eligible under three playbooks appears three times so
-- drilling into any of those playbooks sees its full cohort).
--
-- Row cap: per (playbook, archetype, risk_tier) slice, only the top 500
-- rows by expected_loss = churn_probability × value_at_risk are kept. This
-- is a defense-in-depth guardrail against the dashboard rendering millions
-- of rows — the SnapshotConfig.dashboard_min_churn_probability upstream cut
-- is the real fix. 500 is large enough that the cap is a no-op for typical
-- slices and only bites on pathological cohorts.
--
-- Visibility: respects is_dashboard_visible with COALESCE(..., TRUE) so
-- rows written before the flag existed still surface.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_eligible_all_playbooks AS
WITH latest_run AS (
    SELECT scoring_run_id
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
archetype_names AS (
    SELECT archetype_id, MAX(name) AS archetype_name
    FROM {catalog}.{schema}.archetype_catalog
    WHERE status = 'active'
    GROUP BY archetype_id
),
playbook_names AS (
    SELECT playbook_id, MAX(name) AS playbook_name
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
),
filtered AS (
    SELECT
        s.entity_id,
        s.playbook_id,
        s.playbook_version,
        COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
        s.archetype_id,
        COALESCE(an.archetype_name, s.archetype_id) AS archetype_name,
        s.risk_tier,
        s.churn_probability,
        s.value_at_risk,
        COALESCE(s.churn_probability, 0.0) * COALESCE(s.value_at_risk, 0.0) AS expected_loss,
        s.policy_rank_among_eligible,
        s.priority_rank_within_cohort,
        s.eligible_playbook_count,
        s.eligibility_evidence,
        s.top_shap_features,
        s.recommended,
        s.is_holdout,
        s.playbook_suppressed_reason,
        s.scoring_run_id,
        s.as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot s
    JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
    LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
    LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
    WHERE COALESCE(s.is_dashboard_visible, TRUE) = TRUE
),
ranked AS (
    SELECT
        filtered.*,
        ROW_NUMBER() OVER (
            PARTITION BY playbook_id, archetype_id, risk_tier
            ORDER BY expected_loss DESC, churn_probability DESC
        ) AS slice_rank
    FROM filtered
)
SELECT
    entity_id,
    playbook_id,
    playbook_version,
    playbook_name,
    archetype_id,
    archetype_name,
    risk_tier,
    churn_probability,
    value_at_risk,
    expected_loss,
    policy_rank_among_eligible,
    priority_rank_within_cohort,
    eligible_playbook_count,
    eligibility_evidence,
    top_shap_features,
    recommended,
    is_holdout,
    playbook_suppressed_reason,
    scoring_run_id,
    as_of_date,
    slice_rank
FROM ranked
WHERE slice_rank <= 500;

-- ============================================================================
-- 10. v_account_explanation (L4 — per-account interpretation)
-- ----------------------------------------------------------------------------
-- Per-account detail with archetype + playbook context joined in. Powers the
-- account-detail page: one row per (entity, playbook) with the account's
-- SHAP / evidence AND the archetype's rationale / feature thresholds AND
-- the playbook's description / expected uplift. A single query populates
-- the entire "why is this customer at risk" panel.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_explanation AS
WITH latest_run AS (
    SELECT scoring_run_id
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
active_archetypes AS (
    SELECT
        archetype_id,
        MAX(name) AS archetype_name,
        MAX(description) AS archetype_description,
        MAX(rationale) AS archetype_rationale,
        -- feature_thresholds and top_shap_features are MAP/ARRAY<STRUCT> which
        -- Spark cannot MAX() — use ANY_VALUE() so one active row per archetype_id
        -- wins deterministically (active rows are already filtered in WHERE).
        ANY_VALUE(feature_thresholds) AS archetype_feature_thresholds,
        ANY_VALUE(top_shap_features) AS archetype_top_shap_features,
        MAX(cluster_mean_churn_probability) AS archetype_mean_churn_probability,
        MAX(cluster_size) AS archetype_cluster_size
    FROM {catalog}.{schema}.archetype_catalog
    WHERE status = 'active'
    GROUP BY archetype_id
),
active_playbooks AS (
    SELECT
        playbook_id,
        MAX(name) AS playbook_name,
        MAX(description) AS playbook_description
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
),
active_policies AS (
    SELECT
        playbook_id,
        MAX(eligibility_rules_sql) AS eligibility_rules_sql,
        MAX(expected_uplift_pct) AS expected_uplift_pct,
        MAX(rationale) AS policy_rationale
    FROM {catalog}.{schema}.eligibility_policy
    WHERE status = 'active'
    GROUP BY playbook_id
)
SELECT
    s.entity_id,
    s.playbook_id,
    COALESCE(p.playbook_name, s.playbook_id) AS playbook_name,
    p.playbook_description,
    s.archetype_id,
    COALESCE(a.archetype_name, s.archetype_id) AS archetype_name,
    a.archetype_description,
    a.archetype_rationale,
    a.archetype_feature_thresholds,
    a.archetype_top_shap_features,
    a.archetype_mean_churn_probability,
    a.archetype_cluster_size,
    s.risk_tier,
    s.churn_probability,
    s.value_at_risk,
    COALESCE(s.churn_probability, 0.0) * COALESCE(s.value_at_risk, 0.0) AS expected_loss,
    s.eligibility_evidence,
    s.top_shap_features AS account_top_shap_features,
    pol.eligibility_rules_sql,
    pol.expected_uplift_pct,
    pol.policy_rationale,
    s.recommended,
    s.is_holdout,
    s.policy_rank_among_eligible,
    s.eligible_playbook_count,
    s.scoring_run_id,
    s.as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
LEFT JOIN active_archetypes a ON s.archetype_id = a.archetype_id
LEFT JOIN active_playbooks p ON s.playbook_id = p.playbook_id
LEFT JOIN active_policies pol ON s.playbook_id = pol.playbook_id
WHERE COALESCE(s.is_dashboard_visible, TRUE) = TRUE;

-- ============================================================================
-- 11. v_run_context (app masthead — project-context projection)
-- ----------------------------------------------------------------------------
-- Single-row view exposing the pieces of ``ProjectContext`` the Databricks App
-- needs to build the title bar (horizon, model type, objective, posture) plus
-- the anchor tuple of the latest scoring run. Written to the ``run_context``
-- Delta table by c05 alongside the snapshot build — here we simply pick the row
-- with the most recent ``as_of_date`` so consumers never need to know about
-- history. Fields are nullable — if the writer ran without a ``ProjectContext``
-- to source, the app falls back to a minimal title.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_run_context AS
WITH latest AS (
    SELECT * FROM {catalog}.{schema}.run_context
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.run_context)
)
SELECT
    scoring_run_id,
    as_of_date,
    model_name,
    model_version,
    model_type,
    horizon_days,
    prediction_horizons,
    primary_objective,
    temporal_posture,
    project_name,
    project_run_id,
    target_dataset,
    entity_column,
    dataset_names,
    written_at
FROM latest
ORDER BY written_at DESC
LIMIT 1;
