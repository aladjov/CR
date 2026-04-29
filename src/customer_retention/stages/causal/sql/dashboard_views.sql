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
    COALESCE(s.top_shap_features, d.top_drivers) AS top_shap_features,
    s.scoring_run_id,
    s.as_of_date
FROM {catalog}.{schema}.eligibility_snapshot s
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
LEFT JOIN {catalog}.{schema}.top_shap_drivers d
       ON d.model_name = s.model_name
      AND d.model_version = s.model_version
      AND d.entity_id = s.entity_id
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
-- 6b. v_account_primary_recommendation (entity-grain anchor for the dashboard)
-- ----------------------------------------------------------------------------
-- One row per (scoring_run_id, entity_id) for the latest scoring run -- the
-- account's PRIMARY play (recommended-first, then policy_rank ASC, then
-- expected_loss DESC) plus an ``alternates`` array of every other play the
-- account passed eligibility for, with playbook_name / fit_score /
-- churn_probability / expected_uplift attached.
--
-- Source of truth for portfolio KPIs and every entity-grain aggregation in
-- the app. ``v_portfolio_risk_matrix`` and ``v_playbook_archetype_rollup``
-- are rewritten to source from this view so SUM(eligible_count) at any
-- partition collapses to a DISTINCT entity count instead of an
-- account-x-playbook row count.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_primary_recommendation AS
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
active_policies AS (
    SELECT
        playbook_id,
        MAX(fit_score) AS fit_score,
        MAX(expected_uplift_pct) AS expected_uplift_pct
    FROM {catalog}.{schema}.eligibility_policy
    WHERE status = 'active'
    GROUP BY playbook_id
),
joined AS (
    SELECT
        s.entity_id,
        s.playbook_id,
        s.playbook_version,
        COALESCE(pn.playbook_name, s.playbook_id) AS playbook_name,
        s.archetype_id,
        s.archetype_version,
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
        ap.fit_score,
        ap.expected_uplift_pct,
        s.eligibility_policy_id,
        s.decision_policy_id,
        s.model_name,
        s.model_version,
        s.scoring_run_id,
        s.as_of_date
    FROM {catalog}.{schema}.eligibility_snapshot s
    JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
    LEFT JOIN archetype_names an ON s.archetype_id = an.archetype_id
    LEFT JOIN playbook_names pn ON s.playbook_id = pn.playbook_id
    LEFT JOIN active_policies ap ON ap.playbook_id = s.playbook_id
    WHERE COALESCE(s.is_dashboard_visible, TRUE) = TRUE
),
ranked AS (
    SELECT
        joined.*,
        ROW_NUMBER() OVER (
            PARTITION BY entity_id
            ORDER BY
                -- Recommended plays beat eligible-but-not-recommended.
                CASE WHEN recommended THEN 0 ELSE 1 END,
                -- Lower policy_rank wins (1 = top).
                COALESCE(policy_rank_among_eligible, 2147483647) ASC,
                -- Highest expected loss as tiebreak.
                COALESCE(expected_loss, -1.0) DESC,
                -- Stable final tiebreak so the choice is deterministic.
                playbook_id ASC
        ) AS entity_rank
    FROM joined
),
alternates_collected AS (
    SELECT
        entity_id,
        COLLECT_LIST(NAMED_STRUCT(
            'playbook_id',                playbook_id,
            'playbook_name',              playbook_name,
            'risk_tier',                  risk_tier,
            'churn_probability',          churn_probability,
            'expected_loss',              expected_loss,
            'fit_score',                  fit_score,
            'expected_uplift_pct',        expected_uplift_pct,
            'policy_rank_among_eligible', policy_rank_among_eligible,
            'recommended',                recommended
        )) AS alternates
    FROM ranked
    WHERE entity_rank > 1
    GROUP BY entity_id
)
SELECT
    p.entity_id,
    p.playbook_id,
    p.playbook_version,
    p.playbook_name,
    p.archetype_id,
    p.archetype_version,
    p.archetype_name,
    p.risk_tier,
    p.churn_probability,
    p.value_at_risk,
    p.expected_loss,
    p.policy_rank_among_eligible,
    p.priority_rank_within_cohort,
    p.eligible_playbook_count,
    p.eligibility_evidence,
    p.top_shap_features,
    p.recommended,
    p.is_holdout,
    p.playbook_suppressed_reason,
    p.fit_score,
    p.expected_uplift_pct,
    p.eligibility_policy_id,
    p.decision_policy_id,
    p.model_name,
    p.model_version,
    -- Other plays the account also passed eligibility for, sorted by
    -- expected_loss DESC so the most-impactful alternate sits first.
    ARRAY_SORT(
        COALESCE(alts.alternates, ARRAY()),
        (a1, a2) ->
            CASE
                WHEN COALESCE(a1.expected_loss, -1.0) > COALESCE(a2.expected_loss, -1.0) THEN -1
                WHEN COALESCE(a1.expected_loss, -1.0) < COALESCE(a2.expected_loss, -1.0) THEN 1
                ELSE 0
            END
    ) AS alternates,
    SIZE(COALESCE(alts.alternates, ARRAY())) AS alternate_count,
    p.scoring_run_id,
    p.as_of_date
FROM ranked p
LEFT JOIN alternates_collected alts ON alts.entity_id = p.entity_id
WHERE p.entity_rank = 1;

-- ============================================================================
-- 7. v_portfolio_risk_matrix (L1 — portfolio)
-- ----------------------------------------------------------------------------
-- Per-account (primary playbook, risk_tier) matrix for the latest scoring
-- run. Each row in the source view ``v_account_primary_recommendation`` is
-- one ENTITY -- so SUM(eligible_count) over the whole view = number of
-- accounts in scope, never multi-counted.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_portfolio_risk_matrix AS
SELECT
    p.playbook_id,
    p.playbook_name,
    p.risk_tier,
    COUNT(*) AS eligible_count,
    SUM(CASE WHEN p.recommended THEN 1 ELSE 0 END) AS recommended_count,
    SUM(CASE WHEN p.is_holdout THEN 1 ELSE 0 END) AS holdout_count,
    SUM(CASE WHEN p.playbook_suppressed_reason IS NOT NULL THEN 1 ELSE 0 END) AS suppressed_count,
    SUM(COALESCE(p.value_at_risk, 0)) AS total_value_at_risk,
    AVG(p.churn_probability) AS mean_churn_probability,
    p.scoring_run_id
FROM {catalog}.{schema}.v_account_primary_recommendation p
GROUP BY p.playbook_id, p.playbook_name, p.risk_tier, p.scoring_run_id;

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
SELECT
    p.playbook_id,
    p.playbook_name,
    p.archetype_id,
    p.archetype_name,
    p.risk_tier,
    COUNT(*) AS account_count,
    SUM(CASE WHEN p.recommended THEN 1 ELSE 0 END) AS recommended_count,
    SUM(CASE WHEN p.is_holdout THEN 1 ELSE 0 END) AS holdout_count,
    AVG(p.churn_probability) AS mean_churn_probability,
    SUM(COALESCE(p.value_at_risk, 0)) AS total_value_at_risk,
    p.scoring_run_id
FROM {catalog}.{schema}.v_account_primary_recommendation p
GROUP BY
    p.playbook_id,
    p.playbook_name,
    p.archetype_id,
    p.archetype_name,
    p.risk_tier,
    p.scoring_run_id;

-- ============================================================================
-- 9. v_eligible_all_playbooks (L3 — account cohort, entity-grain)
-- ----------------------------------------------------------------------------
-- Per-account rows for the latest scoring run -- one row per ENTITY at its
-- PRIMARY play, sourced from ``v_account_primary_recommendation``. This is
-- a deliberate semantic shift from the older multi-grain version (one row
-- per (entity, playbook) eligibility match): the dashboard answers "of all
-- customers right now, which ones should the CSM act on?" and that answer
-- is per-account. Other plays an account also passed eligibility for ride
-- along on the ``alternates`` array.
--
-- Row cap: ``slice_rank`` ranks accounts within (playbook, archetype,
-- risk_tier) by expected_loss DESC then churn_probability DESC, so the
-- caller can ``WHERE slice_rank <= K`` for top-K-per-slice surfaces. SHAP
-- drivers come from ``top_shap_features`` on the snapshot row when present
-- and fall back to ``top_shap_drivers`` model-keyed rows otherwise.
--
-- Visibility honored upstream by ``v_account_primary_recommendation``.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_eligible_all_playbooks AS
WITH base AS (
    SELECT
        p.entity_id,
        p.playbook_id,
        p.playbook_version,
        p.playbook_name,
        p.archetype_id,
        p.archetype_name,
        p.risk_tier,
        p.churn_probability,
        p.value_at_risk,
        p.expected_loss,
        p.policy_rank_among_eligible,
        p.priority_rank_within_cohort,
        p.eligible_playbook_count,
        p.eligibility_evidence,
        COALESCE(p.top_shap_features, d.top_drivers) AS top_shap_features,
        p.recommended,
        p.is_holdout,
        p.playbook_suppressed_reason,
        p.alternates,
        p.alternate_count,
        p.scoring_run_id,
        p.as_of_date
    FROM {catalog}.{schema}.v_account_primary_recommendation p
    LEFT JOIN {catalog}.{schema}.top_shap_drivers d
           ON d.model_name = p.model_name
          AND d.model_version = p.model_version
          AND d.entity_id = p.entity_id
)
SELECT
    base.*,
    ROW_NUMBER() OVER (
        PARTITION BY playbook_id, archetype_id, risk_tier
        ORDER BY expected_loss DESC, churn_probability DESC
    ) AS slice_rank
FROM base
WHERE top_shap_features IS NOT NULL;

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
        MAX(description) AS playbook_description,
        MAX(time_zero_definition) AS playbook_time_zero_definition,
        MAX(analysis_population_rule) AS playbook_analysis_population_rule,
        MAX(when_applicable) AS playbook_when_applicable
    FROM {catalog}.{schema}.playbook_catalog
    GROUP BY playbook_id
),
active_policies AS (
    SELECT
        playbook_id,
        MAX(eligibility_rules_sql) AS eligibility_rules_sql,
        MAX(eligibility_rules_prose) AS eligibility_rules_prose,
        MAX(expected_uplift_pct) AS expected_uplift_pct,
        MAX(rationale) AS policy_rationale,
        MAX(fit_score) AS policy_fit_score,
        MAX(fit_tier) AS policy_fit_tier
    FROM {catalog}.{schema}.eligibility_policy
    WHERE status = 'active'
    GROUP BY playbook_id
),
exploded_policies AS (
    -- ``eligibility_policy.archetype_ids`` stores archetype VERSIONS (see
    -- ``derivation.py`` -- the policy is bound to the exact catalog row that
    -- produced it). Keep the column name explicit so the JOIN at the bottom
    -- of the view doesn't accidentally pair it with the snapshot's logical
    -- ``archetype_id`` -- which would be silent and always-empty.
    SELECT
        av AS archetype_version,
        e.playbook_id,
        e.fit_score,
        e.expected_uplift_pct
    FROM {catalog}.{schema}.eligibility_policy e
    LATERAL VIEW EXPLODE(e.archetype_ids) tbl AS av
    WHERE e.status = 'active'
      AND e.archetype_ids IS NOT NULL
),
archetype_playbook_set AS (
    SELECT
        ep.archetype_version,
        COLLECT_LIST(NAMED_STRUCT(
            'playbook_id', ep.playbook_id,
            'playbook_name', COALESCE(p.name, ep.playbook_id),
            'fit_score', ep.fit_score,
            'expected_uplift_pct', ep.expected_uplift_pct
        )) AS playbooks_all
    FROM exploded_policies ep
    LEFT JOIN {catalog}.{schema}.playbook_catalog p ON ep.playbook_id = p.playbook_id
    GROUP BY ep.archetype_version
)
SELECT
    s.entity_id,
    s.playbook_id,
    COALESCE(p.playbook_name, s.playbook_id) AS playbook_name,
    p.playbook_description,
    p.playbook_time_zero_definition,
    p.playbook_analysis_population_rule,
    p.playbook_when_applicable,
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
    COALESCE(s.top_shap_features, d.top_drivers) AS account_top_shap_features,
    pol.eligibility_rules_sql,
    pol.eligibility_rules_prose AS policy_eligibility_rules_prose,
    pol.expected_uplift_pct,
    pol.policy_rationale,
    pol.policy_fit_score,
    pol.policy_fit_tier,
    ARRAY_SORT(
        FILTER(
            COALESCE(aps.playbooks_all, ARRAY()),
            x -> x.playbook_id != s.playbook_id
        ),
        (a1, a2) ->
            CASE
                WHEN COALESCE(a1.fit_score, -1.0) > COALESCE(a2.fit_score, -1.0) THEN -1
                WHEN COALESCE(a1.fit_score, -1.0) < COALESCE(a2.fit_score, -1.0) THEN 1
                ELSE 0
            END
    ) AS alternate_playbooks,
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
LEFT JOIN archetype_playbook_set aps ON s.archetype_version = aps.archetype_version
LEFT JOIN {catalog}.{schema}.top_shap_drivers d
       ON d.model_name = s.model_name
      AND d.model_version = s.model_version
      AND d.entity_id = s.entity_id
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

-- @cr:deviation-block:open
-- ============================================================================
-- 12. v_account_feature_deviation (L4 right-pane — per-account feature z-score)
-- ----------------------------------------------------------------------------
-- One row per (entity_id, feature_name) with both the customer's feature value
-- and the population mean / stddev / quantiles.  This is the SHAP placeholder
-- backing the right-hand panel: a bidirectional bar chart of how far each
-- feature deviates from the training population.  Numeric features only —
-- categorical population stats live as JSON in feature_population_stats and
-- need a different visual.
--
-- Population stats join is "latest writer wins" by ``computed_at`` so the
-- view stays generic across model retrains: the dashboard always reflects the
-- most recent training population, not a stale derivation_run_id.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_feature_deviation AS
WITH latest_run AS (
    SELECT scoring_run_id
    FROM {catalog}.{schema}.eligibility_snapshot
    WHERE as_of_date = (SELECT MAX(as_of_date) FROM {catalog}.{schema}.eligibility_snapshot)
    LIMIT 1
),
gold_long AS (
    SELECT
        g.entity_id,
        kv_key   AS feature_name,
        kv_value AS feature_value
    FROM (
        SELECT
            entity_id,
            FROM_JSON(TO_JSON(STRUCT(*)), 'MAP<STRING,DOUBLE>') AS feature_map
        FROM {catalog}.{schema}.gold_features_{composite_name}
    ) g
    LATERAL VIEW EXPLODE(g.feature_map) tbl AS kv_key, kv_value
    WHERE kv_key NOT IN ('entity_id', 'as_of_date')
),
latest_pop AS (
    SELECT
        feature_name,
        dtype,
        mean, stddev,
        q01, q05, q25, q50, q75, q95, q99,
        ROW_NUMBER() OVER (
            PARTITION BY feature_name ORDER BY computed_at DESC
        ) AS rn
    FROM {catalog}.{schema}.feature_population_stats
    WHERE dtype = 'numeric'
)
SELECT
    s.scoring_run_id,
    s.entity_id,
    g.feature_name,
    g.feature_value,
    p.mean,
    p.stddev,
    p.q05, p.q25, p.q50, p.q75, p.q95,
    CASE WHEN p.stddev IS NOT NULL AND p.stddev > 0
         THEN (g.feature_value - p.mean) / p.stddev END AS z,
    CASE WHEN p.mean IS NOT NULL AND ABS(p.mean) > 1e-9
         THEN (g.feature_value - p.mean) / p.mean   END AS pct_dev
FROM gold_long g
JOIN latest_pop p
       ON p.feature_name = g.feature_name
      AND p.rn = 1
JOIN {catalog}.{schema}.eligibility_snapshot s
       ON s.entity_id = g.entity_id
JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
WHERE g.feature_value IS NOT NULL
  AND COALESCE(s.is_dashboard_visible, TRUE) = TRUE;

-- ============================================================================
-- 13. v_account_feature_deviation_topn (top-12 features by |z| for the panel)
-- ----------------------------------------------------------------------------
-- The dashboard renders only the top-N most-deviating features per customer
-- (mirroring the SHAP plot's row count).  Pre-ranked here so the app issues
-- one indexed query per page hit instead of fetching every feature row.
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_feature_deviation_topn AS
WITH ranked AS (
    SELECT
        d.*,
        ROW_NUMBER() OVER (
            PARTITION BY d.scoring_run_id, d.entity_id
            ORDER BY ABS(COALESCE(d.z, 0.0)) DESC
        ) AS deviation_rank
    FROM {catalog}.{schema}.v_account_feature_deviation d
    WHERE d.z IS NOT NULL
)
SELECT
    scoring_run_id,
    entity_id,
    feature_name,
    feature_value,
    mean, stddev,
    q05, q25, q50, q75, q95,
    z,
    pct_dev,
    deviation_rank
FROM ranked
WHERE deviation_rank <= 12;
-- @cr:deviation-block:close

-- ============================================================================
-- 14. v_feature_provenance (per-feature lineage + business definition)
-- ----------------------------------------------------------------------------
-- One row per feature in the latest scoring run. Surfaces the upstream lineage
-- (source table, source columns, aggregation kind / window) collected at gold-
-- materialization time AND the LLM/curator-authored business phrase so the
-- dashboard can explain "what does this SHAP driver mean" without a notebook.
-- Source-column business definitions are joined in best-effort on column name
-- (no catalog/schema match) — column_descriptions has one canonical row per
-- column_name in this schema, so a name match is sufficient in practice.
-- Falls back to the most recently written row when multiple runs share the
-- same feature_name (CSMs always look at the last published cohort).
-- ============================================================================
CREATE OR REPLACE VIEW {catalog}.{schema}.v_feature_provenance AS
WITH latest_meta AS (
    SELECT MAX(written_at) AS max_written_at
    FROM {catalog}.{schema}.feature_meta
),
latest_rc AS (
    SELECT run_id, composite_name
    FROM {catalog}.{schema}.feature_meta
    WHERE written_at = (SELECT max_written_at FROM latest_meta)
    GROUP BY run_id, composite_name
    LIMIT 1
),
fm AS (
    SELECT
        f.feature_name,
        f.source_columns,
        f.source_table,
        f.aggregation_kind,
        f.window_days,
        f.window_phrase,
        f.target_dependency,
        f.polarity,
        f.business_phrase
    FROM {catalog}.{schema}.feature_meta f
    JOIN latest_rc lr
      ON f.run_id = lr.run_id
     AND f.composite_name = lr.composite_name
),
exploded AS (
    SELECT fm.feature_name, sc AS source_column
    FROM fm
    LATERAL VIEW EXPLODE(COALESCE(fm.source_columns, ARRAY())) tbl AS sc
),
col_defs AS (
    SELECT
        e.feature_name,
        COLLECT_LIST(NAMED_STRUCT(
            'column_name',         e.source_column,
            'business_name',       cd.business_name,
            'business_definition', cd.business_definition,
            'unit',                cd.unit
        )) AS source_column_defs
    FROM exploded e
    LEFT JOIN {catalog}.{schema}.column_descriptions cd
      ON cd.column_name = e.source_column
    GROUP BY e.feature_name
)
SELECT
    fm.feature_name,
    fm.source_columns,
    fm.source_table,
    fm.aggregation_kind,
    fm.window_days,
    fm.window_phrase,
    fm.target_dependency,
    fm.polarity,
    fm.business_phrase,
    cd.source_column_defs
FROM fm
LEFT JOIN col_defs cd ON cd.feature_name = fm.feature_name;
