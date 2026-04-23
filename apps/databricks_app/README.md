# Customer Retention — Databricks App (CSM Triage)

A click-driven, progressive-disclosure Databricks App for customer-retention CSMs.
Lives under `apps/databricks_app/`. Reads the same Unity Catalog views published
by `src/customer_retention/stages/causal/sql/dashboard_views.sql` that the Lakeview
dashboard uses.

A sibling `apps/local_app/` (work in progress) will run the same UI off local
Delta tables via DuckDB for dev and demos without requiring a workspace.

## Why not Lakeview?

Lakeview has real constraints for this workflow:

- Tables can't emit click events, so "pick a customer from the list" must go through a dropdown.
- No dynamic markdown / HTML cards for the customer profile.
- Cross-widget filtering is scoped to the clicking widget's dataset only — cascading through Portfolio → Archetype → Accounts → Detail requires either unified datasets (bad for totals) or dropdown filters (bad UX).

This app reuses the existing `v_*` views, but renders them in Streamlit/Plotly where click events, sortable tables with row-selection, and rich HTML are first-class.

## Flow

1. **Portfolio treemap** — tile per playbook × risk tier, sized by eligible accounts, colored by mean churn probability. Click a playbook to drill.
2. **Archetype treemap** — tiles for archetypes within the selected playbook. Click to narrow the accounts list.
3. **Ranked customer list** — sortable `st.dataframe` sorted by `expected_loss` (churn probability × value at risk). Click a row to open the full profile.
4. **Customer profile** — one rich HTML card rendered from a Handlebars template. The default template ships with the app; `CR_PROFILE_TEMPLATE_PATH` lets you drop a custom one (see "Profile template" below). The template gets every column of `v_account_explanation` plus any additional data sources the template declares in its YAML frontmatter, so `{{entity_id}}`, `{{fmt_currency account.mrr}}`, and `{{#if recommended}}…{{/if}}` all just work.

## Configure

Set via `app.yaml` (or `.env` for local dev):

| Variable | Purpose |
|---|---|
| `CR_CATALOG` | Unity Catalog name (default `churnkit`) |
| `CR_SCHEMA` | Schema holding the `v_*` views (default `analysis`) |
| `CR_WAREHOUSE_ID` | SQL Warehouse id the app runs queries against |
| `CR_PROFILE_TEMPLATE_PATH` | Optional YAML for the "original-data" half of the customer panel (see below). Leave empty to use the pivoted fallback. |

## Profile template (HTML + Handlebars)

The customer-profile panel is rendered from a **single HTML file with YAML frontmatter**. A default template ships with the app (`src/default_profile.html`) and produces the rich card out-of-the-box without any configuration. To customize: copy `examples/customer_profile_template.html` to a location the app can read (a Unity Catalog Volume works), edit, and point `CR_PROFILE_TEMPLATE_PATH` at it.

### Template structure

```html
---
# YAML frontmatter: declare the tables to join on the selected entity.
# Everything under `data:` becomes a nested variable in the template.
data:
  account:
    source: gold_features_cust_emai_aggr__26e8271   # table in CR_CATALOG.CR_SCHEMA
    join_key: account_id                             # column on that table matching entity_id
  latest_email:
    source: bronze_event_email_events
    join_key: account_id
    order_by: event_timestamp DESC
    limit: 1
css: |
  .mrr { color: #047857; font-weight: 700; }
---
<div class="cr-card">
  <h1>{{entity_id}}</h1>
  <span class="cr-pill">{{risk_tier}} risk</span>
  <div>MRR: <span class="mrr">{{fmt_currency account.mrr}}</span></div>
  <div>Last email: {{fmt_datetime latest_email.event_timestamp}}</div>
  {{#if recommended}}
    <div class="ok">✅ <b>{{playbook_name}}</b></div>
  {{/if}}
  <p>{{archetype_rationale}}</p>
  <pre>{{eligibility_rules_sql}}</pre>
</div>
```

### What's available in the template context

- Every column on `v_account_explanation` for the selected entity is a top-level variable: `{{entity_id}}`, `{{churn_probability}}`, `{{risk_tier}}`, `{{archetype_rationale}}`, `{{eligibility_rules_sql}}`, etc. (see the view DDL for the full list).
- Each `data:` entry is a nested variable keyed by its name: `{{account.mrr}}`, `{{latest_email.event_timestamp}}`.
- Missing values are `None` so `{{#if some_col}}...{{/if}}` behaves correctly.

### Built-in Handlebars helpers

| Helper | Usage | Notes |
|---|---|---|
| `fmt_currency` | `{{fmt_currency account.mrr}}` | 2-decimal USD with thousands separator |
| `fmt_pct` | `{{fmt_pct churn_probability}}` | One-decimal %, multiplies by 100 |
| `fmt_int` | `{{fmt_int eligible_playbook_count}}` | Thousands separator |
| `fmt_float` | `{{fmt_float metric 3}}` | N-decimal float |
| `fmt_date` | `{{fmt_date account.contract_start}}` | `YYYY-MM-DD` |
| `fmt_datetime` | `{{fmt_datetime event_ts}}` | `YYYY-MM-DD HH:MM` |
| `risk_tier_class` | `class="cr-tier-{{risk_tier_class risk_tier}}"` | maps `High`/`Medium`/`Low` → CSS class |
| `upper` / `lower` | `{{upper archetype_name}}` | simple case transforms |

Any of these also work inside `{{#if ...}}` conditionals and `{{#each ...}}` loops (Handlebars native).

### Styling

- The default stylesheet (`src/default_profile.css`) is always prepended, so a custom template can rely on `.cr-card`, `.cr-hero`, `.cr-kpis`, `.cr-callout`, etc.
- The `css:` frontmatter key adds template-specific rules on top, so you can override any default without editing the app.

## Local dev

```bash
cd apps/databricks_app
cp .env.example .env        # fill in DATABRICKS_HOST / DATABRICKS_TOKEN / CR_WAREHOUSE_ID
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Databricks Apps

```bash
databricks apps deploy customer_retention_app \
  --source-code-path apps/databricks_app \
  --env CR_WAREHOUSE_ID=<id>
```

The app inherits the workspace user's identity for SQL queries — each viewer only sees what their Unity Catalog grants allow.

## Key files

```
app.py                            # page layout + cascade orchestrator
src/config.py                     # env var wiring
src/data.py                       # cached SQL readers over the v_* views
src/state.py                      # st.session_state keys and breadcrumb
src/treemap.py                    # L1 — portfolio treemap with plotly_chart on_select
src/archetype_view.py             # L2 — archetype treemap
src/accounts_view.py              # L3 — st.dataframe with row-click selection
src/customer_profile.py           # L4 — HTML-template profile renderer
src/template.py                   # Handlebars loader + frontmatter parser + helpers
src/default_profile.html          # default template (no joins, uses only v_account_explanation)
src/default_profile.css           # default stylesheet for `.cr-*` classes
examples/customer_profile_template.html  # example custom template with `account` + `latest_email` joins
```
