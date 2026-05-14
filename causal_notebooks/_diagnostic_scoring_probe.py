# @cr:probe name='causal_scoring_diagnostic_v1' id=cspr0001
# =====================================================================
# CAUSAL-TRACK SCORING DIAGNOSTIC PROBE — single-cell drop-in
# =====================================================================
# Paste this entire cell into a fresh Databricks notebook attached to the
# same cluster c01-c05 run on. Run it standalone — it reads every state
# c04 needs and prints a full diagnostic report so the next iteration is
# designed against real data instead of guesses.
#
# What it checks (in order):
#   1. Namespace resolution (which run is active, via which discovery
#      tier, what's on disk under it).
#   2. project_context.yaml: sample_filters, target dataset, raw entity
#      key per dataset, datasets[*].role.
#   3. Catalog/schema configuration + the composite name training landed
#      on.
#   4. UC table inventory: landing_*, bronze_*, silver_featureset_*,
#      gold_features_*, predictions, customer_features.
#   5. Schema dump for the key tables (which columns survive at each
#      layer; is entity_id present in landing? what's the raw key?).
#   6. MLflow @production model lookup.
#   7. Scope filter shape: raw expression, pandas->SQL translation,
#      sibling-temp-view candidates, target dataset's landing FQN.
#   8. Proposed scoring strategy with a one-line rationale.
#
# The cell is read-only — no Delta writes, no temp-view side effects
# beyond the explicit `createOrReplaceTempView` for sibling-name
# resolution (which is identical to what landing already did at write
# time, so it's a no-op against existing views).
# =====================================================================
from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone


def _hr(title=""):
    line = "=" * 78
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _section(title):
    print(f"\n--- {title} ".ljust(78, '-'))


def _safe(label, fn):
    """Run fn() and print result/error with a clear label. Never raises."""
    try:
        result = fn()
        return result
    except Exception as exc:
        print(f"  [ERR] {label}: {type(exc).__name__}: {exc}")
        return None


_hr("CAUSAL SCORING DIAGNOSTIC PROBE — START")
print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")

# ------------------------------------------------------------------ 1) NAMESPACE
_section("1. Namespace resolution")

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.config.experiments import (
    get_catalog,
    get_experiments_dir,
    get_schema,
    get_workspace_path,
)

_catalog_from_env = _safe("get_catalog()", get_catalog)
_schema_from_env  = _safe("get_schema()", get_schema)
_experiments_dir  = _safe("get_experiments_dir()", get_experiments_dir)
_workspace_path   = _safe("get_workspace_path()", get_workspace_path)

print(f"  CATALOG           = {_catalog_from_env}")
print(f"  SCHEMA            = {_schema_from_env}")
print(f"  experiments_dir   = {_experiments_dir}")
print(f"  workspace_path    = {_workspace_path}")
print()

# Tier-by-tier resolution
_tier_results = []
for tier_name, fn in (
    ("from_env (CR_RUN_ID env var)", RunNamespace.from_env),
    ("from_run_pointer (.cr_active_run.json)", RunNamespace.from_run_pointer),
    ("from_sentinel (runs/.active_run_id)", RunNamespace.from_sentinel),
    ("from_latest (most recent project_context mtime)", RunNamespace.from_latest),
):
    try:
        ns = fn()
        _tier_results.append((tier_name, ns))
        if ns:
            print(f"  [{tier_name}] -> run_id={ns.run_id!r}  root={ns.root}")
        else:
            print(f"  [{tier_name}] -> None")
    except Exception as exc:
        print(f"  [{tier_name}] -> ERR {type(exc).__name__}: {exc}")

print()
_ns = RunNamespace.from_env_or_latest()
if _ns is None:
    print("  !!! from_env_or_latest() returned None — every discovery tier failed.")
    print("  !!! Set ENGAGEMENT_RUN_ID / ENGAGEMENT_EXPERIMENTS_DIR in the c04 config cell.")
else:
    print(f"  >>> RESOLVED: run_id={_ns.run_id!r}  root={_ns.root}")
    print(f"  >>> run_dir={_ns.run_dir}  exists={_ns.run_dir.is_dir()}")
    print(f"  >>> project_context_path={_ns.project_context_path}  exists={_ns.project_context_path.exists()}")

# ------------------------------------------------------------------ 2) PROJECT_CONTEXT
_section("2. project_context.yaml")

_ctx = None
_sample_filters = {}
_target_name = None
_raw_entity_keys = {}
_dataset_roles = {}
if _ns is not None and _ns.project_context_path.exists():
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        _ctx = ProjectContext.load(_ns.project_context_path)
        _sample_filters = getattr(_ctx, "sample_filters", None) or {}
        for name, ds in _ctx.datasets.items():
            _dataset_roles[name] = getattr(ds, "role", None)
            _raw_entity_keys[name] = (
                getattr(ds, "entity_key", None)
                or getattr(ds, "entity_column", None)
                or getattr(ds, "primary_key", None)
                or "(none)"
            )
        _target_name = next(
            (n for n, r in _dataset_roles.items() if r == "target"), None,
        )
        if _target_name is None and len(_ctx.datasets) == 1:
            _target_name = next(iter(_ctx.datasets))

        print(f"  datasets ({len(_ctx.datasets)}):")
        for name in sorted(_ctx.datasets):
            role = _dataset_roles[name] or "(none)"
            key  = _raw_entity_keys[name]
            star = "  <-- target" if name == _target_name else ""
            print(f"    {name:<30}  role={role:<10}  raw entity key={key}{star}")
        print()
        print(f"  sample_filters: {len(_sample_filters)}")
        for name, expr in _sample_filters.items():
            print(f"    [{name}] {expr!r}")
        if not _sample_filters:
            print("    (none — scoring will run unfiltered across full entity population)")
    except Exception as exc:
        print(f"  ERR loading project_context: {type(exc).__name__}: {exc}")
        traceback.print_exc()
else:
    print("  project_context.yaml not found — cannot resolve target dataset / sample_filters.")

# ------------------------------------------------------------------ 3) UC TABLE INVENTORY
_section("3. UC table inventory")

try:
    spark
except NameError:
    from customer_retention.core.compat.detection import get_spark_session
    spark = get_spark_session()

CATALOG = _catalog_from_env or "?"
SCHEMA = _schema_from_env or "?"

# Try to discover COMPOSITE_NAME from gold_metadata.json or training_metadata.json
_composite_name = None
for meta_path_attr in ("gold_metadata_path", "training_metadata_path", "exploration_metadata_path"):
    if _ns is None:
        break
    p = getattr(_ns, meta_path_attr, None)
    if p is None or not p.exists():
        continue
    try:
        meta = json.loads(p.read_text())
        _composite_name = meta.get("composite_name") or _composite_name
        if _composite_name:
            print(f"  composite_name discovered from {meta_path_attr}: {_composite_name}")
            break
    except Exception:
        continue

print()
print(f"  CATALOG.SCHEMA = {CATALOG}.{SCHEMA}")
print(f"  composite_name = {_composite_name or '(unresolved)'}")
print()


def _tbl_exists(fqn):
    try:
        return spark.catalog.tableExists(fqn)
    except Exception:
        return False


def _row_count(fqn):
    try:
        return spark.table(fqn).count()
    except Exception as exc:
        return f"ERR: {type(exc).__name__}: {str(exc)[:60]}"


def _columns(fqn, focus=None):
    try:
        cols = [f.name for f in spark.table(fqn).schema.fields]
        focus = focus or set()
        hits = [c for c in cols if c in focus or any(c.startswith(p) for p in focus if isinstance(p, str) and p.endswith("*"))]
        return cols, hits
    except Exception as exc:
        return None, f"ERR: {type(exc).__name__}: {str(exc)[:60]}"


# Build the list of tables to probe
_to_probe = []
for ds_name in sorted(_ctx.datasets.keys() if _ctx else []):
    _to_probe.append(f"{CATALOG}.{SCHEMA}.landing_{ds_name}")
if _composite_name:
    _to_probe.append(f"{CATALOG}.{SCHEMA}.silver_featureset_{_composite_name}")
    _to_probe.append(f"{CATALOG}.{SCHEMA}.gold_features_{_composite_name}")
_to_probe.extend([
    f"{CATALOG}.{SCHEMA}.predictions",
    f"{CATALOG}.{SCHEMA}.customer_features",
    f"{CATALOG}.{SCHEMA}.eligibility_snapshot",
    f"{CATALOG}.{SCHEMA}.archetype_catalog",
])

print(f"  Probing {len(_to_probe)} candidate tables:\n")
print(f"  {'TABLE':<78} {'EXISTS':<8} {'ROWS'}")
print(f"  {'-'*78} {'-'*8} {'-'*12}")
_existence = {}
for fqn in _to_probe:
    exists = _tbl_exists(fqn)
    _existence[fqn] = exists
    rows = _row_count(fqn) if exists else "-"
    rows_str = f"{rows:,}" if isinstance(rows, int) else str(rows)
    print(f"  {fqn:<78} {str(exists):<8} {rows_str}")

# ------------------------------------------------------------------ 4) SCHEMA DUMP
_section("4. Schema sample for key tables (entity_id / ACCOUNT_ID / filter cols)")

_focus_cols = set()
# Add raw entity keys from project_context
for k in _raw_entity_keys.values():
    if k and k != "(none)":
        _focus_cols.add(k)
# Add column references parsed from sample_filters
_filter_columns = set()
if _sample_filters:
    for expr in _sample_filters.values():
        for m in re.finditer(r"\b([A-Z_][A-Z0-9_]+)\b", expr or ""):
            tok = m.group(1)
            if tok not in ("AND", "OR", "IN", "NOT", "NULL", "IS", "FROM", "SELECT", "WHERE", "JOIN", "ON", "AS"):
                _filter_columns.add(tok)
print(f"  focus_columns: entity_id, original_*, {sorted(_focus_cols)}, filter cols {sorted(_filter_columns)}")

for fqn in _to_probe:
    if not _existence.get(fqn):
        continue
    cols, _ = _columns(fqn)
    if cols is None:
        continue
    relevant = [
        c for c in cols
        if c == "entity_id"
        or c.startswith("original_")
        or c in _focus_cols
        or any(c.startswith(fc) for fc in _filter_columns)
    ]
    print(f"  {fqn} ({len(cols)} cols)")
    print(f"    relevant: {relevant or '(none of entity_id / original_* / filter / raw-key columns)'}")

# ------------------------------------------------------------------ 5) SCOPE FILTER ANALYSIS
_section("5. Scope filter analysis")

_raw_filter = None
if _target_name and _sample_filters:
    _raw_filter = _sample_filters.get(_target_name)

if _raw_filter is None:
    print("  No filter for target dataset — scoring runs over full population.")
else:
    print(f"  RAW filter (from project_context.sample_filters[{_target_name!r}]):")
    print(f"    {_raw_filter}")

    # Pandas->SQL translation
    from customer_retention.core.compat import _spark_safe_query_expr
    _sql_filter = _spark_safe_query_expr(_raw_filter)
    print("  SQL-translated filter:")
    print(f"    {_sql_filter}")

    # Sibling temp views
    _sibling_pattern = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
    _sibling_names = sorted(set(_sibling_pattern.findall(_sql_filter)))
    print(f"  bare-name table references (need temp views): {_sibling_names}")
    for name in _sibling_names:
        fqn = f"{CATALOG}.{SCHEMA}.landing_{name}"
        print(f"    landing_{name} -> {fqn}  exists={_tbl_exists(fqn)}")

    # Target landing table & raw entity key
    _target_landing = f"{CATALOG}.{SCHEMA}.landing_{_target_name}"
    _target_raw_key = _raw_entity_keys.get(_target_name)
    print()
    print(f"  Target landing table:  {_target_landing}")
    print(f"  Target raw entity key: {_target_raw_key}")
    print(f"  Target landing exists: {_tbl_exists(_target_landing)}")
    if _tbl_exists(_target_landing):
        _lcols, _ = _columns(_target_landing)
        print(f"    has 'entity_id' column: {'entity_id' in (_lcols or [])}")
        print(f"    has {_target_raw_key!r} column: {(_target_raw_key or '') in (_lcols or [])}")

    # Dry-run the filter to see if it RESOLVES (no count, just plan)
    if _tbl_exists(_target_landing):
        try:
            _planned = spark.table(_target_landing).filter(_sql_filter)
            # Force analysis without execution
            _planned.printSchema()
            print(f"  [OK] Filter resolves against {_target_landing} — schema printed above.")
        except Exception as exc:
            print(f"  [FAIL] Filter does NOT resolve against {_target_landing}: {type(exc).__name__}")
            print(f"         {str(exc).split(chr(10))[0][:200]}")

# ------------------------------------------------------------------ 6) MLFLOW MODEL LOOKUP
_section("6. MLflow model")

if _composite_name:
    _registered_model = f"{CATALOG}.{SCHEMA}.model_{_composite_name}"
    print(f"  Looking up: {_registered_model}@production")
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(_registered_model, "production")
        print(f"  [OK] version={mv.version}  run_id={mv.run_id}  source={getattr(mv, 'source', '(n/a)')}")
    except Exception as exc:
        print(f"  [ERR] {type(exc).__name__}: {exc}")
else:
    print("  (composite_name unresolved — cannot infer registered model name)")

# ------------------------------------------------------------------ 7) SCORING STRATEGY RECOMMENDATION
_section("7. Proposed scoring strategy")

# Build a decision based on what we found
if _raw_filter is None:
    print("  STRATEGY: No filter — score full gold population directly. No join needed.")
elif not _target_name:
    print("  STRATEGY UNKNOWN: filter exists but target dataset is unresolved.")
    print("    -> Set datasets[<name>].role='target' in project_context, OR run a")
    print("       single-dataset project so the framework infers it.")
else:
    _target_landing = f"{CATALOG}.{SCHEMA}.landing_{_target_name}"
    if _tbl_exists(_target_landing):
        _lcols, _ = _columns(_target_landing)
        _has_entity_id = "entity_id" in (_lcols or [])
        _has_raw_key   = (_target_raw_key or "") in (_lcols or [])
        if _has_entity_id:
            print(f"  STRATEGY: Filter on landing_{_target_name}, project entity_id, inner-join gold.")
            print("    (landing has entity_id directly — this is the path the framework expects.)")
        elif _has_raw_key:
            print(f"  STRATEGY: Filter on landing_{_target_name}, project {_target_raw_key!r},")
            print(f"    rename to entity_id (or join on original_{_target_raw_key} on the gold side),")
            print("    then inner-join gold.")
            print("    !!! THE FRAMEWORK CURRENTLY ASSUMES landing has 'entity_id' but landing")
            print("        keeps the raw key — this is the bug causing today's error. !!!")
        else:
            print(f"  STRATEGY UNKNOWN: landing_{_target_name} has neither entity_id nor "
                  f"{_target_raw_key!r}. Inspect schema above.")
    else:
        print(f"  STRATEGY: landing_{_target_name} not found in UC.")
        print("    -> Either pre-create it or skip cohort filtering at scoring time.")

# ---------------------------------------------------------------- 8) C05 DASHBOARD PREREQS
_section("8. c05 dashboard publishing prerequisites (SPS overrides)")
# Mirrors what `# @cr:user_code name='publish_sps_account_profile' id=du4r-sps-001`
# (in docs/sps_notebook_ux_overrides.md) needs in order to publish
# `v_account_profile_sps` + `v_account_feature_deviation_enriched_sps`.
# c05 builds these views from a LEFT-JOIN chain over:
#   - eligibility_snapshot (anchor — every scored account gets a profile row)
#   - prod_corp_snowflake_provisioning_shared.salesforce.account       (account_meta)
#   - prod_corp_snowflake_provisioning_shared.salesforce.contract       (contract_summary)
#   - prod_corp_snowflake_provisioning_shared.salesforce.case           (case_summary)
#   - prod_corp_snowflake_provisioning_shared.salesforce.contact        (contact_summary, optional)
#   - prod_corp_snowflake_provisioning_shared.salesforce.opportunity    (opportunity_arr_summary, optional)
#   - prod_corp_snowflake_provisioning_shared.revenue_operations_share.contract_arr  (arr_summary)
# The cell degrades gracefully on missing shares — but if `account`,
# `contract`, `case`, or `eligibility_snapshot` are missing the
# publish aborts. ARR + contact + opportunity are best-effort.

_C05_FRAMEWORK_REQ = [
    (f"{CATALOG}.{SCHEMA}.eligibility_snapshot",          "REQUIRED — anchor cohort for v_account_profile_sps"),
    (f"{CATALOG}.{SCHEMA}.archetype_catalog",             "REQUIRED — joined for archetype label/policy"),
    (f"{CATALOG}.{SCHEMA}.predictions",                   "REQUIRED — base risk score per account"),
    (f"{CATALOG}.{SCHEMA}.gold_features_{_composite_name}" if _composite_name else f"{CATALOG}.{SCHEMA}.gold_features_(unset)",
                                                          "REQUIRED — feature deviation enrichment input"),
    (f"{CATALOG}.{SCHEMA}.dashboard_template_overrides",  "WRITE TARGET — apply_profile_override appends rows here"),
    (f"{CATALOG}.{SCHEMA}.v_dashboard_template_active",   "READ — exposed to the Databricks App via SQL warehouse"),
    (f"{CATALOG}.{SCHEMA}.v_account_feature_deviation",   "REQUIRED for *_enriched_sps view — owned by publish_dashboard_views (c05 cell 14)"),
    (f"{CATALOG}.{SCHEMA}.v_feature_provenance",          "OPTIONAL — joined for business phrases / source-column defs"),
    (f"{CATALOG}.{SCHEMA}.v_account_profile_sps",         "OUTPUT (overwritten on publish)"),
    (f"{CATALOG}.{SCHEMA}.v_account_feature_deviation_enriched_sps", "OUTPUT (overwritten on publish; depends on the deviation enrichment chain)"),
]

print(f"  {'TABLE / VIEW':<78} {'EXISTS':<8} ROLE")
print(f"  {'-'*78} {'-'*8} ----")
for fqn, role in _C05_FRAMEWORK_REQ:
    exists = _tbl_exists(fqn)
    print(f"  {fqn:<78} {str(exists):<8} {role}")

print()
print("  Snowflake share access (read via UC federation):")
_C05_SHARES = [
    ("prod_corp_snowflake_provisioning_shared.salesforce.account",                "REQUIRED — account_meta CTE source"),
    ("prod_corp_snowflake_provisioning_shared.salesforce.contract",               "REQUIRED — contract_summary CTE source"),
    ("prod_corp_snowflake_provisioning_shared.salesforce.case",                   "REQUIRED — case_summary CTE source"),
    ("prod_corp_snowflake_provisioning_shared.salesforce.contact",                "OPTIONAL — contact_summary degrades to empty"),
    ("prod_corp_snowflake_provisioning_shared.salesforce.opportunity",            "OPTIONAL — value_at_risk falls back to contract_arr only"),
    ("prod_corp_snowflake_provisioning_shared.revenue_operations_share.contract_arr", "OPTIONAL — value_at_risk fills 33% of cohort"),
]
print(f"  {'SHARE TABLE':<82} {'EXISTS':<8} ROLE")
print(f"  {'-'*82} {'-'*8} ----")
for fqn, role in _C05_SHARES:
    exists = _tbl_exists(fqn)
    print(f"  {fqn:<82} {str(exists):<8} {role}")

print()
# apply_profile_override helper presence
try:
    from customer_retention.stages.causal.dashboard_profile_override import apply_profile_override  # noqa: F401
    print("  [OK] customer_retention.stages.causal.dashboard_profile_override.apply_profile_override is importable")
except Exception as exc:
    print(f"  [ERR] apply_profile_override import failed: {type(exc).__name__}: {exc}")

# c05 will substitute {catalog}/{schema} placeholders in the doc's PROFILE_SQL — verify
# the resolved FQN of the SPS profile view is consistent with what we computed.
_target_view_sps = f"{CATALOG}.{SCHEMA}.v_account_profile_sps"
print(f"  Resolved SPS profile view target: {_target_view_sps}")

# Composite-name view that the framework template publishes (c05 cell `00cf348e`).
if _composite_name:
    print(f"  Composite-name dashboard views the framework will publish under: {_composite_name}")
else:
    print("  (composite_name unresolved — framework dashboard views may not publish until it's set)")

# What c05 will EXECUTE (high level summary, no SQL dump): operator audit trail
print()
print("  c05 execution outline (from docs/sps_notebook_ux_overrides.md):")
print("    1. RunNamespace.from_env_or_latest() resolves active run (already shown in §1)")
print("    2. Introspect Snowflake share schemas above — degrade missing columns to NULL")
print("    3. Build CTEs: scored (anchor on eligibility_snapshot), account_meta,")
print("       contract_summary, case_summary, contact_summary, opportunity_arr_summary, arr_summary")
print("    4. CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile_sps (left-join chain)")
print("    5. CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_feature_deviation_enriched_sps")
print("       (joins v_account_feature_deviation x v_feature_provenance; SKIPPED when either")
print("        upstream is missing — see §8 EXISTS column above)")
print("    6. apply_profile_override(html, composite_name=...) writes a row into")
print("       {catalog}.{schema}.dashboard_template_overrides (append-only, latest wins via")
print("        v_dashboard_template_active)")

_hr("CAUSAL SCORING DIAGNOSTIC PROBE — END")
print("Paste this output into the next message so we can design the right framework fix.")
print("Sections:  1 namespace  2 project_context  3 UC inventory  4 schemas  5 filter")
print("           6 mlflow     7 strategy        8 c05 prereqs (dashboard publishing)")
