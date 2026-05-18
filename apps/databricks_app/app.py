"""Streamlit entry — progressive-disclosure CSM triage.

Four levels, one long scroll, strict detail gating:
  L1 Portfolio        — always visible: churn risk over the model horizon
  L2 Playbook drill   — appears after a playbook treemap click
  L3 Customer list    — appears alongside L2, scoped by the selection
  L4 Customer profile — appears after a row click

Nothing downstream is ever rendered until the upstream decision has been made.
"""
from __future__ import annotations

import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from src.masthead import l1_title_html, masthead_title

from src import (
    accounts_view,
    archetype_view,
    archetypes_view,
    auth,
    customer_profile,
    data,
    diagnostics,
    playbooks_view,
    state,
    treemap,
)


def _load_run_context() -> tuple[dict, str | None]:
    """Return ``(ctx, diagnostic)`` for the masthead.

    ``diagnostic`` is ``None`` on the happy path. When the run-context view
    is missing or empty the diagnostic surfaces the operator-actionable
    reason so the dashboard never silently degrades to a generic title.
    """
    try:
        df = data.run_context()
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to operator
        return {}, f"v_run_context query failed: {type(exc).__name__}: {exc}"
    if df is None or df.empty:
        return {}, "v_run_context returned 0 rows — re-run c05 to publish run_context"
    row = df.iloc[0]

    def _get(col):
        if col not in df.columns:
            return None
        v = row[col]
        return None if pd.isna(v) else v

    ctx = {
        "horizon_days":       _get("horizon_days"),
        "primary_objective":  _get("primary_objective"),
        "temporal_posture":   _get("temporal_posture"),
        "model_type":         _get("model_type"),
        "model_name":         _get("model_name"),
    }
    # Only fields that drive the masthead/L1 title trigger the warning.
    # ``model_type`` is best-effort MLflow metadata and ``model_name`` is
    # decorative -- neither is required to render the dashboard, so a
    # missing one shouldn't surface a "context unavailable" banner.
    _title_driving = ("horizon_days", "primary_objective", "temporal_posture")
    missing = [k for k in _title_driving if ctx.get(k) is None]
    diag = (
        f"v_run_context row has NULL fields: {', '.join(missing)}"
        if missing else None
    )
    return ctx, diag


st.set_page_config(
    page_title="Churn Risk · CSM Triage",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)
state.init()


# ---------------------------------------------------------------------------
# Theme injection
# ---------------------------------------------------------------------------
_theme_path = Path(__file__).parent / "src" / "theme.css"
st.markdown(f"<style>{_theme_path.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead with inline breadcrumb
# ---------------------------------------------------------------------------
def _render_masthead(ctx: dict) -> None:
    parts = state.breadcrumb_parts()
    crumbs: list[str] = []
    for i, (label, level) in enumerate(parts):
        if i > 0:
            crumbs.append('<span class="crumb-sep">›</span>')
        crumbs.append(
            f'<span class="crumb crumb-{level}">{escape(str(label))}</span>'
        )
    trail_html = "".join(crumbs) if crumbs else ""

    title, segments = masthead_title(ctx)
    sub_html = (
        " &middot; ".join(escape(s) for s in segments)
        if segments
        else "CSM Triage"
    )
    st.markdown(
        f"""
        <header class="masthead">
          <div class="masthead-brand">
            <span class="brand-mark"></span>
            <span class="brand-title">{escape(title)}</span>
            <span class="brand-sub">{sub_html}</span>
          </div>
          <nav class="masthead-trail">{trail_html}</nav>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_reset_bar() -> None:
    if len(state.breadcrumb_parts()) <= 1:
        return
    _, right = st.columns([6, 1])
    with right:
        if st.button("Reset drill", key="reset_btn", use_container_width=True):
            state.clear_all()
            st.rerun()


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
def _level_header(
    level: int,
    eyebrow: str,
    title_html: str,
    lead: str | None = None,
) -> None:
    lead_html = f'<p class="level-lead">{escape(lead)}</p>' if lead else ""
    st.markdown(
        f"""
        <section class="level level-{level}">
          <div class="level-eyebrow"><span class="dot"></span>{escape(eyebrow)}</div>
          <h2 class="level-title">{title_html}</h2>
          {lead_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _section_head(eyebrow: str, title: str, lead: str) -> None:
    st.markdown(
        f"""
        <section class="section-head">
          <div class="section-eyebrow">{escape(eyebrow)}</div>
          <h3 class="section-title">{escape(title)}</h3>
          <p class="section-lead">{escape(lead)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _compact_number(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(v):,}"


def _compact_currency(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.0f}"


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _stat_breakdown_html(
    high: float,
    medium: float,
    low: float,
    *,
    formatter,
    high_label: str = "high",
    medium_label: str = "med",
    low_label: str = "low",
) -> str:
    total = high + medium + low
    if total <= 0:
        return ""
    pct_high = high / total * 100
    pct_med = medium / total * 100
    pct_low = max(0.0, 100 - pct_high - pct_med)
    return (
        '<div class="stat-bar" role="img" aria-label="risk-tier composition">'
        f'<span class="stat-bar-seg stat-bar-high" style="width:{pct_high:.4f}%"></span>'
        f'<span class="stat-bar-seg stat-bar-med" style="width:{pct_med:.4f}%"></span>'
        f'<span class="stat-bar-seg stat-bar-low" style="width:{pct_low:.4f}%"></span>'
        '</div>'
        '<div class="stat-tier-line">'
        f'<span class="stat-tier stat-tier-high"><span class="stat-tier-dot"></span>{formatter(high)} {high_label}</span>'
        f'<span class="stat-tier stat-tier-med"><span class="stat-tier-dot"></span>{formatter(medium)} {medium_label}</span>'
        f'<span class="stat-tier stat-tier-low"><span class="stat-tier-dot"></span>{formatter(low)} {low_label}</span>'
        '</div>'
    )


def _render_stat_row() -> None:
    try:
        df = data.portfolio_totals()
    except Exception as exc:
        st.warning(f"Could not load portfolio totals: {exc}")
        return
    if df.empty:
        return
    r = df.iloc[0]

    eligible = _compact_number(r.get("total_eligible"))
    value_at_risk = _compact_currency(r.get("total_value_at_risk"))
    expected_loss = _compact_currency(r.get("total_expected_loss"))
    playbooks = _compact_number(r.get("active_playbooks"))

    eligible_breakdown = _stat_breakdown_html(
        _safe_float(r.get("eligible_high")),
        _safe_float(r.get("eligible_medium")),
        _safe_float(r.get("eligible_low")),
        formatter=_compact_number,
    )
    var_breakdown = _stat_breakdown_html(
        _safe_float(r.get("value_at_risk_high")),
        _safe_float(r.get("value_at_risk_medium")),
        _safe_float(r.get("value_at_risk_low")),
        formatter=lambda v: f"${_compact_currency(v)}",
    )
    el_breakdown = _stat_breakdown_html(
        _safe_float(r.get("expected_loss_high")),
        _safe_float(r.get("expected_loss_medium")),
        _safe_float(r.get("expected_loss_low")),
        formatter=lambda v: f"${_compact_currency(v)}",
    )

    # Expected loss = sum of P(churn) × value_at_risk across eligible
    # accounts. Value-at-risk is "worst case if every eligible account
    # churned"; expected loss is the probability-weighted projection of
    # what we'd actually lose this cycle if we did nothing. The tier
    # breakdown shows whether risk is concentrated (most $ in high
    # tier = act first) or spread (similar weights across tiers = harder
    # prioritization).
    expected_loss_help = (
        "Probability-weighted dollar exposure: Σ(churn_probability × value_at_risk) "
        "across every eligible account. Read it as 'if we did nothing this cycle, "
        "this is the realistic $ projection of what we'd lose'. The breakdown bar "
        "shows how concentrated that risk is across tiers."
    )

    st.markdown(
        f"""
        <div class="stat-row">
          <div class="stat">
            <span class="stat-label">Eligible accounts</span>
            <span class="stat-value">{eligible}</span>
            {eligible_breakdown}
          </div>
          <div class="stat">
            <span class="stat-label">Value at risk</span>
            <span class="stat-value">${value_at_risk}</span>
            {var_breakdown}
          </div>
          <div class="stat" title="{escape(expected_loss_help)}">
            <span class="stat-label">Expected loss</span>
            <span class="stat-value"><em>${expected_loss}</em></span>
            {el_breakdown}
          </div>
          <div class="stat">
            <span class="stat-label">Active playbooks</span>
            <span class="stat-value">{playbooks}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Render
# ===========================================================================
_ctx, _ctx_diagnostic = _load_run_context()
_render_masthead(_ctx)
_render_reset_bar()

if _ctx_diagnostic:
    st.warning(f"Dashboard run context unavailable — {_ctx_diagnostic}")


def _render_assign_button(
    entity_id: str, *, holdout: bool = False, key_namespace: str = "default",
) -> None:
    """Render the self-assign / unassign control inline with the L4 header.

    The button reads the cached ``assignments()`` frame so its label
    reflects current ownership without firing extra SQL on every rerun.
    Four states:
      - holdout            -> "Holdout · no action"   (disabled, regardless of owner)
      - unassigned         -> "Assign to me"          (enabled)
      - assigned to me     -> "Unassign"              (enabled)
      - assigned to other  -> "Claimed · <handle>"    (disabled)

    Holdout accounts are deliberately excluded from CSM intervention so
    we can measure model lift; the disabled label makes that policy
    visible at the point where someone might try to override it.
    """
    me = auth.current_user_email()
    try:
        owner = data.assignment_for(entity_id)
    except Exception as exc:
        st.caption(f"Assignment lookup failed: {exc}")
        return
    mine = bool(me and owner and owner.strip().lower() == me.lower())
    other = bool(owner and not mine)

    if holdout:
        label = "Holdout · no action"
        help_text = "Holdout accounts are excluded from CSM intervention by policy."
    elif mine:
        label = "Unassign"
        help_text = "Release this account."
    elif other:
        handle = owner.split("@", 1)[0] if "@" in (owner or "") else (owner or "")
        label = f"Claimed · {handle}"
        help_text = "Already claimed by another CSM."
    elif not me:
        label = "Assign to me"
        help_text = "Sign in to self-assign."
    else:
        label = "Assign to me"
        help_text = "Claim this account for yourself."

    disabled = holdout or other or not me
    # Streamlit renders both tabs on every run (tabs are display-toggled,
    # not lazy), so without a tab-scoped namespace the same entity selected
    # in both tabs produces duplicate widget keys and the page crashes.
    clicked = st.button(
        label,
        key=f"assign_btn::{key_namespace}::{entity_id}",
        disabled=disabled,
        help=help_text,
        use_container_width=True,
    )
    if clicked and me and not holdout:
        try:
            result = data.toggle_assignment(entity_id, me)
        except Exception as exc:
            st.error(f"Assignment update failed: {exc}")
            return
        data.assignments.clear()
        if result == "claimed_by_other":
            st.warning("Another CSM claimed this account first.")
        st.rerun()


def _render_l4_panel(entity_id: str, *, lead: str, key_namespace: str) -> None:
    """Render the L4 profile (header + assign button + profile body).

    Shared by the Dashboard tab (driven by ``selected_entity``) and the
    Search tab (driven by ``searched_entity``). Header + button live in a
    single columns row so the assign control sits inline with the profile
    title rather than consuming a separate full-width band underneath.
    ``vertical_alignment="center"`` keeps the button visually anchored to
    the title.

    ``key_namespace`` segregates widget keys across tabs. Streamlit
    renders both tabs on every run, so without per-tab namespacing the
    same entity selected in both places (drill picked X, search typed X)
    crashes the page with ``StreamlitDuplicateElementKey``.
    """
    try:
        is_holdout = data.account_is_holdout(entity_id)
    except Exception:
        is_holdout = False
    hdr_col, btn_col = st.columns([6, 1], vertical_alignment="center")
    with hdr_col:
        _level_header(
            level=4,
            eyebrow="Level 04 · Profile",
            title_html=escape(entity_id),
            lead=lead,
        )
    with btn_col:
        _render_assign_button(
            entity_id, holdout=is_holdout, key_namespace=key_namespace,
        )
    try:
        customer_profile.render(entity_id=entity_id)
    except Exception as exc:
        st.error(f"Customer profile failed: {exc}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
# Four intents:
#   - Dashboard  : top-down drill (L1 portfolio -> L2 playbook -> L3 list -> L4)
#   - Playbooks  : readonly catalog of intervention designs (master + detail)
#   - Archetypes : readonly catalog of model-derived clusters (master + detail)
#   - Search     : targeted lookup by entity_id, jumps straight to L4
# State keys are intentionally distinct (`selected_entity` vs `searched_entity`
# vs `pb_detail_id` / `arch_detail_id`) so switching tabs doesn't fold one
# selection into another's breadcrumb / drill state.
# Diagnostics tab is opt-in via CR_SHOW_DIAGNOSTICS=1 in the app env. When
# off (production default), the tab is not even added to the tab strip --
# zero visual footprint, zero per-event cost (diagnostics.record short-
# circuits at the env-var check before touching session_state).
_tab_labels = ["Dashboard", "Playbooks", "Archetypes", "Search"]
if diagnostics.is_enabled():
    _tab_labels.append("Diagnostics")
_tabs = st.tabs(_tab_labels)
_tab_dashboard, _tab_playbooks, _tab_archetypes, _tab_search = _tabs[:4]
_tab_diagnostics = _tabs[4] if diagnostics.is_enabled() else None


with _tab_dashboard:
    # --- Level 1 · Portfolio ------------------------------------------------
    _l1_t0 = time.perf_counter()
    diagnostics.record("L1_begin")
    _level_header(
        level=1,
        eyebrow="Level 01 · Portfolio",
        title_html=l1_title_html(_ctx),
    )
    _render_stat_row()

    _section_head(
        eyebrow="Archetypes breakdown",
        title="Where the cohort sits today",
        lead=(
            "Every eligible customer grouped by the model-derived archetype that "
            "best describes their behaviour. Each account belongs to exactly one "
            "archetype, so tiles never overlap — summing them across the chart "
            "equals the total eligible cohort. Tile size is the cohort count; "
            "tile colour tracks mean churn probability (pale green is safe, "
            "amber-yellow needs a look, deeper hues mean act first)."
        ),
    )

    try:
        treemap.render()
    except Exception as exc:
        st.error(f"Treemap failed: {exc}")
    diagnostics.record("L1_end", elapsed_ms=int((time.perf_counter() - _l1_t0) * 1000))

    st.markdown(
        '<p class="chart-caption">'
        '<span class="accent-green">▬</span>&nbsp;low risk · '
        '<span class="accent-yellow">▬</span>&nbsp;medium · '
        '<span class="accent-blue">▬</span>&nbsp;high · click an archetype or a tier to drill'
        '</p>',
        unsafe_allow_html=True,
    )

    # --- Level 2 · Playbook recommendations for selected archetype ---------
    _selected_archetype = state.get("selected_archetype")
    _selected_risk_tier = state.get("selected_risk_tier")
    if _selected_archetype:
        _l2_t0 = time.perf_counter()
        diagnostics.record(
            "L2_begin",
            archetype=str(_selected_archetype),
            risk_tier=str(_selected_risk_tier) if _selected_risk_tier else None,
        )
        l2_title = escape(str(_selected_archetype))
        if _selected_risk_tier:
            l2_title = f"{l2_title} <span style='color:var(--muted);font-weight:400'>· {escape(_selected_risk_tier)} risk</span>"
        _level_header(
            level=2,
            eyebrow="Level 02 · Playbook recommendations",
            title_html=l2_title,
            lead=(
                "Playbooks the model recommends for this archetype, ranked by "
                "fit score (deeper plum = stronger match) and sized by the count "
                "of accounts in the slice. Hover for the expected uplift the "
                "policy associates with each pairing; click a tile to narrow the "
                "list below."
            ),
        )
        try:
            archetype_view.render()
        except Exception as exc:
            st.error(f"Playbook recommendations view failed: {exc}")
        diagnostics.record("L2_end", elapsed_ms=int((time.perf_counter() - _l2_t0) * 1000))

        if not _selected_risk_tier:
            # Inline swatches: top-right corner of the priority matrix
            # ("best fit × highest risk" = the brightest deepest plum)
            # vs the diagonals so the operator sees what to look for.
            # Hex values mirror _priority_color() outputs for the
            # corner cells (fit=1.0 paired with tier sat 1.0/0.5/0.18
            # and fit=0.0 paired with tier sat 1.0).
            # Swatch hexes are the exact output of archetype_view._priority_color
            # for the corner cells of the matrix (fit=1.0 paired with the three
            # tier saturations, plus fit=0.0 at full saturation). Keep them in
            # sync with _TIER_SATURATION / _PLUM_STOPS if either is retuned.
            st.markdown(
                '<p class="chart-caption">'
                'tile colour = priority &nbsp;·&nbsp; '
                '<span class="prio-swatch" style="background:#6e4a8c"></span>&nbsp;high risk + best fit&nbsp; '
                '<span class="prio-swatch" style="background:#6c5a7c"></span>&nbsp;medium &nbsp; '
                '<span class="prio-swatch" style="background:#6c6571"></span>&nbsp;low &nbsp;·&nbsp; '
                '<span class="prio-swatch" style="background:#f4eff5"></span>&nbsp;weak fit'
                '</p>'
                '<p class="chart-caption chart-caption-note">'
                'Depth of plum tracks how well the playbook fits the archetype; '
                'vividness tracks the risk tier (gray-ish = Low, vivid = High). '
                'The most vivid AND deepest tile is the top-priority recommendation.'
                '</p>',
                unsafe_allow_html=True,
            )

    # --- Level 3 · Customer list -------------------------------------------
    if _selected_archetype:
        _l3_t0 = time.perf_counter()
        _selected_playbook = state.get("selected_playbook")
        diagnostics.record(
            "L3_begin",
            archetype=str(_selected_archetype),
            playbook=str(_selected_playbook) if _selected_playbook else None,
            risk_tier=str(_selected_risk_tier) if _selected_risk_tier else None,
        )
        scope_parts = [str(_selected_archetype)]
        if _selected_playbook:
            scope_parts.append(str(_selected_playbook))
        if _selected_risk_tier:
            scope_parts.append(f"{_selected_risk_tier} risk")
        scope_str = " · ".join(scope_parts)
        lead_l3 = (
            f"Customers in {scope_str}, sorted by expected loss. "
            "Click a row to open the profile below."
        )
        if not _selected_playbook:
            lead_l3 = (
                f"Customers in {scope_str}, sorted by expected loss. "
                "Click a row to open the profile below, or narrow further by "
                "picking a playbook tile above."
            )
        _level_header(
            level=3,
            eyebrow="Level 03 · Customers",
            title_html="In scope",
            lead=lead_l3,
        )
        try:
            accounts_view.render()
        except Exception as exc:
            st.error(f"Accounts list failed: {exc}")
        diagnostics.record("L3_end", elapsed_ms=int((time.perf_counter() - _l3_t0) * 1000))

    # --- Level 4 · Customer profile ----------------------------------------
    _selected_entity = state.get("selected_entity")
    if _selected_entity:
        _render_l4_panel(
            str(_selected_entity),
            lead="Everything we know about this customer and why the model flagged them.",
            key_namespace="dash",
        )


with _tab_playbooks:
    _section_head(
        eyebrow="Playbooks",
        title="Intervention catalog",
        lead=(
            "Every playbook the model can recommend, with its target-trial "
            "definition and the ordered step sequence a CSM works through "
            "once an account is assigned."
        ),
    )
    try:
        playbooks_view.render()
    except Exception as exc:
        st.error(f"Playbooks view failed: {exc}")


with _tab_archetypes:
    _section_head(
        eyebrow="Archetypes",
        title="Behavioural clusters",
        lead=(
            "Active archetypes derived for the current model build. Each "
            "row is a SHAP-space cluster with its driver features and "
            "cluster statistics — the building blocks the eligibility "
            "policy stitches into playbook recommendations."
        ),
    )
    try:
        archetypes_view.render()
    except Exception as exc:
        st.error(f"Archetypes view failed: {exc}")


with _tab_search:
    _section_head(
        eyebrow="Search",
        title="Look up an account by entity ID",
        lead=(
            "Jump straight to a customer's profile. Search covers every entity "
            "scored in the latest run — driver attribution (SHAP) is precomputed "
            "only for the top recommended accounts, so for entities outside that "
            "set the profile shows the prediction and population deviation without "
            "the per-feature contribution chart."
        ),
    )

    with st.form("entity_search_form", clear_on_submit=False):
        _input_col, _submit_col = st.columns([6, 1], vertical_alignment="bottom")
        with _input_col:
            _raw_search = st.text_input(
                "Entity ID",
                key="entity_search_input",
                placeholder="e.g. 1D70F6",
                label_visibility="collapsed",
            )
        with _submit_col:
            _search_submitted = st.form_submit_button(
                "Search", use_container_width=True
            )

    if _search_submitted:
        _cleaned = (_raw_search or "").strip()
        if not _cleaned:
            state.set_searched_entity(None)
        else:
            try:
                _exists = data.entity_exists_in_latest_run(_cleaned)
            except Exception as exc:
                _exists = False
                st.error(f"Lookup failed: {exc}")
            if _exists:
                state.set_searched_entity(_cleaned)
            else:
                # Keep any prior pick on screen rather than blanking the
                # panel; the warning is enough feedback that this typed
                # value didn't resolve.
                st.warning(
                    f"No entity with id `{_cleaned}` was found in the latest scoring run."
                )

    _searched_entity = state.get("searched_entity")
    if _searched_entity:
        _render_l4_panel(
            str(_searched_entity),
            lead=(
                "Everything we know about this customer and why the model flagged them. "
                "If the driver chart is missing this account is outside the top "
                "recommended set — the population deviation panel still shows "
                "where this customer's features sit relative to the cohort."
            ),
            key_namespace="search",
        )


# Hidden diagnostics tab — only rendered when CR_SHOW_DIAGNOSTICS=1 is set
# in the app environment. Surfaces the per-session timing log captured by
# diagnostics.record() throughout data.py and customer_profile.py.
if _tab_diagnostics is not None:
    with _tab_diagnostics:
        _section_head(
            eyebrow="Diagnostics",
            title="Per-session query + render timing log",
            lead=(
                "Every data-layer query and L1-L4 render boundary writes a row here. "
                "Use this to see where a slow page click is actually spending its time "
                "(connection rebuilds, retry cascades, slow individual queries) without "
                "leaving the app."
            ),
        )
        diagnostics.render()
