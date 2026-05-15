"""Playbooks tab — master table of intervention designs + readonly detail.

The detail panel mirrors the structure of the source YAML so an operator
can scan a playbook the way it's authored. Sections, in order:

  Hero            name / id / version / status chips
  Narrative       description · when applicable · policy summary · expected effect
  Operational     cost, uplift, grace period, eligible / recommended (def-list)
  Target-trial    time-zero, follow-up rules, estimand, analysis pop, outcome (def-list)
  Lifecycle       active_from / active_to / outcome_definition_version
  Steps           ordered cadence list
  Operator runbook (optional, when YAML is reachable):
    Circumstance + indicator class
    Things to note
    Gong questions
    Success criteria
    Email templates
    Cancel reasons
    Reports

The runbook block lives in the YAML but not in the ``playbook_catalog``
table (the framework loader deliberately drops it). We read the YAML
directly through ``data.playbook_yaml_runbook`` -- when it's reachable
the rich runbook sections appear; otherwise the catalog-driven layout
still gives a complete picture of every field the table carries.
"""
from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping

import pandas as pd
import streamlit as st

from . import data

_STATE_KEY = "pb_detail_id"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_currency(v: Any) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v: Any) -> str:
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_date(v: Any) -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(v)


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return not str(v).strip()


def _humanize_enum(v: Any) -> str:
    """Render snake_case enum values as 'Snake case' for readability."""
    if _is_blank(v):
        return "—"
    s = str(v).strip().replace("_", " ")
    return s[:1].upper() + s[1:]


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------


def _render_master_table(df: pd.DataFrame) -> int | None:
    display = pd.DataFrame({
        "Playbook":     df["name"].astype(str),
        "ID":           df["playbook_id"].astype(str),
        "Version":      df["version"].astype(str),
        "Steps":        df["step_count"].astype("Int64"),
        "Eligible":     df["eligible_count"].astype("Int64"),
        "Recommended":  df["recommended_count"].astype("Int64"),
        "Default uplift": df["expected_uplift_pct_default"],
        "Cost / cust.": df["cost_per_customer_default"],
    })

    col_config = {
        "Playbook":       st.column_config.TextColumn(width="medium"),
        "ID":             st.column_config.TextColumn(width="small", help="playbook_id"),
        "Version":        st.column_config.TextColumn(width="small"),
        "Steps":          st.column_config.NumberColumn(format="%d", width="small"),
        "Eligible":       st.column_config.NumberColumn(format="%,d", width="small"),
        "Recommended":    st.column_config.NumberColumn(format="%,d", width="small"),
        "Default uplift": st.column_config.NumberColumn(format="%.0f%%", width="small"),
        "Cost / cust.":   st.column_config.NumberColumn(format="$%.2f", width="small"),
    }

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        height=420,
        key="playbooks_master_table",
    )
    rows = (event.selection or {}).get("rows") if event else None
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Detail rendering — catalog-driven (always available)
# ---------------------------------------------------------------------------


def _chip(label: str, value: str, *, accent: bool = False) -> str:
    cls = "catalog-chip accent" if accent else "catalog-chip"
    return (
        f'<span class="{cls}">'
        f'<span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span>'
        f'</span>'
    )


def _prose_section(label: str, text: Any, *, full: bool = False) -> str:
    if _is_blank(text):
        return ""
    cls = "pb-section full" if full else "pb-section"
    return (
        f'<section class="{cls}">'
        f'<div class="label">{escape(label)}</div>'
        f'<p class="prose">{escape(str(text).strip())}</p>'
        '</section>'
    )


def _deflist_section(label: str, items: list[tuple[str, str]]) -> str:
    """Render a definition list inside a pb-section. Empty rows are dropped."""
    filtered = [(k, v) for k, v in items if not _is_blank(v) and v != "—"]
    if not filtered:
        return ""
    rows = ''.join(
        f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in filtered
    )
    return (
        '<section class="pb-section">'
        f'<div class="label">{escape(label)}</div>'
        f'<dl class="pb-deflist">{rows}</dl>'
        '</section>'
    )


def _hero(row: pd.Series) -> None:
    name = str(row.get("name") or row.get("playbook_id"))
    pid = str(row.get("playbook_id"))
    version = str(row.get("version"))

    chips = [
        _chip("ID", pid),
        _chip("Version", version),
        _chip("Eligible", _fmt_int(row.get("eligible_count"))),
        _chip("Recommended", _fmt_int(row.get("recommended_count")), accent=True),
        _chip("Default uplift", _fmt_pct(row.get("expected_uplift_pct_default"))),
        _chip("Steps", _fmt_int(row.get("step_count"))),
    ]
    active_to = row.get("active_to")
    if _is_blank(active_to):
        chips.append(_chip("Status", "Active"))
    else:
        chips.append(_chip("Retired", _fmt_date(active_to)))

    st.markdown(
        f"""
        <article class="catalog-detail">
          <div class="eyebrow">Playbook · readonly</div>
          <h3 class="title">{escape(name)}</h3>
          <p class="subtitle">{escape(pid)} · v{escape(version)}</p>
          <div class="chip-row">{''.join(chips)}</div>
        """,
        unsafe_allow_html=True,
    )


def _close_detail() -> None:
    st.markdown("</article>", unsafe_allow_html=True)


def _narrative_grid(row: pd.Series) -> None:
    """Four-pane prose grid: description, when applicable, policy, expected effect.

    Each pane is independently rendered, so a playbook that only fills
    two of the four still produces a clean layout (the empty panes
    collapse out of the grid).
    """
    panes = [
        _prose_section("Description", row.get("description")),
        _prose_section("When applicable", row.get("when_applicable")),
        _prose_section("Policy summary", row.get("policy_summary")),
        _prose_section("Expected effect", row.get("expected_effect")),
    ]
    panes = [p for p in panes if p]
    if not panes:
        return
    st.markdown(
        f'<div class="pb-detail-grid">{"".join(panes)}</div>',
        unsafe_allow_html=True,
    )


def _operational_block(row: pd.Series) -> str:
    items = [
        ("Cost per customer",    _fmt_currency(row.get("cost_per_customer_default"))),
        ("Default uplift",       _fmt_pct(row.get("expected_uplift_pct_default"))),
        ("Grace period",         f"{_fmt_int(row.get('grace_period_days'))} days"),
        ("Eligible now",         _fmt_int(row.get("eligible_count"))),
        ("Recommended now",      _fmt_int(row.get("recommended_count"))),
        ("Holdout now",          _fmt_int(row.get("holdout_count"))),
    ]
    return _deflist_section("Operational defaults", items)


def _target_trial_block(row: pd.Series) -> str:
    windows = row.get("outcome_windows_days")
    windows_str = "—"
    try:
        if windows is not None:
            seq = list(windows) if not isinstance(windows, str) else []
            if seq:
                windows_str = ", ".join(f"{int(w)}d" for w in seq)
    except TypeError:
        pass

    items = [
        ("Time-zero",            _humanize_enum(row.get("time_zero_definition"))),
        ("Follow-up start",      _humanize_enum(row.get("followup_start_rule"))),
        ("Follow-up end",        _humanize_enum(row.get("followup_end_rule"))),
        ("Default estimand",     _humanize_enum(row.get("default_estimand"))),
        ("Analysis population",  _humanize_enum(row.get("analysis_population_rule"))),
        ("Outcome windows",      windows_str),
        ("Outcome definition",   str(row.get("outcome_definition_version") or "—")),
    ]
    return _deflist_section("Target-trial parameters", items)


def _lifecycle_block(row: pd.Series) -> str:
    items = [
        ("Active from", _fmt_date(row.get("active_from"))),
        ("Active to",   _fmt_date(row.get("active_to")) if not _is_blank(row.get("active_to")) else "open"),
    ]
    return _deflist_section("Lifecycle", items)


def _ops_grid(row: pd.Series) -> None:
    blocks = [
        _operational_block(row),
        _target_trial_block(row),
        _lifecycle_block(row),
    ]
    blocks = [b for b in blocks if b]
    if not blocks:
        return
    # Lifecycle is short -- put it next to operational in a 2-column grid.
    # If all three exist we still render them as three siblings in the grid
    # (CSS wraps to the next row at the 880px breakpoint).
    st.markdown(
        f'<div class="pb-detail-grid">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


_CADENCE_RELATIVE_PREFIX = {
    "relative_to_assignment": "from assignment",
    "relative_to_step":       "from step",
    "immediate":              "immediately",
}


def _step_meta(step: pd.Series) -> str:
    parts: list[str] = []

    action = step.get("action_type")
    automation = step.get("automation_level")
    owner = step.get("owner_role")
    if action:
        parts.append(f"<span><strong>Action</strong> {escape(_humanize_enum(action))}</span>")
    if automation:
        parts.append(f"<span><strong>By</strong> {escape(_humanize_enum(automation))}</span>")
    if owner:
        parts.append(f"<span><strong>Owner</strong> {escape(_humanize_enum(owner))}</span>")

    trigger = step.get("cadence_trigger")
    if trigger:
        prefix = _CADENCE_RELATIVE_PREFIX.get(str(trigger), str(trigger))
        bits = [str(prefix)]
        rel = step.get("cadence_relative_to")
        if not _is_blank(rel):
            bits.append(f"({rel})")
        offset = step.get("cadence_offset_days")
        try:
            if offset is not None and not pd.isna(offset) and int(offset) != 0:
                bits.append(f"+{int(offset)}d")
        except (TypeError, ValueError):
            pass
        cond = step.get("cadence_condition")
        if not _is_blank(cond) and str(cond) != "always":
            bits.append(f"if {cond}")
        parts.append(f"<span><strong>Cadence</strong> {escape(' '.join(bits))}</span>")

    timeout = step.get("timeout_days")
    try:
        if timeout is not None and not pd.isna(timeout):
            parts.append(f"<span><strong>Timeout</strong> {int(timeout)}d</span>")
    except (TypeError, ValueError):
        pass

    schema = step.get("response_schema_id")
    if not _is_blank(schema):
        parts.append(f"<span><strong>Response</strong> {escape(str(schema))}</span>")

    return ''.join(parts)


def _render_steps(pid: str, version: str) -> None:
    try:
        steps = data.playbook_steps_for(pid, version)
    except Exception as exc:
        st.warning(f"Could not load steps for {pid}: {exc}")
        return
    if steps is None or steps.empty:
        return

    rows_html: list[str] = []
    for _, s in steps.iterrows():
        seq = s.get("step_sequence")
        try:
            seq_str = str(int(seq))
        except (TypeError, ValueError):
            seq_str = "—"
        step_name = str(s.get("step_name") or s.get("step_id") or "")
        rows_html.append(
            '<div class="step">'
            f'<div class="step-num">{escape(seq_str)}</div>'
            '<div>'
            f'<div class="step-name">{escape(step_name)}</div>'
            f'<div class="step-meta">{_step_meta(s)}</div>'
            '</div>'
            '</div>'
        )

    st.markdown(
        '<section class="catalog-section">'
        f'<h4>Steps · {len(steps)}</h4>'
        f'{"".join(rows_html)}'
        '</section>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Operator runbook — best-effort YAML read
# ---------------------------------------------------------------------------


def _runbook_bullets(items: Iterable[Any]) -> str:
    if not items:
        return ""
    lis = []
    for it in items:
        if _is_blank(it):
            continue
        lis.append(f'<li>{escape(str(it).strip())}</li>')
    if not lis:
        return ""
    return f'<ul class="runbook-bullets">{"".join(lis)}</ul>'


def _runbook_section(title: str, body_html: str) -> str:
    if not body_html.strip():
        return ""
    return (
        '<section class="catalog-section">'
        f'<h4>{escape(title)}</h4>'
        f'{body_html}'
        '</section>'
    )


def _runbook_circumstance(circumstance: Any, indicator_class: Any, detail_tags: Any) -> str:
    bits: list[str] = []
    if not _is_blank(indicator_class):
        bits.append(f'<span class="tag">{escape(_humanize_enum(indicator_class))}</span>')
    if isinstance(detail_tags, list):
        for tag in detail_tags:
            if isinstance(tag, Mapping):
                label = str(tag.get("tag") or "").strip()
                if label:
                    bits.append(f'<span class="tag">{escape(label)}</span>')
            elif isinstance(tag, str) and tag.strip():
                bits.append(f'<span class="tag">{escape(tag.strip())}</span>')

    head_html = ''
    if bits:
        head_html = f'<div class="runbook-meta">{"".join(bits)}</div>'
    quote_html = ''
    if not _is_blank(circumstance):
        quote_html = (
            f'<blockquote class="runbook-circumstance">{escape(str(circumstance).strip())}</blockquote>'
        )
    if not head_html and not quote_html:
        return ""
    return _runbook_section("Circumstance", head_html + quote_html)


def _runbook_criteria(success_criteria: Any) -> str:
    if not isinstance(success_criteria, list) or not success_criteria:
        return ""
    cards = []
    for sc in success_criteria:
        if not isinstance(sc, Mapping):
            continue
        outcome = str(sc.get("outcome") or "").strip()
        definition = str(sc.get("definition") or "").strip()
        if not outcome and not definition:
            continue
        cards.append(
            '<div class="runbook-card">'
            f'<div class="head"><span class="name">{escape(_humanize_enum(outcome))}</span></div>'
            f'<p class="body">{escape(definition)}</p>'
            '</div>'
        )
    if not cards:
        return ""
    return _runbook_section(
        "Success criteria",
        f'<div class="runbook-criteria">{"".join(cards)}</div>',
    )


def _runbook_cancel_reasons(cancel_reasons: Any) -> str:
    if not isinstance(cancel_reasons, list) or not cancel_reasons:
        return ""
    cards = []
    for cr in cancel_reasons:
        if not isinstance(cr, Mapping):
            continue
        category = str(cr.get("category") or "").strip()
        code = str(cr.get("code") or "").strip()
        label = str(cr.get("label") or "").strip()
        if not (category or code or label):
            continue
        cat_class = escape(category) if category else ""
        pill = (
            f'<span class="pill {cat_class}">{escape(_humanize_enum(category))}</span>'
            if category else ""
        )
        cards.append(
            '<div class="runbook-card">'
            '<div class="head">'
            f'<span class="name">{pill}{escape(label or code)}</span>'
            f'<span class="meta">{escape(code)}</span>'
            '</div>'
            '</div>'
        )
    if not cards:
        return ""
    return _runbook_section(
        "Cancel reasons",
        f'<div class="runbook-cancel">{"".join(cards)}</div>',
    )


def _runbook_reports(reports: Any) -> str:
    if not isinstance(reports, list) or not reports:
        return ""
    cards = []
    for rp in reports:
        if not isinstance(rp, Mapping):
            continue
        rid = str(rp.get("report_id") or "").strip()
        segment = str(rp.get("segment") or "").strip()
        url = str(rp.get("url") or "").strip()
        note = str(rp.get("filter_note") or "").strip()
        cadence = str(rp.get("cadence") or "").strip()
        subscribe = rp.get("subscribe")
        if not (rid or url or note):
            continue
        meta_bits = []
        if segment:
            meta_bits.append(escape(segment))
        if cadence:
            meta_bits.append(escape(_humanize_enum(cadence)))
        if subscribe is True:
            meta_bits.append("subscribe")
        meta_html = ' · '.join(meta_bits)
        link_html = (
            f'<a class="link" href="{escape(url)}" target="_blank" rel="noopener">{escape(url)}</a>'
            if url and url.lower().startswith(("http://", "https://"))
            else ""
        )
        body_html = f'<p class="body">{escape(note)}</p>' if note else ""
        cards.append(
            '<div class="runbook-card">'
            '<div class="head">'
            f'<span class="name">{escape(rid or "Report")}</span>'
            f'<span class="meta">{meta_html}</span>'
            '</div>'
            f'{body_html}'
            f'{link_html}'
            '</div>'
        )
    if not cards:
        return ""
    return _runbook_section(
        "Reports",
        f'<div class="runbook-reports">{"".join(cards)}</div>',
    )


def _runbook_emails(email_templates: Any) -> str:
    if not isinstance(email_templates, list) or not email_templates:
        return ""
    cards = []
    for tpl in email_templates:
        if not isinstance(tpl, Mapping):
            continue
        tid = str(tpl.get("template_id") or "").strip()
        scenario = str(tpl.get("scenario") or "").strip()
        subject = str(tpl.get("subject") or "").strip()
        body = str(tpl.get("body") or "").strip()
        if not (tid or subject or body):
            continue
        subject_html = (
            f'<p class="runbook-email-subject">Subject &nbsp;{escape(subject)}</p>'
            if subject else ""
        )
        body_html = (
            f'<pre class="runbook-email-body">{escape(body)}</pre>'
            if body else ""
        )
        cards.append(
            '<div class="runbook-card">'
            '<div class="head">'
            f'<span class="name">{escape(tid or "template")}</span>'
            f'<span class="meta">{escape(_humanize_enum(scenario))}</span>'
            '</div>'
            f'{subject_html}{body_html}'
            '</div>'
        )
    if not cards:
        return ""
    return _runbook_section(
        "Email templates",
        f'<div class="runbook-emails">{"".join(cards)}</div>',
    )


def _runbook_taxonomy(label: str, taxonomy: Any) -> str:
    """Flatten nested taxonomies (``dsat_taxonomy`` etc.) into one card list.

    Each top-level key becomes a sub-heading; each entry under it becomes
    a card with its code as the meta tag and its label as the body. Used
    for DSAT / engagement reason taxonomies that vary playbook-to-playbook.
    """
    if not isinstance(taxonomy, Mapping) or not taxonomy:
        return ""
    blocks: list[str] = []
    for group, entries in taxonomy.items():
        if not isinstance(entries, list) or not entries:
            continue
        cards = []
        for e in entries:
            if not isinstance(e, Mapping):
                continue
            code = str(e.get("code") or "").strip()
            etext = str(e.get("label") or "").strip()
            if not (code or etext):
                continue
            cards.append(
                '<div class="runbook-card">'
                '<div class="head">'
                f'<span class="name">{escape(etext or code)}</span>'
                f'<span class="meta">{escape(code)}</span>'
                '</div>'
                '</div>'
            )
        if cards:
            blocks.append(
                f'<h5 style="font-family:var(--font-mono);font-size:0.7rem;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);'
                f'margin:0.9rem 0 0.4rem 0;">{escape(_humanize_enum(group))}</h5>'
                f'<div class="runbook-cancel">{"".join(cards)}</div>'
            )
    if not blocks:
        return ""
    return _runbook_section(label, ''.join(blocks))


def _render_runbook(runbook: Mapping[str, Any]) -> None:
    if not isinstance(runbook, Mapping):
        return

    sections = [
        _runbook_circumstance(
            runbook.get("circumstance"),
            runbook.get("indicator_class"),
            runbook.get("detail_tags"),
        ),
        _runbook_section("Things to note", _runbook_bullets(runbook.get("things_to_note") or [])),
        _runbook_section(
            "Gong questions",
            _runbook_bullets(runbook.get("gong_questions") or []),
        ),
        _runbook_section(
            "Gong smart trackers",
            _runbook_bullets(runbook.get("gong_smart_trackers") or []),
        ),
        _runbook_criteria(runbook.get("success_criteria")),
        _runbook_emails(runbook.get("email_templates")),
        _runbook_cancel_reasons(runbook.get("cancel_reasons")),
        _runbook_reports(runbook.get("reports")),
        _runbook_taxonomy("DSAT taxonomy", runbook.get("dsat_taxonomy")),
        _runbook_taxonomy("Engagement reason codes", runbook.get("engagement_reason_codes")),
        _runbook_taxonomy("Feature picklist", runbook.get("feature_picklist_covered_during_call")),
    ]
    sections = [s for s in sections if s]
    if not sections:
        return
    st.markdown(
        '<section class="catalog-section" style="margin-top:2rem">'
        '<div class="eyebrow" style="font-family:var(--font-mono);font-size:0.7rem;'
        'letter-spacing:0.18em;text-transform:uppercase;color:var(--orange-600);'
        'font-weight:500;margin-bottom:0.4rem">'
        'Operator runbook'
        '</div>'
        '</section>',
        unsafe_allow_html=True,
    )
    st.markdown(''.join(sections), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Detail entry point
# ---------------------------------------------------------------------------


def _render_detail(row: pd.Series) -> None:
    _hero(row)
    _close_detail()
    _narrative_grid(row)
    _ops_grid(row)
    _render_steps(str(row.get("playbook_id")), str(row.get("version")))

    try:
        runbook_doc = data.playbook_yaml_runbook(str(row.get("playbook_id")))
    except Exception:
        runbook_doc = None
    if isinstance(runbook_doc, Mapping):
        runbook = runbook_doc.get("operator_runbook")
        if isinstance(runbook, Mapping):
            _render_runbook(runbook)


def render() -> None:
    st.markdown(
        '<p class="catalog-lead">'
        'Every intervention design currently registered with the model. '
        'Click a row to inspect the playbook’s narrative, operational '
        'defaults, target-trial parameters, ordered step sequence, and — '
        'where the YAML is reachable — the operator runbook the CSM works '
        'against (circumstance, things to note, success criteria, email '
        'templates). Readonly view; edits happen in the YAML catalog and '
        'are picked up on next ingest.'
        '</p>',
        unsafe_allow_html=True,
    )

    try:
        df = data.playbook_catalog_all()
    except Exception as exc:
        st.error(f"Could not load playbook catalog: {exc}")
        return
    if df is None or df.empty:
        st.info("No playbooks found in the catalog yet.")
        return

    idx = _render_master_table(df)
    if idx is None:
        prior = st.session_state.get(_STATE_KEY)
        if not prior:
            return
        match = df.index[df["playbook_id"] == prior]
        if len(match) == 0:
            return
        idx = int(match[0])
    else:
        st.session_state[_STATE_KEY] = str(df.iloc[idx]["playbook_id"])

    _render_detail(df.iloc[idx])
