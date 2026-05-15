"""Playbooks tab — master table of intervention designs + readonly detail.

Master row click reveals a detail panel below with the full playbook
content: description, prose blocks (when applicable, time-zero, estimand),
operational defaults (cost, uplift, grace period), and the ordered step
list. Selection lives in its own session-state key so it does not
collide with the dashboard drill-down's ``selected_playbook``.
"""
from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from . import data

_STATE_KEY = "pb_detail_id"


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_currency(v: Any) -> str:
    try:
        return f"${float(v):,.0f}"
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


def _chip(label: str, value: str, *, accent: bool = False) -> str:
    cls = "catalog-chip accent" if accent else "catalog-chip"
    return (
        f'<span class="{cls}">'
        f'<span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span>'
        f'</span>'
    )


def _prose_block(title: str, text: Any) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(text).strip()
    if not s:
        return ""
    return (
        '<section class="catalog-section">'
        f'<h4>{escape(title)}</h4>'
        f'<p class="prose">{escape(s)}</p>'
        '</section>'
    )


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


def _render_detail(row: pd.Series) -> None:
    name = str(row.get("name") or row.get("playbook_id"))
    pid = str(row.get("playbook_id"))
    version = str(row.get("version"))

    chips = [
        _chip("ID", pid),
        _chip("Version", version),
        _chip("Eligible", _fmt_int(row.get("eligible_count"))),
        _chip("Recommended", _fmt_int(row.get("recommended_count")), accent=True),
        _chip("Default uplift", _fmt_pct(row.get("expected_uplift_pct_default"))),
        _chip("Cost / customer", _fmt_currency(row.get("cost_per_customer_default"))),
        _chip("Grace period", f"{_fmt_int(row.get('grace_period_days'))} d"),
        _chip("Active from", _fmt_date(row.get("active_from"))),
    ]

    sections = [
        _prose_block("Description", row.get("description")),
        _prose_block("When applicable", row.get("when_applicable")),
    ]

    # Target-trial parameters — operator-facing prose rendered as a compact
    # mono block so the raw enum values stay legible.
    trial_lines = []
    for label, key in (
        ("Time-zero definition",      "time_zero_definition"),
        ("Follow-up start rule",      "followup_start_rule"),
        ("Follow-up end rule",        "followup_end_rule"),
        ("Default estimand",          "default_estimand"),
        ("Analysis population rule",  "analysis_population_rule"),
        ("Outcome definition",        "outcome_definition_version"),
    ):
        v = row.get(key)
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        s = str(v).strip()
        if not s:
            continue
        trial_lines.append(f"{label}: {s}")

    windows = row.get("outcome_windows_days")
    if windows is not None:
        try:
            seq = list(windows) if not isinstance(windows, str) else []
        except TypeError:
            seq = []
        if seq:
            trial_lines.append(
                "Outcome windows: " + ", ".join(f"{int(w)}d" for w in seq)
            )

    if trial_lines:
        sections.append(
            '<section class="catalog-section">'
            '<h4>Target-trial parameters</h4>'
            f'<div class="mono">{escape(chr(10).join(trial_lines))}</div>'
            '</section>'
        )

    st.markdown(
        f"""
        <article class="catalog-detail">
          <div class="eyebrow">Playbook · readonly</div>
          <h3 class="title">{escape(name)}</h3>
          <p class="subtitle">{escape(pid)} · v{escape(version)}</p>
          <div class="chip-row">{''.join(chips)}</div>
          {''.join(sections)}
        </article>
        """,
        unsafe_allow_html=True,
    )

    _render_steps(pid, version)


_CADENCE_RELATIVE_PREFIX = {
    "relative_to_assignment": "from assignment",
    "relative_to_step":       "from step",
    "immediate":              "immediately",
    None:                     "",
}


def _step_meta(step: pd.Series) -> str:
    parts: list[str] = []

    action = step.get("action_type")
    automation = step.get("automation_level")
    owner = step.get("owner_role")
    if action:
        parts.append(f"<span><strong>Action</strong> {escape(str(action))}</span>")
    if automation:
        parts.append(f"<span><strong>By</strong> {escape(str(automation))}</span>")
    if owner:
        parts.append(f"<span><strong>Owner</strong> {escape(str(owner))}</span>")

    trigger = step.get("cadence_trigger")
    if trigger:
        prefix = _CADENCE_RELATIVE_PREFIX.get(str(trigger), str(trigger))
        cadence_bits = [str(prefix) if prefix else str(trigger)]
        rel = step.get("cadence_relative_to")
        if rel:
            try:
                if not pd.isna(rel):
                    cadence_bits.append(f"({rel})")
            except (TypeError, ValueError):
                cadence_bits.append(f"({rel})")
        offset = step.get("cadence_offset_days")
        try:
            if offset is not None and not pd.isna(offset) and int(offset) != 0:
                cadence_bits.append(f"+{int(offset)}d")
        except (TypeError, ValueError):
            pass
        cond = step.get("cadence_condition")
        if cond and str(cond) not in ("always", "None"):
            cadence_bits.append(f"if {cond}")
        parts.append(
            f"<span><strong>Cadence</strong> {escape(' '.join(cadence_bits))}</span>"
        )

    timeout = step.get("timeout_days")
    try:
        if timeout is not None and not pd.isna(timeout):
            parts.append(
                f"<span><strong>Timeout</strong> {int(timeout)}d</span>"
            )
    except (TypeError, ValueError):
        pass

    schema = step.get("response_schema_id")
    if schema:
        parts.append(f"<span><strong>Response</strong> {escape(str(schema))}</span>")

    return ''.join(parts)


def _render_steps(pid: str, version: str) -> None:
    try:
        steps = data.playbook_steps_for(pid, version)
    except Exception as exc:
        st.warning(f"Could not load steps for {pid}: {exc}")
        return
    if steps is None or steps.empty:
        st.markdown(
            '<section class="catalog-section">'
            '<h4>Steps</h4>'
            '<p class="prose"><em>No steps recorded for this playbook version.</em></p>'
            '</section>',
            unsafe_allow_html=True,
        )
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


def render() -> None:
    st.markdown(
        '<p class="catalog-lead">'
        'Every intervention design currently registered with the model. '
        'Click a row to inspect the playbook’s description, target-trial '
        'parameters, and the ordered step sequence. Readonly view — edits '
        'happen in the YAML catalog and are picked up on next ingest.'
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
        # No row clicked this rerun but a prior selection exists — keep it
        # on screen by resolving the playbook_id back to the latest frame.
        match = df.index[df["playbook_id"] == prior]
        if len(match) == 0:
            return
        idx = int(match[0])
    else:
        st.session_state[_STATE_KEY] = str(df.iloc[idx]["playbook_id"])

    _render_detail(df.iloc[idx])
