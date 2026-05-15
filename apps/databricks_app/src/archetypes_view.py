"""Archetypes tab — master table of model-derived archetypes + readonly detail.

The detail view splits driver features into two parallel columns —
risk-driving on the left, protective on the right — with proportional
bars that mirror the L4 SHAP visual: yellow grows rightward from the
centre, green grows leftward, so the heaviest drivers on both sides
bracket the divider line. A per-row "tech" disclosure surfaces the raw
column name, cluster mean value, mean SHAP, p25/p50/p75 thresholds, and
source-dataset metadata from feature provenance.

Mean-churn handling: when ``cluster_mean_churn_probability`` is 0 for
every active archetype the value is treated as upstream-missing rather
than a true 0% rate. The master-table cell renders "—", the detail chip
is suppressed, and a banner explains the gap. This avoids flat 0% bars
that read as a UI bug while honoring real values when they're present.
"""
from __future__ import annotations

from html import escape
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from . import data

_STATE_KEY = "arch_detail_id"

_DIRECTION_POSITIVE = "positive"
_DIRECTION_NEGATIVE = "negative"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
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


def _fmt_float(v: Any, *, precision: int = 3) -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(v):.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mean-churn-zero detection
# ---------------------------------------------------------------------------


def _mean_churn_all_zero(df: pd.DataFrame) -> bool:
    """Return True when every archetype reports churn probability ~0.

    The signal pattern we care about is "all rows are exactly 0 or null
    after a fresh derivation run" -- a true cluster with a 0% historical
    churn rate is implausible for any active archetype, so seeing it
    uniformly across the catalog is almost always an upstream-data gap
    (target column missing from the join, or rows filtered out before
    the per-cluster mean was computed).
    """
    if df is None or df.empty:
        return False
    col = df.get("cluster_mean_churn_probability")
    if col is None:
        return True
    series = pd.to_numeric(col, errors="coerce").fillna(0.0)
    return bool((series.abs() < 1e-9).all())


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------


def _render_master_table(df: pd.DataFrame, *, hide_mean_churn: bool) -> int | None:
    """Master table. When ``hide_mean_churn`` the column collapses to "—"
    text so the row no longer renders a uniform 0% progress bar that
    reads as a UI bug."""

    if hide_mean_churn:
        mean_churn_col_data = pd.Series(["—"] * len(df), dtype="string")
        mean_churn_cfg = st.column_config.TextColumn(
            width="small",
            help=(
                "Cluster mean churn probability is 0 for every active archetype — "
                "likely upstream gap (target column missing from the derivation "
                "join). Re-run c02 archetype_derivation against training data that "
                "carries the binary target."
            ),
        )
    else:
        mean_churn_col_data = df["cluster_mean_churn_probability"]
        mean_churn_cfg = st.column_config.ProgressColumn(
            format="%.2f", min_value=0.0, max_value=1.0, width="small"
        )

    display = pd.DataFrame({
        "Archetype":   df["archetype_name"].astype(str),
        "ID":          df["archetype_id"].astype(str),
        "Version":     df["archetype_version"].astype(str),
        "Cluster":     df["cluster_size"].astype("Int64"),
        "Mean churn":  mean_churn_col_data,
        "Stability":   df.get("stability_vs_prior_version", pd.Series([None] * len(df))),
        "Method":      df["derivation_method"].astype(str),
    })

    col_config = {
        "Archetype":  st.column_config.TextColumn(width="medium"),
        "ID":         st.column_config.TextColumn(width="small", help="archetype_id"),
        "Version":    st.column_config.TextColumn(width="small"),
        "Cluster":    st.column_config.NumberColumn(format="%,d", width="small"),
        "Mean churn": mean_churn_cfg,
        "Stability":  st.column_config.NumberColumn(
            format="%.2f", width="small",
            help="Jaccard-like overlap with the same archetype in the prior version (1.0 = identical)",
        ),
        "Method":     st.column_config.TextColumn(width="small"),
    }

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        height=420,
        key="archetypes_master_table",
    )
    rows = (event.selection or {}).get("rows") if event else None
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Feature struct helpers
# ---------------------------------------------------------------------------


def _as_dict(item: Any) -> dict[str, Any]:
    """Coerce a row of ``top_shap_features`` (struct / Row / dict) to dict."""
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "asDict"):
        try:
            return item.asDict()
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in ("feature", "mean_shap", "mean_value", "direction"):
        if hasattr(item, key):
            out[key] = getattr(item, key)
    return out


def _split_drivers(features: Any) -> tuple[list[dict], list[dict]]:
    """Return ``(positive, negative)`` driver dicts, ordered by |mean_shap|.

    The framework writes an explicit ``direction`` of "positive" or
    "negative"; when it's missing we fall back to the sign of ``mean_shap``
    so older catalogs still split cleanly.
    """
    if features is None:
        return [], []
    try:
        seq: Iterable[Any] = list(features)
    except TypeError:
        return [], []
    if not seq:
        return [], []

    pos: list[dict] = []
    neg: list[dict] = []
    for raw in seq:
        d = _as_dict(raw)
        if not d.get("feature"):
            continue
        direction = (d.get("direction") or "").strip().lower()
        mean_shap = d.get("mean_shap")
        if direction == _DIRECTION_POSITIVE:
            pos.append(d)
        elif direction == _DIRECTION_NEGATIVE:
            neg.append(d)
        else:
            try:
                if mean_shap is not None and float(mean_shap) >= 0:
                    pos.append(d)
                else:
                    neg.append(d)
            except (TypeError, ValueError):
                pos.append(d)

    def _abs_shap(d: dict) -> float:
        v = d.get("mean_shap")
        try:
            return abs(float(v))
        except (TypeError, ValueError):
            return 0.0

    pos.sort(key=_abs_shap, reverse=True)
    neg.sort(key=_abs_shap, reverse=True)
    return pos, neg


def _coerce_map(item: Any) -> dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "asDict"):
        try:
            return item.asDict()
        except Exception:
            return {}
    return {}


def _threshold_for(thresholds: dict, feature_name: str) -> dict[str, Any]:
    raw = thresholds.get(feature_name)
    if raw is None:
        return {}
    return _coerce_map(raw)


def _phrase_lookup(phrases_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if phrases_df is None or phrases_df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, r in phrases_df.iterrows():
        name = r.get("feature_name")
        if not name:
            continue
        out[str(name)] = {
            "business_phrase": r.get("business_phrase"),
            "source_dataset":  r.get("source_dataset"),
            "aggregation_kind": r.get("aggregation_kind"),
            "window_phrase":   r.get("window_phrase"),
            "polarity":        r.get("polarity"),
        }
    return out


# ---------------------------------------------------------------------------
# Two-column drivers — chart-style layout
# ---------------------------------------------------------------------------


def _max_abs_shap(*driver_lists: list[dict]) -> float:
    """Joint |mean_shap| max across both columns. Used to scale bars so
    risk and protective sides share the same x-axis: a 0.8-shap protective
    feature reads as wider than a 0.3-shap risk feature."""
    cap = 0.0
    for dl in driver_lists:
        for d in dl:
            v = d.get("mean_shap")
            try:
                a = abs(float(v))
            except (TypeError, ValueError):
                continue
            if a > cap:
                cap = a
    return cap


def _driver_row_html(
    driver: dict,
    *,
    direction: str,
    thresholds: dict,
    phrases: dict[str, dict[str, Any]],
    cap: float,
) -> str:
    feature = str(driver.get("feature") or "")
    phrase_meta = phrases.get(feature, {})
    display_phrase = (
        str(phrase_meta.get("business_phrase") or "").strip()
        or feature
    )
    mean_shap = driver.get("mean_shap")
    abs_shap = 0.0
    try:
        abs_shap = abs(float(mean_shap))
    except (TypeError, ValueError):
        pass
    pct = (abs_shap / cap * 100.0) if cap > 0 else 0.0

    # Tech-detail body — definition list of every field we have, written
    # only when at least one piece of metadata is non-empty.
    tech_rows: list[tuple[str, str]] = []
    tech_rows.append(("Column", feature))
    mean_value = driver.get("mean_value")
    if mean_value is not None:
        tech_rows.append(("Cluster mean value", _fmt_float(mean_value, precision=4)))
    if mean_shap is not None:
        tech_rows.append(("Cluster mean SHAP", _fmt_float(mean_shap, precision=6)))
    explicit_dir = driver.get("direction")
    if explicit_dir:
        tech_rows.append(("Direction", str(explicit_dir)))
    threshold = _threshold_for(thresholds, feature)
    if threshold:
        t_bits = []
        for tk in ("p25", "p50", "p75"):
            if tk in threshold and threshold[tk] is not None:
                t_bits.append(f"{tk}={_fmt_float(threshold[tk], precision=3)}")
        if t_bits:
            tech_rows.append(("Cluster quantiles", " · ".join(t_bits)))
    for key, label in (
        ("source_dataset",   "Source dataset"),
        ("aggregation_kind", "Aggregation"),
        ("window_phrase",    "Window"),
        ("polarity",         "Training polarity"),
    ):
        val = phrase_meta.get(key)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if s:
            tech_rows.append((label, s))

    tech_dl = ''.join(
        f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in tech_rows
    )

    name_html = (
        f'<span class="arch-name" title="{escape(feature)}">{escape(display_phrase)}</span>'
    )
    shap_html = f'<span class="arch-shap">{_fmt_float(abs_shap, precision=4)}</span>'
    bar_html = (
        '<span class="arch-bar-track">'
        f'<span class="arch-bar" style="width:{pct:.2f}%"></span>'
        '</span>'
    )
    tech_html = (
        '<details class="arch-tech">'
        '<summary>technical detail</summary>'
        f'<div class="arch-tech-body"><dl>{tech_dl}</dl></div>'
        '</details>'
    )

    # For the protective column we mirror the row: value on the left,
    # name on the right, bar anchored right and growing leftward toward
    # the divider. We control that purely by DOM order here -- earlier
    # attempts via CSS ``order:`` collided with the full-width
    # ``arch-bar-track`` / ``arch-tech`` grid items and pushed them
    # above the labels.
    if direction == _DIRECTION_NEGATIVE:
        return (
            '<div class="arch-row">'
            f'{shap_html}{name_html}{bar_html}{tech_html}'
            '</div>'
        )
    return (
        '<div class="arch-row">'
        f'{name_html}{shap_html}{bar_html}{tech_html}'
        '</div>'
    )


def _render_drivers_two_col(
    pos: list[dict],
    neg: list[dict],
    thresholds: dict,
    phrases: dict[str, dict[str, Any]],
) -> None:
    cap = _max_abs_shap(pos, neg)

    def _col(drivers: list[dict], side: str, head_class: str, head_label: str) -> str:
        if drivers:
            rows = ''.join(
                _driver_row_html(d, direction=side, thresholds=thresholds, phrases=phrases, cap=cap)
                for d in drivers
            )
        else:
            rows = '<div class="arch-col-empty">No drivers in this direction.</div>'
        head_count = (
            f'<span class="count">{len(drivers)}</span>' if drivers else ''
        )
        return (
            f'<div class="arch-col {head_class}">'
            f'<div class="arch-col-head {head_class}">'
            f'<span class="dot"></span><span>{escape(head_label)}</span>{head_count}'
            '</div>'
            f'{rows}'
            '</div>'
        )

    st.markdown(
        '<section class="catalog-section" style="margin-top:1.25rem">'
        '<h4>Driver features</h4>'
        '<div class="arch-drivers">'
        f'{_col(pos, "risk", "risk", "Risk-driving")}'
        f'{_col(neg, "prot", "prot", "Protective")}'
        '</div>'
        '</section>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero card + prose
# ---------------------------------------------------------------------------


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


def _render_detail(
    row: pd.Series,
    phrases: dict[str, dict[str, Any]],
    *,
    mean_churn_unreliable: bool,
) -> None:
    name = str(row.get("archetype_name") or row.get("archetype_id"))
    aid = str(row.get("archetype_id"))
    version = str(row.get("archetype_version"))

    # Hero bubbles: the archetype name is shown as a soft pill so the
    # full LLM-authored phrase is always visible (no h3 truncation), and
    # the version sits in a more prominent accent bubble next to it --
    # the version is what changes most frequently as new derivation runs
    # ship, so an operator scanning across runs needs to spot it first.
    name_bubble = (
        f'<span class="arch-name-bubble" title="{escape(aid)}">{escape(name)}</span>'
    )
    version_bubble = (
        f'<span class="arch-version-bubble">v{escape(version)}</span>'
    )

    # Secondary chips (everything except name/version, which now live in
    # the hero bubbles). ID drops too — it's surfaced via the hover
    # tooltip on the name bubble — but cluster size, mean churn, method,
    # stability and model stay as informational chips.
    chips = [_chip("Cluster size", _fmt_int(row.get("cluster_size")))]
    if not mean_churn_unreliable:
        mean_churn = _safe_float(row.get("cluster_mean_churn_probability"))
        if mean_churn is not None and mean_churn > 0:
            chips.append(_chip("Mean churn", _fmt_pct(mean_churn), accent=True))
    chips.append(_chip("Method", str(row.get("derivation_method") or "—")))
    stability = _safe_float(row.get("stability_vs_prior_version"))
    if stability is not None:
        chips.append(_chip("Stability vs prior", _fmt_float(stability, precision=2)))
    model_name = row.get("model_name")
    model_version = row.get("model_version")
    if model_name:
        mv = f"{model_name}"
        if model_version:
            mv = f"{mv} · v{model_version}"
        chips.append(_chip("Model", mv))

    sections_html = ''.join([
        _prose_block("Description", row.get("archetype_description")),
        _prose_block("Rationale", row.get("rationale")),
    ])

    st.markdown(
        f"""
        <article class="catalog-detail">
          <div class="eyebrow">Archetype · readonly</div>
          <div class="arch-hero-row">{name_bubble}{version_bubble}</div>
          <p class="subtitle">{escape(aid)}</p>
          <div class="chip-row">{''.join(chips)}</div>
          {sections_html}
        </article>
        """,
        unsafe_allow_html=True,
    )

    pos, neg = _split_drivers(row.get("top_shap_features"))
    thresholds = _coerce_map(row.get("feature_thresholds"))

    if not pos and not neg:
        st.markdown(
            '<section class="catalog-section">'
            '<h4>Driver features</h4>'
            '<p class="prose"><em>No SHAP drivers recorded for this archetype.</em></p>'
            '</section>',
            unsafe_allow_html=True,
        )
        return

    _render_drivers_two_col(pos, neg, thresholds, phrases)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render() -> None:
    st.markdown(
        '<p class="catalog-lead">'
        'Behavioural clusters the model has identified in the current '
        'feature space. Click a row to see the archetype’s description, '
        'rationale, and the SHAP drivers — risk-driving on the left, '
        'protective on the right, scaled to the same axis so a long '
        'protective bar reads as stronger than a short risk bar. Expand '
        'any driver for its technical detail.'
        '</p>',
        unsafe_allow_html=True,
    )

    try:
        df = data.archetype_catalog_all()
    except Exception as exc:
        st.error(f"Could not load archetype catalog: {exc}")
        return
    if df is None or df.empty:
        st.info("No active archetypes — run c03 / c04 to derive and publish.")
        return

    mean_churn_unreliable = _mean_churn_all_zero(df)
    if mean_churn_unreliable:
        st.markdown(
            '<div class="arch-data-warn">'
            '<strong>Mean churn probability unavailable.</strong> '
            'Every active archetype reports 0 — almost certainly an '
            'upstream gap (the binary target column was missing from '
            'the derivation join, or rows were filtered out before the '
            'per-cluster mean was computed). The column is hidden until '
            'c02 archetype_derivation is re-run against training data '
            'that carries the target.'
            '</div>',
            unsafe_allow_html=True,
        )

    try:
        phrases_df = data.feature_business_phrases()
    except Exception:
        phrases_df = pd.DataFrame()
    phrases = _phrase_lookup(phrases_df)

    idx = _render_master_table(df, hide_mean_churn=mean_churn_unreliable)
    if idx is None:
        prior = st.session_state.get(_STATE_KEY)
        if not prior:
            return
        match = df.index[df["archetype_id"] == prior]
        if len(match) == 0:
            return
        idx = int(match[0])
    else:
        st.session_state[_STATE_KEY] = str(df.iloc[idx]["archetype_id"])

    _render_detail(df.iloc[idx], phrases, mean_churn_unreliable=mean_churn_unreliable)
