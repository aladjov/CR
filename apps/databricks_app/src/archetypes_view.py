"""Archetypes tab — master table of model-derived archetypes + readonly detail.

Master row click reveals a complex profile: summary header (cluster size,
mean churn, derivation), prose blocks (description, rationale), then the
top SHAP drivers split into positive (risk-pushing) and negative
(protective) lists. Each driver shows the readable business phrase on
the surface; an Expander reveals the raw technical metadata (column name,
mean SHAP, mean value, p25/p50/p75 from feature_thresholds, source
dataset / window from feature provenance).
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
        "Archetype":   df["archetype_name"].astype(str),
        "ID":          df["archetype_id"].astype(str),
        "Version":     df["archetype_version"].astype(str),
        "Cluster":     df["cluster_size"].astype("Int64"),
        "Mean churn":  df["cluster_mean_churn_probability"],
        "Stability":   df.get("stability_vs_prior_version", pd.Series([None] * len(df))),
        "Method":      df["derivation_method"].astype(str),
    })

    col_config = {
        "Archetype":  st.column_config.TextColumn(width="medium"),
        "ID":         st.column_config.TextColumn(width="small", help="archetype_id"),
        "Version":    st.column_config.TextColumn(width="small"),
        "Cluster":    st.column_config.NumberColumn(format="%,d", width="small"),
        "Mean churn": st.column_config.ProgressColumn(
            format="%.2f", min_value=0.0, max_value=1.0, width="small"
        ),
        "Stability":  st.column_config.NumberColumn(format="%.2f", width="small",
            help="Jaccard-like overlap with the same archetype in the prior version (1.0 = identical)"),
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
    """Coerce a row of ``top_shap_features`` (struct or pyspark Row) to dict."""
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "asDict"):
        try:
            return item.asDict()
        except Exception:
            pass
    # plain object with attribute access (e.g. namedtuple-like)
    out: dict[str, Any] = {}
    for key in ("feature", "mean_shap", "mean_value", "direction"):
        if hasattr(item, key):
            out[key] = getattr(item, key)
    return out


def _split_drivers(features: Any) -> tuple[list[dict], list[dict]]:
    """Return ``(positive, negative)`` SHAP-driver dicts, ordered by |mean_shap|."""
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
            # No explicit direction — fall back to sign of mean_shap.
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


def _render_driver_row(
    driver: dict,
    *,
    direction: str,
    thresholds: dict,
    phrases: dict[str, dict[str, Any]],
    expander_key: str,
) -> None:
    feature = str(driver.get("feature") or "")
    phrase_meta = phrases.get(feature, {})
    display_phrase = (
        str(phrase_meta.get("business_phrase") or "").strip()
        or feature
    )
    mean_shap = driver.get("mean_shap")
    glyph = "▲" if direction == _DIRECTION_POSITIVE else "▼"
    cls = "pos" if direction == _DIRECTION_POSITIVE else "neg"

    # Summary row (always visible)
    st.markdown(
        f'<div class="feature-row {cls}">'
        f'<span class="glyph">{glyph}</span>'
        f'<span class="phrase">{escape(display_phrase)}</span>'
        f'<span class="shap">SHAP {_fmt_float(mean_shap, precision=4)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Technical detail (click-to-expand)
    threshold = _threshold_for(thresholds, feature)
    with st.expander("technical detail", expanded=False):
        meta_lines: list[str] = [f"**Column name** &nbsp;`{feature}`"]
        mean_value = driver.get("mean_value")
        if mean_value is not None:
            meta_lines.append(f"**Cluster mean value** &nbsp;{_fmt_float(mean_value, precision=4)}")
        if mean_shap is not None:
            meta_lines.append(f"**Cluster mean SHAP** &nbsp;{_fmt_float(mean_shap, precision=6)}")
        explicit_dir = driver.get("direction")
        if explicit_dir:
            meta_lines.append(f"**Direction** &nbsp;{explicit_dir}")
        if threshold:
            t_bits = []
            for tk in ("p25", "p50", "p75"):
                if tk in threshold and threshold[tk] is not None:
                    t_bits.append(f"{tk}={_fmt_float(threshold[tk], precision=3)}")
            if t_bits:
                meta_lines.append(f"**Cluster quantiles** &nbsp;{' · '.join(t_bits)}")
        for key, label in (
            ("source_dataset",   "Source dataset"),
            ("aggregation_kind", "Aggregation"),
            ("window_phrase",    "Window"),
            ("polarity",         "Polarity (training)"),
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
            if not s:
                continue
            meta_lines.append(f"**{label}** &nbsp;{s}")
        st.markdown("  \n".join(meta_lines))
    # The key is referenced to keep streamlit expander state stable per row.
    _ = expander_key  # noqa: F841 — kept for future per-row keying if needed


def _render_drivers_section(
    title: str,
    drivers: list[dict],
    *,
    direction: str,
    thresholds: dict,
    phrases: dict[str, dict[str, Any]],
    archetype_id: str,
) -> None:
    if not drivers:
        return
    st.markdown(
        f'<section class="catalog-section">'
        f'<h4>{escape(title)} · {len(drivers)}</h4>'
        '</section>',
        unsafe_allow_html=True,
    )
    for i, d in enumerate(drivers):
        _render_driver_row(
            d,
            direction=direction,
            thresholds=thresholds,
            phrases=phrases,
            expander_key=f"{archetype_id}::{direction}::{i}",
        )


def _render_detail(row: pd.Series, phrases: dict[str, dict[str, Any]]) -> None:
    name = str(row.get("archetype_name") or row.get("archetype_id"))
    aid = str(row.get("archetype_id"))
    version = str(row.get("archetype_version"))

    chips = [
        _chip("ID", aid),
        _chip("Version", version),
        _chip("Cluster size", _fmt_int(row.get("cluster_size"))),
        _chip("Mean churn", _fmt_pct(row.get("cluster_mean_churn_probability")), accent=True),
        _chip("Method", str(row.get("derivation_method") or "—")),
    ]
    stability = row.get("stability_vs_prior_version")
    try:
        if stability is not None and not pd.isna(stability):
            chips.append(_chip("Stability vs prior", _fmt_float(stability, precision=2)))
    except (TypeError, ValueError):
        pass
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
          <h3 class="title">{escape(name)}</h3>
          <p class="subtitle">{escape(aid)} · v{escape(version)}</p>
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

    _render_drivers_section(
        "Risk-driving features",
        pos,
        direction=_DIRECTION_POSITIVE,
        thresholds=thresholds,
        phrases=phrases,
        archetype_id=aid,
    )
    _render_drivers_section(
        "Protective features",
        neg,
        direction=_DIRECTION_NEGATIVE,
        thresholds=thresholds,
        phrases=phrases,
        archetype_id=aid,
    )


def render() -> None:
    st.markdown(
        '<p class="catalog-lead">'
        'Behavioural clusters the model has identified in the current '
        'feature space. Click a row to see the archetype’s description, '
        'rationale, and the SHAP drivers — split into risk-driving '
        '(pushes prediction toward churn) and protective. Expand any '
        'driver for its technical detail.'
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

    try:
        phrases_df = data.feature_business_phrases()
    except Exception:
        phrases_df = pd.DataFrame()
    phrases = _phrase_lookup(phrases_df)

    idx = _render_master_table(df)
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

    _render_detail(df.iloc[idx], phrases)
