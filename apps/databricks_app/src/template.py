"""Customer-profile template — HTML + Handlebars + YAML frontmatter.

A template is ONE HTML body that declares:
  * its data dependencies (tables to join on the selected entity), and
  * its HTML/CSS layout with Handlebars interpolation.

The default template shipped with the app renders a rich card using only
``v_account_explanation`` columns — no joins required. A dataset-specific
template stored in Unity Catalog (table
``{catalog}.{schema}.dashboard_template_overrides``, exposed by
``v_dashboard_template_active``) can declare additional UC tables to join
on ``entity_id``, and those rows become nested contexts accessible by name.

The HTML body itself is no longer read from a Volume FUSE path — the App
service principal cannot consistently access Volume mounts even with
``READ_VOLUME`` granted, so we route through the SQL warehouse instead.
The orchestrator in ``customer_profile.py`` fetches the body string via
``data.load_template_html_from_uc(...)`` and passes it to
``load_template_from_text`` here.

Built-in helpers: ``fmt_currency``, ``fmt_pct``, ``fmt_int``, ``fmt_float``,
``fmt_date``, ``fmt_datetime``, ``risk_tier_class``, ``upper``, ``lower``,
plus the SHAP/deviation helpers documented inline below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ``pybars`` is imported lazily inside render_html so that the frontmatter parser
# and helper utilities are usable in unit tests without the dependency.


@dataclass
class DataSource:
    """Declares a table to join against the currently-selected entity.

    ``as_list`` controls the shape passed into the template context:
        * ``False`` (default) — first matching row as ``dict[str, Any]``,
          matching the historical contract used by the bundled profile.
        * ``True`` — list of row-dicts. Use for fan-out children (e.g. the
          deviation-bar feature list) where the template iterates with
          ``{{#each}}``.
    """
    name: str
    source: str          # table name in {catalog}.{schema}
    join_key: str        # column on `source` to match against the selected entity_id
    order_by: Optional[str] = None
    limit: int = 1
    as_list: bool = False


@dataclass
class Template:
    body: str
    data_sources: list[DataSource] = field(default_factory=list)
    css: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a file into (frontmatter_dict, body_html). Tolerant of no frontmatter."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("---"):
        return {}, text
    close = None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\n\r") == "---":
            close = i
            break
    if close is None:
        return {}, text
    fm_raw = "".join(lines[1:close])
    body = "".join(lines[close + 1:])
    try:
        front = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError:
        front = {}
    return front, body


def _default_template_path() -> Path:
    return Path(__file__).parent / "default_profile.html"


def _default_css_path() -> Path:
    return Path(__file__).parent / "default_profile.css"


def _build_template_from_text(text: str) -> Template:
    front, body = _parse_frontmatter(text)
    data_sources: list[DataSource] = []
    for name, cfg in (front.get("data") or {}).items():
        as_list = bool(cfg.get("as_list", False))
        # When the template asks for a list, default to a generous limit
        # rather than the single-row default — most array uses (top-N
        # deviation rows, history slices) want more than one row.
        default_limit = 50 if as_list else 1
        data_sources.append(DataSource(
            name=name,
            source=cfg["source"],
            join_key=cfg["join_key"],
            order_by=cfg.get("order_by"),
            limit=int(cfg.get("limit", default_limit)),
            as_list=as_list,
        ))
    return Template(
        body=body,
        data_sources=data_sources,
        css=front.get("css") or "",
    )


def load_template_from_text(text: Optional[str]) -> Template:
    """Build a Template from raw HTML text. Empty/None falls back to the bundled default.

    This is the single entry point for template loading. The Streamlit
    customer-profile orchestrator fetches the HTML body via
    ``data.load_template_html_from_uc()`` and passes the result here; on
    miss (None) we read the bundled default file.
    """
    if not text:
        text = _default_template_path().read_text(encoding="utf-8")
        logger.info("no UC override available — using bundled default template")
    return _build_template_from_text(text)


# ---------------------------------------------------------------------------
# Handlebars helpers — available to every template
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return False


def _h_fmt_currency(this, v):
    if _is_missing(v):
        return "—"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)


def _h_fmt_pct(this, v):
    if _is_missing(v):
        return "—"
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return str(v)


def _h_fmt_int(this, v):
    if _is_missing(v):
        return "—"
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)


def _h_fmt_float(this, v, places=3):
    if _is_missing(v):
        return "—"
    try:
        return f"{float(v):,.{int(places)}f}"
    except Exception:
        return str(v)


def _h_fmt_date(this, v):
    if _is_missing(v):
        return "—"
    try:
        return pd.to_datetime(v).strftime("%Y-%m-%d")
    except Exception:
        return str(v)


def _h_fmt_datetime(this, v):
    if _is_missing(v):
        return "—"
    try:
        return pd.to_datetime(v).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(v)


def _h_risk_tier_class(this, tier):
    return {
        "High":   "risk-high",
        "Medium": "risk-medium",
        "Low":    "risk-low",
    }.get(tier, "risk-unknown")


_FIT_TIER_LABELS = {
    "auto":      "Auto-fit",
    "review":    "Manual review",
    "manual":    "Manual",
    "catch_all": "Default",
}


def _h_fit_tier_label(this, tier):
    if _is_missing(tier):
        return ""
    return _FIT_TIER_LABELS.get(str(tier), str(tier).replace("_", " ").title())


def _h_fit_tier_class(this, tier):
    if _is_missing(tier):
        return "fit-unknown"
    key = str(tier).strip().lower().replace(" ", "_")
    if key in _FIT_TIER_LABELS:
        return f"fit-{key}"
    return "fit-unknown"


def _h_upper(this, s):
    return (str(s) if not _is_missing(s) else "").upper()


def _h_lower(this, s):
    return (str(s) if not _is_missing(s) else "").lower()


# Deviation panel — turns a z-score into a percentage width and a sign-class
# so the template can draw bidirectional bars without inline conditionals.

_DEVIATION_BAR_CAP = 3.0  # |z| above this saturates the bar at 100%


def _h_dev_bar_pct(this, z):
    """Map |z| → 0–100% bar width, capped at ``_DEVIATION_BAR_CAP`` sigma."""
    if _is_missing(z):
        return "0"
    try:
        magnitude = min(abs(float(z)) / _DEVIATION_BAR_CAP, 1.0)
        return f"{magnitude * 100:.1f}"
    except Exception:
        return "0"


def _h_dev_sign_class(this, z):
    if _is_missing(z):
        return "cr-dev-zero"
    try:
        zf = float(z)
    except Exception:
        return "cr-dev-zero"
    if zf > 0:
        return "cr-dev-pos"
    if zf < 0:
        return "cr-dev-neg"
    return "cr-dev-zero"


def _h_fmt_signed_z(this, z):
    if _is_missing(z):
        return "—"
    try:
        zf = float(z)
        return f"{zf:+.2f}σ"
    except Exception:
        return str(z)


# SHAP panel — turns a per-feature signed contribution into a bar width
# (relative to the largest |contribution| in the row) and a sign class so
# the template can draw bidirectional bars and a signed label.
#
# Bar normalization: scaling by the row's own max absolute contribution
# keeps the panel readable across models with very different SHAP
# magnitudes (logit vs. probability vs. raw margin) and always shows the
# top driver at 100%. ``shap_bar_pct(contribution, drivers)`` is the
# template signature -- we look at every contribution in ``drivers`` to
# find the local max.

def _shap_max_abs(drivers) -> float:
    """Largest ``|shap_contribution|`` among the items in ``drivers``.

    Falls back to ``1.0`` so a degenerate row (all zeros / all None) still
    renders a non-broken bar element instead of dividing by zero.
    """
    if not drivers:
        return 1.0
    best = 0.0
    for item in drivers:
        try:
            val = item.get("shap_contribution") if hasattr(item, "get") else getattr(item, "shap_contribution", None)
        except Exception:
            val = None
        if val is None:
            continue
        try:
            mag = abs(float(val))
        except Exception:
            continue
        if mag > best:
            best = mag
    return best if best > 0 else 1.0


def _shap_total_abs(drivers) -> float:
    """Sum of ``|shap_contribution|`` across ``drivers``.

    Used by ``shap_share_pct`` so labelled percentages sum to 100% across
    the visible rows. Falls back to ``1.0`` to keep arithmetic safe when
    the panel is empty / all-zero.
    """
    if not drivers:
        return 1.0
    total = 0.0
    for item in drivers:
        try:
            val = item.get("shap_contribution") if hasattr(item, "get") else getattr(item, "shap_contribution", None)
        except Exception:
            val = None
        if val is None:
            continue
        try:
            total += abs(float(val))
        except Exception:
            continue
    return total if total > 0 else 1.0


def _h_shap_bar_pct(this, contribution, drivers):
    """Map a signed ``shap_contribution`` to a 0–100% bar width, scaled by
    the row's own largest absolute contribution. ``drivers`` is the full
    array of structs (so the template can pass ``account_top_shap_features``
    as the second argument and the helper figures out the local max).
    """
    if _is_missing(contribution):
        return "0"
    try:
        cf = abs(float(contribution))
    except Exception:
        return "0"
    cap = _shap_max_abs(drivers)
    return f"{min(cf / cap, 1.0) * 100:.1f}"


def _h_shap_share_pct(this, contribution, drivers):
    """Row's share of total visible |SHAP| as a formatted percentage.

    Sums to 100% across the rows shown in the panel so the label honestly
    reads "this feature accounts for X% of what's pushing the prediction
    away from baseline among the top features shown." Renders as ``12%``
    for normal magnitudes and ``<1%`` for tiny tails so a barely-visible
    bar isn't accompanied by a mathematically misleading ``0%``.
    Empty string when the contribution is missing -- callers pair the
    helper with an ``{{#if shap_contribution}}`` so the label disappears
    cleanly on rows that have no SHAP data.
    """
    if _is_missing(contribution):
        return ""
    try:
        cf = abs(float(contribution))
    except Exception:
        return ""
    total = _shap_total_abs(drivers)
    share = (cf / total) * 100.0
    if share < 1.0 and share > 0.0:
        return "<1%"
    return f"{round(share)}%"


def _h_shap_sign_class(this, contribution):
    """``cr-shap-pos`` (pushes toward churn / target=1) / ``cr-shap-neg``
    (pushes away) / ``cr-shap-zero``. Class names are ``cr-`` prefixed to
    match the CSS selectors in ``default_profile.css``."""
    if _is_missing(contribution):
        return "cr-shap-zero"
    try:
        cf = float(contribution)
    except Exception:
        return "cr-shap-zero"
    if cf > 0:
        return "cr-shap-pos"
    if cf < 0:
        return "cr-shap-neg"
    return "cr-shap-zero"


def _h_fmt_signed_shap(this, contribution):
    """Two-decimal-place signed contribution (e.g. ``+0.14``).

    Two decimals is the right operational precision for SHAP log-odds in
    this dashboard: the upstream contributions sit in roughly [-1, 1],
    so .2f keeps the meaningful magnitude while dropping a trailing
    decimal that read as false precision.
    """
    if _is_missing(contribution):
        return "—"
    try:
        cf = float(contribution)
        return f"{cf:+.2f}"
    except Exception:
        return str(contribution)


def _h_fmt_shap_value(this, value):
    """Raw feature value as displayed alongside its SHAP contribution.

    Uses 3 decimal places for floats / scientific notation for very small
    values and integer formatting for whole numbers; falls back to
    ``str()`` for non-numeric values (booleans, categorical strings).
    """
    if _is_missing(value):
        return "—"
    try:
        vf = float(value)
    except Exception:
        return str(value)
    if abs(vf - int(vf)) < 1e-9 and abs(vf) < 1e9:
        return f"{int(vf):,}"
    if abs(vf) < 1e-3 and vf != 0.0:
        return f"{vf:.2e}"
    return f"{vf:.3f}"


# Quantile-band classification: where does this customer's raw feature value
# sit relative to the training population? 5 buckets (very low / low /
# typical / high / very high) computed from q05/q25/q75/q95. Falls back to
# an EMPTY string when any quantile is missing or the value is non-numeric
# so the band-pill template block can use ``{{#if quantile_band_text}}`` to
# skip rendering altogether — no em-dash placeholder reaches the panel.

_BAND_VERY_LOW = "very low"
_BAND_LOW = "low"
_BAND_TYPICAL = "typical"
_BAND_HIGH = "high"
_BAND_VERY_HIGH = "very high"
_BAND_UNKNOWN = ""


def _h_quantile_band(this, value, q05, q25, q75, q95):
    if _is_missing(value):
        return _BAND_UNKNOWN
    try:
        vf = float(value)
    except (TypeError, ValueError):
        return _BAND_UNKNOWN
    quantiles = (q05, q25, q75, q95)
    if any(_is_missing(q) for q in quantiles):
        return _BAND_UNKNOWN
    try:
        a, b, c, d = (float(q) for q in quantiles)
    except (TypeError, ValueError):
        return _BAND_UNKNOWN
    if vf < a:
        return _BAND_VERY_LOW
    if vf < b:
        return _BAND_LOW
    if vf < c:
        return _BAND_TYPICAL
    if vf < d:
        return _BAND_HIGH
    return _BAND_VERY_HIGH


_BAND_CLASS = {
    _BAND_VERY_LOW:  "cr-band-very-low",
    _BAND_LOW:       "cr-band-low",
    _BAND_TYPICAL:   "cr-band-typical",
    _BAND_HIGH:      "cr-band-high",
    _BAND_VERY_HIGH: "cr-band-very-high",
    "":              "cr-band-unknown",
}


def _h_band_class(this, band):
    if _is_missing(band):
        return _BAND_CLASS[""]
    return _BAND_CLASS.get(str(band), _BAND_CLASS[""])


# Direction phrase + glyph + tooltip — the bar's colour already shows
# direction, so we no longer render a permanent ``risk-driving`` /
# ``protective`` pill column. The phrase moves to a hover tooltip
# (``title`` attribute on the bar) and is summarized visually by an
# arrow glyph rendered inside the bar's leading edge.

_DIR_RISK = "risk-driving"
_DIR_PROTECTIVE = "protective"
_DIR_NEUTRAL = "neutral"

# Anything below this magnitude (in log-odds) is rounded to "neutral" — the
# bar would be barely visible at this level anyway, and labelling a 0.001
# contribution as "risk-driving" overstates its weight.
_DIR_NEUTRAL_EPSILON = 1e-3

_DIR_GLYPH = {
    _DIR_RISK:       "↑",
    _DIR_PROTECTIVE: "↓",
    _DIR_NEUTRAL:    "",
}

_DIR_TOOLTIP = {
    _DIR_RISK:       "Risk-driving — pushes the prediction toward churn",
    _DIR_PROTECTIVE: "Protective — pushes the prediction toward retention",
    _DIR_NEUTRAL:    "Near-zero contribution",
}


def _h_direction_phrase(this, contribution):
    if _is_missing(contribution):
        return _DIR_NEUTRAL
    try:
        cf = float(contribution)
    except (TypeError, ValueError):
        return _DIR_NEUTRAL
    if abs(cf) < _DIR_NEUTRAL_EPSILON:
        return _DIR_NEUTRAL
    return _DIR_RISK if cf > 0 else _DIR_PROTECTIVE


_DIR_CLASS = {
    _DIR_RISK:       "cr-dir-risk",
    _DIR_PROTECTIVE: "cr-dir-protective",
    _DIR_NEUTRAL:    "cr-dir-neutral",
}


def _h_direction_class(this, contribution):
    return _DIR_CLASS[_h_direction_phrase(this, contribution)]


def _h_direction_glyph(this, contribution):
    """Up arrow / down arrow / empty string depending on contribution sign.

    Rendered inside the bar's leading edge so direction is encoded
    visually even when the per-row tooltip isn't being hovered. Empty
    string for neutral so degenerate contributions don't add visual noise.
    """
    return _DIR_GLYPH[_h_direction_phrase(this, contribution)]


def _h_direction_tooltip(this, contribution):
    """Long-form tooltip text for the bar's ``title`` attribute.

    Native browser tooltip (no JS): hovering the bar surfaces the
    risk-driving / protective phrase. Used to be a permanent pill column;
    moved to hover so the panel reclaims a column for the percentage label.
    """
    return _DIR_TOOLTIP[_h_direction_phrase(this, contribution)]


def _h_fmt_raw_value(this, value):
    """Customer's raw, unscaled feature value.

    Returns an EMPTY string for missing values so the raw-value cell
    collapses cleanly via ``{{#if raw_value_text}}`` rather than rendering
    a ``—`` placeholder. Numeric values format as integer / 2-decimal
    float / scientific. Two decimals (was three) matches the operator
    expectation that the dashboard reads as round numbers, not pseudo-
    precise SHAP scaffolding.
    """
    if _is_missing(value):
        return ""
    try:
        vf = float(value)
    except Exception:
        return str(value)
    if abs(vf - int(vf)) < 1e-9 and abs(vf) < 1e9:
        return f"{int(vf):,}"
    if abs(vf) < 1e-3 and vf != 0.0:
        return f"{vf:.1e}"
    return f"{vf:.2f}"


# Derivation prose — translate the curated ``aggregation_kind`` from
# ``feature_meta`` into a CSM-readable phrase. Falls back to a name-based
# inference when the meta row is absent (older runs that never wrote
# feature_meta). Window context comes from the meta row's
# ``window_phrase`` when available.

_AGG_KIND_PHRASES = {
    "count":           "Count of {source} occurrences {window}",
    "count_distinct":  "Distinct values of {source} observed {window}",
    "sum":             "Total {source} {window}",
    "avg":             "Average {source} {window}",
    "max":             "Maximum {source} observed {window}",
    "min":             "Minimum {source} observed {window}",
    "last":            "Most recent value of {source} {window}",
    "first":           "First observed value of {source} {window}",
    "ratio":           "Ratio derived from {source} {window}",
    "recency_days":    "Days since the most recent {source} event",
    "derived_datetime":"Calendar-derived from {source} (e.g. day-of-week, hour)",
    "passthrough":     "Raw value of {source} from the source table",
}

# Heuristics for inferring derivation when feature_meta is missing — based
# on conventions used in the framework's gold materializer.
_DERIVATION_NAME_HINTS = (
    ("_velocity",   "Rate of change in {source} over time"),
    ("_per_day",    "Per-day rate of {source}"),
    ("_recency",    "Days since the last {source} event"),
    ("_days_since", "Days since the last {source} event"),
    ("_ratio",      "Ratio derived from {source}"),
    ("_pct",        "Percentage derived from {source}"),
    ("_count",      "Count of {source} occurrences"),
    ("_sum",        "Total {source}"),
    ("_avg",        "Average {source}"),
    ("_mean",       "Average {source}"),
    ("_max",        "Maximum {source} observed"),
    ("_min",        "Minimum {source} observed"),
    ("_first",      "First observed value of {source}"),
    ("_last",       "Most recent value of {source}"),
    ("_cos",        "Cyclical encoding (cosine) of {source}"),
    ("_sin",        "Cyclical encoding (sine) of {source}"),
)


def _derivation_window_clause(window_phrase):
    if _is_missing(window_phrase):
        return ""
    text = str(window_phrase).strip()
    if not text:
        return ""
    return f"over {text}"


def _h_derivation_phrase(this, aggregation_kind, feature_name, source_columns=None, window_phrase=None):
    """Return one-sentence derivation prose. ``source_columns`` may be a
    list (the typical case) or a single string; the first non-empty entry
    is used as ``{source}``. Falls back to the feature name when the
    source list is empty/missing."""
    source = ""
    if isinstance(source_columns, (list, tuple)):
        for entry in source_columns:
            if entry:
                source = str(entry)
                break
    elif source_columns:
        source = str(source_columns)
    if not source and not _is_missing(feature_name):
        source = str(feature_name)
    window = _derivation_window_clause(window_phrase)
    template = None
    if not _is_missing(aggregation_kind):
        template = _AGG_KIND_PHRASES.get(str(aggregation_kind))
    if template is None and not _is_missing(feature_name):
        lower = str(feature_name).lower()
        for hint, phrase in _DERIVATION_NAME_HINTS:
            if hint in lower:
                template = phrase
                break
    if template is None:
        return f"Derived from {source}" if source else ""
    out = template.format(source=source, window=window).strip()
    while "  " in out:
        out = out.replace("  ", " ")
    return out


HELPERS = {
    "fmt_currency":      _h_fmt_currency,
    "fmt_pct":           _h_fmt_pct,
    "fmt_int":           _h_fmt_int,
    "fmt_float":         _h_fmt_float,
    "fmt_date":          _h_fmt_date,
    "fmt_datetime":      _h_fmt_datetime,
    "risk_tier_class":   _h_risk_tier_class,
    "fit_tier_label":    _h_fit_tier_label,
    "fit_tier_class":    _h_fit_tier_class,
    "dev_bar_pct":       _h_dev_bar_pct,
    "dev_sign_class":    _h_dev_sign_class,
    "fmt_signed_z":      _h_fmt_signed_z,
    "shap_bar_pct":      _h_shap_bar_pct,
    "shap_share_pct":    _h_shap_share_pct,
    "shap_sign_class":   _h_shap_sign_class,
    "fmt_signed_shap":   _h_fmt_signed_shap,
    "fmt_shap_value":    _h_fmt_shap_value,
    "fmt_raw_value":     _h_fmt_raw_value,
    "quantile_band":     _h_quantile_band,
    "band_class":        _h_band_class,
    "direction_phrase":  _h_direction_phrase,
    "direction_class":   _h_direction_class,
    "direction_glyph":   _h_direction_glyph,
    "direction_tooltip": _h_direction_tooltip,
    "derivation_phrase": _h_derivation_phrase,
    "upper":             _h_upper,
    "lower":             _h_lower,
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_html(template: Template, context: dict) -> str:
    """Compile the template body and render with the supplied context dict."""
    import pybars  # lazy — avoids hard dep in unit tests / parser-only usage
    compiler = pybars.Compiler()
    compiled = compiler.compile(template.body)
    return compiled(context, helpers=HELPERS)


def bundle_css(template: Template) -> str:
    """Return a `<style>…</style>` block: default CSS + template-specific CSS."""
    default = _default_css_path().read_text(encoding="utf-8") if _default_css_path().exists() else ""
    css = default + "\n" + (template.css or "")
    return f"<style>{css}</style>"
