"""Customer-profile template — HTML + Handlebars + YAML frontmatter.

A template is ONE file that declares:
  * its data dependencies (tables to join on the selected entity), and
  * its HTML/CSS layout with Handlebars interpolation (`{{col}}`, `{{nested.col}}`, helpers, `{{#if}}`).

The default template shipped with the app renders a rich card using only
`v_account_explanation` columns — no joins required. A custom template at
`CR_PROFILE_TEMPLATE_PATH` can declare additional Unity Catalog tables to join
on `entity_id`, and those rows become nested contexts accessible by name.

Example custom template (save as `.html`):

```html
---
data:
  account:
    source: "gold_features_cust_emai_aggr__26e8271"
    join_key: "account_id"
  latest_email:
    source: "bronze_event_email_events"
    join_key: "account_id"
    order_by: "event_timestamp DESC"
    limit: 1
css: |
  .mrr { color: green; font-weight: bold; }
---
<div class="cr-card">
  <h1>{{entity_id}}</h1>
  <p>MRR: <span class="mrr">{{fmt_currency account.mrr}}</span></p>
  {{#if recommended}}<span class="ok">✅ recommended</span>{{/if}}
</div>
```

Built-in helpers: `fmt_currency`, `fmt_pct`, `fmt_int`, `fmt_float`, `fmt_date`,
`fmt_datetime`, `risk_tier_class`, `upper`, `lower`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# `pybars` is imported lazily inside render_html so that the frontmatter parser
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
    # Diagnostic captured at load time so the empty-state panel can show why
    # a per-dataset template wasn't loaded (path missing, permission denied,
    # parse error, ...) — operators can't always reach the App's Logs tab.
    diagnostic: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a file into (frontmatter_dict, body_html). Tolerant of no frontmatter."""
    if not text.startswith("---"):
        return {}, text
    # Find the closing delimiter after the first line
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


def load_template(path: Optional[str]) -> Template:
    """Load an HTML template. Empty/missing path falls back to the bundled default.

    The fallback used to be silent — a permission error on the requested path
    looked identical to "no path configured", which made the empty-state panel
    impossible to diagnose without local repro. We now log the exact reason at
    INFO/WARNING level so it shows up in the Databricks Apps "Logs" tab AND
    we record the same info in ``Template.diagnostic`` so the empty-state
    panel can render it (operators sometimes can't reach the Logs tab).
    """
    text: Optional[str] = None
    diagnostic: dict = {
        "requested_path": path or "",
        "default_template_path": str(_default_template_path()),
        "loaded_from": "",
        "status": "no_path_set",
        "error_type": "",
        "error_message": "",
        "parent_dir_exists": False,
        "parent_dir_listing": "",
    }
    if path:
        diagnostic["status"] = "attempting"
        try:
            parent = Path(path).parent
            diagnostic["parent_dir_exists"] = parent.exists()
            if parent.exists():
                try:
                    listing = sorted(p.name for p in parent.iterdir())[:20]
                    diagnostic["parent_dir_listing"] = ", ".join(listing) if listing else "(empty)"
                except OSError as exc:
                    diagnostic["parent_dir_listing"] = f"(listing failed: {exc})"
        except OSError as exc:
            diagnostic["parent_dir_listing"] = f"(parent stat failed: {exc})"
        try:
            text = Path(path).read_text(encoding="utf-8")
            diagnostic["status"] = "loaded"
            diagnostic["loaded_from"] = path
            logger.info("loaded profile template -> %s", path)
        except FileNotFoundError as exc:
            diagnostic["status"] = "fallback_default"
            diagnostic["error_type"] = "FileNotFoundError"
            diagnostic["error_message"] = str(exc)
            logger.warning(
                "profile template path does not exist: %s — falling back to bundled default",
                path,
            )
        except PermissionError as exc:
            diagnostic["status"] = "fallback_default"
            diagnostic["error_type"] = "PermissionError"
            diagnostic["error_message"] = str(exc)
            logger.warning(
                "profile template read denied (%s) for %s — grant READ_VOLUME to "
                "the app's service principal on the parent volume (or wait for "
                "the deployed grant to propagate, which may require a Stop/Start "
                "of the app)",
                exc,
                path,
            )
        except OSError as exc:
            diagnostic["status"] = "fallback_default"
            diagnostic["error_type"] = type(exc).__name__
            diagnostic["error_message"] = str(exc)
            logger.warning(
                "profile template read failed (%s) for %s — falling back to bundled default",
                exc,
                path,
            )
    if text is None:
        text = _default_template_path().read_text(encoding="utf-8")
        diagnostic["loaded_from"] = str(_default_template_path())
        if not path:
            diagnostic["status"] = "no_path_set"
            logger.info("no CR_PROFILE_TEMPLATE_PATH set — using bundled default template")

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
        diagnostic=diagnostic,
    )


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
    """Three-significant-digit signed contribution (e.g. ``+0.142``)."""
    if _is_missing(contribution):
        return "—"
    try:
        cf = float(contribution)
        return f"{cf:+.3f}"
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
# "—" when any quantile is missing or the value is non-numeric.

_BAND_VERY_LOW = "very low"
_BAND_LOW = "low"
_BAND_TYPICAL = "typical"
_BAND_HIGH = "high"
_BAND_VERY_HIGH = "very high"
_BAND_UNKNOWN = "—"


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
    _BAND_UNKNOWN:   "cr-band-unknown",
}


def _h_band_class(this, band):
    if _is_missing(band):
        return _BAND_CLASS[_BAND_UNKNOWN]
    return _BAND_CLASS.get(str(band), _BAND_CLASS[_BAND_UNKNOWN])


# Direction phrase — translate the SHAP sign into business language.
# The bar's colour already shows direction; this label puts a word on it
# so a CSM doesn't have to interpret a log-odds number.

_DIR_RISK = "risk-driving"
_DIR_PROTECTIVE = "protective"
_DIR_NEUTRAL = "neutral"

# Anything below this magnitude (in log-odds) is rounded to "neutral" — the
# bar would be barely visible at this level anyway, and labelling a 0.001
# contribution as "risk-driving" overstates its weight.
_DIR_NEUTRAL_EPSILON = 1e-3


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


def _h_fmt_raw_value(this, value):
    """Customer's raw, unscaled feature value. Integer for whole numbers,
    3 decimals for floats, scientific notation for very small magnitudes."""
    return _h_fmt_shap_value(this, value)


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
    # Compact double spaces left over when window is empty.
    while "  " in out:
        out = out.replace("  ", " ")
    return out


HELPERS = {
    "fmt_currency":     _h_fmt_currency,
    "fmt_pct":          _h_fmt_pct,
    "fmt_int":          _h_fmt_int,
    "fmt_float":        _h_fmt_float,
    "fmt_date":         _h_fmt_date,
    "fmt_datetime":     _h_fmt_datetime,
    "risk_tier_class":  _h_risk_tier_class,
    "fit_tier_label":   _h_fit_tier_label,
    "fit_tier_class":   _h_fit_tier_class,
    "dev_bar_pct":      _h_dev_bar_pct,
    "dev_sign_class":   _h_dev_sign_class,
    "fmt_signed_z":     _h_fmt_signed_z,
    "shap_bar_pct":     _h_shap_bar_pct,
    "shap_sign_class":  _h_shap_sign_class,
    "fmt_signed_shap":  _h_fmt_signed_shap,
    "fmt_shap_value":   _h_fmt_shap_value,
    "fmt_raw_value":    _h_fmt_raw_value,
    "quantile_band":    _h_quantile_band,
    "band_class":       _h_band_class,
    "direction_phrase": _h_direction_phrase,
    "direction_class":  _h_direction_class,
    "derivation_phrase":_h_derivation_phrase,
    "upper":            _h_upper,
    "lower":            _h_lower,
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
