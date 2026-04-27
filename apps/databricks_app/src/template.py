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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

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
    """Load an HTML template. Empty/missing path falls back to the bundled default."""
    if path and os.path.exists(path):
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = _default_template_path().read_text(encoding="utf-8")

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
        return "dev-zero"
    try:
        zf = float(z)
    except Exception:
        return "dev-zero"
    if zf > 0:
        return "dev-pos"
    if zf < 0:
        return "dev-neg"
    return "dev-zero"


def _h_fmt_signed_z(this, z):
    if _is_missing(z):
        return "—"
    try:
        zf = float(z)
        return f"{zf:+.2f}σ"
    except Exception:
        return str(z)


HELPERS = {
    "fmt_currency":    _h_fmt_currency,
    "fmt_pct":         _h_fmt_pct,
    "fmt_int":         _h_fmt_int,
    "fmt_float":       _h_fmt_float,
    "fmt_date":        _h_fmt_date,
    "fmt_datetime":    _h_fmt_datetime,
    "risk_tier_class": _h_risk_tier_class,
    "fit_tier_label":  _h_fit_tier_label,
    "fit_tier_class":  _h_fit_tier_class,
    "dev_bar_pct":     _h_dev_bar_pct,
    "dev_sign_class":  _h_dev_sign_class,
    "fmt_signed_z":    _h_fmt_signed_z,
    "upper":           _h_upper,
    "lower":           _h_lower,
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
