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
    """Declares a table to join against the currently-selected entity."""
    name: str
    source: str          # table name in {catalog}.{schema}
    join_key: str        # column on `source` to match against the selected entity_id
    order_by: Optional[str] = None
    limit: int = 1


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
        data_sources.append(DataSource(
            name=name,
            source=cfg["source"],
            join_key=cfg["join_key"],
            order_by=cfg.get("order_by"),
            limit=int(cfg.get("limit", 1)),
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


def _h_upper(this, s):
    return (str(s) if not _is_missing(s) else "").upper()


def _h_lower(this, s):
    return (str(s) if not _is_missing(s) else "").lower()


HELPERS = {
    "fmt_currency":    _h_fmt_currency,
    "fmt_pct":         _h_fmt_pct,
    "fmt_int":         _h_fmt_int,
    "fmt_float":       _h_fmt_float,
    "fmt_date":        _h_fmt_date,
    "fmt_datetime":    _h_fmt_datetime,
    "risk_tier_class": _h_risk_tier_class,
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
