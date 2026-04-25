from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

from jinja2 import BaseLoader, Environment

from .models import (
    BronzeEventConfig,
    BronzeLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    TransformationStep,
)

SECTION_MAP = {
    PipelineTransformationType.IMPUTE_NULL: "Missing Value Analysis",
    PipelineTransformationType.DROP_COLUMN: "Missing Value Analysis",
    PipelineTransformationType.CAP_OUTLIER: "Global Outlier Detection",
    PipelineTransformationType.WINSORIZE: "Global Outlier Detection",
    PipelineTransformationType.SEGMENT_AWARE_CAP: "Segment-Aware Outlier Analysis",
    PipelineTransformationType.LOG_TRANSFORM: "Feature Distributions",
    PipelineTransformationType.SQRT_TRANSFORM: "Feature Distributions",
    PipelineTransformationType.YEO_JOHNSON: "Feature Distributions",
    PipelineTransformationType.CAP_THEN_LOG: "Feature Distributions",
    PipelineTransformationType.ZERO_INFLATION_HANDLING: "Feature Distributions",
    PipelineTransformationType.ENCODE: "Categorical Feature Analysis",
    PipelineTransformationType.SCALE: "Feature-Target Correlations",
    PipelineTransformationType.FEATURE_SELECT: "Feature Selection Recommendations",
    PipelineTransformationType.DERIVED_COLUMN: "Feature Engineering Recommendations",
    PipelineTransformationType.TYPE_CAST: "Data Consistency Checks",
    PipelineTransformationType.FILTER: "Data Quality Filters",
}

ANCHOR_MAP = {
    PipelineTransformationType.IMPUTE_NULL: "2.5-Missing-Value-Analysis",
    PipelineTransformationType.DROP_COLUMN: "2.5-Missing-Value-Analysis",
    PipelineTransformationType.CAP_OUTLIER: "2.8-Global-Outlier-Detection",
    PipelineTransformationType.WINSORIZE: "2.8-Global-Outlier-Detection",
    PipelineTransformationType.SEGMENT_AWARE_CAP: "2.7-Segment-Aware-Outlier-Analysis",
    PipelineTransformationType.LOG_TRANSFORM: "5.4-Feature-Distributions-by-Retention-Status",
    PipelineTransformationType.SQRT_TRANSFORM: "5.4-Feature-Distributions-by-Retention-Status",
    PipelineTransformationType.YEO_JOHNSON: "5.4-Feature-Distributions-by-Retention-Status",
    PipelineTransformationType.CAP_THEN_LOG: "5.4-Feature-Distributions-by-Retention-Status",
    PipelineTransformationType.ZERO_INFLATION_HANDLING: "5.4-Feature-Distributions-by-Retention-Status",
    PipelineTransformationType.ENCODE: "5.6-Categorical-Feature-Analysis",
    PipelineTransformationType.SCALE: "5.5-Feature-Target-Correlations",
    PipelineTransformationType.FEATURE_SELECT: "5.9.1-Feature-Selection-Recommendations",
    PipelineTransformationType.DERIVED_COLUMN: "5.9.4-Feature-Engineering-Recommendations",
    PipelineTransformationType.TYPE_CAST: "2.11-Data-Consistency-Checks",
}

DEFAULT_NOTEBOOK_MAP = {
    PipelineTransformationType.IMPUTE_NULL: "02_source_integrity",
    PipelineTransformationType.DROP_COLUMN: "02_source_integrity",
    PipelineTransformationType.CAP_OUTLIER: "02_source_integrity",
    PipelineTransformationType.WINSORIZE: "02_source_integrity",
    PipelineTransformationType.SEGMENT_AWARE_CAP: "02_source_integrity",
    PipelineTransformationType.TYPE_CAST: "02_source_integrity",
    PipelineTransformationType.LOG_TRANSFORM: "05_relationship_analysis",
    PipelineTransformationType.SQRT_TRANSFORM: "05_relationship_analysis",
    PipelineTransformationType.YEO_JOHNSON: "05_relationship_analysis",
    PipelineTransformationType.CAP_THEN_LOG: "05_relationship_analysis",
    PipelineTransformationType.ZERO_INFLATION_HANDLING: "05_relationship_analysis",
    PipelineTransformationType.ENCODE: "05_relationship_analysis",
    PipelineTransformationType.SCALE: "05_relationship_analysis",
    PipelineTransformationType.FEATURE_SELECT: "05_relationship_analysis",
    PipelineTransformationType.DERIVED_COLUMN: "05_relationship_analysis",
    PipelineTransformationType.FILTER: "04_column_deep_dive",
}


_docs_base: str = "docs"


def render_python_literal(value) -> str:
    import math
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('-inf')" if value < 0 else "float('inf')"
        return repr(value)
    if isinstance(value, dict):
        body = ", ".join(
            f"{render_python_literal(k)}: {render_python_literal(v)}"
            for k, v in value.items()
        )
        return "{" + body + "}"
    if isinstance(value, (list, tuple)):
        body = ", ".join(render_python_literal(v) for v in value)
        if isinstance(value, tuple):
            suffix = "," if len(value) == 1 else ""
            return "(" + body + suffix + ")"
        return "[" + body + "]"
    return repr(value)


def _notebook_title(notebook: str) -> str:
    name = notebook.split("_", 1)[1] if "_" in notebook else notebook
    return name.replace("_", " ").title()


def provenance_docstring(step: TransformationStep) -> str:
    notebook = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
    if not notebook:
        return ""
    title = _notebook_title(notebook)
    anchor = ANCHOR_MAP.get(step.type)
    section = SECTION_MAP.get(step.type)
    base = _docs_base
    if anchor:
        return f"{title} {section}\n    {base}/{notebook}.html#{anchor}"
    return f"{title}\n    {base}/{notebook}.html"


def provenance_docstring_block(steps) -> str:
    seen = set()
    entries = []
    for step in steps:
        key = provenance_key(step)
        if not key or key in seen:
            continue
        seen.add(key)
        entry = provenance_docstring(step)
        if entry:
            entries.append(entry)
    if not entries:
        return ""
    body = "\n    ".join(entries)
    return f'    """\n    {body}\n    """'


def provenance_key(step: TransformationStep) -> str:
    notebook = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
    section = SECTION_MAP.get(step.type, "")
    return f"{notebook}:{section}" if notebook else ""


class StepGrouper:
    _TYPE_TO_FUNC = {
        PipelineTransformationType.DROP_COLUMN: "drop_unusable_columns",
        PipelineTransformationType.IMPUTE_NULL: "impute_remaining_nulls",
        PipelineTransformationType.CAP_OUTLIER: "cap_outliers",
        PipelineTransformationType.TYPE_CAST: "apply_type_casts",
        PipelineTransformationType.WINSORIZE: "winsorize_outliers",
        PipelineTransformationType.SEGMENT_AWARE_CAP: "cap_segment_aware_outliers",
        PipelineTransformationType.LOG_TRANSFORM: "apply_log_transforms",
        PipelineTransformationType.SQRT_TRANSFORM: "apply_sqrt_transforms",
        PipelineTransformationType.ZERO_INFLATION_HANDLING: "handle_zero_inflation",
        PipelineTransformationType.CAP_THEN_LOG: "apply_cap_then_log_transforms",
        PipelineTransformationType.YEO_JOHNSON: "apply_power_transforms",
        PipelineTransformationType.FEATURE_SELECT: "apply_feature_selection",
        PipelineTransformationType.FILTER: "apply_filters",
    }

    _DERIVED_ACTION_TO_FUNC = {
        "ratio": "create_ratio_features",
        "interaction": "create_interaction_features",
        "composite": "create_composite_features",
    }

    @classmethod
    def group(cls, steps: List[TransformationStep]) -> List[Tuple[str, List[TransformationStep]]]:
        if not steps:
            return []
        groups: OrderedDict[str, List[TransformationStep]] = OrderedDict()
        for step in steps:
            groups.setdefault(cls._func_name(step), []).append(step)
        return list(groups.items())

    @classmethod
    def _func_name(cls, step: TransformationStep) -> str:
        if step.type == PipelineTransformationType.DERIVED_COLUMN:
            action = step.parameters.get("action", "ratio")
            return cls._DERIVED_ACTION_TO_FUNC.get(action, f"create_{action}_features")
        return cls._TYPE_TO_FUNC.get(step.type, f"apply_{step.type.value}")


group_steps = StepGrouper.group


class InlineLoader(BaseLoader):
    def __init__(self, templates: dict):
        self._templates = templates

    def get_source(self, environment, template):
        if template in self._templates:
            return self._templates[template], template, lambda: True
        raise Exception(f"Template {template} not found")


TEMPLATES = {
    "config.py.j2": """import os
from pathlib import Path

PIPELINE_NAME = "{{ config.name }}"
COMPOSITE_NAME = "{{ config.composite_name or config.name }}"
TARGET_COLUMN = "{{ config.target_column }}"
TIMESTAMP_COLUMN = "event_timestamp"
OUTPUT_DIR = Path("{{ config.output_dir }}")

# Iteration tracking
ITERATION_ID = {{ '"%s"' % config.iteration_id if config.iteration_id else 'None' }}
PARENT_ITERATION_ID = {{ '"%s"' % config.parent_iteration_id if config.parent_iteration_id else 'None' }}

# Recommendations hash for experiment tracking
RECOMMENDATIONS_HASH = {{ '"%s"' % config.recommendations_hash if config.recommendations_hash else 'None' }}


def _find_project_root():
    path = Path(__file__).parent
    for _ in range(10):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
        path = path.parent
    return Path(__file__).parent


PROJECT_ROOT = _find_project_root()

# Experiments directory - all artifacts (data, mlruns, feast) go here
# Override with CR_EXPERIMENTS_DIR environment variable for Databricks/custom locations
_default_experiments = {{ '"%s"' % config.experiments_dir if config.experiments_dir else '"experiments"' }}
EXPERIMENTS_DIR = Path(os.environ.get("CR_EXPERIMENTS_DIR", str(PROJECT_ROOT / _default_experiments)))

# Documentation base URL for provenance links in generated code
# Local: file:// URI to HTML docs (from export_tutorial_html.py)
# Databricks: set to workspace notebook path for exploration report
DOCS_BASE_URL = os.environ.get("CR_DOCS_BASE_URL", str(EXPERIMENTS_DIR / "docs"))

# Production output directory - all pipeline writes go here
# Override with CR_PRODUCTION_DIR environment variable
_default_production = {{ '"%s"' % config.production_dir if config.production_dir else 'str(EXPERIMENTS_DIR)' }}
PRODUCTION_DIR = Path(os.environ.get("CR_PRODUCTION_DIR", _default_production))

# MLflow tracking - using SQLite backend (recommended over deprecated file-based backend)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{EXPERIMENTS_DIR / 'mlruns.db'}")
MLFLOW_ARTIFACT_ROOT = str(EXPERIMENTS_DIR / "mlruns" / "artifacts")

# Feast feature store configuration - stored in experiments directory
FEAST_REPO_PATH = str(PRODUCTION_DIR / "feature_repo")
FEAST_FEATURE_VIEW = "{{ config.feast.feature_view_name if config.feast else 'featureset_' + (config.composite_name or config.name) }}"
FEAST_ENTITY_NAME = "{{ config.feast.entity_name if config.feast else 'customer' }}"
ENTITY_KEY = "{{ config.feast.entity_key if config.feast else 'entity_id' }}"
FEAST_TIMESTAMP_COL = "{{ config.feast.timestamp_column if config.feast else 'event_timestamp' }}"
FEAST_TTL_DAYS = {{ config.feast.ttl_days if config.feast else 365 }}

# Source paths - findings directory is a subfolder of experiments
FINDINGS_DIR = EXPERIMENTS_DIR / "findings"

SOURCES = {
{% for source in config.sources %}
    "{{ source.name }}": {
        "path": "{{ source.raw_source_path }}",
        "format": "{{ source.format }}",
        "entity_key": "{{ source.entity_key }}",
{% if source.time_column %}
        "time_column": "{{ source.time_column }}",
{% endif %}
        "is_event_level": {{ source.is_event_level }},
    },
{% endfor %}
}


def get_bronze_path(source_name: str) -> Path:
    return PRODUCTION_DIR / "data" / "bronze" / source_name


def get_silver_path() -> Path:
    return PRODUCTION_DIR / "data" / "silver" / f"silver_featureset_{COMPOSITE_NAME}"


def get_gold_path() -> Path:
    return PRODUCTION_DIR / "data" / "gold" / f"gold_features_{COMPOSITE_NAME}"


get_feast_data_path = get_gold_path


# Fit mode configuration for training vs scoring separation
FIT_MODE = {{ 'True' if config.fit_mode else 'False' }}
ARTIFACTS_PATH = {{ '"%s"' % config.artifacts_path if config.artifacts_path else 'str(PRODUCTION_DIR / "artifacts" / (RECOMMENDATIONS_HASH or "default"))' }}

RAW_SOURCES = {
{% for name, landing in config.landing.items() %}
    "{{ name }}": {
        "path": "{{ landing.raw_source_path }}",
        "format": "{{ landing.raw_source_format }}",
        "entity_key": "{{ landing.entity_column }}",
        "time_column": "{{ landing.time_column }}",
    },
{% endfor %}
}

EXCLUDED_SOURCES = [
{% for source in config.sources %}
{% if source.excluded %}
    "{{ source.name }}",
{% endif %}
{% endfor %}
]

EXPLORATION_ARTIFACTS = {
    "bronze": {name: str(EXPERIMENTS_DIR / "data" / "bronze" / name) for name in SOURCES},
    "silver": str(EXPERIMENTS_DIR / "data" / "silver" / f"silver_featureset_{COMPOSITE_NAME}"),
    "gold": str(EXPERIMENTS_DIR / "data" / "gold" / f"gold_features_{COMPOSITE_NAME}"),
    "scoring": str(EXPERIMENTS_DIR / "data" / "scoring" / "predictions"),
}
""",
    "bronze.py.j2": """import pandas as pd
import numpy as np
from pathlib import Path
{% set ops, fitted = collect_imports(config.transformations, False) %}
{% if ops %}
from customer_retention.transforms import {{ ops | sort | join(', ') }}
{% endif %}
from customer_retention.core.compat import ensure_timestamp, safe_to_datetime, timedelta_to_days
from config import SOURCES, get_bronze_path{{ ', RAW_SOURCES' if config.lifecycle else '' }}

SOURCE_NAME = "{{ source }}"


def load_{{ source }}():
    source_config = SOURCES[SOURCE_NAME]
    path = Path(source_config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if source_config["format"] == "csv":
        return pd.read_csv(str(path))
    if source_config["format"] == "parquet":
        return pd.read_parquet(str(path))
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(path))


{% set groups = group_steps(config.transformations) %}

def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
{%- if groups %}
{%- for func_name, steps in groups %}
    df = {{ func_name }}(df)
{%- endfor %}
{%- endif %}
    return df

{% for func_name, steps in groups %}

def {{ func_name }}(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    # {{ action_description(t) }}
    df = {{ render_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}

{% if config.lifecycle %}

ENTITY_COLUMN = "{{ config.entity_column or config.source.entity_key }}"
TIME_COLUMN = "{{ config.time_column or config.source.time_column }}"


def _load_raw_events():
    source = RAW_SOURCES[SOURCE_NAME]
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"Raw source not found: {path}")
    if source["format"] == "csv":
        return pd.read_csv(str(path))
    if source["format"] == "parquet":
        return pd.read_parquet(str(path))
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(path))

{% if config.lifecycle.include_recency_bucket %}

def add_recency_tenure(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    reference_date = raw_df[TIME_COLUMN].max()
    _grp = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN]
    entity_stats = _grp.min().to_frame("_time_min")
    entity_stats["_time_max"] = _grp.max()
    entity_stats["days_since_last"] = timedelta_to_days(reference_date - entity_stats["_time_max"])
    entity_stats["days_since_first"] = timedelta_to_days(reference_date - entity_stats["_time_min"])
    df = df.merge(entity_stats[["days_since_last", "days_since_first"]], left_on=ENTITY_COLUMN, right_index=True, how="left")
    return df


def add_recency_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_last" in df.columns:
        df["recency_bucket"] = pd.cut(df["days_since_last"], bins={{ config.lifecycle.recency_bucket_edges }} + [float("inf")],
                                       labels={{ config.lifecycle.recency_bucket_labels }})
    return df

{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}

def add_lifecycle_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_first" not in df.columns or "days_since_last" not in df.columns:
        return df
    event_count_cols = sorted(c for c in df.columns if c.startswith("event_count_"))
    if not event_count_cols:
        return df
    event_count_col = (
        "event_count_all_time" if "event_count_all_time" in event_count_cols
        else event_count_cols[-1]
    )
    duration = (df["days_since_first"] - df["days_since_last"]).astype(float)
    intensity = df[event_count_col].astype(float) / duration.clip(lower=1.0)
    tenure_med = float(duration.median())
    intensity_med = float(intensity.median())
    conditions = [
        (duration >= tenure_med) & (intensity >= intensity_med),
        (duration >= tenure_med) & (intensity < intensity_med),
        (duration < tenure_med) & (intensity >= intensity_med),
        (duration < tenure_med) & (intensity < intensity_med),
    ]
    labels = ["steady_loyal_lifecycle", "occasional_loyal_lifecycle", "intense_brief_lifecycle", "one_shot_lifecycle"]
    df["lifecycle_quadrant"] = np.select(conditions, labels, default="unknown")
    return df

{% endif %}
{% if config.lifecycle.include_cyclical_features %}

def add_cyclical_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    mean_dow = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: x.dt.dayofweek.mean())
    df = df.merge(mean_dow.rename("mean_dow"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["dow_sin"] = np.sin(2 * np.pi * df["mean_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["mean_dow"] / 7)
    df = df.drop(columns=["mean_dow"], errors="ignore")
    return df

{% endif %}
{% if config.lifecycle.include_month_cyclical %}

def add_month_quarter_cyclical(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    mean_month = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: x.dt.month.mean())
    df = df.merge(mean_month.rename("mean_month"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["month_sin"] = np.sin(2 * np.pi * df["mean_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["mean_month"] / 12)
{% if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: ((x.dt.month - 1) // 3).mean())
    df = df.merge(mean_quarter.rename("mean_quarter"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["quarter_sin"] = np.sin(2 * np.pi * df["mean_quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["mean_quarter"] / 4)
    df = df.drop(columns=["mean_month", "mean_quarter"], errors="ignore")
{% else %}
    df = df.drop(columns=["mean_month"], errors="ignore")
{% endif %}
    return df

{% endif %}
{% if config.lifecycle.include_trend_features %}

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    window_cols = sorted([c for c in df.columns if c.startswith("event_count_") and c != "event_count_all_time"])
    all_time_col = "event_count_all_time" if "event_count_all_time" in df.columns else None
    if window_cols and all_time_col:
        df["recent_vs_overall_ratio"] = df[window_cols[0]] / df[all_time_col].replace(0, float("nan"))
    if len(window_cols) >= 2:
        window_values = df[window_cols].values
        x = np.arange(len(window_cols), dtype=float)
        slopes = np.array([np.polyfit(x, row, 1)[0] if not np.any(np.isnan(row)) else 0.0 for row in window_values])
        df["entity_trend_slope"] = slopes
    return df

{% endif %}
{% if config.lifecycle.include_cohort_features %}

def add_cohort_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    first_event = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].min()
    cohort_data = pd.DataFrame({"first_event": first_event})
    cohort_data["cohort_year"] = cohort_data["first_event"].dt.year
    cohort_data["cohort_quarter"] = ((cohort_data["first_event"].dt.month - 1) // 3 + 1)
    df = df.merge(cohort_data[["cohort_year", "cohort_quarter"]], left_on=ENTITY_COLUMN, right_index=True, how="left")
    return df

{% endif %}
{% if config.lifecycle.momentum_pairs %}

def add_momentum_ratios(df: pd.DataFrame) -> pd.DataFrame:
{% for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df["momentum_{{ pair.short_window }}_{{ pair.long_window }}"] = df[short_col] / df[long_col].replace(0, float("nan"))
{% endfor %}
    return df

{% endif %}

def enrich_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    raw_df = _load_raw_events()
{% if config.raw_time_column %}
    raw_df = raw_df.rename(columns={"{{ config.raw_time_column }}": TIME_COLUMN})
{% endif %}
{% if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df, raw_df)
    df = add_recency_buckets(df)
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{% endif %}
{% if config.lifecycle.include_cyclical_features %}
    df = add_cyclical_features(df, raw_df)
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df, raw_df)
{% endif %}
{% if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{% endif %}
{% if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df, raw_df)
{% endif %}
{% if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{% endif %}
    return df
{% endif %}

{% if config.text_features %}

def compute_text_features_entity(df):
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
{% for tf in config.text_features %}
    if "{{ tf.column }}" in df.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        df, result = processor.process_column(df, "{{ tf.column }}", fit=True)
{% endfor %}
    return df
{% endif %}


def run_bronze_entity_{{ source }}():
    df = load_{{ source }}()
    df = apply_transformations(df)
{% if config.lifecycle %}
    df = enrich_lifecycle(df)
{% endif %}
{% if config.text_features %}
    df = compute_text_features_entity(df)
{% endif %}
    output_path = get_bronze_path(SOURCE_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from customer_retention.integrations.adapters.factory import get_delta
    _delta = get_delta(force_local=True)
    _delta.write(df, str(output_path))
    _z_cols = [c for c in ["{{ config.source.entity_key }}"] if c in df.columns]
    if _z_cols:
        _delta.optimize(str(output_path), _z_cols)
    else:
        _delta.optimize(str(output_path))
    return df


if __name__ == "__main__":
    run_bronze_entity_{{ source }}()
""",
    "silver.py.j2": '''import time

import pandas as pd
from pathlib import Path
{% set ops, fitted = collect_imports(config.silver.derived_columns, False) %}
{% if ops %}
from customer_retention.transforms import {{ ops | sort | join(', ') }}
{% endif %}
{% if config.silver.grid_dates %}
from customer_retention.stages.temporal.temporal_merger import TemporalMerger, MergeConfig, DatasetMergeInput
from customer_retention.core.config.column_config import DatasetGranularity
{% endif %}
{% set has_key_resolution = config.silver.merge_sources | selectattr('key_resolution_steps') | list | length > 0 %}
{% if has_key_resolution %}
from customer_retention.analysis.auto_explorer.key_resolver import resolve_entity_keys
from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
{% endif %}
from config import SOURCES, get_bronze_path, get_silver_path, TARGET_COLUMN, EXPERIMENTS_DIR, FINDINGS_DIR
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

_NAMESPACE = RunNamespace.from_env_or_latest(EXPERIMENTS_DIR)

{% if config.silver.grid_dates %}
GRID_DATES = {{ config.silver.grid_dates }}

MERGE_SOURCE_META = [
{% for src in config.silver.merge_sources %}
    {"name": "{{ src.name }}", "granularity": "{{ src.granularity }}"{{ ', "feature_timestamp_column": "' + src.feature_timestamp_column + '"' if src.feature_timestamp_column else '' }}, "key_resolution_steps": [{% for kr in src.key_resolution_steps %}{"bridge_dataset": "{{ kr.bridge_dataset }}", "source_key": "{{ kr.source_key }}", "bridge_key": "{{ kr.bridge_key }}", "resolve_column": "{{ kr.resolve_column }}"}{{ ", " if not loop.last else "" }}{% endfor %}]},
{% endfor %}
]
{% endif %}


def _load_artifact(path):
    from customer_retention.integrations.adapters.factory import get_delta
    df = get_delta(force_local=True).read(str(path))
    ref_path = Path(str(path) + ".reference_date")
    if ref_path.exists():
        df.attrs["aggregation_reference_date"] = ref_path.read_text().strip()
    return df


def _bronze_output_name(name: str) -> str:
    return f"{name}_aggregated" if SOURCES[name].get("is_event_level") else name


def load_bronze_outputs() -> dict:
    return {name: _load_artifact(get_bronze_path(_bronze_output_name(name)))
            for name in SOURCES if not SOURCES[name].get("excluded")}


{% if config.silver.grid_dates %}
def merge_sources(bronze_outputs: dict) -> pd.DataFrame:
    raw_entity_key = "{{ config.silver.entity_key or config.sources[0].entity_key }}"
    base_source = "{{ config.sources[0].name }}"
    entity_ids = bronze_outputs[base_source][raw_entity_key].unique()
{% if has_key_resolution %}
    resolutions = {}
    for meta in MERGE_SOURCE_META:
        kr_steps = meta.get("key_resolution_steps", [])
        if kr_steps:
            resolutions[meta["name"]] = [
                KeyResolutionStep(**s) for s in kr_steps
            ]
    if resolutions:
        bronze_outputs = resolve_entity_keys(bronze_outputs, resolutions)
{% endif %}
    for name, df in bronze_outputs.items():
        if raw_entity_key in df.columns and raw_entity_key != "entity_id":
            bronze_outputs[name] = df.rename(columns={raw_entity_key: "entity_id"})
    merger = TemporalMerger(MergeConfig(entity_key="entity_id"))
    spine = merger.build_spine(entity_ids, GRID_DATES)
    inputs = []
    for meta in MERGE_SOURCE_META:
        name = meta["name"]
        if name not in bronze_outputs:
            continue
        granularity = DatasetGranularity(meta["granularity"])
        inputs.append(DatasetMergeInput(
            name=name,
            df=bronze_outputs[name],
            granularity=granularity,
            feature_timestamp_column=meta.get("feature_timestamp_column"),
        ))
    merged, _report = merger.merge_all(spine, inputs)
    return merged
{% else %}
def merge_sources(bronze_outputs: dict) -> pd.DataFrame:
    raw_entity_key = "{{ config.silver.entity_key or config.sources[0].entity_key }}"
    base_source = "{{ config.sources[0].name }}"
    merged = bronze_outputs[base_source]
{% for join in config.silver.joins %}
    merged = merged.merge(
        bronze_outputs["{{ join.right_source }}"],
        left_on={{ join.left_keys }},
        right_on={{ join.right_keys }},
        how="{{ join.how }}"
    )
{% endfor %}
    if raw_entity_key in merged.columns and raw_entity_key != "entity_id":
        merged = merged.rename(columns={raw_entity_key: "entity_id"})
    return merged
{% endif %}


def create_holdout_mask(df: pd.DataFrame, holdout_fraction: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Create entity-level holdout by masking target for a fraction of entities.

    If a pre-computed holdout_entity_ids.json exists in FINDINGS_DIR, those IDs
    are used (preserving the same stratification/filters as the exploration sample).
    Otherwise falls back to random entity sampling.

    IMPORTANT: This must happen in the silver layer (BEFORE gold layer feature
    computation) to prevent temporal leakage.
    """
    import json as _json

    ORIGINAL_COLUMN = f"original_{TARGET_COLUMN}"

    if ORIGINAL_COLUMN in df.columns:
        print(f"  Holdout already exists ({ORIGINAL_COLUMN}), skipping creation")
        return df

    if TARGET_COLUMN not in df.columns:
        print(f"  Warning: TARGET_COLUMN \\'{TARGET_COLUMN}\\' not found, skipping holdout creation")
        return df

    df = df.copy()
    entity_ids = df["entity_id"].drop_duplicates()

    _holdout_ids_path = FINDINGS_DIR / "holdout_entity_ids.json"
    if _holdout_ids_path.exists():
        _saved_holdout = _json.loads(_holdout_ids_path.read_text())
        holdout_entities = entity_ids[entity_ids.isin(_saved_holdout)]
        n_holdout_entities = len(holdout_entities)
        print(f"Creating holdout set from pre-computed IDs ({n_holdout_entities:,} entities)...")
    else:
        n_holdout_entities = max(1, int(len(entity_ids) * holdout_fraction))
        holdout_entities = entity_ids.sample(n=n_holdout_entities, random_state=random_state)
        print(f"Creating holdout set ({holdout_fraction:.0%} of entities)...")

    holdout_mask = df["entity_id"].isin(holdout_entities)

    df[ORIGINAL_COLUMN] = pd.NA
    df.loc[holdout_mask, ORIGINAL_COLUMN] = df.loc[holdout_mask, TARGET_COLUMN]
    df.loc[holdout_mask, TARGET_COLUMN] = pd.NA

    n_holdout_rows = int(holdout_mask.sum())
    print(f"  Holdout entities: {n_holdout_entities:,} / {len(entity_ids):,}")
    print(f"  Holdout rows: {n_holdout_rows:,}, Training rows: {len(df) - n_holdout_rows:,}")

    return df


{% set derived_groups = group_steps(config.silver.derived_columns) %}

def create_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
{%- if derived_groups %}
{%- for func_name, steps in derived_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
{%- endif %}
    return df

{% for func_name, steps in derived_groups %}

def {{ func_name }}(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for dc in steps %}
    # {{ dc.rationale }}
    # {{ action_description(dc) }}
    df = {{ render_step_call(dc) }}
{%- endfor %}
    return df
{% endfor %}


def run_silver_merge(create_holdout: bool = True, holdout_fraction: float = 0.1):
    _t0 = time.perf_counter()
    bronze_outputs = load_bronze_outputs()
    _t_load = time.perf_counter() - _t0
    print(f"  Load bronze outputs: {_t_load:.1f}s")
    ref_date = next(
        (v.attrs["aggregation_reference_date"] for v in bronze_outputs.values()
         if "aggregation_reference_date" in v.attrs),
        None,
    )
    _t1 = time.perf_counter()
    silver = merge_sources(bronze_outputs)
    print(f"  Merge sources: {time.perf_counter() - _t1:.1f}s")
    _t2 = time.perf_counter()
    silver = create_derived_columns(silver)
    print(f"  Derived columns: {time.perf_counter() - _t2:.1f}s")

    if create_holdout:
        _t3 = time.perf_counter()
        silver = create_holdout_mask(silver, holdout_fraction=holdout_fraction)
        print(f"  Create holdout: {time.perf_counter() - _t3:.1f}s")

    if ref_date:
        silver.attrs["aggregation_reference_date"] = ref_date

    output_path = get_silver_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from customer_retention.integrations.adapters.factory import get_delta
    _delta = get_delta(force_local=True)
    _t4 = time.perf_counter()
    _delta.write(silver, str(output_path))
    _z_cols = [c for c in ["entity_id", "as_of_date"] if c in silver.columns]
    if _z_cols:
        _delta.optimize(str(output_path), _z_cols)
    else:
        _delta.optimize(str(output_path))
    print(f"  Write + optimize: {time.perf_counter() - _t4:.1f}s")
    if ref_date:
        Path(str(output_path) + ".reference_date").write_text(ref_date)
    _silver_elapsed = time.perf_counter() - _t0
    print(f"Silver total: {_silver_elapsed:.1f}s")
    if _NAMESPACE is not None:
        import json as _json
        _silver_meta = {
            "rows": len(silver), "columns": len(silver.columns),
            "column_list": list(silver.columns), "elapsed_seconds": round(_silver_elapsed, 1),
            "source_count": len(bronze_outputs), "sources": list(bronze_outputs.keys()),
        }
        _NAMESPACE.silver_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _NAMESPACE.silver_metadata_path.write_text(_json.dumps(_silver_meta))
    return silver


if __name__ == "__main__":
    run_silver_merge()
''',
    "gold.py.j2": r"""import time

import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
{% set parts = partition_gold_steps(config.gold) %}
{% set fitted_steps = parts['fitted_transforms'] + parts['fitted_encodings'] + parts['fitted_scalings'] %}
{% set stateless_steps = parts['stateless_transforms'] + parts['stateless_encodings'] %}
{% set all_gold_steps = config.gold.transformations + config.gold.encodings + config.gold.scalings %}
{% set ops, fitted = collect_imports(all_gold_steps, True) %}
from customer_retention.transforms import ArtifactStore{{ (', ' + (ops | sort | join(', '))) if ops }}
{% if fitted %}
from customer_retention.transforms.fitted import {{ fitted | sort | join(', ') }}
{% endif %}
from config import (get_silver_path, get_gold_path, get_feast_data_path,
                    COMPOSITE_NAME, TARGET_COLUMN, TIMESTAMP_COLUMN, RECOMMENDATIONS_HASH,
                    FEAST_REPO_PATH, FEAST_FEATURE_VIEW, ENTITY_KEY, FEAST_TIMESTAMP_COL,
                    EXPERIMENTS_DIR, ARTIFACTS_PATH, FIT_MODE)
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

_NAMESPACE = RunNamespace.from_env_or_latest(EXPERIMENTS_DIR)

{% if not fitted_steps %}
{% if config.fit_mode %}
_store = ArtifactStore(Path(ARTIFACTS_PATH))
{% else %}
_store = ArtifactStore.from_manifest(Path(ARTIFACTS_PATH) / "manifest.yaml")
{% endif %}
{% endif %}

from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)

ENCODINGS = [
{% for enc in config.gold.encodings %}
    TransformationStep(type=PipelineTransformationType.ENCODE, column="{{ enc.column }}", parameters={{ enc.parameters }}, rationale="{{ enc.rationale }}"),
{% endfor %}
]

SCALINGS = [
{% for scale in config.gold.scalings %}
    TransformationStep(type=PipelineTransformationType.SCALE, column="{{ scale.column }}", parameters={{ scale.parameters }}, rationale="{{ scale.rationale }}"),
{% endfor %}
]


def load_silver() -> pd.DataFrame:
    from customer_retention.integrations.adapters.factory import get_delta
    path = get_silver_path()
    df = get_delta(force_local=True).read(str(path))
    ref_path = Path(str(path) + ".reference_date")
    if ref_path.exists():
        df.attrs["aggregation_reference_date"] = ref_path.read_text().strip()
    return df


def load_gold() -> pd.DataFrame:
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(get_gold_path()))


{% set transform_groups = group_steps(parts['stateless_transforms']) %}

def apply_gold_transformations(df: pd.DataFrame) -> pd.DataFrame:
{%- if transform_groups %}
{%- for func_name, steps in transform_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
{%- endif %}
    return df

{% for func_name, steps in transform_groups %}

def {{ func_name }}(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    # {{ action_description(t) }}
    df = {{ render_step_call(t, config.fit_mode) }}
{%- endfor %}
    return df
{% endfor %}


def apply_encodings(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(parts['stateless_encodings']) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- if parts['stateless_encodings'] %}
{%- for enc in parts['stateless_encodings'] %}
    # {{ enc.rationale }}
    # {{ action_description(enc) }}
    df = {{ render_step_call(enc, config.fit_mode) }}
{%- endfor %}
{%- endif %}
    return df


def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
    return df


def apply_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
{% if config.gold.feature_exclusion_prefixes %}
    from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
    _prefix_drops = FindingsParser.find_leakage_excluded_columns(df.columns, {{ config.gold.feature_exclusion_prefixes }})
    if _prefix_drops:
        df = df.drop(columns=_prefix_drops)
        print(f"  Dropped {len(_prefix_drops)} leakage-excluded columns")
{% endif %}
{% if config.gold.feature_selections %}
    _drop_cols = {{ config.gold.feature_selections }}
    df = df.drop(columns=[c for c in _drop_cols if c in df.columns])
{% endif %}
    return df

{% if fitted_steps %}

def apply_fitted_transforms(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(fitted_steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{% if config.fit_mode %}
    _store = ArtifactStore(Path(ARTIFACTS_PATH))
    from customer_retention.core.compat import temporal_quantile
    from datetime import timedelta
    _ts_col = FEAST_TIMESTAMP_COL
    if _ts_col not in df.columns:
        for _candidate in ["as_of_date", "feature_timestamp", "event_timestamp"]:
            if _candidate in df.columns:
                _ts_col = _candidate
                break
    _cutoff = temporal_quantile(df[_ts_col], 1 - {{ config.training.test_size if config.training else 0.2 }})
{% if config.training and config.training.purge_gap_days %}
    _train_mask = df[_ts_col] < (_cutoff - timedelta(days={{ config.training.purge_gap_days }}))
{% else %}
    _train_mask = df[_ts_col] < _cutoff
{% endif %}
    _fit_subset = df[_train_mask].copy()
{%- for step in fitted_steps %}
    # {{ step.rationale }}
    # {{ action_description(step) }}
    _fit_subset = {{ render_step_call(step, fit_mode=True) }}
{%- endfor %}
    del _fit_subset
{%- for step in fitted_steps %}
    df = {{ render_step_call(step, fit_mode=False) }}
{%- endfor %}
    _store.save_manifest()
    print(f"Fit artifacts saved to: {ARTIFACTS_PATH} (fitted on {_train_mask.sum():,} train rows, transforming {len(df):,} total)")
{% else %}
    _store = ArtifactStore.from_manifest(Path(ARTIFACTS_PATH) / "manifest.yaml")
{%- for step in fitted_steps %}
    # {{ step.rationale }}
    # {{ action_description(step) }}
    df = {{ render_step_call(step, fit_mode=False) }}
{%- endfor %}
{% endif %}
    return df
{% endif %}


def get_feature_version_tag() -> str:
    if RECOMMENDATIONS_HASH:
        return f"v1.0.0_{RECOMMENDATIONS_HASH}"
    return "v1.0.0"


def add_feast_timestamp(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    if FEAST_TIMESTAMP_COL in df.columns:
        return df
    if "as_of_date" in df.columns:
        return df.rename(columns={"as_of_date": FEAST_TIMESTAMP_COL})
    if "feature_timestamp" in df.columns:
        return df.rename(columns={"feature_timestamp": FEAST_TIMESTAMP_COL})
    if "aggregation_reference_date" in df.attrs:
        timestamp = pd.to_datetime(df.attrs["aggregation_reference_date"])
        print(f"  Using aggregation reference_date for Feast timestamp: {timestamp}")
    elif reference_date is not None:
        timestamp = reference_date
        print(f"  Using provided reference_date for Feast timestamp: {timestamp}")
    else:
        timestamp = datetime.now()
        warnings.warn(
            f"No reference_date available for Feast timestamp. Using datetime.now() ({timestamp}). "
            "This may cause temporal leakage - features should use actual aggregation dates. "
            "Set aggregation_reference_date in DataFrame.attrs during aggregation.",
            UserWarning
        )
    df[FEAST_TIMESTAMP_COL] = timestamp
    return df


def run_gold_features():
    _t0 = time.perf_counter()
    silver = load_silver()
    print(f"  Load silver: {time.perf_counter() - _t0:.1f}s")
    _t1 = time.perf_counter()
    gold = apply_gold_transformations(silver)
    print(f"  Gold transformations: {time.perf_counter() - _t1:.1f}s")
    _t2 = time.perf_counter()
    gold = apply_feature_selection(gold)
    print(f"  Feature selection: {time.perf_counter() - _t2:.1f}s")
    _t3 = time.perf_counter()
    gold = apply_encodings(gold)
    print(f"  Encodings: {time.perf_counter() - _t3:.1f}s")
    _t4 = time.perf_counter()
    gold = apply_scaling(gold)
    print(f"  Scaling: {time.perf_counter() - _t4:.1f}s")
{% if fitted_steps %}
    _t5 = time.perf_counter()
    gold = apply_fitted_transforms(gold)
    print(f"  Fitted transforms: {time.perf_counter() - _t5:.1f}s")
{% endif %}
    gold = add_feast_timestamp(gold)
{% if not fitted_steps and config.fit_mode %}
    _store.save_manifest()
    print(f"Fit artifacts saved to: {ARTIFACTS_PATH}")
{% endif %}
    output_path = get_gold_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.attrs["recommendations_hash"] = RECOMMENDATIONS_HASH
    gold.attrs["feature_version"] = get_feature_version_tag()
    _numeric = gold.select_dtypes(include=["number"]).columns
    if len(_numeric) > 0:
        gold[_numeric] = gold[_numeric].astype("float32")
    from customer_retention.integrations.adapters.factory import get_delta
    _delta = get_delta(force_local=True)
    _t6 = time.perf_counter()
    _delta.write(gold, str(output_path))
    _z_cols = [c for c in ["entity_id", TIMESTAMP_COLUMN] if c in gold.columns]
    if _z_cols:
        _delta.optimize(str(output_path), _z_cols)
    else:
        _delta.optimize(str(output_path))
    print(f"  Write + optimize: {time.perf_counter() - _t6:.1f}s")
    _gold_elapsed = time.perf_counter() - _t0
    print(f"Gold total: {_gold_elapsed:.1f}s")
    print(f"Gold features saved with version: {get_feature_version_tag()}")
    if _NAMESPACE is not None:
        import json as _json
        _meta_cols = {TARGET_COLUMN, FEAST_TIMESTAMP_COL, ENTITY_KEY, "as_of_date", "feature_timestamp"}
        _gold_meta = {
            "rows": len(gold), "columns": len(gold.columns),
            "feature_count": len([c for c in gold.columns if c not in _meta_cols]),
            "feature_version": get_feature_version_tag(),
            "elapsed_seconds": round(_gold_elapsed, 1),
        }
        _NAMESPACE.gold_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _NAMESPACE.gold_metadata_path.write_text(_json.dumps(_gold_meta))
        try:
            from customer_retention.stages.causal.interpretation import (
                FeatureLineage as _FeatureLineage,
                build_feature_meta_rows as _build_feature_meta_rows,
                load_column_descriptions_sidecar as _load_column_descriptions_sidecar,
                parse_aggregation_feature_name as _parse_aggregation_feature_name,
                write_feature_meta_sidecar as _write_feature_meta_sidecar,
            )
            _fm_lineages = []
            for _col in gold.columns:
                if _col in _meta_cols:
                    continue
                _lineage = _parse_aggregation_feature_name(_col)
                if _lineage is None:
                    # Defensive fallback — always emit a non-empty source_columns
                    # so compile_predicate_prose has a column to look up in
                    # column_descriptions even for unrecognised feature patterns.
                    _lineage = _FeatureLineage(
                        feature_name=_col,
                        source_columns=[_col],
                        aggregation_kind="passthrough",
                    )
                _fm_lineages.append(_lineage)
            _fm_rows = _build_feature_meta_rows(
                composite_name=COMPOSITE_NAME,
                lineages=_fm_lineages,
                column_descriptions=_load_column_descriptions_sidecar(_NAMESPACE),
            )
            _write_feature_meta_sidecar(_NAMESPACE, COMPOSITE_NAME, _fm_rows)
        except Exception as _fm_exc:
            print(f"feature_meta sidecar aborted: {type(_fm_exc).__name__}: {_fm_exc}")
    return gold


if __name__ == "__main__":
    run_gold_features()
""",
    "training.py.j2": '''{% set best_model_type = config.training.best_model_type if config.training else None %}
{% set production_cv_folds = config.training.production_cv_folds if config.training else None %}
{% set feature_spec_path = config.feature_spec_path %}
import json as _json
import logging
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
import mlflow.sklearn
{% if best_model_type is none or best_model_type == "xgboost" %}
import mlflow.xgboost
import xgboost as xgb
{% endif %}
from pathlib import Path
from feast import FeatureStore
{% if best_model_type is none or best_model_type == "random_forest" %}
from sklearn.ensemble import RandomForestClassifier
{% endif %}
{% if best_model_type is none or best_model_type == "logistic_regression" %}
from sklearn.linear_model import LogisticRegression
{% endif %}
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score)
from customer_retention.stages.modeling.data_splitter import DataSplitter, SplitStrategy
{% if production_cv_folds %}
from customer_retention.stages.modeling.cross_validator import CrossValidator, CVStrategy
{% endif %}
from customer_retention.stages.modeling.feature_profile import FeatureProfile, ColumnProfile, build_feature_profile, compare_feature_profiles
{% if feature_spec_path %}
from customer_retention.stages.modeling.feature_spec import FeatureSpec
from customer_retention.transforms import ArtifactStore
from customer_retention.transforms.executor import TransformExecutor
{% endif %}
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat.timing import log_timing
from customer_retention.core.config.column_config import GOLD_METADATA_COLUMNS
{% if config.training and config.training.imbalance_strategy == "smote" %}
from customer_retention.stages.modeling.imbalance_handler import ImbalanceHandler, ImbalanceStrategy
{% endif %}
from config import (TARGET_COLUMN, PIPELINE_NAME, COMPOSITE_NAME, RECOMMENDATIONS_HASH, MLFLOW_TRACKING_URI,
                    MLFLOW_ARTIFACT_ROOT, FEAST_REPO_PATH, FEAST_FEATURE_VIEW, ENTITY_KEY,
                    FEAST_TIMESTAMP_COL, EXPERIMENTS_DIR, get_feast_data_path)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger = logging.getLogger("training")
_EXCLUDE_COLS = {TARGET_COLUMN, FEAST_TIMESTAMP_COL, ENTITY_KEY} | GOLD_METADATA_COLUMNS
{% if config.training and config.training.exploration_feature_profile %}
_EXPLORATION_PROFILE = {{ config.training.exploration_feature_profile | py_source }}
{% else %}
_EXPLORATION_PROFILE = None
{% endif %}
_NAMESPACE = RunNamespace.from_env_or_latest(EXPERIMENTS_DIR)

{% if feature_spec_path %}
_FEATURE_SPEC_PATH = Path(r"{{ feature_spec_path }}")
PRODUCTION_TEST_SIZE = {{ config.training.production_internal_split_test_size if config.training else 0.1 }}
PRODUCTION_RANDOM_STATE = 42
_HARD_BLOCK_VERDICTS = {"overfit", "leaky"}
{% else %}
_FEATURE_SPEC_PATH = None
{% endif %}

def _assert_rows(count, stage):
    if count == 0:
        raise ValueError(f"[TRAINING] {stage}: 0 rows remaining — cannot proceed")
    return count


def _load_feast_data():
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(get_feast_data_path()))


def get_training_data_from_feast() -> pd.DataFrame:
    """Retrieve training data from Feast for training/serving consistency.

    Uses get_historical_features for point-in-time correct feature retrieval.
    This ensures training uses the exact same feature retrieval path as inference.
    """
    feast_path = Path(FEAST_REPO_PATH)

    if not (feast_path / "feature_store.yaml").exists():
        print("Feast repo not initialized, falling back to data file")
        return _load_feast_data()

    try:
        store = FeatureStore(repo_path=str(feast_path))

        features_df = _load_feast_data()

        entity_df = features_df[[ENTITY_KEY, FEAST_TIMESTAMP_COL]].copy()

        exclude_cols = {ENTITY_KEY, FEAST_TIMESTAMP_COL, TARGET_COLUMN}
        feature_cols = [c for c in features_df.columns
                        if c not in exclude_cols and not c.startswith("original_")]

        feature_refs = [f"{FEAST_FEATURE_VIEW}:{col}" for col in feature_cols]

        print(f"Retrieving {len(feature_refs)} features from Feast...")
        print(f"  Feature view: {FEAST_FEATURE_VIEW}")
        print(f"  Entity key: {ENTITY_KEY}")

        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()

        training_df = training_df.merge(
            features_df[[ENTITY_KEY, TARGET_COLUMN]],
            on=ENTITY_KEY,
            how="left"
        )

        print(f"  Retrieved {len(training_df):,} rows, {len(training_df.columns)} columns")
        return training_df

    except (ImportError, ConnectionError, RuntimeError, KeyError) as e:
        import warnings
        warnings.warn(f"Feast retrieval failed ({e}), loading directly from gold data")
        return _load_feast_data()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c in _EXCLUDE_COLS or c.startswith("original_")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).fillna(0)


def compute_metrics(y_true, y_proba, y_pred) -> dict:
    both_classes = len(np.unique(y_true)) > 1
    return {
        "roc_auc": roc_auc_score(y_true, y_proba) if both_classes else 0.0,
        "pr_auc": average_precision_score(y_true, y_proba) if both_classes else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def log_feature_importance(model, feature_names):
    fi = get_feature_importance(model, feature_names)
    if fi is None:
        return
    fi.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")


{% if best_model_type is none or best_model_type == "xgboost" %}
def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    mlflow.xgboost.autolog(disable=True)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    params = {"objective": "binary:logistic", "eval_metric": ["auc", "logloss"],
              "max_depth": 6, "learning_rate": 0.1, "seed": 42}
    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dtrain, "train"), (dtest, "eval")], verbose_eval=False)
    return model
{% endif %}

{% if production_cv_folds %}
def _on_fold(detail, fold_num, total_folds):
    print(f"  CV fold {fold_num}/{total_folds}: roc_auc={detail['score']:.4f} ({detail['elapsed_seconds']:.0f}s)", flush=True)
{% endif %}


def get_model_name_with_hash(base_name: str) -> str:
    if RECOMMENDATIONS_HASH:
        return f"{base_name}_{RECOMMENDATIONS_HASH}"
    return base_name


def run_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _experiment_name = f"training_{COMPOSITE_NAME}"
    experiment = mlflow.get_experiment_by_name(_experiment_name)
    if experiment is None:
        mlflow.create_experiment(_experiment_name, artifact_location=MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(_experiment_name)
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print(f"Artifacts: {MLFLOW_ARTIFACT_ROOT}")
    _results = {"models": {}, "feature_profile": {}}

    with log_timing("load_gold_data", logger):
        training_data = get_training_data_from_feast()
    raw_count = _assert_rows(len(training_data), "gold_data")
    type_summary = dict(training_data.dtypes.value_counts())
    print(f"[TRAINING] Gold data: {raw_count:,} rows, {len(training_data.columns)} columns")
    print(f"[TRAINING] Column types: {type_summary}")
    _results["gold_data"] = {"rows": raw_count, "columns": len(training_data.columns), "column_types": {str(k): v for k, v in type_summary.items()}}

    with log_timing("prepare_features", logger):
        y = training_data[TARGET_COLUMN]
        X = prepare_features(training_data.drop(columns=[TARGET_COLUMN]))
        feature_names = list(X.columns)
    print(f"[TRAINING] Features: {len(feature_names)} columns after preparation")
    _results["feature_count"] = len(feature_names)
    train_mask = y.notna()
    X, y = X.loc[train_mask], y.loc[train_mask]
    filtered_count = _assert_rows(len(X), "after_null_label_filter")
    print(f"[TRAINING] After null-label filter: {filtered_count:,} rows")
    _results["filtered_rows"] = filtered_count

    with log_timing("feature_profile", logger):
        excluded_cols = {}
        for c in training_data.columns:
            if c in _EXCLUDE_COLS:
                excluded_cols[c] = "metadata"
            elif c.startswith("original_"):
                excluded_cols[c] = "original_prefix"
{% if config.gold.feature_selections %}
        for _fs_col in {{ config.gold.feature_selections }}:
            excluded_cols[_fs_col] = "feature_selection"
{% endif %}
{% if feature_spec_path %}
        _SPEC = FeatureSpec.load(_FEATURE_SPEC_PATH)
        if _SPEC.verdict.is_hard_block() and not os.environ.get("CR_OVERRIDE_UNSTABLE_SPEC"):
            raise RuntimeError(
                f"FeatureSpec verdict={_SPEC.verdict.status!r}; refusing to train. "
                "Set CR_OVERRIDE_UNSTABLE_SPEC=1 to override."
            )
        if _SPEC.verdict.status == "unstable":
            warnings.warn(
                f"FeatureSpec verdict=unstable (cv_std={_SPEC.verdict.cv_std:.3f}); proceeding.",
                stacklevel=1,
            )
        _leakage_excluded = {e.column for e in _SPEC.leakage_exclusions}
        _leakage_drops = [c for c in X.columns if c in _leakage_excluded]
        if _leakage_drops:
            X = X.drop(columns=_leakage_drops)
            for _c in _leakage_drops:
                excluded_cols[_c] = "leakage_exclusion"
        _missing = [c for c in _SPEC.selected_features if c not in X.columns]
        if _missing:
            raise RuntimeError(
                f"FeatureSpec parity violation: gold missing {len(_missing)} declared "
                f"features: {_missing[:10]}. Bronze/silver/gold derivation is out of sync "
                "with exploration — regenerate gold or re-run NB08."
            )
        for _c in X.columns:
            if _c not in set(_SPEC.selected_features):
                excluded_cols.setdefault(_c, "not_in_spec")
        X = X[list(_SPEC.selected_features)].copy()
        feature_names = list(_SPEC.selected_features)
        print(
            f"[TRAINING] FeatureSpec applied: {len(feature_names)} features, "
            f"verdict={_SPEC.verdict.status}, leakage_drops={len(_leakage_drops)}"
        )
{% else %}
        if _NAMESPACE is not None:
            _rec_path = _NAMESPACE.merged_recommendations_path
            if _rec_path.exists():
                import yaml as _rec_yaml
                from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
                with _rec_path.open() as _rf:
                    _drop_recs = RecommendationRegistry.from_dict(_rec_yaml.safe_load(_rf))
                _runtime_drops = set()
                for _rec in getattr(getattr(_drop_recs, 'gold', None), 'feature_selection', []):
                    if _rec.action in ('drop_multicollinear', 'drop_weak', 'drop_l1_zero', 'drop_chi_squared', 'drop_gbdt_importance', 'drop_rescue_consensus', 'drop_availability', 'drop_zero_variance'):
                        excluded_cols[_rec.target_column] = _rec.action
                    if _rec.action in ('drop_l1_zero', 'drop_chi_squared', 'drop_gbdt_importance', 'drop_rescue_consensus', 'drop_zero_variance'):
                        _runtime_drops.add(_rec.target_column)
                _actual_runtime = [c for c in _runtime_drops if c in feature_names]
                if _actual_runtime:
                    X = X.drop(columns=_actual_runtime)
                    feature_names = [c for c in feature_names if c not in _runtime_drops]
                    print(f"[TRAINING] Runtime L1/variance drops: {len(_actual_runtime)} features")
{% endif %}
        feature_stats = {}
        for c in feature_names:
            null_count = int(X[c].isna().sum())
            feature_stats[c] = ColumnProfile(dtype=str(X[c].dtype), non_null_count=filtered_count - null_count, null_count=null_count)
        prod_profile = build_feature_profile("production", TARGET_COLUMN, filtered_count, feature_stats, excluded_cols)
        print(f"[TRAINING] Production profile: {prod_profile.feature_count} features, {filtered_count:,} rows")
        if _NAMESPACE is not None:
            prod_profile.save(_NAMESPACE.production_feature_profile_path)
            print(f"[TRAINING] Production profile saved to {_NAMESPACE.production_feature_profile_path}")
        _results["feature_profile"]["production_features"] = prod_profile.feature_count
        _results["feature_profile"]["production_rows"] = filtered_count
        _results["feature_profile"]["excluded_details"] = excluded_cols
        if _EXPLORATION_PROFILE is not None:
            _exp_feats = _EXPLORATION_PROFILE.get("features", {})
            _exp_excl = _EXPLORATION_PROFILE.get("excluded", {})
            _prod_feature_set = set(feature_names)
            for _pc in _prod_feature_set:
                if _pc not in _exp_feats and _pc not in _exp_excl:
                    _exp_feats[_pc] = {"dtype": "float32", "non_null": _EXPLORATION_PROFILE.get("row_count", 0), "null_count": 0}
            for _sc in [c for c in list(_exp_feats) if c not in _prod_feature_set and c not in _exp_excl]:
                _exp_excl[_sc] = "not_in_pipeline"
                del _exp_feats[_sc]
            _EXPLORATION_PROFILE["features"] = _exp_feats
            _EXPLORATION_PROFILE["excluded"] = _exp_excl
            _EXPLORATION_PROFILE["feature_count"] = len(_exp_feats)
            exp_profile = FeatureProfile.from_dict(_EXPLORATION_PROFILE)
            discrepancies = compare_feature_profiles(exp_profile, prod_profile)
            _results["feature_profile"]["exploration_features"] = exp_profile.feature_count
            _results["feature_profile"]["discrepancies"] = discrepancies
            if discrepancies:
                print(f"[TRAINING] WARNING: {len(discrepancies)} feature discrepancies vs exploration:")
                for d in discrepancies:
                    print(f"[TRAINING]   {d}")
            else:
                print("[TRAINING] Feature profile matches exploration")
        else:
            print("[TRAINING] WARNING: No exploration feature profile available for comparison")
        if _NAMESPACE is not None and '_drop_recs' in dir():
            _gold_fs = getattr(getattr(_drop_recs, 'gold', None), 'feature_selection', [])
            _prioritized_rationales = {r.target_column: r.rationale for r in _gold_fs if r.action == 'prioritize'}
            _correlated_map = {}
            for _r in _gold_fs:
                if _r.action == 'drop_multicollinear' and _r.parameters:
                    _correlated_map.setdefault(_r.target_column, []).append(
                        (_r.parameters.get('correlated_with', ''), _r.parameters.get('correlation', 0)))
            _dropped_strong = {c: _prioritized_rationales[c] for c in _prioritized_rationales if c in excluded_cols}
            if _dropped_strong:
                print(f"[TRAINING] NOTE: {len(_dropped_strong)} strong predictors dropped (signal preserved via correlated survivors):")
                for _c in sorted(_dropped_strong):
                    _survivors = [f"{p} (r={r:.2f})" for p, r in _correlated_map.get(_c, []) if p not in excluded_cols]
                    _surv_str = ", ".join(_survivors[:3]) if _survivors else "none found"
                    print(f"[TRAINING]   {_c}: {_dropped_strong[_c]} -> survived by: {_surv_str}")
{% if config.training and config.training.recommended_training_start %}
    if FEAST_TIMESTAMP_COL in training_data.columns:
        time_mask = training_data.loc[train_mask, FEAST_TIMESTAMP_COL] >= pd.to_datetime("{{ config.training.recommended_training_start }}")
        X, y = X.loc[time_mask], y.loc[time_mask]
{% endif %}
{% if config.training and config.training.filter_future_dates %}
    if FEAST_TIMESTAMP_COL in training_data.columns:
        future_mask = training_data.loc[X.index, FEAST_TIMESTAMP_COL] <= datetime.now()
        X, y = X.loc[future_mask], y.loc[future_mask]
{% endif %}
    with log_timing("temporal_split", logger):
        splitter = DataSplitter(
            target_column=TARGET_COLUMN,
            strategy=SplitStrategy.TEMPORAL,
            temporal_column=FEAST_TIMESTAMP_COL,
            group_column=ENTITY_KEY,
{% if config.training and config.training.purge_gap_days %}
            purge_gap_days={{ config.training.purge_gap_days }},
{% endif %}
            test_size={% if feature_spec_path %}PRODUCTION_TEST_SIZE{% else %}{{ config.training.test_size if config.training else 0.2 }}{% endif %},
            exclude_columns=[FEAST_TIMESTAMP_COL, ENTITY_KEY],
        )
        split_df = training_data.loc[X.index].copy()
        split_df[TARGET_COLUMN] = y
        for col in X.columns:
            split_df[col] = X[col]
        splits = splitter.split(split_df)
    X_train, X_test = splits.X_train[feature_names], splits.X_test[feature_names]
    y_train, y_test = splits.y_train, splits.y_test
    _assert_rows(len(X_train), "train_set_after_split")
    _assert_rows(len(X_test), "test_set_after_split")
    print(f"[TRAINING] Split: train={len(X_train):,}, test={len(X_test):,}")
    print(f"[TRAINING] Split info: {splits.split_info}")
    _results["split"] = {"train": len(X_train), "test": len(X_test), **splits.split_info}
    label_dist = dict(y_train.value_counts())
    print(f"[TRAINING] Label distribution: {label_dist}")
    _results["label_distribution"] = {str(k): int(v) for k, v in label_dist.items()}
    if y_train.nunique() < 2:
        raise ValueError(f"[TRAINING] Only {y_train.nunique()} class(es) — Need at least 2 for binary classification")
{% if config.training and config.training.imbalance_strategy == "smote" %}
    handler = ImbalanceHandler(strategy=ImbalanceStrategy.SMOTE)
    _imb_result = handler.fit_transform(X_train, y_train)
    X_train, y_train = _imb_result.X_resampled, _imb_result.y_resampled
{% endif %}

{% set class_weight_param = ', class_weight="balanced"' if config.training and config.training.imbalance_strategy == "class_weight" else '' %}
    sklearn_models = {}
{% if best_model_type is none or best_model_type == "logistic_regression" %}
    sklearn_models["logistic_regression"] = LogisticRegression(max_iter=5000, random_state=42{{ class_weight_param }})
{% endif %}
{% if best_model_type is none or best_model_type == "random_forest" %}
    sklearn_models["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=42{{ class_weight_param }})
{% endif %}

    run_name = get_model_name_with_hash(f"training_{COMPOSITE_NAME}")
    _logged_models = []
    with mlflow.start_run(run_name=run_name) as _parent_run:
        mlflow.log_params({"train_samples": len(X_train), "test_samples": len(X_test), "n_features": X.shape[1]})
        mlflow.set_tag("feature_source", "feast")
        mlflow.set_tag("feast_feature_view", FEAST_FEATURE_VIEW)
        mlflow.set_tag("composite_name", COMPOSITE_NAME)
        mlflow.set_tag("pipeline_name", PIPELINE_NAME)
        if RECOMMENDATIONS_HASH:
            mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
        best_model, best_auc = None, -1

        for name, model in sklearn_models.items():
            with mlflow.start_run(run_name=name, nested=True) as _nested_run:
                if RECOMMENDATIONS_HASH:
                    mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
                mlflow.set_tag("feature_source", "feast")
                with log_timing(f"fit_{name}", logger):
                    model.fit(X_train, y_train)
                model_artifact_name = get_model_name_with_hash(f"model_{name}")
                _log_info = mlflow.sklearn.log_model(model, model_artifact_name)
                _logged_models.append({
                    "artifact_path": model_artifact_name,
                    "model_uri": _log_info.model_uri,
                    "flavor": "sklearn",
                    "run_id": _nested_run.info.run_id,
                    "display_name": name,
                    "wrapper_meta_artifact_path": None,
                })
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_proba, y_pred)
{% if production_cv_folds %}
                _cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits={{ production_cv_folds }}, scoring="roc_auc")
                _cv_result = _cv.run(model, X_train, y_train, on_fold_complete=_on_fold)
                print(f"  CV: {_cv_result.cv_mean:.4f} +/- {_cv_result.cv_std:.4f}", flush=True)
                mlflow.log_metrics({**metrics, "cv_mean": _cv_result.cv_mean, "cv_std": _cv_result.cv_std})
                _results["models"][name] = {**metrics, "cv_mean": _cv_result.cv_mean, "cv_std": _cv_result.cv_std}
{% else %}
                mlflow.log_metrics(metrics)
                _results["models"][name] = metrics
{% endif %}
                log_feature_importance(model, feature_names)
                print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}", flush=True)
                if metrics["roc_auc"] > best_auc:
                    best_auc, best_model = metrics["roc_auc"], name

{% if best_model_type is none or best_model_type == "xgboost" %}
        with mlflow.start_run(run_name="xgboost", nested=True) as _xgb_run:
            if RECOMMENDATIONS_HASH:
                mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
            mlflow.set_tag("feature_source", "feast")
            xgb_model = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
            dtest = xgb.DMatrix(X_test, feature_names=feature_names)
            y_proba = xgb_model.predict(dtest)
            y_pred = (y_proba > 0.5).astype(int)
            metrics = compute_metrics(y_test, y_proba, y_pred)
            xgb_model_name = get_model_name_with_hash("model_xgboost")
            _xgb_log_info = mlflow.xgboost.log_model(xgb_model, xgb_model_name)
            _logged_models.append({
                "artifact_path": xgb_model_name,
                "model_uri": _xgb_log_info.model_uri,
                "flavor": "xgboost",
                "run_id": _xgb_run.info.run_id,
                "display_name": "xgboost",
                "wrapper_meta_artifact_path": None,
            })
            mlflow.log_metrics(metrics)
            importance = xgb_model.get_score(importance_type="gain")
            fi = pd.DataFrame({"feature": importance.keys(), "importance": importance.values()})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            fi.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            print(f"xgboost: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
            _results["models"]["xgboost"] = metrics
            if metrics["roc_auc"] > best_auc:
                best_auc, best_model = metrics["roc_auc"], "xgboost"
{% endif %}

        if best_model is not None:
            mlflow.set_tag("best_model", best_model)
            mlflow.log_metric("best_roc_auc", best_auc)
        mlflow.log_dict({"feature_columns": feature_names, "count": len(feature_names)}, "features.json")
        print(f"Best: {best_model} (ROC-AUC={best_auc:.4f})")

    _results["best_model"] = best_model
    _results["best_model_name"] = best_model
    _results["best_roc_auc"] = best_auc
    _results["mlflow_run_id"] = _parent_run.info.run_id
    _results["mlflow_experiment_name"] = _experiment_name
    _results["pipeline_name"] = PIPELINE_NAME
    _results["composite_name"] = COMPOSITE_NAME
    _results["target_column"] = TARGET_COLUMN
    _results["entity_key"] = ENTITY_KEY
    _results["timestamp_column"] = FEAST_TIMESTAMP_COL
    _results["recommendations_hash"] = RECOMMENDATIONS_HASH or ""
    _results["feature_columns"] = feature_names
    _results["logged_models"] = _logged_models
    _results["registered_model_name"] = ""
    if _NAMESPACE is not None:
        _NAMESPACE.training_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _NAMESPACE.training_metadata_path.write_text(_json.dumps(_results, default=str))
{% if feature_spec_path %}
    if _NAMESPACE is not None:
        _label_rate_train = float(pd.Series(y_train).mean()) if len(y_train) else 0.0
        _label_rate_test = float(pd.Series(y_test).mean()) if len(y_test) else 0.0
        _prod_diag = {
            "run_type": "production",
            "exploration_run_id": _SPEC.exploration_run_id,
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "split": {"train": int(len(X_train)), "test": int(len(X_test)), **_results.get("split", {})},
            "test_metrics": {n: v for n, v in _results["models"].items()},
            "label_distribution": _results.get("label_distribution", {}),
            "label_rate_train": _label_rate_train,
            "label_rate_test": _label_rate_test,
            "best_model_name": best_model,
            "best_roc_auc": best_auc,
        }
        _NAMESPACE.production_diagnostics_path.write_text(_json.dumps(_prod_diag, default=str))
        print(f"[TRAINING] Production diagnostics saved to {_NAMESPACE.production_diagnostics_path}")
{% endif %}
    return _results


if __name__ == "__main__":
    _training_results = run_experiment()
    print("\\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(_json.dumps(_training_results, indent=2, default=str))
''',
    "runner.py.j2": """import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

from config import PIPELINE_NAME, COMPOSITE_NAME, EXPERIMENTS_DIR, PRODUCTION_DIR
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
{% for name in sorted_landing_names(config.landing) %}
from landing.landing_{{ name }} import run_landing_{{ name }}
{% endfor %}
{% for name in config.bronze %}
from bronze.bronze_entity_{{ name }} import run_bronze_entity_{{ name }}
{% endfor %}
{% for name in config.bronze_event %}
from bronze.bronze_event_{{ name }} import run_bronze_event_{{ name }}
from bronze.bronze_entity_{{ name }}_aggregated import run_bronze_entity_{{ name }}_aggregated
{% endfor %}
{% set cn = config.composite_name or config.name %}
from silver.silver_featureset_{{ cn }} import run_silver_merge
from gold.gold_features_{{ cn }} import run_gold_features
from training.ml_experiment import run_experiment


def setup_experiments_dir():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "mlruns").mkdir(parents=True, exist_ok=True)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "bronze").mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "silver").mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "gold").mkdir(parents=True, exist_ok=True)


def run_pipeline(validate=False):
    import json as _json
    print(f"Starting pipeline: {PIPELINE_NAME}")
    setup_experiments_dir()
    _ns = RunNamespace.from_env_or_latest(EXPERIMENTS_DIR)
{% if config.landing %}

    print("\\n[1/6] Landing (event sources)...")
{% for name in sorted_landing_names(config.landing) %}
    run_landing_{{ name }}()
{% endfor %}
    print("Landing complete")
    if validate:
        from validation.validate_pipeline import validate_landing
        validate_landing()
{% endif %}

    _bronze_results = {}
    print("\\n[{{ '2/6' if config.landing else '1/4' }}] Bronze event...")
{% for name in config.bronze_event %}
    _df = run_bronze_event_{{ name }}()
    _bronze_results["event_{{ name }}"] = {"rows": len(_df), "columns": len(_df.columns)}
{% endfor %}

    print("\\n[{{ '3/6' if config.landing else '2/4' }}] Bronze entity (parallel)...")
    with ThreadPoolExecutor(max_workers={{ (config.bronze | length) + (config.bronze_event | length) }}) as executor:
        _futures = {
{% for name in config.bronze %}
            "{{ name }}": executor.submit(run_bronze_entity_{{ name }}),
{% endfor %}
{% for name in config.bronze_event %}
            "{{ name }}_aggregated": executor.submit(run_bronze_entity_{{ name }}_aggregated),
{% endfor %}
        }
        for _name, _fut in _futures.items():
            _df = _fut.result()
            _bronze_results[_name] = {"rows": len(_df), "columns": len(_df.columns)}
    _bronze_meta = {"sources": _bronze_results, "total_sources": len(_bronze_results)}
    if _ns is not None:
        _ns.bronze_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _ns.bronze_metadata_path.write_text(_json.dumps(_bronze_meta))
    print("Bronze complete")
    for _src, _info in _bronze_results.items():
        print(f"  {_src}: {_info['rows']:,} rows, {_info['columns']} columns")
    if validate:
        from validation.validate_pipeline import validate_bronze
        validate_bronze()

    print("\\n[{{ '4/6' if config.landing else '3/4' }}] Silver...")
    _silver_df = run_silver_merge()
    print(f"Silver complete: {len(_silver_df):,} rows, {len(_silver_df.columns)} columns")
    if validate:
        from validation.validate_pipeline import validate_silver
        validate_silver()

    print("\\n[{{ '5/6' if config.landing else '4/4' }}] Gold...")
    _gold_df = run_gold_features()
    print(f"Gold complete: {len(_gold_df):,} rows, {len(_gold_df.columns)} columns")
    if validate:
        from validation.validate_pipeline import validate_gold
        validate_gold()

    print("\\n[{{ '6/6' if config.landing else '4/4' }}] Training...")
    _training_results = run_experiment()
    print("Training complete")
    print("\\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(_json.dumps(_training_results, indent=2, default=str))
    if validate:
        from validation.validate_pipeline import validate_training
        validate_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    run_pipeline(validate=args.validate)
""",
    "run_all.py.j2": '''"""{{ config.name }} - Pipeline Runner with MLflow UI

All artifacts (data, mlruns, feast) are stored in the experiments directory.
Override location with CR_EXPERIMENTS_DIR environment variable.
"""
import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

from config import PIPELINE_NAME, COMPOSITE_NAME, SOURCES, MLFLOW_TRACKING_URI, EXPERIMENTS_DIR, PRODUCTION_DIR, FINDINGS_DIR
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
{% for name in sorted_landing_names(config.landing) %}
from landing.landing_{{ name }} import run_landing_{{ name }}
{% endfor %}
{% for name in config.bronze %}
from bronze.bronze_entity_{{ name }} import run_bronze_entity_{{ name }}
{% endfor %}
{% for name in config.bronze_event %}
from bronze.bronze_event_{{ name }} import run_bronze_event_{{ name }}
from bronze.bronze_entity_{{ name }}_aggregated import run_bronze_entity_{{ name }}_aggregated
{% endfor %}
{% set cn = config.composite_name or config.name %}
from silver.silver_featureset_{{ cn }} import run_silver_merge
from gold.gold_features_{{ cn }} import run_gold_features
from training.ml_experiment import run_experiment


def setup_experiments_dir():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "mlruns").mkdir(parents=True, exist_ok=True)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "bronze").mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "silver").mkdir(parents=True, exist_ok=True)
    (PRODUCTION_DIR / "data" / "gold").mkdir(parents=True, exist_ok=True)
    print(f"Experiments directory: {EXPERIMENTS_DIR}")
    print(f"Production directory: {PRODUCTION_DIR}")
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print(f"Findings directory: {FINDINGS_DIR}")


def run_landing():
{% for name in sorted_landing_names(config.landing) %}
    run_landing_{{ name }}()
{% endfor %}
    pass


def run_bronze_event():
    _results = {}
{% for name in config.bronze_event %}
    _df = run_bronze_event_{{ name }}()
    _results["event_{{ name }}"] = {"rows": len(_df), "columns": len(_df.columns)}
{% endfor %}
    return _results


def run_bronze_entity_parallel():
    bronze_funcs = {
{% for name in config.bronze %}
        "{{ name }}": run_bronze_entity_{{ name }},
{% endfor %}
{% for name in config.bronze_event %}
        "{{ name }}_aggregated": run_bronze_entity_{{ name }}_aggregated,
{% endfor %}
    }
    _results = {}
    with ThreadPoolExecutor(max_workers={{ (config.bronze | length) + (config.bronze_event | length) }}) as ex:
        futures = {name: ex.submit(fn) for name, fn in bronze_funcs.items()}
        for name, fut in futures.items():
            _df = fut.result()
            _results[name] = {"rows": len(_df), "columns": len(_df.columns)}
    return _results


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_mlflow_ui():
    port = 5050
    if is_port_in_use(port):
        print(f"\\n⚠ Port {port} is already in use.")
        print(f"  Either mlflow is already running, or kill the old process:")
        print(f"  pkill -f 'mlflow ui'")
        print(f"\\n  Opening browser to existing server...")
        webbrowser.open(f"http://localhost:{port}")
        return None

    print(f"\\nStarting MLflow UI (tracking: {MLFLOW_TRACKING_URI})...")
    process = subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    for _ in range(10):
        time.sleep(1)
        if process.poll() is not None:
            err = process.stderr.read().decode() if process.stderr else ""
            print(f"\\n✗ MLflow UI failed to start (exit code {process.returncode})")
            if err:
                print(err)
            return None
        if is_port_in_use(port):
            break
    webbrowser.open(f"http://localhost:{port}")
    print(f"MLflow UI running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    return process


def run_pipeline():
    import json as _json
    print(f"Running {PIPELINE_NAME}")
    print("=" * 50)

    setup_experiments_dir()
    _ns = RunNamespace.from_env_or_latest(EXPERIMENTS_DIR)
{% if config.landing %}

    print("\\n[1/6] Landing (event sources)...")
    run_landing()
    print("Landing complete")
{% endif %}

    print("\\n[{{ '2/6' if config.landing else '1/4' }}] Bronze event...")
    _bronze_event_results = run_bronze_event()

    print("\\n[{{ '3/6' if config.landing else '2/4' }}] Bronze entity (parallel)...")
    _bronze_entity_results = run_bronze_entity_parallel()

    _bronze_meta = {"sources": {**_bronze_event_results, **_bronze_entity_results}, "total_sources": len(_bronze_event_results) + len(_bronze_entity_results)}
    if _ns is not None:
        _ns.bronze_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _ns.bronze_metadata_path.write_text(_json.dumps(_bronze_meta))
    print("Bronze complete")
    for _src, _info in _bronze_meta["sources"].items():
        print(f"  {_src}: {_info['rows']:,} rows, {_info['columns']} columns")

    print("\\n[{{ '4/6' if config.landing else '3/4' }}] Silver...")
    _silver_df = run_silver_merge()
    print(f"Silver complete: {len(_silver_df):,} rows, {len(_silver_df.columns)} columns")

    print("\\n[{{ '5/6' if config.landing else '4/4' }}] Gold...")
    _gold_df = run_gold_features()
    print(f"Gold complete: {len(_gold_df):,} rows, {len(_gold_df.columns)} columns")

    print("\\n[{{ '6/6' if config.landing else '5/4' }}] Training...")
    _training_results = run_experiment()
    print("Training complete")

    print("\\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(_json.dumps(_training_results, indent=2, default=str))

    print("\\n" + "=" * 50)
    print("Pipeline finished!")

    mlflow_process = start_mlflow_ui()
    if mlflow_process:
        try:
            mlflow_process.wait()
        except KeyboardInterrupt:
            mlflow_process.terminate()
            print("\\nMLflow UI stopped")


if __name__ == "__main__":
    run_pipeline()
''',
    "workflow.json.j2": """{
{% set cn = config.composite_name or config.name %}
  "name": "{{ config.name }}_pipeline",
  "tasks": [
{% for name, lcfg in config.landing.items() %}
    {
      "task_key": "landing_{{ name }}",
{% if lcfg.key_resolution_steps %}
      "depends_on": [
{% for step in lcfg.key_resolution_steps %}
{% if step.bridge_dataset in config.landing %}
        {"task_key": "landing_{{ step.bridge_dataset }}"}{{ "," if not loop.last else "" }}
{% endif %}
{% endfor %}
      ],
{% endif %}
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/landing/landing_{{ name }}"
      }
    },
{% endfor %}
{% for name in config.bronze_event %}
    {
      "task_key": "bronze_event_{{ name }}",
{% if config.landing %}
      "depends_on": [
{% for lname in config.landing %}
        {"task_key": "landing_{{ lname }}"}{{ "," if not loop.last else "" }}
{% endfor %}
      ],
{% endif %}
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/bronze/bronze_event_{{ name }}"
      }
    },
    {
      "task_key": "bronze_entity_{{ name }}_aggregated",
      "depends_on": [{"task_key": "bronze_event_{{ name }}"}],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/bronze/bronze_entity_{{ name }}_aggregated"
      }
    },
{% endfor %}
{% for name in config.bronze %}
    {
      "task_key": "bronze_entity_{{ name }}",
{% if config.landing %}
      "depends_on": [
{% for lname in config.landing %}
        {"task_key": "landing_{{ lname }}"}{{ "," if not loop.last else "" }}
{% endfor %}
      ],
{% endif %}
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/bronze/bronze_entity_{{ name }}"
      }
    },
{% endfor %}
    {
      "task_key": "silver_featureset_{{ cn }}",
      "depends_on": [
{% for name in config.bronze_event %}
        {"task_key": "bronze_entity_{{ name }}_aggregated"},
{% endfor %}
{% for name in config.bronze %}
        {"task_key": "bronze_entity_{{ name }}"}{{ "," if not loop.last else "" }}
{% endfor %}
      ],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/silver/silver_featureset_{{ cn }}"
      }
    },
    {
      "task_key": "gold_features_{{ cn }}",
      "depends_on": [{"task_key": "silver_featureset_{{ cn }}"}],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/gold/gold_features_{{ cn }}"
      }
    },
    {
      "task_key": "ml_experiment",
      "depends_on": [{"task_key": "gold_features_{{ cn }}"}],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/training/ml_experiment"
      }
    }
  ]
}
""",
    "feature_store.yaml.j2": """project: {{ config.name }}
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
""",
    "features.py.j2": """from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

{% set fv_name = config.feast.feature_view_name if config.feast else 'featureset_' + (config.composite_name or config.name) %}
{% set entity_name = config.feast.entity_name if config.feast else 'customer' %}
{% set entity_key = config.feast.entity_key if config.feast else config.sources[0].entity_key %}
{% set ts_col = config.feast.timestamp_column if config.feast else 'event_timestamp' %}
{% set ttl = config.feast.ttl_days if config.feast else 365 %}

{{ entity_name }} = Entity(
    name="{{ entity_name }}",
    join_keys=["{{ entity_key }}"],
)

{{ fv_name }}_source = FileSource(
    path="../data/gold/gold_features_{{ config.composite_name or config.name }}",
    timestamp_field="{{ ts_col }}"
)

{{ fv_name }} = FeatureView(
    name="{{ fv_name }}",
    entities=[{{ entity_name }}],
    ttl=timedelta(days={{ ttl }}),
    source={{ fv_name }}_source,
    tags={
        "pipeline": "{{ config.name }}",
        "composite_name": "{{ config.composite_name or '' }}",
        "recommendations_hash": "{{ config.recommendations_hash or 'none' }}",
    }
)
""",
    "landing.py.j2": """import pandas as pd
import numpy as np
from pathlib import Path
from customer_retention.core.compat import safe_to_datetime
from config import RAW_SOURCES, PRODUCTION_DIR

SOURCE_NAME = "{{ name }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"
TARGET_COLUMN = "{{ config.target_column }}"


def load_raw_data() -> pd.DataFrame:
    source = RAW_SOURCES[SOURCE_NAME]
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"Raw source not found: {path}")
    if source["format"] == "csv":
        return pd.read_csv(str(path))
    if source["format"] == "parquet":
        return pd.read_parquet(str(path))
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(path))


def derive_feature_timestamp(df: pd.DataFrame) -> pd.DataFrame:
{% if config.timestamp_coalesce %}
{% set cols = config.timestamp_coalesce.datetime_columns_ordered %}
    df["feature_timestamp"] = safe_to_datetime(df["{{ cols[-1] }}"], errors="coerce")
{% for col in cols[:-1] | reverse %}
    df["feature_timestamp"] = df["feature_timestamp"].fillna(safe_to_datetime(df["{{ col }}"], errors="coerce"))
{% endfor %}
{% else %}
    if TIME_COLUMN in df.columns:
        df["feature_timestamp"] = safe_to_datetime(df[TIME_COLUMN], errors="coerce")
    elif "feature_timestamp" not in df.columns:
        raise KeyError(f"Time column '{TIME_COLUMN}' not found in columns: {list(df.columns)}")
{% endif %}
    return df


def derive_label_timestamp(df: pd.DataFrame) -> pd.DataFrame:
{% if config.label_timestamp %}
{% set lt = config.label_timestamp %}
{% if lt.label_column %}
    df["label_timestamp"] = safe_to_datetime(df["{{ lt.label_column }}"], errors="coerce")
    df["label_timestamp"] = df["label_timestamp"].fillna(
        df["feature_timestamp"] + pd.Timedelta(days={{ lt.fallback_window_days }})
    )
{% else %}
    df["label_timestamp"] = df["feature_timestamp"] + pd.Timedelta(days={{ lt.fallback_window_days }})
{% endif %}
{% else %}
    df["label_timestamp"] = df["feature_timestamp"] + pd.Timedelta(days=180)
{% endif %}
    return df


def derive_label_available_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["label_available_flag"] = df[TARGET_COLUMN].notna() if TARGET_COLUMN in df.columns else False
    return df

{% if config.datetime_derivation %}

def derive_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    from customer_retention.stages.profiling import derive_extra_datetime_features
    source_columns = {{ config.datetime_derivation.source_columns }}
    mask_future_columns = {{ config.datetime_derivation.mask_future_columns }}
    existing = [c for c in source_columns if c in df.columns]
    if existing:
        df, _ = derive_extra_datetime_features(
            df, "{{ config.datetime_derivation.reference_column }}", existing,
            mask_future_columns=[c for c in mask_future_columns if c in existing],
        )
    return df
{% endif %}


def derive_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = derive_feature_timestamp(df)
    df = derive_label_timestamp(df)
    df = derive_label_available_flag(df)
{% if config.datetime_derivation %}
    df = derive_datetime_features(df)
{% endif %}
    return df


{% if config.history_window %}

def apply_history_window(df: pd.DataFrame) -> pd.DataFrame:
    time_col = "feature_timestamp"
{% if config.history_window.upper_limit %}
    upper = safe_to_datetime("{{ config.history_window.upper_limit }}")
{% else %}
    upper = df[time_col].max()
{% endif %}
{% if config.history_window.lookback_periods %}
    lookback_days = {{ config.history_window.lookback_periods }} * {{ config.history_window.cadence_days }}
    lower = upper - pd.Timedelta(days=lookback_days)
    df = df[df[time_col].isna() | (df[time_col] >= lower)]
{% endif %}
{% if config.history_window.upper_limit %}
    df = df[df[time_col].isna() | (df[time_col] <= upper)]
{% endif %}
    print(f"  History window: {len(df):,} records after filtering")
    return df
{% endif %}


{% if config.key_resolution_steps %}

def resolve_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    from customer_retention.integrations.adapters.factory import get_delta
{% for step in config.key_resolution_steps %}
    _bridge_path = PRODUCTION_DIR / "data" / "landing" / "{{ step.bridge_dataset }}"
    _bridge = get_delta(force_local=True).read(str(_bridge_path))
    _bridge = _bridge[["{{ step.bridge_key }}", "{{ step.resolve_column }}"]].drop_duplicates(subset=["{{ step.bridge_key }}"])
    df = df.merge(_bridge, left_on="{{ step.source_key }}", right_on="{{ step.bridge_key }}", how="inner")
{% if step.source_key != step.bridge_key %}
    df = df.drop(columns=["{{ step.bridge_key }}"])
{% endif %}
{% endfor %}
    return df
{% endif %}


def get_landing_output_path() -> Path:
    return PRODUCTION_DIR / "data" / "landing" / SOURCE_NAME


def run_landing_{{ name }}():
    print(f"Landing: {SOURCE_NAME}")
    df = load_raw_data()
    print(f"  Raw records: {len(df):,}")
{% if config.raw_time_column %}
    df = df.rename(columns={"{{ config.raw_time_column }}": TIME_COLUMN})
{% endif %}
{% if config.original_target_column %}
    df = df.rename(columns={"{{ config.original_target_column }}": TARGET_COLUMN})
{% endif %}
{% if config.filters %}
    from customer_retention.core.compat import apply_sql_predicate
{% for step in config.filters %}
    df = apply_sql_predicate(df, {{ step.parameters.predicate | python_repr }})
    print(f"  After user filter: {len(df):,}")
{% endfor %}
{% endif %}
{% if config.lifecycle_enrichments %}
    from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig
    from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset
{% for step in config.lifecycle_enrichments %}
    df = enrich_lifecycle_dataset(df, LifecycleEnrichmentConfig.from_dict({{ step.parameters.config | python_repr }}))
    print(f"  After lifecycle enrichment: {len(df):,}")
{% endfor %}
{% endif %}
{% if config.key_resolution_steps %}
    df = resolve_entity_key(df)
    print(f"  After key resolution: {len(df):,}")
{% endif %}
    df = derive_temporal_columns(df)
{% if config.history_window %}
    df = apply_history_window(df)
{% endif %}
    output_path = get_landing_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from customer_retention.integrations.adapters.factory import get_delta
    _delta = get_delta(force_local=True)
    _delta.write(df, str(output_path))
    _z_cols = [c for c in [ENTITY_COLUMN, TIME_COLUMN] if c in df.columns]
    if _z_cols:
        _delta.optimize(str(output_path), _z_cols)
    else:
        _delta.optimize(str(output_path))
    print(f"  Records: {len(df):,}")
    print(f"  Output: {output_path}")
    return df


if __name__ == "__main__":
    run_landing_{{ name }}()
""",
    "bronze_event.py.j2": """import pandas as pd
import numpy as np
from pathlib import Path
{% set ops, fitted = collect_imports(config.pre_shaping, False) %}
{% if ops %}
from customer_retention.transforms import {{ ops | sort | join(', ') }}
{% endif %}
from customer_retention.core.compat import ensure_timestamp, safe_to_datetime, as_tz_naive
from customer_retention.core.naming import sanitize_column_token
from pandas.api.types import is_numeric_dtype
{% if config.text_features %}
from config import PRODUCTION_DIR, TARGET_COLUMN, FIT_MODE
{% else %}
from config import PRODUCTION_DIR, TARGET_COLUMN
{% endif %}

SOURCE_NAME = "{{ source }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

{% set pre_groups = group_steps(config.pre_shaping) %}

def apply_pre_shaping(df: pd.DataFrame) -> pd.DataFrame:
{% if config.deduplicate %}
{% if config.deduplicate is not true and config.deduplicate.strategy is defined and config.deduplicate.strategy == "keep_most_complete" %}
{% if config.deduplicate.conflict_columns %}
    _subset = {{ config.deduplicate.conflict_columns }}
{% else %}
    _subset = [ENTITY_COLUMN, TIME_COLUMN]
{% endif %}
    _null_counts = df[df.columns.difference(_subset)].isnull().sum(axis=1)
    df = df.assign(_null_count=_null_counts).sort_values("_null_count").drop_duplicates(subset=_subset, keep="first").drop(columns=["_null_count"])
{% elif config.deduplicate is not true and config.deduplicate.conflict_columns is defined and config.deduplicate.conflict_columns %}
    df = df.drop_duplicates(subset={{ config.deduplicate.conflict_columns }}, keep="first")
{% else %}
    df = df.drop_duplicates(subset=[ENTITY_COLUMN, TIME_COLUMN], keep="first")
{% endif %}
{% endif %}
{%- if pre_groups %}
{%- for func_name, steps in pre_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
{%- endif %}
    return df

{% for func_name, steps in pre_groups %}

def {{ func_name }}(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    # {{ action_description(t) }}
    df = {{ render_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}

{% if config.datetime_derivation %}

def derive_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    from customer_retention.stages.profiling import derive_extra_datetime_features
    source_columns = {{ config.datetime_derivation.source_columns }}
    existing = [c for c in source_columns if c in df.columns]
    if existing:
        df, _ = derive_extra_datetime_features(df, TIME_COLUMN, existing)
    return df
{% endif %}

{% if config.aggregation %}
def _parse_window(window_str):
    if window_str == "all_time":
        return None
    if window_str.endswith("d"):
        return pd.Timedelta(days=int(window_str[:-1]))
    if window_str.endswith("h"):
        return pd.Timedelta(hours=int(window_str[:-1]))
    if window_str.endswith("w"):
        return pd.Timedelta(weeks=int(window_str[:-1]))
    return pd.Timedelta(days=int(window_str))


def _safe_mode(x):
    if len(x) == 0:
        return None
    return x.value_counts().idxmax()


AGGREGATION_WINDOWS = {{ config.aggregation.windows }}
VALUE_COLUMNS = {{ config.aggregation.value_columns }}
AGG_FUNCS = {{ config.aggregation.agg_funcs }}
CATEGORICAL_COLUMNS = {{ config.aggregation.categorical_columns }}
CATEGORICAL_AGG_FUNCS = {{ config.aggregation.categorical_agg_funcs }}
BINARY_COLUMNS = {{ config.aggregation.binary_columns }}
COLUMN_BLOCKED_FUNCS = {{ config.aggregation.column_blocked_funcs }}
CATEGORICAL_VALUE_COUNTS = {{ config.aggregation.categorical_value_counts }}
{% endif %}
{%- if config.per_grid_date_mode %}
GRID_DATES = {{ grid_dates }}
VALUE_COUNTS_COLUMNS = {{ config.value_counts_columns | list }}


def apply_event_aggregation_per_grid_date(df: pd.DataFrame) -> pd.DataFrame:
    # Per-grid-date count aggregation using pandas merge_asof.
    #
    # For each (entity, value-of-VALUE_COUNTS_COLUMN, window) we compute the
    # cumulative event count at each grid date and at (grid_date - window).
    # The window count is the difference. Output has one row per (entity,
    # as_of_date) which downstream temporal_merger handles via equi-join.
    ensure_timestamp(df, TIME_COLUMN)
    df[TIME_COLUMN] = as_tz_naive(df[TIME_COLUMN])

    grid = pd.to_datetime(GRID_DATES)
    entities = df[[ENTITY_COLUMN]].drop_duplicates()
    spine = entities.assign(_key=1).merge(
        pd.DataFrame({"as_of_date": grid, "_key": 1}), on="_key"
    ).drop(columns=["_key"])
    spine = spine.sort_values([ENTITY_COLUMN, "as_of_date"]).reset_index(drop=True)

    output = spine.copy()

    for vc_col in VALUE_COUNTS_COLUMNS:
        if vc_col not in df.columns:
            continue
        distinct_values = sorted(df[vc_col].dropna().unique().tolist())
        for value in distinct_values:
            sub = df[df[vc_col] == value][[ENTITY_COLUMN, TIME_COLUMN]].copy()
            if sub.empty:
                continue
            sub = sub.sort_values([ENTITY_COLUMN, TIME_COLUMN]).reset_index(drop=True)
            sub["_running"] = sub.groupby(ENTITY_COLUMN).cumcount() + 1
            sub_running = sub.rename(columns={TIME_COLUMN: "_event_ts"})

            cum_at_G = pd.merge_asof(
                spine.sort_values("as_of_date"),
                sub_running.sort_values("_event_ts"),
                left_on="as_of_date",
                right_on="_event_ts",
                by=ENTITY_COLUMN,
                direction="backward",
            )[[ENTITY_COLUMN, "as_of_date", "_running"]].rename(
                columns={"_running": "_cum_G"}
            )
            cum_at_G["_cum_G"] = cum_at_G["_cum_G"].fillna(0)

            for window_str in AGGREGATION_WINDOWS:
                col_name = f"{vc_col}_{sanitize_column_token(value)}_count_{window_str}"
                if window_str == "all_time":
                    merged = cum_at_G.rename(columns={"_cum_G": col_name})
                else:
                    td = _parse_window(window_str)
                    spine_shifted = spine.copy()
                    spine_shifted["as_of_date_shifted"] = (
                        spine_shifted["as_of_date"] - td
                    )
                    cum_at_minus = pd.merge_asof(
                        spine_shifted.sort_values("as_of_date_shifted"),
                        sub_running.sort_values("_event_ts"),
                        left_on="as_of_date_shifted",
                        right_on="_event_ts",
                        by=ENTITY_COLUMN,
                        direction="backward",
                    )[[ENTITY_COLUMN, "as_of_date", "_running"]].rename(
                        columns={"_running": "_cum_minus"}
                    )
                    cum_at_minus["_cum_minus"] = cum_at_minus["_cum_minus"].fillna(0)
                    merged = cum_at_G.merge(
                        cum_at_minus, on=[ENTITY_COLUMN, "as_of_date"]
                    )
                    merged[col_name] = merged["_cum_G"] - merged["_cum_minus"]
                    merged = merged[[ENTITY_COLUMN, "as_of_date", col_name]]

                output = output.merge(
                    merged, on=[ENTITY_COLUMN, "as_of_date"], how="left"
                )
                output[col_name] = output[col_name].fillna(0).astype("int64")

    return output
{%- endif %}


def apply_event_aggregation(df: pd.DataFrame) -> pd.DataFrame:
{% if config.aggregation %}
    ensure_timestamp(df, TIME_COLUMN)
    df[TIME_COLUMN] = as_tz_naive(df[TIME_COLUMN])
    reference_date = df[TIME_COLUMN].max()
    numeric_value_columns = [c for c in VALUE_COLUMNS if c in df.columns and is_numeric_dtype(df[c])]
    for _cvc_col in CATEGORICAL_VALUE_COUNTS:
        if _cvc_col not in df.columns:
            raise ValueError(
                f"categorical_value_counts declares column {_cvc_col!r} but it is not "
                f"present in the dataframe; fix upstream config or enrichment"
            )
    base = df.groupby(ENTITY_COLUMN).agg("first")[[]]
    parts = []
    if TARGET_COLUMN in df.columns:
        parts.append(df.groupby(ENTITY_COLUMN)[TARGET_COLUMN].first())
    for window in AGGREGATION_WINDOWS:
        td = _parse_window(window)
        window_df = df if td is None else df[df[TIME_COLUMN] >= (reference_date - td)]
        for col in numeric_value_columns:
            _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
            for func in AGG_FUNCS:
                if func not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].agg(func).rename(f"{col}_{func}_{window}"))
        for col in CATEGORICAL_COLUMNS:
            if col in window_df.columns:
                _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
                if "nunique" not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].nunique().rename(f"{col}_nunique_{window}"))
                if "mode" not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].agg(_safe_mode).rename(f"{col}_mode_{window}"))
        for col in BINARY_COLUMNS:
            if col in window_df.columns:
                _blocked = COLUMN_BLOCKED_FUNCS.get(col, [])
                if "rate" not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].mean().rename(f"{col}_rate_{window}"))
                if "count" not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].sum().rename(f"{col}_count_{window}"))
                if "any" not in _blocked:
                    parts.append(window_df.groupby(ENTITY_COLUMN)[col].max().rename(f"{col}_any_{window}"))
        for _cvc_col, _cvc_values in CATEGORICAL_VALUE_COUNTS.items():
            for _cvc_value in _cvc_values:
                _cvc_flag = (window_df[_cvc_col] == _cvc_value).astype(int)
                parts.append(_cvc_flag.groupby(window_df[ENTITY_COLUMN]).sum().rename(f"{_cvc_col}_{sanitize_column_token(_cvc_value)}_count_{window}"))
        parts.append(window_df.groupby(ENTITY_COLUMN).size().rename(f"event_count_{window}"))
    if "feature_timestamp" in df.columns:
        parts.append(df.groupby(ENTITY_COLUMN)["feature_timestamp"].max().rename("feature_timestamp"))
    df = pd.concat([base] + parts, axis=1).reset_index()
    _fill_cols = [c for c in df.columns if any(c.endswith(s) for s in ("_count", "_sum", "_rate")) or c.startswith("event_count_")]
    if _fill_cols:
        df[_fill_cols] = df[_fill_cols].fillna(0)
    df.attrs["aggregation_reference_date"] = str(reference_date)
{% endif %}
    return df


{% if config.temporal_features and config.temporal_features.has_renderable_content() %}

def compute_temporal_features(agg_df, raw_df):
    from customer_retention.stages.profiling.temporal_feature_engineer import (
        TemporalAggregationConfig, TemporalFeatureEngineer,
    )
    from customer_retention.core.compat import ensure_timestamp
    ensure_timestamp(raw_df, TIME_COLUMN)
    value_cols = {{ config.temporal_features.lag_columns or (config.aggregation.value_columns if config.aggregation else []) }}
    numeric_vals = [c for c in value_cols if c in raw_df.columns]
    eng_config = TemporalAggregationConfig(
        lag_window_days={{ config.temporal_features.lag_window_days }},
        num_lags={{ config.temporal_features.num_lags }},
        lag_aggregations={{ config.temporal_features.lag_agg_funcs }},
        compute_velocity={{ 'velocity' in config.temporal_features.feature_groups }},
        compute_acceleration={{ 'acceleration' in config.temporal_features.feature_groups }},
        compute_lifecycle={{ 'lifecycle' in config.temporal_features.feature_groups }},
        compute_recency={{ 'recency' in config.temporal_features.feature_groups }},
        compute_regularity={{ 'regularity' in config.temporal_features.feature_groups }},
        compute_cohort={{ 'cohort_comparison' in config.temporal_features.feature_groups }},
    )
    engineer = TemporalFeatureEngineer(eng_config)
    result = engineer.compute(raw_df, ENTITY_COLUMN, TIME_COLUMN, numeric_vals)
    temporal_df = result.features_df
    merge_cols = [c for c in temporal_df.columns if c != ENTITY_COLUMN]
    agg_df = agg_df.merge(temporal_df[[ENTITY_COLUMN] + merge_cols], on=ENTITY_COLUMN, how="left")
    return agg_df
{% endif %}

{% if config.text_features %}

def compute_text_features(df, raw_df):
    from customer_retention.stages.profiling.text_processor import TextColumnProcessor, TextProcessingConfig
    from customer_retention.transforms import ArtifactStore
    from pathlib import Path
    store = ArtifactStore(Path(PRODUCTION_DIR / "artifacts" / "text"))
{% for tf in config.text_features %}
    if "{{ tf.column }}" in raw_df.columns:
        processor = TextColumnProcessor(TextProcessingConfig(embedding_model="{{ tf.embedding_model }}"), registry=None)
        text_data = raw_df.groupby(ENTITY_COLUMN)["{{ tf.column }}"].first().reset_index()
        if FIT_MODE:
            text_data, result = processor.process_column(text_data, "{{ tf.column }}", fit=True)
        else:
            text_data, result = processor.process_column(text_data, "{{ tf.column }}", fit=False)
        component_cols = result.component_columns
        df = df.merge(text_data[[ENTITY_COLUMN] + component_cols], on=ENTITY_COLUMN, how="left")
{% endfor %}
    return df
{% endif %}

def run_bronze_event_{{ source }}():
    from customer_retention.integrations.adapters.factory import get_delta
    storage = get_delta(force_local=True)
    landing_path = str(PRODUCTION_DIR / "data" / "landing" / SOURCE_NAME)
    if not storage.exists(landing_path):
        raise FileNotFoundError(f"Landing output not found: {landing_path}")
    raw_df = storage.read(landing_path)
    df = apply_pre_shaping(raw_df.copy())
{% if config.datetime_derivation %}
    df = derive_datetime_features(df)
{% endif %}
{%- if config.per_grid_date_mode %}
    df = apply_event_aggregation_per_grid_date(df)
{%- else %}
    df = apply_event_aggregation(df)
{%- endif %}
{% if config.temporal_features and config.temporal_features.has_renderable_content() %}
    df = compute_temporal_features(df, raw_df)
{% endif %}
{% if config.text_features %}
    df = compute_text_features(df, raw_df)
{% endif %}
    output_name = f"{SOURCE_NAME}_aggregated"
    bronze_dir = PRODUCTION_DIR / "data" / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    _output_path = str(bronze_dir / output_name)
    storage.write(df, _output_path)
    _z_cols = [c for c in [ENTITY_COLUMN, "as_of_date"] if c in df.columns]
    if _z_cols:
        storage.optimize(_output_path, _z_cols)
    else:
        storage.optimize(_output_path)
    if "aggregation_reference_date" in df.attrs:
        (bronze_dir / f"{output_name}.reference_date").write_text(df.attrs["aggregation_reference_date"])
    return df


if __name__ == "__main__":
    run_bronze_event_{{ source }}()
""",
    "bronze_entity.py.j2": """import pandas as pd
import numpy as np
from pathlib import Path
{% set ops, fitted = collect_imports(config.post_shaping, False) %}
{% if ops %}
from customer_retention.transforms import {{ ops | sort | join(', ') }}
{% endif %}
from customer_retention.core.compat import ensure_timestamp, safe_to_datetime, timedelta_to_days
from config import PRODUCTION_DIR, RAW_SOURCES, get_bronze_path

SOURCE_NAME = "{{ source }}"
ENTITY_COLUMN = "{{ config.entity_column }}"
TIME_COLUMN = "{{ config.time_column }}"

{% if config.lifecycle %}

def _load_raw_events():
    source = RAW_SOURCES["{{ raw_source }}"]
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"Raw source not found: {path}")
    if source["format"] == "csv":
        return pd.read_csv(str(path))
    if source["format"] == "parquet":
        return pd.read_parquet(str(path))
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(path))

{% if config.lifecycle.include_recency_bucket %}

def add_recency_tenure(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    reference_date = raw_df[TIME_COLUMN].max()
    _grp = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN]
    entity_stats = _grp.min().to_frame("_time_min")
    entity_stats["_time_max"] = _grp.max()
    entity_stats["days_since_last"] = timedelta_to_days(reference_date - entity_stats["_time_max"])
    entity_stats["days_since_first"] = timedelta_to_days(reference_date - entity_stats["_time_min"])
    df = df.merge(entity_stats[["days_since_last", "days_since_first"]], left_on=ENTITY_COLUMN, right_index=True, how="left")
    return df


def add_recency_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_last" in df.columns:
        df["recency_bucket"] = pd.cut(df["days_since_last"], bins={{ config.lifecycle.recency_bucket_edges }} + [float("inf")],
                                       labels={{ config.lifecycle.recency_bucket_labels }})
    return df

{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}

def add_lifecycle_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_first" not in df.columns or "days_since_last" not in df.columns:
        return df
    event_count_cols = sorted(c for c in df.columns if c.startswith("event_count_"))
    if not event_count_cols:
        return df
    event_count_col = (
        "event_count_all_time" if "event_count_all_time" in event_count_cols
        else event_count_cols[-1]
    )
    duration = (df["days_since_first"] - df["days_since_last"]).astype(float)
    intensity = df[event_count_col].astype(float) / duration.clip(lower=1.0)
    tenure_med = float(duration.median())
    intensity_med = float(intensity.median())
    conditions = [
        (duration >= tenure_med) & (intensity >= intensity_med),
        (duration >= tenure_med) & (intensity < intensity_med),
        (duration < tenure_med) & (intensity >= intensity_med),
        (duration < tenure_med) & (intensity < intensity_med),
    ]
    labels = ["steady_loyal_lifecycle", "occasional_loyal_lifecycle", "intense_brief_lifecycle", "one_shot_lifecycle"]
    df["lifecycle_quadrant"] = np.select(conditions, labels, default="unknown")
    return df

{% endif %}
{% if config.lifecycle.include_cyclical_features %}

def add_cyclical_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    mean_dow = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: x.dt.dayofweek.mean())
    df = df.merge(mean_dow.rename("mean_dow"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["dow_sin"] = np.sin(2 * np.pi * df["mean_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["mean_dow"] / 7)
    df = df.drop(columns=["mean_dow"], errors="ignore")
    return df

{% endif %}
{% if config.lifecycle.include_month_cyclical %}

def add_month_quarter_cyclical(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    mean_month = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: x.dt.month.mean())
    df = df.merge(mean_month.rename("mean_month"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["month_sin"] = np.sin(2 * np.pi * df["mean_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["mean_month"] / 12)
{% if config.lifecycle.include_quarter_cyclical %}
    mean_quarter = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].apply(lambda x: ((x.dt.month - 1) // 3).mean())
    df = df.merge(mean_quarter.rename("mean_quarter"), left_on=ENTITY_COLUMN, right_index=True, how="left")
    df["quarter_sin"] = np.sin(2 * np.pi * df["mean_quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["mean_quarter"] / 4)
    df = df.drop(columns=["mean_month", "mean_quarter"], errors="ignore")
{% else %}
    df = df.drop(columns=["mean_month"], errors="ignore")
{% endif %}
    return df

{% endif %}
{% if config.lifecycle.include_trend_features %}

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    window_cols = sorted([c for c in df.columns if c.startswith("event_count_") and c != "event_count_all_time"])
    all_time_col = "event_count_all_time" if "event_count_all_time" in df.columns else None
    if window_cols and all_time_col:
        df["recent_vs_overall_ratio"] = df[window_cols[0]] / df[all_time_col].replace(0, float("nan"))
    if len(window_cols) >= 2:
        window_values = df[window_cols].values
        x = np.arange(len(window_cols), dtype=float)
        slopes = np.array([np.polyfit(x, row, 1)[0] if not np.any(np.isnan(row)) else 0.0 for row in window_values])
        df["entity_trend_slope"] = slopes
    return df

{% endif %}
{% if config.lifecycle.include_cohort_features %}

def add_cohort_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    ensure_timestamp(raw_df, TIME_COLUMN)
    first_event = raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].min()
    cohort_data = pd.DataFrame({"first_event": first_event})
    cohort_data["cohort_year"] = cohort_data["first_event"].dt.year
    cohort_data["cohort_quarter"] = ((cohort_data["first_event"].dt.month - 1) // 3 + 1)
    df = df.merge(cohort_data[["cohort_year", "cohort_quarter"]], left_on=ENTITY_COLUMN, right_index=True, how="left")
    return df

{% endif %}
{% if config.lifecycle.momentum_pairs %}

def add_momentum_ratios(df: pd.DataFrame) -> pd.DataFrame:
{% for pair in config.lifecycle.momentum_pairs %}
    short_col = "event_count_{{ pair.short_window }}"
    long_col = "event_count_{{ pair.long_window }}"
    if short_col in df.columns and long_col in df.columns:
        df["momentum_{{ pair.short_window }}_{{ pair.long_window }}"] = df[short_col] / df[long_col].replace(0, float("nan"))
{% endfor %}
    return df

{% endif %}

def enrich_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    raw_df = _load_raw_events()
{% if config.raw_time_column %}
    raw_df = raw_df.rename(columns={"{{ config.raw_time_column }}": TIME_COLUMN})
{% endif %}
{% if config.lifecycle.include_recency_bucket %}
    df = add_recency_tenure(df, raw_df)
    df = add_recency_buckets(df)
{% endif %}
{% if config.lifecycle.include_lifecycle_quadrant %}
    df = add_lifecycle_quadrant(df)
{% endif %}
{% if config.lifecycle.include_cyclical_features %}
    df = add_cyclical_features(df, raw_df)
{% endif %}
{% if config.lifecycle.include_month_cyclical %}
    df = add_month_quarter_cyclical(df, raw_df)
{% endif %}
{% if config.lifecycle.include_trend_features %}
    df = add_trend_features(df)
{% endif %}
{% if config.lifecycle.include_cohort_features %}
    df = add_cohort_features(df, raw_df)
{% endif %}
{% if config.lifecycle.momentum_pairs %}
    df = add_momentum_ratios(df)
{% endif %}
    return df
{% endif %}

{% set post_groups = group_steps(config.post_shaping) %}

def apply_quality_transforms(df: pd.DataFrame) -> pd.DataFrame:
{% if config.lifecycle %}
    df = enrich_lifecycle(df)
{% endif %}
{%- if post_groups %}
{%- for func_name, steps in post_groups %}
    df = {{ func_name }}(df)
{%- endfor %}
{%- endif %}
    return df

{% for func_name, steps in post_groups %}

def {{ func_name }}(df: pd.DataFrame) -> pd.DataFrame:
{%- set _prov = provenance_docstring_block(steps) %}
{%- if _prov %}
{{ _prov }}
{%- endif %}
{%- for t in steps %}
    # {{ t.rationale }}
    # {{ action_description(t) }}
    df = {{ render_step_call(t) }}
{%- endfor %}
    return df
{% endfor %}


def run_bronze_entity_{{ source }}():
    from customer_retention.integrations.adapters.factory import get_delta
    storage = get_delta(force_local=True)
    bronze_path = get_bronze_path("{{ bronze_input_name }}")
    if not storage.exists(str(bronze_path)):
        raise FileNotFoundError(f"Bronze input not found: {bronze_path}")
    df = storage.read(str(bronze_path))
    df = apply_quality_transforms(df)
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    storage.write(df, str(bronze_path))
    return df


if __name__ == "__main__":
    run_bronze_entity_{{ source }}()
""",
    "validate.py.j2": """import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config import (SOURCES, EXPLORATION_ARTIFACTS, EXPERIMENTS_DIR, PRODUCTION_DIR,
                    TARGET_COLUMN, get_silver_path, get_gold_path)


def _load_artifact(path):
    from customer_retention.integrations.adapters.factory import get_delta
    return get_delta(force_local=True).read(str(path))


def _categorize_missing(columns):
    categories = {}
    for col in columns:
        if any(col.startswith(p) for p in ["pca_", "text_pca_"]) or col.endswith("_pca"):
            categories.setdefault("text_embedding_pca", set()).add(col)
        elif any(k in col for k in ["_lag", "velocity", "acceleration"]):
            categories.setdefault("lag_velocity", set()).add(col)
        elif col in ("month_sin", "month_cos", "quarter_sin", "quarter_cos"):
            categories.setdefault("cyclical_month_quarter", set()).add(col)
        elif col in ("cohort_year", "cohort_quarter"):
            categories.setdefault("cohort", set()).add(col)
        elif col in ("recent_vs_overall_ratio", "entity_trend_slope"):
            categories.setdefault("trend", set()).add(col)
        elif col in ("dow_sin", "dow_cos"):
            categories.setdefault("cyclical_dow", set()).add(col)
        else:
            categories.setdefault("other", set()).add(col)
    return categories


def _compare_dataframes(stage, production_path, exploration_path, entity_key=None, tolerance=1e-5):
    from customer_retention.integrations.adapters.factory import get_delta
    storage = get_delta(force_local=True)
    if not storage.exists(str(production_path)):
        raise FileNotFoundError(f"[{stage}] Production output not found: {production_path}")
    if not storage.exists(str(exploration_path)):
        print(f"[{stage}] SKIP - exploration artifact not found: {exploration_path}")
        return True

    prod = _load_artifact(production_path)
    expl = _load_artifact(exploration_path)

    if entity_key and entity_key in prod.columns and entity_key in expl.columns:
        prod = prod.sort_values(entity_key).reset_index(drop=True)
        expl = expl.sort_values(entity_key).reset_index(drop=True)

    if prod.shape[0] != expl.shape[0]:
        raise AssertionError(f"[{stage}] Row count: production={prod.shape[0]} vs exploration={expl.shape[0]}")

    prod_cols = set(prod.columns)
    expl_cols = set(expl.columns)
    missing = expl_cols - prod_cols
    extra = prod_cols - expl_cols
    if missing:
        categories = _categorize_missing(missing)
        for category, cols in categories.items():
            print(f"[{stage}] MISSING ({category}): {sorted(cols)}")
    if extra:
        print(f"[{stage}] INFO: extra columns: {extra}")

    common = sorted(prod_cols & expl_cols)
    for col in common:
        if pd.api.types.is_numeric_dtype(prod[col]) and pd.api.types.is_numeric_dtype(expl[col]):
            try:
                pd.testing.assert_series_equal(prod[col], expl[col], check_exact=False, rtol=tolerance, check_names=False)
            except AssertionError as e:
                delta = (prod[col].astype(float) - expl[col].astype(float)).abs()
                max_idx = delta.idxmax()
                raise AssertionError(
                    f"[{stage}] Column '{col}' diverges at row {max_idx}: "
                    f"production={prod[col].iloc[max_idx]} vs exploration={expl[col].iloc[max_idx]} "
                    f"(max delta={delta.max():.2e})"
                ) from None

    print(f"[{stage}] PASS - {prod.shape[0]} rows, {len(common)} common cols, tolerance={tolerance}")
    return True


def validate_landing(tolerance=1e-5):
    landing_dir = PRODUCTION_DIR / "data" / "landing"
    if not landing_dir.exists():
        print("[Landing] SKIP - no landing directory")
        return True
    from customer_retention.integrations.adapters.factory import get_delta
    storage = get_delta(force_local=True)
    for path in landing_dir.iterdir():
        if storage.exists(str(path)):
            name = path.name
            expl_key = f"landing_{name}" if f"landing_{name}" in EXPLORATION_ARTIFACTS else "landing"
            if expl_key in EXPLORATION_ARTIFACTS:
                _compare_dataframes(f"Landing/{name}", str(path), EXPLORATION_ARTIFACTS[expl_key])
    return True


def validate_bronze(tolerance=1e-5):
    bronze_artifacts = EXPLORATION_ARTIFACTS.get("bronze", {})
    for name, expl_path in bronze_artifacts.items():
        prod_path = PRODUCTION_DIR / "data" / "bronze" / name
        _compare_dataframes(f"Bronze/{name}", str(prod_path), expl_path, tolerance=tolerance)
    return True


def validate_silver(tolerance=1e-5):
    prod_path = get_silver_path()
    expl_path = EXPLORATION_ARTIFACTS.get("silver", "")
    entity_key = list(SOURCES.values())[0]["entity_key"] if SOURCES else None
    _compare_dataframes("Silver", str(prod_path), expl_path, entity_key=entity_key, tolerance=tolerance)
    return True


def validate_gold(tolerance=1e-5):
    prod_path = get_gold_path()
    expl_path = EXPLORATION_ARTIFACTS.get("gold", "")
    entity_key = list(SOURCES.values())[0]["entity_key"] if SOURCES else None
    _compare_dataframes("Gold", str(prod_path), expl_path, entity_key=entity_key, tolerance=tolerance)
    return True


def validate_training():
    print("[Training] Split strategy: temporal (with DataSplitter)")
    print("[Training] PASS - training validation requires MLflow comparison (not yet implemented)")
    return True


def validate_scoring(tolerance=1e-5):
    prod_path = PRODUCTION_DIR / "data" / "scoring" / "predictions"
    expl_path = EXPLORATION_ARTIFACTS.get("scoring", "")
    _compare_dataframes("Scoring", str(prod_path), expl_path, tolerance=tolerance)
    return True


def run_all_validations(tolerance=1e-5):
    stages = [
        ("Landing", lambda: validate_landing(tolerance)),
        ("Bronze", lambda: validate_bronze(tolerance)),
        ("Silver", lambda: validate_silver(tolerance)),
        ("Gold", lambda: validate_gold(tolerance)),
        ("Training", validate_training),
        ("Scoring", lambda: validate_scoring(tolerance)),
    ]
    results = []
    for name, fn in stages:
        try:
            fn()
            results.append((name, "PASS"))
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            break

    print("\\nStage Validation Report")
    print("=" * 50)
    for name, status in results:
        print(f"[{status.split(':')[0]:4s}] {name}")
    return results
""",
    "run_validation.py.j2": '''"""{{ config.name }} - Standalone Validation Runner

Compares pipeline outputs against exploration artifacts.
Run after pipeline completes to verify correctness.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.validate_pipeline import run_all_validations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate pipeline outputs")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    results = run_all_validations(tolerance=args.tolerance)
    failures = [r for r in results if not r[1].startswith("PASS")]
    sys.exit(1 if failures else 0)
''',
    "exploration_report.py.j2": '''"""Exploration Report Viewer

Opens HTML documentation for the exploration notebooks that informed
the pipeline transformations. Works both locally (file:// URI) and
on Databricks (displayHTML with scroll-to-anchor injection).
"""
import os
import webbrowser
from pathlib import Path

# Known notebooks referenced by pipeline provenance comments
KNOWN_NOTEBOOKS = [
{% for nb in notebooks %}
    "{{ nb }}",
{% endfor %}
]

DOCS_DIR = Path(os.environ.get("CR_DOCS_BASE_URL", str(Path(__file__).parent)))


def _is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def list_reports():
    for nb in KNOWN_NOTEBOOKS:
        html_path = DOCS_DIR / f"{nb}.html"
        status = "available" if html_path.exists() else "missing"
        print(f"  {nb}: {status}")


if __name__ == "__main__":
    print("Available exploration reports:")
    list_reports()
''',
}


def _sorted_landing_names(landing_dict):
    bridge_deps = {}
    for name, cfg in landing_dict.items():
        deps = set()
        for step in getattr(cfg, "key_resolution_steps", []) or []:
            if step.bridge_dataset in landing_dict:
                deps.add(step.bridge_dataset)
        bridge_deps[name] = deps
    ordered = []
    remaining = set(landing_dict)
    while remaining:
        ready = [n for n in remaining if not bridge_deps[n] - set(ordered)]
        if not ready:
            ordered.extend(sorted(remaining))
            break
        for n in sorted(ready):
            ordered.append(n)
            remaining.discard(n)
    return ordered


class CodeRenderer:
    _TEMPLATE_MAP = {
        "config": "config.py.j2",
        "silver": "silver.py.j2",
        "gold": "gold.py.j2",
        "training": "training.py.j2",
        "runner": "runner.py.j2",
        "workflow": "workflow.json.j2",
        "run_all": "run_all.py.j2",
        "feast_config": "feature_store.yaml.j2",
        "feast_features": "features.py.j2",
        "landing": "landing.py.j2",
        "bronze_event": "bronze_event.py.j2",
        "validation": "validate.py.j2",
        "run_validation": "run_validation.py.j2",
        "exploration_report": "exploration_report.py.j2",
    }

    def __init__(self):
        self._env = Environment(loader=InlineLoader(TEMPLATES))
        self._env.globals["action_description"] = action_description
        self._env.globals["render_step_call"] = render_step_call
        self._env.globals["collect_imports"] = collect_imports
        self._env.globals["group_steps"] = group_steps
        self._env.globals["provenance_docstring_block"] = provenance_docstring_block
        self._env.globals["provenance_key"] = provenance_key
        self._env.globals["partition_gold_steps"] = partition_gold_steps
        self._env.globals["sorted_landing_names"] = _sorted_landing_names
        self._env.filters["python_repr"] = repr
        self._env.filters["py_source"] = render_python_literal

    def set_docs_base(self, experiments_dir: str | None) -> None:
        global _docs_base
        if experiments_dir:
            _docs_base = f"file://{Path(experiments_dir).resolve() / 'docs'}"
        else:
            _docs_base = "docs"

    def _render(self, template_key: str, **context) -> str:
        return self._env.get_template(self._TEMPLATE_MAP[template_key]).render(**context)

    def template_versions(self) -> Dict[str, str]:
        from .generation_manifest import template_versions_for
        return template_versions_for(TEMPLATES)

    def render_config(self, config: PipelineConfig) -> str:
        return self._render("config", config=config)

    def render_bronze(self, source_name: str, bronze_config: BronzeLayerConfig) -> str:
        return self._env.get_template("bronze.py.j2").render(source=source_name, config=bronze_config)

    def render_silver(self, config: PipelineConfig) -> str:
        return self._render("silver", config=config)

    def render_gold(self, config: PipelineConfig) -> str:
        return self._render("gold", config=config)

    def render_training(self, config: PipelineConfig) -> str:
        return self._render("training", config=config)

    def render_runner(self, config: PipelineConfig) -> str:
        return self._render("runner", config=config)

    def render_workflow(self, config: PipelineConfig) -> str:
        return self._render("workflow", config=config)

    def render_run_all(self, config: PipelineConfig) -> str:
        return self._render("run_all", config=config)

    def render_feast_config(self, config: PipelineConfig) -> str:
        return self._render("feast_config", config=config)

    def render_feast_features(self, config: PipelineConfig) -> str:
        return self._render("feast_features", config=config)

    def render_landing(self, name: str, config: LandingLayerConfig) -> str:
        return self._env.get_template("landing.py.j2").render(name=name, config=config)

    def render_bronze_event(
        self,
        source_name: str,
        config: BronzeEventConfig,
        pipeline_config: "PipelineConfig | None" = None,
    ) -> str:
        grid_dates = []
        if pipeline_config is not None and pipeline_config.silver is not None:
            grid_dates = list(pipeline_config.silver.grid_dates or [])
        return self._env.get_template("bronze_event.py.j2").render(
            source=source_name, config=config, grid_dates=grid_dates
        )

    def render_bronze_entity(
        self, source_name: str, config: BronzeEventConfig, bronze_input_name: str, raw_source_name: str = ""
    ) -> str:
        return self._env.get_template("bronze_entity.py.j2").render(
            source=source_name,
            config=config,
            bronze_input_name=bronze_input_name,
            raw_source=raw_source_name or source_name,
        )

    def render_validation(self, config: PipelineConfig) -> str:
        return self._render("validation", config=config)

    def render_run_validation(self, config: PipelineConfig) -> str:
        return self._render("run_validation", config=config)

    def render_exploration_report(self, config: PipelineConfig) -> str:
        notebooks = set()
        for bronze in config.bronze.values():
            for step in bronze.transformations:
                nb = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
                if nb:
                    notebooks.add(nb)
        for step in config.gold.transformations + config.gold.encodings + config.gold.scalings:
            nb = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
            if nb:
                notebooks.add(nb)
        for step in config.silver.derived_columns:
            nb = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
            if nb:
                notebooks.add(nb)
        for be in config.bronze_event.values():
            for step in be.pre_shaping + be.post_shaping:
                nb = step.source_notebook or DEFAULT_NOTEBOOK_MAP.get(step.type)
                if nb:
                    notebooks.add(nb)
        return self._render("exploration_report", notebooks=sorted(notebooks))


_StepMeta = namedtuple("_StepMeta", ["desc_tpl", "call_tpl", "import_name", "param_defaults"])

_STATELESS_REGISTRY = {
    PipelineTransformationType.IMPUTE_NULL: _StepMeta(
        "impute nulls in {col} with {value}",
        "apply_impute_null(df, '{col}', value='{value}')",
        "apply_impute_null",
        {"value": 0},
    ),
    PipelineTransformationType.CAP_OUTLIER: _StepMeta(
        "cap outliers in {col} to [{lower}, {upper}]",
        "apply_cap_outlier(df, '{col}', lower={lower}, upper={upper})",
        "apply_cap_outlier",
        {"lower": 0, "upper": 1000000},
    ),
    PipelineTransformationType.TYPE_CAST: _StepMeta(
        "cast {col} to {dtype}", "apply_type_cast(df, '{col}', dtype='{dtype}')", "apply_type_cast", {"dtype": "float"}
    ),
    PipelineTransformationType.DROP_COLUMN: _StepMeta(
        "drop column {col}", "apply_drop_column(df, '{col}')", "apply_drop_column", {}
    ),
    PipelineTransformationType.WINSORIZE: _StepMeta(
        "winsorize {col} to [{lower_bound}, {upper_bound}]",
        "apply_winsorize(df, '{col}', lower_bound={lower_bound}, upper_bound={upper_bound})",
        "apply_winsorize",
        {"lower_bound": 0, "upper_bound": 1000000},
    ),
    PipelineTransformationType.SEGMENT_AWARE_CAP: _StepMeta(
        "segment-aware outlier cap on {col} ({n_segments} segments)",
        "apply_segment_aware_cap(df, '{col}', n_segments={n_segments})",
        "apply_segment_aware_cap",
        {"n_segments": 2},
    ),
    PipelineTransformationType.LOG_TRANSFORM: _StepMeta(
        "log-transform {col}", "apply_log_transform(df, '{col}')", "apply_log_transform", {}
    ),
    PipelineTransformationType.SQRT_TRANSFORM: _StepMeta(
        "sqrt-transform {col}", "apply_sqrt_transform(df, '{col}')", "apply_sqrt_transform", {}
    ),
    PipelineTransformationType.ZERO_INFLATION_HANDLING: _StepMeta(
        "handle zero-inflation in {col}",
        "apply_zero_inflation_handling(df, '{col}')",
        "apply_zero_inflation_handling",
        {},
    ),
    PipelineTransformationType.CAP_THEN_LOG: _StepMeta(
        "cap at p99 then log-transform {col}", "apply_cap_then_log(df, '{col}')", "apply_cap_then_log", {}
    ),
    PipelineTransformationType.FEATURE_SELECT: _StepMeta(
        "drop {col} (feature selection)", "apply_feature_select(df, '{col}')", "apply_feature_select", {}
    ),
}


def _extract_params(step, meta):
    return {k: step.parameters.get(k, v) for k, v in meta.param_defaults.items()}


def action_description(step: TransformationStep) -> str:
    t, col, p = step.type, step.column, step.parameters
    meta = _STATELESS_REGISTRY.get(t)
    if meta is not None:
        return meta.desc_tpl.format(col=col, **_extract_params(step, meta))
    if t == PipelineTransformationType.YEO_JOHNSON:
        return f"yeo-johnson transform {col}"
    if t == PipelineTransformationType.ENCODE:
        method = p.get("method", "one_hot")
        if method in ("one_hot", "onehot"):
            return f"one-hot encode {col}"
        return f"label-encode {col}"
    if t == PipelineTransformationType.SCALE:
        method = p.get("method", "standard")
        if method == "minmax":
            return f"min-max scale {col}"
        return f"standard-scale {col}"
    if t == PipelineTransformationType.DERIVED_COLUMN:
        action = p.get("action", "ratio")
        if action == "ratio":
            return f"create {col} = {p.get('numerator', '?')} / {p.get('denominator', '?')}"
        if action == "interaction":
            features = p.get("features", [])
            col_a = features[0] if len(features) > 0 else p.get("col_a", "?")
            col_b = features[1] if len(features) > 1 else p.get("col_b", "?")
            return f"create {col} = {col_a} * {col_b}"
        if action == "composite":
            return f"create {col} = mean({', '.join(p.get('columns', []))})"
    return f"transform {col}"


def render_step_call(step: TransformationStep, fit_mode: bool = True) -> str:
    t, col, p = step.type, step.column, step.parameters
    meta = _STATELESS_REGISTRY.get(t)
    if meta is not None:
        return meta.call_tpl.format(col=col, **_extract_params(step, meta))
    if t == PipelineTransformationType.YEO_JOHNSON:
        method = "fit_transform" if fit_mode else "transform"
        return f"FittedPowerTransform().{method}(df, '{col}', _store)"
    if t == PipelineTransformationType.ENCODE:
        method = p.get("method", "one_hot")
        if method in ("one_hot", "onehot"):
            return f"apply_one_hot_encode(df, '{col}')"
        fit_method = "fit_transform" if fit_mode else "transform"
        return f"FittedEncoder().{fit_method}(df, '{col}', _store)"
    if t == PipelineTransformationType.SCALE:
        method = p.get("method", "standard")
        fit_method = "fit_transform" if fit_mode else "transform"
        return f"FittedScaler('{method}').{fit_method}(df, '{col}', _store)"
    if t == PipelineTransformationType.DERIVED_COLUMN:
        action = p.get("action", "ratio")
        if action == "ratio":
            return f"apply_derived_ratio(df, '{col}', numerator='{p.get('numerator', '')}', denominator='{p.get('denominator', '')}')"
        if action == "interaction":
            features = p.get("features", [])
            col_a = features[0] if len(features) > 0 else p.get("col_a", "")
            col_b = features[1] if len(features) > 1 else p.get("col_b", "")
            return f"apply_derived_interaction(df, '{col}', col_a='{col_a}', col_b='{col_b}')"
        if action == "composite":
            columns = p.get("columns", [])
            if not columns:
                raise ValueError(f"Composite derived column '{col}' requires non-empty 'columns' parameter")
            return f"apply_derived_composite(df, '{col}', columns={columns})"
    if t == PipelineTransformationType.FILTER:
        condition = p.get("condition", "non_negative")
        if condition == "non_negative":
            return f"df[df['{col}'] >= 0]"
        if condition == "range":
            min_val = p.get("min_value", 0)
            max_val = p.get("max_value", 1000000)
            return f"df[(df['{col}'] >= {min_val}) & (df['{col}'] <= {max_val})]"
        if condition == "valid_values":
            valid = p.get("valid_values", [])
            return f"df[df['{col}'].isin({valid})]"
        return f"df[df['{col}'].notna()]"
    raise ValueError(f"Unknown transformation type: {step.type}")


def collect_imports(steps, include_fitted):
    ops = set()
    fitted = set()
    _OPS_MAP = {k: v.import_name for k, v in _STATELESS_REGISTRY.items()}
    for step in steps:
        t, p = step.type, step.parameters
        if t in _OPS_MAP:
            ops.add(_OPS_MAP[t])
        elif t == PipelineTransformationType.ENCODE:
            method = p.get("method", "one_hot")
            if method in ("one_hot", "onehot"):
                ops.add("apply_one_hot_encode")
            elif include_fitted:
                fitted.add("FittedEncoder")
        elif t == PipelineTransformationType.SCALE:
            if include_fitted:
                fitted.add("FittedScaler")
        elif t == PipelineTransformationType.YEO_JOHNSON:
            if include_fitted:
                fitted.add("FittedPowerTransform")
        elif t == PipelineTransformationType.DERIVED_COLUMN:
            action = p.get("action", "ratio")
            if action == "ratio":
                ops.add("apply_derived_ratio")
            elif action == "interaction":
                ops.add("apply_derived_interaction")
            elif action == "composite":
                ops.add("apply_derived_composite")
    return ops, fitted


def partition_gold_steps(gold):
    FITTED_TYPES = {PipelineTransformationType.SCALE, PipelineTransformationType.YEO_JOHNSON}

    stateless_transforms, fitted_transforms = [], []
    for t in gold.transformations:
        (fitted_transforms if t.type in FITTED_TYPES else stateless_transforms).append(t)

    stateless_encodings, fitted_encodings = [], []
    for e in gold.encodings:
        method = e.parameters.get("method", "one_hot")
        if method in ("one_hot", "onehot"):
            stateless_encodings.append(e)
        else:
            fitted_encodings.append(e)

    fitted_scalings = list(gold.scalings)

    return {
        "stateless_transforms": stateless_transforms,
        "stateless_encodings": stateless_encodings,
        "fitted_transforms": fitted_transforms,
        "fitted_encodings": fitted_encodings,
        "fitted_scalings": fitted_scalings,
    }
