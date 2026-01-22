"""
Customer Retention ML Pipeline Framework.

A modular, production-ready framework for building customer retention
and churn prediction pipelines that work in both local and Databricks
environments.

Main module categories:
- core: Base infrastructure (config, utils, compatibility)
- stages: Pipeline stages (ingestion, profiling, cleaning, transformation,
          features, modeling, deployment, monitoring, validation, temporal)
- analysis: Analysis tools (diagnostics, interpretability, visualization,
            business, auto_explorer, discovery, recommendations)
- generators: Auto-generation tools (notebook_generator, pipeline_generator,
              spec_generator, orchestration)
- integrations: External system adapters (adapters, feature_store, streaming,
                llm_context, iteration)
"""

__version__ = "1.0.0"

# Environment utilities (always available)
from .core.compat import (
    is_databricks,
    is_notebook,
    is_spark_available,
    pd,
)

__all__ = [
    "__version__",
    # Environment
    "pd",
    "is_spark_available",
    "is_databricks",
    "is_notebook",
]
