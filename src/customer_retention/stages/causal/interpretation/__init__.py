"""Deterministic business-interpretation layer.

Modules here translate structured lineage and population statistics into
prose the LLM namer and dashboard can consume without inventing values.

See ``docs/business_interpretation_layer_plan.md`` for the full design.
"""
from customer_retention.stages.causal.interpretation.business_phrase import (
    render_business_phrase,
    render_window_phrase,
)

__all__ = ["render_business_phrase", "render_window_phrase"]
