"""Pure helpers for masthead and L1 title rendering.

Lives in its own module so unit tests can import it without triggering the
Streamlit side effects in ``app.py`` (page config, state init, theme injection).
"""
from __future__ import annotations

from html import escape

OBJECTIVE_LABELS = {
    "immediate_risk":   "Immediate risk",
    "renewal_risk":     "Renewal risk",
    "disengagement":    "Disengagement",
}
POSTURE_LABELS = {
    "long_memory":      "Stable posture",
    "short_memory":     "Reactive posture",
}
MODEL_TYPE_LABELS = {
    "xgboost":          "XGBoost",
    "lightgbm":         "LightGBM",
    "catboost":         "CatBoost",
    "sklearn":          "scikit-learn",
    "pytorch":          "PyTorch",
    "tensorflow":       "TensorFlow",
}


def _objective_label(value) -> str | None:
    if not value:
        return None
    return OBJECTIVE_LABELS.get(value, str(value).replace("_", " ").title())


def _posture_label(value) -> str | None:
    if not value:
        return None
    return POSTURE_LABELS.get(value, str(value).replace("_", " ").title())


def _model_type_label(value) -> str | None:
    if not value:
        return None
    return MODEL_TYPE_LABELS.get(value, str(value).title())


def horizon_phrase(ctx: dict) -> str | None:
    horizon = ctx.get("horizon_days")
    if horizon is None:
        return None
    try:
        return f"Churn Risk in next {int(horizon)} days"
    except (TypeError, ValueError):
        return None


def context_segments(ctx: dict) -> list[str]:
    segments: list[str] = []
    for label in (
        _objective_label(ctx.get("primary_objective")),
        _posture_label(ctx.get("temporal_posture")),
        _model_type_label(ctx.get("model_type")),
    ):
        if label:
            segments.append(label)
    return segments


def masthead_title(ctx: dict) -> tuple[str, list[str]]:
    """Return (main_title, [subtitle segments]) for the small masthead."""
    title = horizon_phrase(ctx) or "Churn Risk"
    segments: list[str] = []
    for label in (
        _model_type_label(ctx.get("model_type")),
        _objective_label(ctx.get("primary_objective")),
        _posture_label(ctx.get("temporal_posture")),
    ):
        if label:
            segments.append(label)
    return title, segments


def l1_title_html(ctx: dict) -> str:
    """L1 hero headline. Always dynamic: when run context is unavailable
    (``v_run_context`` missing or empty), surfaces a "context unavailable"
    flourish so the operator immediately sees that c05 needs republishing
    rather than a misleading editorial fallback.
    """
    main = horizon_phrase(ctx)
    if main is None:
        return 'Churn Risk &middot; <em>Actionable insights</em>'
    extras = context_segments(ctx)
    if not extras:
        return escape(main)
    flourish = " &middot; ".join(escape(s) for s in extras)
    return f"{escape(main)} <em>{flourish}</em>"
