import logging
from pathlib import Path
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


def _merge_registry_bronze_overrides(
    namespace, nb10_overrides: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    if namespace is None:
        return dict(nb10_overrides or {})
    recs_path = getattr(namespace, "merged_recommendations_path", None)
    if recs_path is None or not Path(recs_path).exists():
        return dict(nb10_overrides or {})
    try:
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        registry = RecommendationRegistry.load(recs_path)
    except Exception as e:
        _logger.warning(
            "override_merge: failed to load registry from %s (%s: %s); "
            "falling back to NB10-provided overrides only",
            recs_path, type(e).__name__, e,
        )
        return dict(nb10_overrides or {})
    return registry.merge_bronze_aggregation_overrides(nb10_overrides)
