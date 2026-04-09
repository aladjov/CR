"""Lifecycle enrichment for service-unit datasets.

Transforms an entity-with-lifespan dataset (e.g. contracts with start +
optional termination dates) into a doubled event stream so the framework's
existing event-level pipeline can capture the lifecycle gradient.

Quick Start:
    >>> from customer_retention.stages.lifecycle import (
    ...     LifecycleEnrichmentConfig, enrich_lifecycle_dataset,
    ... )
    >>> config = LifecycleEnrichmentConfig(
    ...     enriched_view_name="enriched_contract",
    ...     parent_entity_key="account_id",
    ...     sub_entity_key="contract_id",
    ...     valid_from_column="start_date",
    ...     valid_to_columns=("termination_date",),
    ...     status_column="status",
    ...     terminal_status_values=("cancelled", "terminated", "expired"),
    ... )
    >>> doubled = enrich_lifecycle_dataset(raw_contract_df, config=config)
"""

from .config import LifecycleEnrichmentConfig
from .enrich import enrich_lifecycle_dataset

__all__ = [
    "LifecycleEnrichmentConfig",
    "enrich_lifecycle_dataset",
]
