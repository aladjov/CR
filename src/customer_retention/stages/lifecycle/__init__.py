"""Lifecycle enrichment for service-unit datasets.

Transforms an entity-with-lifespan dataset (e.g. contracts with start +
optional termination dates) into a doubled event stream so the framework's
existing event-level pipeline can capture the lifecycle gradient.

Quick Start:
    >>> from customer_retention.stages.lifecycle import (
    ...     LifecycleEnrichmentConfig, enrich_lifecycle_dataset,
    ... )
    >>> config = LifecycleEnrichmentConfig(
    ...     enriched_view_name="sps_enriched_contract",
    ...     parent_entity_key="ACCOUNT_ID",
    ...     sub_entity_key="CONTRACT_ID",
    ...     valid_from_column="CONTRACT_START_DATE",
    ...     valid_to_columns=("BILLING_TERMINATION_DATE",),
    ...     status_column="CONTRACT_STATUS",
    ...     terminal_status_values=("Cancelled", "Terminated", "Expired"),
    ... )
    >>> doubled = enrich_lifecycle_dataset(raw_contract_df, config=config)
"""

from .config import LifecycleEnrichmentConfig
from .enrich import enrich_lifecycle_dataset

__all__ = [
    "LifecycleEnrichmentConfig",
    "enrich_lifecycle_dataset",
]
