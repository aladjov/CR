"""Runtime configuration — reads env vars set by app.yaml (or .env locally).

The app's catalog / schema / warehouse all come from process env. The
customer-profile HTML body is no longer resolved from a file path -- it
lives as a row in ``{catalog}.{schema}.dashboard_template_overrides``
and is fetched by ``data.load_template_html_from_uc()`` over the SQL
warehouse. The previous Volume-FUSE auto-discovery path was unreliable
from Databricks Apps even with READ_VOLUME granted, so it's been removed.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    catalog: str
    schema: str
    warehouse_id: str

    @property
    def fqn_prefix(self) -> str:
        return f"{self.catalog}.{self.schema}"


def load_config() -> AppConfig:
    catalog = os.environ.get("CR_CATALOG", "churnkit")
    schema = os.environ.get("CR_SCHEMA", "analysis")
    return AppConfig(
        catalog=catalog,
        schema=schema,
        warehouse_id=os.environ.get("CR_WAREHOUSE_ID", ""),
    )
