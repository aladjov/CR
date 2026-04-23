"""Runtime configuration — reads env vars set by app.yaml (or .env locally)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    catalog: str
    schema: str
    warehouse_id: str
    profile_template_path: str

    @property
    def fqn_prefix(self) -> str:
        return f"{self.catalog}.{self.schema}"


def load_config() -> AppConfig:
    return AppConfig(
        catalog=os.environ.get("CR_CATALOG", "churnkit"),
        schema=os.environ.get("CR_SCHEMA", "analysis"),
        warehouse_id=os.environ.get("CR_WAREHOUSE_ID", ""),
        profile_template_path=os.environ.get("CR_PROFILE_TEMPLATE_PATH", ""),
    )
