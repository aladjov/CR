import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from customer_retention.core.config.source_config import DataSourceConfig

from .load_result import LoadResult


class LoadHistoryEntry(BaseModel):
    timestamp: str
    row_count: int
    duration_seconds: float
    success: bool
    warnings: list[str] = []
    errors: list[str] = []


class SourceRegistration(BaseModel):
    source_config: DataSourceConfig
    registered_at: str
    registered_by: str
    last_loaded_at: Optional[str] = None
    last_row_count: Optional[int] = None
    last_load_duration: Optional[float] = None
    load_history: list[LoadHistoryEntry] = []

    def update_from_load(self, load_result: LoadResult) -> None:
        entry = LoadHistoryEntry(
            timestamp=datetime.now().isoformat(),
            row_count=load_result.row_count,
            duration_seconds=load_result.duration_seconds,
            success=load_result.success,
            warnings=load_result.warnings,
            errors=load_result.errors
        )

        self.load_history.append(entry)
        if len(self.load_history) > 100:
            self.load_history = self.load_history[-100:]

        self.last_loaded_at = entry.timestamp
        self.last_row_count = load_result.row_count
        self.last_load_duration = load_result.duration_seconds


class DataSourceRegistry:
    def __init__(self):
        self._sources: dict[str, SourceRegistration] = {}

    def register(self, config: DataSourceConfig, registered_by: str = "system",
                overwrite: bool = False) -> None:
        if config.name in self._sources and not overwrite:
            raise ValueError(f"Source '{config.name}' already registered. Use overwrite=True to replace.")

        self._sources[config.name] = SourceRegistration(
            source_config=config,
            registered_at=datetime.now().isoformat(),
            registered_by=registered_by
        )

    def get(self, name: str) -> Optional[SourceRegistration]:
        return self._sources.get(name)

    def list_sources(self) -> list[str]:
        return list(self._sources.keys())

    def record_load(self, source_name: str, load_result: LoadResult) -> None:
        registration = self.get(source_name)
        if not registration:
            raise ValueError(f"Source '{source_name}' not found in registry")
        registration.update_from_load(load_result)

    def get_load_stats(self, source_name: str) -> dict:
        registration = self.get(source_name)
        if not registration:
            raise ValueError(f"Source '{source_name}' not found in registry")

        total_loads = len(registration.load_history)
        successful_loads = sum(1 for entry in registration.load_history if entry.success)
        failed_loads = total_loads - successful_loads

        return {
            "source_name": source_name,
            "total_loads": total_loads,
            "successful_loads": successful_loads,
            "failed_loads": failed_loads,
            "last_loaded_at": registration.last_loaded_at,
            "last_row_count": registration.last_row_count,
            "last_load_duration": registration.last_load_duration
        }

    def save_to_file(self, path: str) -> None:
        data = {name: reg.model_dump() for name, reg in self._sources.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
        self._sources = {
            name: SourceRegistration(**reg_data)
            for name, reg_data in data.items()
        }

    def validate_source(self, config: DataSourceConfig) -> list[str]:
        errors = []

        if not config.name:
            errors.append("Source name is required")
        if not config.primary_key:
            errors.append("Primary key is required")

        duplicate_columns = self.find_duplicate_column_names(config)
        if duplicate_columns:
            errors.append(f"Duplicate column names found: {', '.join(duplicate_columns)}")

        return errors

    def find_duplicate_column_names(self, config: DataSourceConfig) -> list[str]:
        column_names = [c.name for c in config.columns]
        seen = set()
        duplicates = set()
        for name in column_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        return list(duplicates)
