from dataclasses import dataclass
from typing import Optional

from .point_in_time_registry import PointInTimeRegistry
from .timestamp_manager import TimestampConfig, TimestampStrategy


@dataclass
class SyntheticCoordinationParams:
    base_date: str = "2024-01-01"
    range_days: int = 365
    observation_window_days: int = 90
    strategy: TimestampStrategy = TimestampStrategy.SYNTHETIC_INDEX


class SyntheticTimestampCoordinator:
    def __init__(self, params: Optional[SyntheticCoordinationParams] = None):
        self._params = params or SyntheticCoordinationParams()
        self._registered_configs: dict[str, TimestampConfig] = {}

    def create_config(self, dataset_name: str) -> TimestampConfig:
        config = TimestampConfig(
            strategy=self._params.strategy,
            synthetic_base_date=self._params.base_date,
            synthetic_range_days=self._params.range_days,
            observation_window_days=self._params.observation_window_days,
        )
        self._registered_configs[dataset_name] = config
        return config

    @property
    def registered_datasets(self) -> list[str]:
        return list(self._registered_configs.keys())

    def validate_compatibility(self) -> tuple[bool, str]:
        if len(self._registered_configs) <= 1:
            return True, "compatible"
        configs = list(self._registered_configs.values())
        reference = configs[0]
        for name, config in self._registered_configs.items():
            if config.synthetic_base_date != reference.synthetic_base_date:
                return False, (
                    f"Incompatible base_date: '{name}' has {config.synthetic_base_date}, "
                    f"expected {reference.synthetic_base_date}"
                )
            if config.observation_window_days != reference.observation_window_days:
                return False, (
                    f"Incompatible observation_window_days: '{name}' has {config.observation_window_days}, "
                    f"expected {reference.observation_window_days}"
                )
            if config.strategy != reference.strategy:
                return False, (
                    f"Incompatible strategy: '{name}' has {config.strategy.value}, "
                    f"expected {reference.strategy.value}"
                )
        return True, "compatible"

    @classmethod
    def from_registry(cls, registry: PointInTimeRegistry) -> "SyntheticTimestampCoordinator":
        reference_cutoff = registry.get_reference_cutoff()
        if reference_cutoff is None:
            return cls()
        params = SyntheticCoordinationParams(
            base_date=reference_cutoff.strftime("%Y-%m-%d"),
        )
        return cls(params)
