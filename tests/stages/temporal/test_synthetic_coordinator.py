import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from customer_retention.stages.temporal.point_in_time_registry import PointInTimeRegistry
from customer_retention.stages.temporal.synthetic_coordinator import (
    SyntheticCoordinationParams,
    SyntheticTimestampCoordinator,
)
from customer_retention.stages.temporal.timestamp_manager import TimestampConfig, TimestampStrategy


class TestSyntheticCoordinationParams:
    def test_defaults_to_synthetic_index_strategy(self):
        params = SyntheticCoordinationParams()
        assert params.strategy == TimestampStrategy.SYNTHETIC_INDEX

    def test_defaults_base_date_to_2024(self):
        params = SyntheticCoordinationParams()
        assert params.base_date == "2024-01-01"

    def test_custom_params_override_all_defaults(self):
        params = SyntheticCoordinationParams(
            base_date="2023-06-15", range_days=180,
            observation_window_days=60, strategy=TimestampStrategy.SYNTHETIC_RANDOM,
        )
        assert params.base_date == "2023-06-15"
        assert params.range_days == 180
        assert params.observation_window_days == 60
        assert params.strategy == TimestampStrategy.SYNTHETIC_RANDOM


class TestSyntheticTimestampCoordinator:
    def test_create_config_returns_timestamp_config(self):
        coordinator = SyntheticTimestampCoordinator()
        config = coordinator.create_config("dataset_a")
        assert isinstance(config, TimestampConfig)

    def test_create_config_uses_coordinator_base_date(self):
        params = SyntheticCoordinationParams(base_date="2023-03-01")
        coordinator = SyntheticTimestampCoordinator(params)
        config = coordinator.create_config("ds")
        assert config.synthetic_base_date == "2023-03-01"

    def test_create_config_uses_coordinator_strategy(self):
        params = SyntheticCoordinationParams(strategy=TimestampStrategy.SYNTHETIC_RANDOM)
        coordinator = SyntheticTimestampCoordinator(params)
        config = coordinator.create_config("ds")
        assert config.strategy == TimestampStrategy.SYNTHETIC_RANDOM

    def test_create_config_uses_coordinator_observation_window(self):
        params = SyntheticCoordinationParams(observation_window_days=45)
        coordinator = SyntheticTimestampCoordinator(params)
        config = coordinator.create_config("ds")
        assert config.observation_window_days == 45

    def test_multiple_configs_share_same_base_date(self):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("dataset_a")
        config_b = coordinator.create_config("dataset_b")
        assert config_a.synthetic_base_date == config_b.synthetic_base_date

    def test_multiple_configs_share_same_observation_window(self):
        params = SyntheticCoordinationParams(observation_window_days=120)
        coordinator = SyntheticTimestampCoordinator(params)
        config_a = coordinator.create_config("a")
        config_b = coordinator.create_config("b")
        assert config_a.observation_window_days == config_b.observation_window_days == 120

    def test_registered_datasets_tracks_created_configs(self):
        coordinator = SyntheticTimestampCoordinator()
        coordinator.create_config("retail")
        coordinator.create_config("bank")
        coordinator.create_config("telecom")
        assert set(coordinator.registered_datasets) == {"retail", "bank", "telecom"}

    def test_validate_compatibility_passes_for_consistent_configs(self):
        coordinator = SyntheticTimestampCoordinator()
        coordinator.create_config("a")
        coordinator.create_config("b")
        valid, msg = coordinator.validate_compatibility()
        assert valid is True

    def test_validate_compatibility_single_config_is_valid(self):
        coordinator = SyntheticTimestampCoordinator()
        coordinator.create_config("only_one")
        valid, _ = coordinator.validate_compatibility()
        assert valid is True

    def test_validate_compatibility_detects_observation_window_mismatch(self):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("a")
        coordinator.create_config("b")
        config_a.observation_window_days = 999
        valid, msg = coordinator.validate_compatibility()
        assert valid is False
        assert "observation_window" in msg

    def test_validate_compatibility_detects_strategy_mismatch(self):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("a")
        coordinator.create_config("b")
        config_a.strategy = TimestampStrategy.SYNTHETIC_RANDOM
        valid, msg = coordinator.validate_compatibility()
        assert valid is False
        assert "strategy" in msg

    def test_validate_compatibility_detects_external_config_modification(self):
        coordinator = SyntheticTimestampCoordinator()
        config_a = coordinator.create_config("a")
        coordinator.create_config("b")
        config_a.synthetic_base_date = "1999-01-01"
        valid, msg = coordinator.validate_compatibility()
        assert valid is False
        assert "base_date" in msg or "1999" in msg


class TestCoordinatorFromRegistry:
    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_from_empty_registry_uses_defaults(self, temp_dir):
        registry = PointInTimeRegistry(temp_dir)
        coordinator = SyntheticTimestampCoordinator.from_registry(registry)
        config = coordinator.create_config("ds")
        assert config.synthetic_base_date == "2024-01-01"

    def test_from_registry_with_snapshots_uses_reference_cutoff(self, temp_dir):
        registry = PointInTimeRegistry(temp_dir)
        registry.register_snapshot("existing", "snap_1", datetime(2023, 7, 15), "/data/x.csv", 1000)
        coordinator = SyntheticTimestampCoordinator.from_registry(registry)
        config = coordinator.create_config("new_ds")
        assert config.synthetic_base_date == "2023-07-15"

    def test_from_registry_creates_configs_compatible_with_registry_cutoff(self, temp_dir):
        registry = PointInTimeRegistry(temp_dir)
        registry.register_snapshot("prod", "snap_1", datetime(2023, 9, 1), "/data/p.csv", 5000)
        coordinator = SyntheticTimestampCoordinator.from_registry(registry)
        config_a = coordinator.create_config("synthetic_a")
        config_b = coordinator.create_config("synthetic_b")
        assert config_a.synthetic_base_date == config_b.synthetic_base_date == "2023-09-01"
