import json

import pytest

from customer_retention.core.config import ColumnConfig, ColumnType, DataSourceConfig, FileFormat, SourceType
from customer_retention.stages.ingestion import DataSourceRegistry, LoadResult


class TestDataSourceRegistry:
    def test_create_empty_registry(self):
        registry = DataSourceRegistry()
        assert len(registry.list_sources()) == 0

    def test_register_source(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        assert len(registry.list_sources()) == 1
        assert "customer_master" in registry.list_sources()

    def test_register_source_with_registered_by(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config, registered_by="test_user")

        registration = registry.get("customer_master")
        assert registration.registered_by == "test_user"

    def test_register_duplicate_raises_error(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(sample_source_config)

    def test_register_duplicate_with_overwrite(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config, registered_by="user1")
        registry.register(sample_source_config, registered_by="user2", overwrite=True)

        registration = registry.get("customer_master")
        assert registration.registered_by == "user2"

    def test_get_existing_source(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        registration = registry.get("customer_master")
        assert registration is not None
        assert registration.source_config.name == "customer_master"

    def test_get_nonexistent_source(self):
        registry = DataSourceRegistry()
        registration = registry.get("nonexistent")
        assert registration is None

    def test_list_sources_empty(self):
        registry = DataSourceRegistry()
        assert registry.list_sources() == []

    def test_list_sources_multiple(self):
        registry = DataSourceRegistry()
        config1 = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV
        )
        config2 = DataSourceConfig(
            name="source2",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test2.csv",
            file_format=FileFormat.CSV
        )
        registry.register(config1)
        registry.register(config2)

        sources = registry.list_sources()
        assert len(sources) == 2
        assert "source1" in sources
        assert "source2" in sources

    def test_record_load(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        load_result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="customer_master"
        )
        registry.record_load("customer_master", load_result)

        registration = registry.get("customer_master")
        assert registration.last_row_count == 100
        assert registration.last_load_duration == 1.5
        assert len(registration.load_history) == 1

    def test_record_load_nonexistent_source(self):
        registry = DataSourceRegistry()
        load_result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="nonexistent"
        )

        with pytest.raises(ValueError, match="not found in registry"):
            registry.record_load("nonexistent", load_result)

    def test_record_multiple_loads(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        for i in range(3):
            load_result = LoadResult(
                success=True,
                row_count=100 + i,
                column_count=10,
                duration_seconds=1.0 + i * 0.1,
                source_name="customer_master"
            )
            registry.record_load("customer_master", load_result)

        registration = registry.get("customer_master")
        assert len(registration.load_history) == 3
        assert registration.last_row_count == 102

    def test_load_history_limit_100(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        for i in range(150):
            load_result = LoadResult(
                success=True,
                row_count=100,
                column_count=10,
                duration_seconds=1.0,
                source_name="customer_master"
            )
            registry.record_load("customer_master", load_result)

        registration = registry.get("customer_master")
        assert len(registration.load_history) == 100

    def test_get_load_stats(self, sample_source_config):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        load_result1 = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="customer_master"
        )
        load_result2 = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="customer_master",
            errors=["Error"]
        )

        registry.record_load("customer_master", load_result1)
        registry.record_load("customer_master", load_result2)

        stats = registry.get_load_stats("customer_master")
        assert stats["source_name"] == "customer_master"
        assert stats["total_loads"] == 2
        assert stats["successful_loads"] == 1
        assert stats["failed_loads"] == 1
        assert stats["last_row_count"] == 0

    def test_get_load_stats_nonexistent_source(self):
        registry = DataSourceRegistry()
        with pytest.raises(ValueError, match="not found in registry"):
            registry.get_load_stats("nonexistent")

    def test_save_to_file(self, sample_source_config, temp_registry_file):
        registry = DataSourceRegistry()
        registry.register(sample_source_config)

        load_result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="customer_master"
        )
        registry.record_load("customer_master", load_result)

        registry.save_to_file(temp_registry_file)

        with open(temp_registry_file, 'r') as f:
            data = json.load(f)

        assert "customer_master" in data
        assert data["customer_master"]["source_config"]["name"] == "customer_master"

    def test_load_from_file(self, sample_source_config, temp_registry_file):
        registry1 = DataSourceRegistry()
        registry1.register(sample_source_config)
        registry1.save_to_file(temp_registry_file)

        registry2 = DataSourceRegistry()
        registry2.load_from_file(temp_registry_file)

        assert len(registry2.list_sources()) == 1
        registration = registry2.get("customer_master")
        assert registration is not None
        assert registration.source_config.name == "customer_master"

    def test_validate_source_valid(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        registry = DataSourceRegistry()
        errors = registry.validate_source(config)
        assert len(errors) == 0

    def test_validate_source_missing_name(self):
        config = DataSourceConfig(
            name="",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        registry = DataSourceRegistry()
        errors = registry.validate_source(config)
        assert len(errors) > 0
        assert any("name" in err.lower() for err in errors)

    def test_validate_source_missing_primary_key(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        registry = DataSourceRegistry()
        errors = registry.validate_source(config)
        assert len(errors) > 0
        assert any("primary key" in err.lower() for err in errors)

    def test_validate_source_duplicate_columns(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        registry = DataSourceRegistry()
        errors = registry.validate_source(config)
        assert len(errors) > 0
        assert any("duplicate" in err.lower() for err in errors)

    def test_find_duplicate_column_names_none(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        registry = DataSourceRegistry()
        duplicates = registry.find_duplicate_column_names(config)
        assert len(duplicates) == 0

    def test_find_duplicate_column_names_exists(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        registry = DataSourceRegistry()
        duplicates = registry.find_duplicate_column_names(config)
        assert len(duplicates) == 1
        assert "age" in duplicates
