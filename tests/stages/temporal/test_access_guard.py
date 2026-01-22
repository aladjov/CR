import os
from pathlib import Path

import pytest

from customer_retention.stages.temporal.access_guard import (
    AccessContext,
    DataAccessGuard,
    guarded_read,
    require_context,
)


class TestAccessContext:
    def test_context_values(self):
        assert AccessContext.EXPLORATION.value == "exploration"
        assert AccessContext.TRAINING.value == "training"
        assert AccessContext.INFERENCE.value == "inference"
        assert AccessContext.BACKFILL.value == "backfill"
        assert AccessContext.ADMIN.value == "admin"


class TestDataAccessGuard:
    def test_exploration_blocks_raw_data(self):
        guard = DataAccessGuard(AccessContext.EXPLORATION)

        with pytest.raises(PermissionError, match="blocked"):
            guard.validate_access("/data/raw/customers.csv")

    def test_exploration_allows_snapshots(self):
        guard = DataAccessGuard(AccessContext.EXPLORATION)
        assert guard.validate_access("/data/snapshots/training_v1.parquet")

    def test_training_blocks_raw(self):
        guard = DataAccessGuard(AccessContext.TRAINING)

        with pytest.raises(PermissionError):
            guard.validate_access("/data/raw/customers.csv")

    def test_training_allows_gold(self):
        guard = DataAccessGuard(AccessContext.TRAINING)
        assert guard.validate_access("/data/gold/features.parquet")

    def test_inference_blocks_snapshots(self):
        guard = DataAccessGuard(AccessContext.INFERENCE)

        with pytest.raises(PermissionError):
            guard.validate_access("/data/snapshots/training_v1.parquet")

    def test_inference_allows_feature_store(self):
        guard = DataAccessGuard(AccessContext.INFERENCE)
        assert guard.validate_access("/data/feature_store/customer_features.parquet")

    def test_backfill_allows_raw(self):
        guard = DataAccessGuard(AccessContext.BACKFILL)
        assert guard.validate_access("/data/raw/customers.csv")

    def test_backfill_blocks_snapshots(self):
        guard = DataAccessGuard(AccessContext.BACKFILL)

        with pytest.raises(PermissionError):
            guard.validate_access("/data/snapshots/training_v1.parquet")

    def test_admin_allows_everything(self):
        guard = DataAccessGuard(AccessContext.ADMIN)
        assert guard.validate_access("/data/raw/customers.csv")
        assert guard.validate_access("/data/snapshots/training_v1.parquet")
        assert guard.validate_access("/data/bronze/cleaned.parquet")


class TestIsAllowed:
    def test_is_allowed_returns_true_for_valid_paths(self):
        guard = DataAccessGuard(AccessContext.EXPLORATION)
        assert guard.is_allowed("/data/snapshots/training_v1.parquet") is True

    def test_is_allowed_returns_false_for_invalid_paths(self):
        guard = DataAccessGuard(AccessContext.EXPLORATION)
        assert guard.is_allowed("/data/raw/customers.csv") is False

    def test_admin_is_allowed_everything(self):
        guard = DataAccessGuard(AccessContext.ADMIN)
        assert guard.is_allowed("/any/path/here.csv") is True


class TestContextManagement:
    def teardown_method(self):
        if "DATA_ACCESS_CONTEXT" in os.environ:
            del os.environ["DATA_ACCESS_CONTEXT"]

    def test_set_context(self):
        DataAccessGuard.set_context(AccessContext.TRAINING)
        assert os.environ.get("DATA_ACCESS_CONTEXT") == "training"

    def test_get_current_context_default(self):
        context = DataAccessGuard.get_current_context()
        assert context == AccessContext.EXPLORATION

    def test_get_current_context_from_env(self):
        os.environ["DATA_ACCESS_CONTEXT"] = "inference"
        context = DataAccessGuard.get_current_context()
        assert context == AccessContext.INFERENCE

    def test_from_environment(self):
        os.environ["DATA_ACCESS_CONTEXT"] = "backfill"
        guard = DataAccessGuard.from_environment()
        assert guard.context == AccessContext.BACKFILL


class TestContextManager:
    def teardown_method(self):
        if "DATA_ACCESS_CONTEXT" in os.environ:
            del os.environ["DATA_ACCESS_CONTEXT"]

    def test_context_manager_sets_and_restores(self):
        os.environ["DATA_ACCESS_CONTEXT"] = "exploration"

        with DataAccessGuard(AccessContext.ADMIN) as guard:
            assert os.environ["DATA_ACCESS_CONTEXT"] == "admin"
            assert guard.context == AccessContext.ADMIN

        assert os.environ["DATA_ACCESS_CONTEXT"] == "exploration"

    def test_context_manager_cleans_up_when_no_previous(self):
        with DataAccessGuard(AccessContext.TRAINING):
            assert os.environ["DATA_ACCESS_CONTEXT"] == "training"

        assert "DATA_ACCESS_CONTEXT" not in os.environ


class TestRequireContextDecorator:
    def teardown_method(self):
        if "DATA_ACCESS_CONTEXT" in os.environ:
            del os.environ["DATA_ACCESS_CONTEXT"]

    def test_require_context_allows_matching_context(self):
        @require_context(AccessContext.TRAINING, AccessContext.ADMIN)
        def training_function():
            return "success"

        os.environ["DATA_ACCESS_CONTEXT"] = "training"
        assert training_function() == "success"

    def test_require_context_blocks_wrong_context(self):
        @require_context(AccessContext.TRAINING)
        def training_only_function():
            return "success"

        os.environ["DATA_ACCESS_CONTEXT"] = "exploration"

        with pytest.raises(PermissionError, match="requires context"):
            training_only_function()


class TestGuardedRead:
    def teardown_method(self):
        if "DATA_ACCESS_CONTEXT" in os.environ:
            del os.environ["DATA_ACCESS_CONTEXT"]

    def test_guarded_read_with_valid_context(self):
        path = guarded_read("/data/snapshots/training_v1.parquet", AccessContext.EXPLORATION)
        assert path == Path("/data/snapshots/training_v1.parquet")

    def test_guarded_read_with_invalid_context_raises(self):
        with pytest.raises(PermissionError):
            guarded_read("/data/raw/customers.csv", AccessContext.EXPLORATION)

    def test_guarded_read_uses_environment(self):
        os.environ["DATA_ACCESS_CONTEXT"] = "admin"
        path = guarded_read("/data/raw/customers.csv")
        assert path == Path("/data/raw/customers.csv")
