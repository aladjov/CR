"""Data access control based on execution context.

This module provides path-based access control to prevent accidental
data leakage by restricting which data paths are accessible in different
execution contexts (exploration, training, inference, etc.).

Key concepts:
    - AccessContext: The current execution mode
    - DataAccessGuard: Validates path access against context rules
    - require_context: Decorator to enforce context requirements

Example:
    >>> from customer_retention.stages.temporal import AccessContext, DataAccessGuard
    >>> # Set context for the session
    >>> with DataAccessGuard(AccessContext.TRAINING):
    ...     # Can access snapshots/ and gold/
    ...     df = pd.read_parquet("output/snapshots/training_v1.parquet")
    ...     # This would raise PermissionError:
    ...     # df = pd.read_parquet("output/raw/customers.csv")
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional


class AccessContext(Enum):
    """Execution context for data access control.

    Attributes:
        EXPLORATION: Interactive data exploration (can access snapshots)
        TRAINING: Model training (can access snapshots and gold)
        INFERENCE: Production inference (can access gold and feature_store)
        BACKFILL: Historical data processing (can access raw through gold)
        ADMIN: Administrative access (unrestricted)
    """
    EXPLORATION = "exploration"
    TRAINING = "training"
    INFERENCE = "inference"
    BACKFILL = "backfill"
    ADMIN = "admin"


class DataAccessGuard:
    """Guards data access based on the current execution context.

    The DataAccessGuard prevents accidental data leakage by restricting
    which paths can be accessed based on the execution context. For example,
    during training, raw data paths are blocked to ensure only properly
    prepared snapshots are used.

    Can be used as a context manager to temporarily set the access context:

        >>> with DataAccessGuard(AccessContext.TRAINING):
        ...     # Only training-appropriate paths accessible here
        ...     pass

    Or used directly for path validation:

        >>> guard = DataAccessGuard(AccessContext.EXPLORATION)
        >>> guard.validate_access("output/snapshots/v1.parquet")  # OK
        >>> guard.validate_access("output/raw/data.csv")  # Raises PermissionError
    """

    ALLOWED_PATHS = {
        AccessContext.EXPLORATION: ["snapshots/"],
        AccessContext.TRAINING: ["snapshots/", "gold/"],
        AccessContext.INFERENCE: ["gold/", "feature_store/"],
        AccessContext.BACKFILL: ["raw/", "bronze/", "silver/", "gold/"],
        AccessContext.ADMIN: ["*"],
    }

    BLOCKED_PATHS = {
        AccessContext.EXPLORATION: ["raw/", "bronze/", "silver/"],
        AccessContext.TRAINING: ["raw/", "bronze/"],
        AccessContext.INFERENCE: ["snapshots/", "raw/", "bronze/", "silver/"],
        AccessContext.BACKFILL: ["snapshots/"],
        AccessContext.ADMIN: [],
    }

    def __init__(self, context: AccessContext):
        self.context = context

    def validate_access(self, path: str) -> bool:
        path_str = str(path)
        for blocked in self.BLOCKED_PATHS[self.context]:
            if blocked in path_str:
                raise PermissionError(
                    f"Access to '{path}' blocked in {self.context.value} context. "
                    f"Blocked patterns: {self.BLOCKED_PATHS[self.context]}"
                )
        return True

    def is_allowed(self, path: str) -> bool:
        if "*" in self.ALLOWED_PATHS[self.context]:
            return True
        path_str = str(path)
        return any(allowed in path_str for allowed in self.ALLOWED_PATHS[self.context])

    def guard_read(self, path: str) -> Path:
        self.validate_access(path)
        return Path(path)

    @staticmethod
    def set_context(context: AccessContext) -> None:
        os.environ["DATA_ACCESS_CONTEXT"] = context.value

    @staticmethod
    def get_current_context() -> AccessContext:
        ctx = os.environ.get("DATA_ACCESS_CONTEXT", "exploration")
        return AccessContext(ctx)

    @classmethod
    def from_environment(cls) -> "DataAccessGuard":
        return cls(cls.get_current_context())

    def __enter__(self) -> "DataAccessGuard":
        self._previous_context = os.environ.get("DATA_ACCESS_CONTEXT")
        os.environ["DATA_ACCESS_CONTEXT"] = self.context.value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._previous_context:
            os.environ["DATA_ACCESS_CONTEXT"] = self._previous_context
        elif "DATA_ACCESS_CONTEXT" in os.environ:
            del os.environ["DATA_ACCESS_CONTEXT"]


def require_context(*allowed_contexts: AccessContext):
    """Decorator to enforce execution context requirements on functions.

    Use this decorator to restrict a function to specific execution contexts.
    If called from a disallowed context, raises PermissionError.

    Args:
        *allowed_contexts: One or more AccessContext values that are permitted

    Example:
        >>> @require_context(AccessContext.TRAINING, AccessContext.INFERENCE)
        ... def predict(features):
        ...     return model.predict(features)
        >>>
        >>> # Only works in TRAINING or INFERENCE context
        >>> DataAccessGuard.set_context(AccessContext.TRAINING)
        >>> predict(X)  # OK
        >>> DataAccessGuard.set_context(AccessContext.EXPLORATION)
        >>> predict(X)  # Raises PermissionError
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current = DataAccessGuard.get_current_context()
            if current not in allowed_contexts:
                raise PermissionError(
                    f"Function requires context {[c.value for c in allowed_contexts]}, "
                    f"but current context is {current.value}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def guarded_read(path: str, context: Optional[AccessContext] = None) -> Path:
    """Validate path access and return a Path object.

    Convenience function that validates a path against access rules
    and returns a Path object if access is allowed.

    Args:
        path: Path to validate
        context: Optional context override (uses environment if None)

    Returns:
        Path object for the validated path

    Raises:
        PermissionError: If access is not allowed in the current context
    """
    guard = DataAccessGuard(context) if context else DataAccessGuard.from_environment()
    return guard.guard_read(path)
