"""Validation helpers for SCD history augmentation."""
from __future__ import annotations

from typing import Any, Dict

from .config import SCDHistoryReconstructionConfig


def validate_scd_sources(
    scd_sources: Dict[str, str],
    datasets: Dict[str, Any],
    configs: Dict[str, SCDHistoryReconstructionConfig],
) -> None:
    """Fail-fast if SCD sources reference unknown datasets or missing configs.

    Raises :class:`KeyError` with a diagnostic message on the first mismatch.
    """
    for parent_name in scd_sources:
        if parent_name not in datasets:
            raise KeyError(
                f"namespace.scd_history_sources references {parent_name!r} "
                f"which is not in `datasets`. Add it to the dataset "
                f"registration first."
            )
        if parent_name not in configs:
            raise KeyError(
                f"namespace.scd_history_sources has an entry for "
                f"{parent_name!r} but SCD_RECONSTRUCTION_CONFIGS does not — "
                f"add a SCDHistoryReconstructionConfig above."
            )
