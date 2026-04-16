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
                f"SCD_HISTORY_SOURCES references {parent_name!r} which is not "
                f"in `datasets`. Add it to the dataset registration first."
            )
        if parent_name not in configs:
            raise KeyError(
                f"SCD_HISTORY_SOURCES has an entry for {parent_name!r} but "
                f"SCD_RECONSTRUCTION_CONFIGS does not — add a "
                f"SCDHistoryReconstructionConfig above."
            )
