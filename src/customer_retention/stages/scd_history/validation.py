"""Validation helpers for SCD history augmentation."""
from __future__ import annotations

from typing import Any, Dict

from .config import SCDHistoryReconstructionConfig


def validate_scd_sources(
    scd_sources: Dict[str, str],
    datasets: Dict[str, Any],
    configs: Dict[str, SCDHistoryReconstructionConfig],
) -> None:
    """Fail-fast when SCD source and config dicts disagree.

    Two NB00-local dicts drive the §0.11.5 augmentation loop and must stay
    in lock-step:

    * ``namespace.scd_history_sources`` — parent-name → history-table path.
    * ``SCD_RECONSTRUCTION_CONFIGS``    — parent-name → reconstruction config.

    The loop iterates ``scd_history_sources.items()``. If an operator
    comments out a source entry but leaves its config live (or vice versa),
    the loop silently skips the parent and the landing Delta is never
    augmented — the exact failure shape reproduced on run
    ``spschurn-e4ad6e1b`` (no tracked SCD fields, no ``as_of_date`` on
    ``landing/case``).

    Raises :class:`KeyError` on the first mismatch, in either direction:

    * A source whose parent is not registered in ``datasets``.
    * A source without a matching reconstruction config.
    * A config without a matching source (silent-no-op regression).

    Both dicts empty is a legitimate state (the pipeline does not use SCD
    reconstruction); both dicts with identical key sets is the other
    legitimate state.
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
    for parent_name in configs:
        if parent_name not in scd_sources:
            raise KeyError(
                f"SCD_RECONSTRUCTION_CONFIGS has an entry for "
                f"{parent_name!r} but namespace.scd_history_sources does "
                f"not — add the history-table path, or remove the config "
                f"entry. Half-commented state silently skips augmentation."
            )
