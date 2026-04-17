"""Emit `user_extensions.py` next to generated pipeline scripts.

Per plan § 6.1: when a non-empty `HarvestResult` is threaded into a
generator, concatenate every `RegisteredFunction.source` into a single
`user_extensions.py` file with a banner + `__all__`. When the harvest is
empty (the no-user-extensions default), no file is written — that's the
byte-parity guarantee for projects that don't adopt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from customer_retention.runtime.harvest import HarvestResult
from customer_retention.runtime.registry import RegisteredFunction

USER_EXTENSIONS_FILENAME = "user_extensions.py"

_STAGE_RANK = {
    "landing_post": 10,
    "bronze_merge": 20,
    "bronze_post": 30,
    "silver_post": 40,
    "target_derive": 50,
    "training": 60,
}


def write_user_extensions(
    output_dir: Path,
    harvest_result: Optional[HarvestResult],
    *,
    now: Optional[datetime] = None,
) -> Optional[Path]:
    if harvest_result is None or harvest_result.is_empty():
        return None
    functions = _ordered_functions(harvest_result)
    if not functions:
        return None
    path = output_dir / USER_EXTENSIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(functions, now or datetime.now(timezone.utc)))
    return path


def _ordered_functions(harvest_result: HarvestResult) -> List[RegisteredFunction]:
    enumerated: List[Tuple[int, int, str, int, RegisteredFunction]] = []
    dataset_index: dict = {}
    for i, rf in enumerate(harvest_result.all_functions()):
        stage = rf.inferred_stage or ""
        stage_rank = _STAGE_RANK.get(stage, 999)
        dataset = rf.dataset or rf.primary or ""
        dataset_order = dataset_index.setdefault(dataset, len(dataset_index))
        enumerated.append((stage_rank, dataset_order, dataset, i, rf))
    enumerated.sort(key=lambda t: (t[0], t[1], t[3]))
    return [t[-1] for t in enumerated]


def _render(functions: List[RegisteredFunction], now: datetime) -> str:
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: List[str] = []
    lines.append("# === generated — do not edit ===")
    lines.append(f"# Generated: {ts}")
    lines.append("# Sources:")
    for rf in functions:
        nb = rf.notebook_path.as_posix() if rf.notebook_path else "<unknown notebook>"
        cell = rf.cell_id or "<unknown cell>"
        lines.append(f"#   - {rf.name}  ({nb}:{cell})")
    lines.append("")
    for rf in functions:
        lines.append(rf.source.rstrip())
        lines.append("")
    lines.append("__all__ = [")
    for rf in functions:
        lines.append(f"    {rf.name!r},")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)
