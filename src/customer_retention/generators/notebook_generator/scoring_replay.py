"""Scoring-pipeline replay helper — Phase 7 splice contract.

Per plan § 0a.5 Phase 7 bailout: all current SPS cells have
`replay_at_scoring=False`, so no call-site gets emitted today. The
helper here captures the shape of the import + call block that will go
into the scoring notebook (s10_batch_inference) when the first real
replay function lands. Tests exercise the helper directly so the
contract is frozen even before a caller plugs it in.
"""
from __future__ import annotations

from typing import Iterable, List

from customer_retention.runtime.registry import RegisteredFunction


def filter_replay_functions(
    functions: Iterable[RegisteredFunction],
) -> List[RegisteredFunction]:
    return [rf for rf in functions if rf.replay_at_scoring]


def render_scoring_replay_block(
    functions: Iterable[RegisteredFunction],
) -> str:
    """Return the text of the scoring-replay cell, or '' when no
    function opts in. Empty-set byte parity: callers that always call
    this helper append an empty string to their cell list when nothing
    replays."""
    replay = filter_replay_functions(functions)
    if not replay:
        return ""
    names = [rf.name for rf in replay]
    import_block = _import_block(names)
    call_block = _call_block(replay)
    return f"{import_block}\n\n{call_block}"


def _import_block(names: List[str]) -> str:
    if len(names) == 1:
        return f"from user_extensions import {names[0]}"
    joined = ",\n    ".join(names)
    return f"from user_extensions import (\n    {joined},\n)"


def _call_block(functions: List[RegisteredFunction]) -> str:
    lines: List[str] = []
    for rf in functions:
        stage = rf.inferred_stage or rf.expected_stage or "<unknown stage>"
        lines.append(
            f"# TODO wire replay call-site for {rf.name} "
            f"(scope={rf.scope}, stage={stage})"
        )
    return "\n".join(lines)
