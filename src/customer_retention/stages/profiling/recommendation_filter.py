from __future__ import annotations

from typing import Dict, Iterable, List, Set, TypeVar

from customer_retention.generators.pipeline_generator.findings_parser import _matches_any_prefix

_T = TypeVar("_T")


def collect_opt_in_prefixes(opt_in: Dict[str, List[str]]) -> List[str]:
    if opt_in is None:
        raise ValueError("collect_opt_in_prefixes: opt_in config cannot be None")
    prefixes: Set[str] = set()
    for cols in opt_in.values():
        for col in cols:
            prefixes.add(col)
            prefixes.add(f"{col}_")
    return sorted(prefixes)


def filter_recommendations_by_opt_in(
    recs: Iterable[_T],
    opt_in: Dict[str, List[str]],
) -> List[_T]:
    if opt_in is None:
        raise ValueError("filter_recommendations_by_opt_in: opt_in config cannot be None")
    prefixes = collect_opt_in_prefixes(opt_in)
    return [
        rec for rec in recs
        if getattr(rec, "action", None) != "zero_inflation_handling"
        or _matches_any_prefix(getattr(rec, "target_column", ""), prefixes)
    ]
