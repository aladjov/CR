"""@cr.register decorator — decoration-time metadata capture.

Per plan § 3.1: the decorator is pure metadata capture. It does NOT wrap
the returned function (wrapping confuses inspect.getsource on
re-decoration). Source is captured verbatim including the decorator
line, because that line is the harvest-time contract.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, List, Optional, TypeVar, Union

from .registry import RegisteredFunction, Scope, registry

F = TypeVar("F", bound=Callable)


_REGISTERED_MARKER = "__cr_registered__"


def register(
    *,
    dataset: Optional[str] = None,
    datasets: Union[List[str], str, None] = None,
    primary: Optional[str] = None,
    replay_at_scoring: bool = False,
    name: Optional[str] = None,
    expected_stage: Optional[str] = None,
) -> Callable[[F], F]:
    def _decorate(func: F) -> F:
        if getattr(func, _REGISTERED_MARKER, False):
            raise TypeError(
                f"@cr.register on '{func.__qualname__}': function already "
                f"decorated with @cr.register; stacking is not permitted"
            )
        fn_name = name if name is not None else func.__name__
        _validate_params(
            func_qualname=func.__qualname__,
            dataset=dataset,
            datasets=datasets,
            primary=primary,
            name=name,
        )
        scope, ds_single, ds_list = _resolve_scope(dataset, datasets)

        source = _capture_source(func)
        notebook_path, cell_id = _capture_caller_location()

        rf = RegisteredFunction(
            name=fn_name,
            source=source,
            scope=scope,
            dataset=ds_single,
            datasets=ds_list,
            primary=primary,
            replay_at_scoring=replay_at_scoring,
            expected_stage=expected_stage,
            notebook_path=notebook_path,
            cell_id=cell_id,
        )
        registry.register(rf)
        try:
            setattr(func, _REGISTERED_MARKER, True)
        except (AttributeError, TypeError):
            pass
        return func
    return _decorate


def _validate_params(
    *,
    func_qualname: str,
    dataset: Optional[str],
    datasets: Union[List[str], str, None],
    primary: Optional[str],
    name: Optional[str],
) -> None:
    if (dataset is None) == (datasets is None):
        raise TypeError(
            f"@cr.register on '{func_qualname}': exactly one of dataset= or "
            f"datasets= is required (got dataset={dataset!r}, "
            f"datasets={datasets!r})"
        )
    if datasets is not None and not isinstance(datasets, (list, str)):
        raise TypeError(
            f"@cr.register on '{func_qualname}': datasets= must be a list of "
            f"strings or the literal '*' (got {type(datasets).__name__})"
        )
    if isinstance(datasets, str) and datasets != "*":
        raise TypeError(
            f"@cr.register on '{func_qualname}': datasets= as string must be "
            f"exactly '*' for wildcard scope (got {datasets!r})"
        )
    if dataset is not None and primary is not None:
        raise TypeError(
            f"@cr.register on '{func_qualname}': primary= is forbidden when "
            f"dataset= is used (primary only applies to multi-dataset cells)"
        )
    if datasets == "*" and primary is not None:
        raise TypeError(
            f"@cr.register on '{func_qualname}': primary= is forbidden when "
            f"datasets='*' (wildcard scope has no primary dataset)"
        )
    if isinstance(datasets, list):
        if len(datasets) == 0:
            raise TypeError(
                f"@cr.register on '{func_qualname}': datasets= list must not "
                f"be empty"
            )
        if len(datasets) > 1 and primary is None:
            raise TypeError(
                f"@cr.register on '{func_qualname}': primary= is required when "
                f"datasets= has more than one entry (got {datasets!r})"
            )
        if primary is not None and primary not in datasets:
            raise TypeError(
                f"@cr.register on '{func_qualname}': primary={primary!r} must "
                f"appear in datasets={datasets!r}"
            )
    if name is not None and not name.isidentifier():
        raise TypeError(
            f"@cr.register on '{func_qualname}': name={name!r} is not a valid "
            f"Python identifier"
        )


def _resolve_scope(
    dataset: Optional[str],
    datasets: Union[List[str], str, None],
) -> tuple[Scope, Optional[str], Optional[List[str]]]:
    if dataset is not None:
        return "dataset", dataset, None
    if datasets == "*":
        return "wildcard", None, None
    assert isinstance(datasets, list)
    return "datasets", None, list(datasets)


def _capture_source(func: Callable) -> str:
    try:
        return inspect.getsource(func)
    except OSError:
        source = getattr(func, "__source__", None)
        if source:
            return source
        raise RuntimeError(
            f"cannot capture source for '{func.__qualname__}' via "
            f"inspect.getsource; add expected_stage= and paste the function "
            f"into a tracked user_extensions_manual.py file"
        )


def _capture_caller_location() -> tuple[Optional[Path], Optional[str]]:
    """Best-effort notebook path + cell id via stack walk.

    Returns `(path, cell_id)` where either may be None. An IPython-input
    filename like `<ipython-input-3-abcd>` is NOT a stable cell id — it
    changes every kernel restart — so we decline to persist it. Dedup in
    `Registry.register` relies on cell_id stability to identify
    re-executions of the same cell across the same kernel; falling back to
    None here forces the caller to supply `expected_stage=` in that
    environment and avoids the false-match trap.
    """
    try:
        frames = inspect.stack()
    except (RuntimeError, OSError):
        return None, None
    for frame_info in frames:
        fname = frame_info.filename or ""
        if fname.endswith(".ipynb"):
            return Path(fname), None
        if fname.startswith("<ipython-input-") or "ipykernel" in fname:
            return None, None
    return None, None
