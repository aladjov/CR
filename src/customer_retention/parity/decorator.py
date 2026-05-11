"""`@apply_op` decorator, `APPLY_REGISTRY`, and `apply_context` manager.

Decoration is a one-time framework act: pair every framework function whose
primary effect is producing a transformed DataFrame with `@apply_op(kind=…)`.

Rules of thumb (codified in `docs/wiki/Adding-a-New-Apply-Op.md`):

1. Decorate **orchestrators**, never internal helpers. `enrich_lifecycle_dataset`
   gets the decorator; `_enrich_pandas` / `_enrich_distributed` do not.
2. Decorate **leaf transforms**, not dispatchers. `apply_log_transform` gets
   it; `TransformExecutor.apply_all` does not.
3. Functions returning `dict`/`tuple` of metrics or saving to disk are NOT
   apply primitives — they are sinks or analysers.

Side effects of the decorator:

- Registers `(qualified_name, ApplyOpDescriptor)` in `APPLY_REGISTRY` so the
  AST walker can resolve call sites by qualname.
- Wraps the function so that, when `CR_PARITY_TRACE=1`, each invocation
  appends to a thread-local trace buffer (Phase 6 — see `trace.py`).

The trace path is opt-in. With the env var unset the wrapper is a single
`if trace_active()` branch returning the original call.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Set, TypeVar

from .kinds import ApplyOpKind

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ApplyOpDescriptor:
    kind: ApplyOpKind
    qualified_name: str
    dataset_kwarg: Optional[str]
    capture_kwargs: Optional[Set[str]]
    gate: Optional[str]


APPLY_REGISTRY: dict[str, ApplyOpDescriptor] = {}


_DATASET_HINT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_cr_apply_dataset_hint", default=None
)


def apply_op(
    kind: ApplyOpKind,
    *,
    dataset_kwarg: Optional[str] = None,
    capture_kwargs: Optional[Set[str]] = None,
    gate: Optional[str] = None,
) -> Callable[[F], F]:
    def _decorate(func: F) -> F:
        qualified_name = f"{func.__module__}.{func.__qualname__}"
        if qualified_name in APPLY_REGISTRY:
            raise TypeError(
                f"@apply_op: {qualified_name} is already registered; "
                "decorating twice is not supported"
            )
        descriptor = ApplyOpDescriptor(
            kind=kind,
            qualified_name=qualified_name,
            dataset_kwarg=dataset_kwarg,
            capture_kwargs=capture_kwargs,
            gate=gate,
        )
        APPLY_REGISTRY[qualified_name] = descriptor

        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if not trace_active():
                return func(*args, **kwargs)
            from . import trace as _trace
            return _trace.record_call(descriptor, func, args, kwargs)

        _wrapped.__cr_apply_descriptor__ = descriptor  # type: ignore[attr-defined]
        return _wrapped  # type: ignore[return-value]

    return _decorate


@contextlib.contextmanager
def apply_context(*, dataset: str) -> Iterator[None]:
    token = _DATASET_HINT.set(dataset)
    try:
        yield
    finally:
        _DATASET_HINT.reset(token)


def active_dataset_hint() -> Optional[str]:
    return _DATASET_HINT.get()


def trace_active() -> bool:
    return os.environ.get("CR_PARITY_TRACE") == "1"
