"""In-process registry of @cr.register-decorated functions.

Decoration appends to the singleton `registry`; harvesters read from it
at codegen time. Re-executing the same cell updates its record in place
— the dedup key is `(name, notebook_path, cell_id)` per plan § 3.1.5.

FW-15b — disk-backed persistence: `register()` writes to a JSON file
under the active `RunNamespace` so a fresh kernel (e.g. NB10 in a
multi-task Databricks job) can rehydrate the registry. Decoration is
the trigger; reads happen via `Registry.load_from_disk()` invoked from
`Harvester.harvest()`.

FW-15a — Lane-2 execution tracker: every imperative `@cr.register` cell
must call `registry.mark_lane2_executed(name)` after running its
in-cell action so subsequent cells (and the FW-15a guard) know the
exploration state already reflects the registered behavior.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional

Scope = Literal["dataset", "datasets", "wildcard"]


@dataclass
class RegisteredFunction:
    name: str
    source: str
    scope: Scope
    dataset: Optional[str] = None
    datasets: Optional[List[str]] = None
    primary: Optional[str] = None
    replay_at_scoring: bool = False
    expected_stage: Optional[str] = None
    notebook_path: Optional[Path] = None
    cell_id: Optional[str] = None
    inferred_stage: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Path is not JSON-serialisable.
        if d["notebook_path"] is not None:
            d["notebook_path"] = str(d["notebook_path"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RegisteredFunction":
        nb = d.get("notebook_path")
        return cls(
            name=d["name"],
            source=d["source"],
            scope=d["scope"],
            dataset=d.get("dataset"),
            datasets=d.get("datasets"),
            primary=d.get("primary"),
            replay_at_scoring=d.get("replay_at_scoring", False),
            expected_stage=d.get("expected_stage"),
            notebook_path=Path(nb) if nb else None,
            cell_id=d.get("cell_id"),
            inferred_stage=d.get("inferred_stage"),
        )


# Environment override for the persistence path. Mainly used in tests and
# multi-task Databricks jobs where `RunNamespace.from_env()` resolves to a
# stable per-run location.
_PERSISTENCE_ENV = "CR_RUNTIME_REGISTRY_PATH"


def _resolve_persistence_path() -> Optional[Path]:
    explicit = os.environ.get(_PERSISTENCE_ENV)
    if explicit:
        return Path(explicit)
    # Lazy import to avoid circulars at module load. ImportError is the only
    # legitimate fallback reason — any other failure should surface.
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import (  # noqa: PLC0415
            RunNamespace,
        )
    except ImportError:
        return None
    # `RunNamespace.from_env()` returns None when no run is bound. Other
    # failures (e.g. malformed env vars) are real bugs and must propagate.
    ns = RunNamespace.from_env()
    if ns is None:
        return None
    return ns.registered_functions_path


def _resolve_persistence_path_for_load() -> Optional[Path]:
    """Resolve the on-disk registry path for an EXPLICIT load attempt.

    Used by `load_from_disk()` when no explicit path is passed — i.e. by
    NB10 codegen's `_auto_harvest()` in a fresh kernel. Falls back to the
    file-tracked discovery chain (sentinel → project pointer → latest)
    that survives Databricks task boundaries: NB00 sets `CR_RUN_ID` in its
    own kernel and writes the registry; NB10 runs in a separate task with
    no env var, so the env-only resolver would return None and silently
    produce a pipeline without `target_derive/run_target_derive.py`
    (engagement spschurn-e34b8ec5 failure mode).

    Kept distinct from `_resolve_persistence_path()` (used by writes and
    auto-hydration) because the latter intentionally returns None when
    no run is explicitly bound — auto-hydrating from a stray local
    sentinel/pointer would break tests and surprise callers in
    development environments where multiple unrelated runs may exist.
    """
    explicit = os.environ.get(_PERSISTENCE_ENV)
    if explicit:
        return Path(explicit)
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import (  # noqa: PLC0415
            RunNamespace,
        )
    except ImportError:
        return None
    ns = RunNamespace.from_env_or_latest()
    if ns is None:
        return None
    return ns.registered_functions_path


class Registry:
    def __init__(self) -> None:
        self._records: List[RegisteredFunction] = []
        self._lane2_executed: Dict[str, bool] = {}
        self._lock = Lock()
        self._auto_hydrated = False

    def register(self, rf: RegisteredFunction) -> None:
        with self._lock:
            key = (rf.name, rf.notebook_path, rf.cell_id)
            for i, existing in enumerate(self._records):
                if (existing.name, existing.notebook_path, existing.cell_id) == key:
                    self._records[i] = rf
                    self._persist_unlocked()
                    return
            self._records.append(rf)
            self._persist_unlocked()

    def get_registered(self) -> List[RegisteredFunction]:
        self._maybe_auto_hydrate()
        return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._lane2_executed.clear()
            self._auto_hydrated = False

    def _maybe_auto_hydrate(self) -> None:
        """Hydrate from the on-disk sidecar on first read in a fresh kernel.

        Exploration notebooks 01-09 start in their own kernels (Databricks
        multi-task jobs, separate `papermill` invocations, manual reruns).
        Without auto-hydration, `get_registered()` returns [] in those
        kernels even when NB00 successfully wrote the JSON sidecar — and
        every diagnostic that uses ``rf.expected_stage`` /
        ``is_lane2_executed`` becomes structurally wrong. We auto-load
        once on the first read and only when the in-memory registry is
        empty (so explicit ``register()`` calls in tests are not shadowed).
        """
        if self._auto_hydrated or self._records:
            return
        path = _resolve_persistence_path()
        self._auto_hydrated = True
        if path is None or not path.exists():
            return
        self.load_from_disk(path)

    def validate(self):
        from .validation import validate_registered_function
        errors = []
        for rf in self._records:
            err = validate_registered_function(rf)
            if err is not None:
                errors.append(err)
        return errors

    # ---- FW-15a: Lane-2 tracker -------------------------------------------

    def mark_lane2_executed(self, name: str) -> None:
        """Record that the imperative Lane-2 path for `name` has run.

        Called from inside a `@cr.register`-decorated cell after the cell
        mutates the shared exploration state (e.g. rebinds `datasets[...]`,
        applies a filter, swaps a view). The FW-15a guard reads this flag
        when verifying that registered functions actually executed in
        exploration; codegen-track replay is verified separately via
        `user_extensions.py` emission.

        Persists the updated flag map to the on-disk sidecar so a fresh
        NB01 kernel can hydrate Lane-2 state via auto-hydrate. Without
        this call, the ``lane2_executed`` flag in the FW-14 diagnostic
        always reads False after the kernel boundary.
        """
        with self._lock:
            self._lane2_executed[name] = True
            self._persist_unlocked()

    def is_lane2_executed(self, name: str) -> bool:
        return self._lane2_executed.get(name, False)

    def lane2_status(self) -> Dict[str, bool]:
        return dict(self._lane2_executed)

    # ---- FW-15b: disk persistence -----------------------------------------

    def _persist_unlocked(self) -> None:
        """Write the registry snapshot to disk. I/O errors propagate so a
        failed persist is visible at the @cr.register callsite — silent
        failure would let a fresh NB10 kernel rehydrate stale data and
        break exploration↔production parity (engagement spschurn-01bb634c
        failure mode).

        Uses only the Path methods that `RemotePath` (Databricks Connect /
        shared cluster Volumes paths) also exposes: `mkdir`, `write_text`.
        Atomic-rename via `Path.replace()` is intentionally avoided — it
        is not part of the `RemotePath` surface, and the file is small
        enough that a partial write would be caught by the load-time JSON
        parse, which raises with an actionable message rather than
        silently zero-rehydrating.

        ``lane2_executed`` is persisted alongside ``records`` so the FW-14
        diagnostic in a fresh NB01 kernel can tell whether NB00 actually
        ran the in-cell Lane-2 path. Without persistence the flag resets
        on every kernel boundary and the diagnostic always reports
        ``lane2_executed=False`` even on a fully-correct run.
        """
        path = _resolve_persistence_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "records": [rf.to_dict() for rf in self._records],
            "lane2_executed": dict(self._lane2_executed),
        }
        path.write_text(json.dumps(payload, indent=2))

    def load_from_disk(self, path: Optional[Path] = None) -> int:
        """Hydrate the in-memory registry from `path` (defaults to the
        env / RunNamespace-resolved location). Returns the number of
        records loaded. Callers in fresh kernels (NB10 codegen, harvester)
        invoke this before reading. Existing in-memory records are kept
        and merged using the normal `register()` dedup key.

        Corrupt JSON or malformed records raise rather than silently
        zero-rehydrating — a parse failure here would otherwise produce a
        codegen run with no user_extensions.py and silently break parity.
        """
        target = path or _resolve_persistence_path_for_load()
        if target is None or not target.exists():
            return 0
        try:
            payload = json.loads(target.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"runtime.registry: cannot parse {target} ({exc}). "
                f"Delete the file to start fresh, or fix the corruption — "
                f"silently treating it as empty would let codegen produce "
                f"a pipeline with no user_extensions.py."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"runtime.registry: {target} is not a JSON object "
                f"(got {type(payload).__name__})."
            )
        records = payload.get("records")
        if not isinstance(records, list):
            raise RuntimeError(
                f"runtime.registry: {target} is missing the 'records' list "
                f"(got {type(records).__name__})."
            )
        # Hydrate Lane-2 flags BEFORE register() calls trigger
        # _persist_unlocked. Each register() call writes the in-memory
        # state to disk; if we registered first, the first persist would
        # carry an empty `_lane2_executed` dict and clobber the markers
        # the caller is trying to load. Order matters here.
        lane2 = payload.get("lane2_executed")
        if isinstance(lane2, dict):
            with self._lock:
                for name, flag in lane2.items():
                    if flag:
                        self._lane2_executed[str(name)] = True
        loaded = 0
        for d in records:
            rf = RegisteredFunction.from_dict(d)
            self.register(rf)
            loaded += 1
        return loaded


registry = Registry()
