"""Probe-session binding for the engagement_e4ad6e1b cycle suite.

Single source of truth for run/experiment/cluster config across every
probe notebook in `debug/engagement_e4ad6e1b/cycles/`. Operator binds
config once in `00_probe_config.ipynb`; downstream probes call
`bind_probe_session()` and receive a fully-resolved `ProbeSession`.

Resolution precedence for every config key:
    1. Databricks job task values (dbutils.jobs.taskValues.get)
    2. Databricks notebook widget (dbutils.widgets.get)
    3. Environment variable (CR_<KEY> or <KEY>)
    4. Hard-coded default

Output layout under `<namespace.session_dir>`:
    probe_runs/
        <UTC_TIMESTAMP>__<RUN_ID>/
            session_manifest.json
            cycle_017/result.json
            cycle_019/result.json
            ...
        <UTC_TIMESTAMP>__<RUN_ID>.zip   (written by bundle_results())

The UTC timestamp + run-id label means multiple iterations of the same
probe suite (before/after a parser fix) live in distinct folders; the
zip sibling is the operator's UI-downloadable artifact.
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_TASK_VALUES_KEY = "probe_config"


def _get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore
        return dbutils
    except Exception:
        try:
            import IPython  # type: ignore
            ip = IPython.get_ipython()
            if ip is not None and "dbutils" in ip.user_ns:
                return ip.user_ns["dbutils"]
        except Exception:
            pass
    return None


def _current_notebook_path() -> Optional[str]:
    """Workspace path of the currently-executing notebook (Databricks only)."""
    dbutils = _get_dbutils()
    if dbutils is None:
        return None
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        path = ctx.notebookPath().get()
        return str(path) if path else None
    except Exception:
        return None


def _autodetect_workspace_root() -> Optional[str]:
    """Derive the workspace folder containing ``exploration_notebooks/``.

    The probe notebook lives somewhere under the operator's working copy
    (typical layout: ``<workspace_root>/debug/<engagement>/cycles/<probe>``).
    The deployed framework notebooks live at
    ``<workspace_root>/exploration_notebooks/<nb>``. Walk up from the
    currently-executing probe path to find ``<workspace_root>``.

    Returns None when not on Databricks (no dbutils) or when neither
    ``debug/`` nor ``exploration_notebooks/`` appears in the path
    (operator is running from an unexpected location — fall through to
    explicit override).
    """
    nb_path = _current_notebook_path()
    if not nb_path:
        return None
    parts = nb_path.split("/")
    # Anchor 1: probe paths typically contain a ``debug/`` folder.
    if "debug" in parts:
        idx = parts.index("debug")
        return "/".join(parts[:idx])
    # Anchor 2: someone running from inside exploration_notebooks/ directly.
    if "exploration_notebooks" in parts:
        idx = parts.index("exploration_notebooks")
        return "/".join(parts[:idx])
    return None


def _resolve_executed_notebook(framework_root: str, rel_subpath: str) -> str:
    """Pick the deployed copy of a framework notebook over the master clone.

    The probe's `framework_root` typically resolves to
    ``/Workspace/Repos/<user>/customer_retention`` — the git-synced master
    clone, which is project-agnostic by design (no engagement-specific
    cells). The notebooks the operator actually *executes* live in the
    deployed user-workspace mirror at
    ``/Workspace/Users/<user>/customer_retention``, where the SPS paste
    cells (00010001, 00025001-4, 00026001, 00046001, 00115001, 06065001-2,
    etc.) physically exist.

    When ``framework_root`` matches the ``Repos/`` shape, derive the
    sibling ``Users/`` path and prefer it when the file actually exists.
    Falls back to the framework_root copy when no deployed mirror is
    present (fresh engagement, or local non-Databricks runs).

    The argument ``rel_subpath`` is the trailing path under the workspace
    root (e.g. ``"exploration_notebooks/00_start_here.ipynb"``).
    """
    fr = str(framework_root)
    candidates = []
    if "/Workspace/Repos/" in fr:
        users_root = fr.replace("/Workspace/Repos/", "/Workspace/Users/", 1)
        candidates.append(f"{users_root}/{rel_subpath}")
    candidates.append(f"{fr}/{rel_subpath}")
    for c in candidates:
        if Path(c).exists():
            return c
    # Nothing exists yet — return the deployed-user variant as the default
    # so error messages point operators at the path they're expected to
    # populate, rather than the master clone.
    return candidates[0]


def discover_pipeline_dirs(
    run_root: Optional[Path],
    framework_root: Optional[Path],
    *,
    verbose: bool = False,
) -> list[Path]:
    """Bounded BFS for every dir matching the per-target pipeline shape
    (``landing/`` + ``silver/`` siblings) under both ``<run_root>/generated_pipelines/``
    and ``<framework_root>/generated_pipelines/``. Also walks the sibling
    ``Workspace/Users/...`` tree when ``framework_root`` is the ``Repos/``
    clone, since NB10 commonly writes to the user-workspace mirror.

    De-duplicates by path. Sorted by mtime (newest first).
    """
    roots: list[Path] = []
    if run_root is not None:
        roots.append(Path(run_root) / "generated_pipelines")
    if framework_root is not None:
        roots.append(Path(framework_root) / "generated_pipelines")
        fr_str = str(framework_root)
        if "/Workspace/Repos/" in fr_str:
            roots.append(
                Path(fr_str.replace("/Workspace/Repos/", "/Workspace/Users/", 1))
                / "generated_pipelines"
            )
    if verbose:
        for r in roots:
            print(f"  [diag] discover_pipeline_dirs root: {r} "
                  f"{'EXISTS' if r.exists() else 'MISSING'}")
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        stack = [(root, 0)]
        while stack:
            cur, depth = stack.pop()
            if depth > 4:
                continue
            try:
                entries = [c for c in cur.iterdir() if c.is_dir()]
            except (OSError, PermissionError):
                continue
            if (cur / "landing").is_dir() and (cur / "silver").is_dir():
                out.append(cur)
                continue
            stack.extend((c, depth + 1) for c in entries)
    seen: set[str] = set()
    uniq: list[Path] = []
    for d in out:
        s = str(d)
        if s not in seen:
            seen.add(s)
            uniq.append(d)

    def _mtime(p: Path) -> float:
        manifest = p / "generation_manifest.json"
        try:
            return manifest.stat().st_mtime if manifest.exists() else p.stat().st_mtime
        except OSError:
            return 0.0

    uniq.sort(key=_mtime, reverse=True)
    return uniq


def active_pipeline_dirs(
    run_root: Optional[Path],
    framework_root: Optional[Path],
    *,
    pipeline_name: Optional[str] = None,
    verbose: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Return ``(active, stale)`` lists from the discovered pipeline dirs.

    "Active" = matches ``pipeline_name`` (when supplied) OR the freshest
    ``generation_manifest.json`` mtime when no name is given. Probes that
    iterate "active" emit FAILs for the latest NB10 codegen output;
    probes that surface the "stale" list emit INFO for residue from
    earlier framework versions or different pipeline configurations.

    Filters out directories that share a `generated_pipelines/` parent
    with the active dir but were not written by the latest NB10 run —
    typically a `customer_churn` left over after the operator switched
    to `PIPELINE_NAME = "sps_churn"`.

    Resolution priority for ``pipeline_name``:
        1. argument
        2. ``CR_PIPELINE_NAME`` env / widget / task value
        3. None → freshest-by-mtime fallback
    """
    if pipeline_name is None:
        pipeline_name = _resolve("pipeline_name") or None

    discovered = discover_pipeline_dirs(
        run_root, framework_root, verbose=verbose
    )
    if not discovered:
        return [], []

    active: list[Path] = []
    stale: list[Path] = []
    if pipeline_name:
        for d in discovered:
            (active if d.name == pipeline_name else stale).append(d)
        # Pipeline_name was set but no match — fall back to freshest, treat
        # all remaining as stale so probes still report something.
        if not active:
            active = [discovered[0]]
            stale = discovered[1:]
    else:
        # No pipeline_name binding: take the freshest (already sorted by
        # mtime in discover_pipeline_dirs) as active, the rest stale.
        active = [discovered[0]]
        stale = discovered[1:]
    return active, stale


def _autodetect_catalog_schema(experiments_root: str) -> tuple[Optional[str], Optional[str]]:
    """Parse ``(catalog, schema)`` from ``/Volumes/<catalog>/<schema>/<volume>/..``.

    Returns ``(None, None)`` for non-UC paths so the operator can still
    pass an explicit override via env vars / widgets / task values.
    """
    if not experiments_root:
        return None, None
    parts = str(experiments_root).split("/")
    if len(parts) >= 5 and parts[1] == "Volumes":
        return parts[2], parts[3]
    return None, None


def _resolve(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a config key by precedence: task values -> widget -> env -> default."""
    dbutils = _get_dbutils()
    if dbutils is not None:
        try:
            v = dbutils.jobs.taskValues.get(
                taskKey=_TASK_VALUES_KEY, key=name, debugValue=""
            )
            if v:
                return str(v)
        except Exception:
            pass
        try:
            v = dbutils.widgets.get(name)
            if v:
                return str(v)
        except Exception:
            pass
    return (
        os.environ.get(f"CR_{name.upper()}")
        or os.environ.get(name)
        or default
    )


def _set_task_value(name: str, value: str) -> bool:
    """Best-effort task-value publish from 00_probe_config. Returns True on success."""
    dbutils = _get_dbutils()
    if dbutils is None:
        return False
    try:
        dbutils.jobs.taskValues.set(key=name, value=value)
        return True
    except Exception:
        return False


@dataclass
class ProbeSession:
    """Resolved per-session config bound by ``bind_probe_session()``."""

    run_id: str
    experiments_root: Path
    framework_root: Path
    catalog: str
    schema: str
    nb10_export_path: Optional[Path]
    nb00_path: Path
    nb06_path: Path
    session_label: str
    namespace: Any
    bundle_root: Path

    @property
    def run_root(self) -> Path:
        """Per-run artifact root: ``<experiments_root>/runs/<run_id>``.

        Probe cells reference this when looking for generated pipeline output.
        NB10 actually writes to ``<framework_root>/generated_pipelines/<target>/<name>``,
        so probes that probe ``run_root / 'pipeline'`` etc. will gracefully
        degrade to INFO when no NB10 output exists at the run root.
        """
        return self.namespace.run_dir

    def cycle_dir(self, cycle_id: int) -> Path:
        d = self.bundle_root / f"cycle_{cycle_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_result(self, cycle_id: int, result: Dict[str, Any]) -> Path:
        result.setdefault("run_id", self.run_id)
        result.setdefault("session_label", self.session_label)
        p = self.cycle_dir(cycle_id) / "result.json"
        p.write_text(json.dumps(result, indent=2, default=str))
        return p

    def write_session_manifest(self) -> Path:
        manifest = {
            "run_id": self.run_id,
            "session_label": self.session_label,
            "experiments_root": str(self.experiments_root),
            "framework_root": str(self.framework_root),
            "catalog": self.catalog,
            "schema": self.schema,
            "nb10_export_path": (
                str(self.nb10_export_path) if self.nb10_export_path else None
            ),
            "nb00_path": str(self.nb00_path),
            "nb06_path": str(self.nb06_path),
            "cycles_present": sorted(
                p.name for p in self.bundle_root.glob("cycle_*") if p.is_dir()
            ),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        p = self.bundle_root / "session_manifest.json"
        p.write_text(json.dumps(manifest, indent=2))
        return p

    def collect_results(self) -> list:
        rows = []
        for p in sorted(self.bundle_root.glob("cycle_*/result.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except Exception as exc:
                rows.append({"path": str(p), "parse_error": repr(exc)})
        return rows

    def bundle_results(self) -> Path:
        """Zip session_manifest.json + every cycle_*/result.json into a single
        archive at <session_dir>/probe_runs/<session_label>.zip. Operator
        downloads via the Databricks file browser.

        Databricks Volume FUSE mounts reject zipfile's close-time seek/truncate
        with ``OSError: [Errno 95] Operation not supported``. Build the archive
        in memory and write the final bytes in a single call, which the FUSE
        mount accepts.
        """
        import io

        self.write_session_manifest()
        archive = self.bundle_root.parent / f"{self.session_label}.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(self.bundle_root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(self.bundle_root.parent))
        archive.write_bytes(buf.getvalue())
        return archive


def bind_probe_session(extra_defaults: Optional[Dict[str, str]] = None) -> ProbeSession:
    """Resolve config + return a ProbeSession.

    Auto-detection takes care of paths and UC namespace derivation; the
    operator only needs ``run_id`` + ``experiments_root`` (anything that
    pins the run identity). All other fields fall back through:

        1. ``extra_defaults`` (caller override)
        2. Databricks task values (set by ``publish_probe_config``)
        3. Databricks widgets (manual operator binding)
        4. Environment variables (``CR_<KEY>`` or ``<KEY>``)
        5. **Auto-detection**:
            - ``framework_root`` ← workspace folder hosting the probe
              (derived from ``dbutils.notebook.getContext().notebookPath()``)
            - ``nb00_path`` / ``nb06_path`` / ``nb10_export_path`` ←
              ``<framework_root>/exploration_notebooks/<nb>.ipynb``
            - ``catalog`` / ``schema`` ← parsed from ``experiments_root``
              when it is a ``/Volumes/<catalog>/<schema>/...`` path

    The previous hard-coded defaults (engagement-specific framework_root,
    catalog, schema) have been removed — those values were a usability
    cliff every new engagement bumped into.
    """

    extras = extra_defaults or {}

    # Auto-detect workspace root from the currently-executing probe path.
    autodetected_root = _autodetect_workspace_root()

    framework_root = _resolve(
        "framework_root", extras.get("framework_root", autodetected_root)
    )
    if not framework_root:
        raise RuntimeError(
            "probe_session: framework_root could not be auto-detected from the "
            "probe notebook path. Either run the probe inside a workspace folder "
            "containing a sibling `exploration_notebooks/` directory, or set "
            "framework_root explicitly via env / widget / task value."
        )

    src = f"{framework_root}/src"
    if src not in sys.path:
        sys.path.insert(0, src)

    from customer_retention.analysis.auto_explorer import RunNamespace

    run_id = _resolve("run_id") or os.environ.get("CR_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "probe_session: run_id is not bound. Run "
            "debug/engagement_e4ad6e1b/cycles/000_probe_config.ipynb first, "
            "or set the CR_RUN_ID environment variable."
        )

    experiments_root_raw = _resolve(
        "experiments_root", extras.get("experiments_root")
    )
    if not experiments_root_raw:
        raise RuntimeError(
            "probe_session: experiments_root is not bound. Run "
            "debug/engagement_e4ad6e1b/cycles/000_probe_config.ipynb first, "
            "or set CR_EXPERIMENTS_ROOT."
        )
    experiments_root = Path(experiments_root_raw)

    # Catalog/schema auto-derived from /Volumes/<catalog>/<schema>/... paths.
    auto_cat, auto_sch = _autodetect_catalog_schema(str(experiments_root))
    catalog = _resolve("catalog", extras.get("catalog", auto_cat))
    schema = _resolve("schema", extras.get("schema", auto_sch))

    # Notebook paths: prefer the executed copy (deployed user-workspace
    # mirror) over the framework_root master clone. When the operator runs
    # NB00/NB06 from /Workspace/Users/<user>/customer_retention/... and the
    # probes scan /Workspace/Repos/<user>/customer_retention/..., every
    # cell-presence check FAILs with false positives on the project-agnostic
    # master clone. `_resolve_executed_notebook` picks the deployed copy
    # whenever it exists; falls back to the master path for fresh runs.
    nb10_raw = _resolve(
        "nb10_export_path",
        extras.get(
            "nb10_export_path",
            _resolve_executed_notebook(
                framework_root, "exploration_notebooks/10_spec_generation.ipynb"
            ),
        ),
    )
    nb00 = Path(
        _resolve(
            "nb00_path",
            extras.get(
                "nb00_path",
                _resolve_executed_notebook(
                    framework_root, "exploration_notebooks/00_start_here.ipynb"
                ),
            ),
        )
    )
    nb06 = Path(
        _resolve(
            "nb06_path",
            extras.get(
                "nb06_path",
                _resolve_executed_notebook(
                    framework_root, "exploration_notebooks/06_feature_opportunities.ipynb"
                ),
            ),
        )
    )

    label = _resolve("session_label") or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"__{run_id}"
    )

    namespace = RunNamespace(root=experiments_root, run_id=run_id)
    bundle_root = Path(namespace.session_dir) / "probe_runs" / label
    bundle_root.mkdir(parents=True, exist_ok=True)

    return ProbeSession(
        run_id=run_id,
        experiments_root=experiments_root,
        framework_root=Path(framework_root),
        catalog=catalog,
        schema=schema,
        nb10_export_path=Path(nb10_raw) if nb10_raw else None,
        nb00_path=nb00,
        nb06_path=nb06,
        session_label=label,
        namespace=namespace,
        bundle_root=bundle_root,
    )


def publish_probe_config(
    *,
    run_id: str,
    experiments_root: str,
    catalog: str = "",
    schema: str = "",
    framework_root: str = "",
    nb10_export_path: str = "",
    nb00_path: str = "",
    nb06_path: str = "",
    session_label: str = "",
) -> Dict[str, str]:
    """Publish task values (when running inside a Databricks job) AND set env
    vars (so the same config survives interactive single-notebook execution).
    Called from ``000_probe_config``'s Cell 1.

    Required: ``run_id`` + ``experiments_root``.

    Everything else auto-detects:
    - ``framework_root`` from the currently-executing notebook path
    - ``catalog`` / ``schema`` from the ``/Volumes/<cat>/<sch>/...`` shape of
      ``experiments_root``
    - ``nb00_path`` / ``nb06_path`` / ``nb10_export_path`` from
      ``<framework_root>/exploration_notebooks/<nb>.ipynb``

    Pass any of those explicitly only when the auto-detection produces the
    wrong answer for your workspace layout.
    """

    if not framework_root:
        framework_root = _autodetect_workspace_root() or ""
    if not (catalog and schema):
        auto_cat, auto_sch = _autodetect_catalog_schema(experiments_root)
        catalog = catalog or (auto_cat or "")
        schema = schema or (auto_sch or "")

    payload: Dict[str, str] = {
        "run_id": run_id,
        "experiments_root": experiments_root,
    }
    if framework_root:
        payload["framework_root"] = framework_root
    if catalog:
        payload["catalog"] = catalog
    if schema:
        payload["schema"] = schema
    if nb10_export_path:
        payload["nb10_export_path"] = nb10_export_path
    if nb00_path:
        payload["nb00_path"] = nb00_path
    if nb06_path:
        payload["nb06_path"] = nb06_path
    if not session_label:
        session_label = (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"__{run_id}"
        )
    payload["session_label"] = session_label

    published_via_task_values = False
    for k, v in payload.items():
        if _set_task_value(k, v):
            published_via_task_values = True
        os.environ[f"CR_{k.upper()}"] = v

    payload["_publish_mode"] = (
        "task_values+env" if published_via_task_values else "env_only"
    )
    return payload
