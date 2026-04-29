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

_DEFAULT_FRAMEWORK_ROOT = (
    "/Workspace/Repos/haladjov@spscommerce.com/customer_retention"
)
_DEFAULT_EXPERIMENTS_ROOT = (
    "/Volumes/prod_tradingpartnermap_internal_use1/customer_churn_model/"
    "experiments_secondary"
)
_DEFAULT_CATALOG = "prod_tradingpartnermap_internal_use1"
_DEFAULT_SCHEMA = "customer_churn_model"
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
        downloads via the Databricks file browser."""
        self.write_session_manifest()
        archive = self.bundle_root.parent / f"{self.session_label}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(self.bundle_root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(self.bundle_root.parent))
        return archive


def bind_probe_session(extra_defaults: Optional[Dict[str, str]] = None) -> ProbeSession:
    """Resolve config + return a ProbeSession.

    ``extra_defaults`` lets a specific notebook override the default
    fallback for one or more keys (e.g. NB00_PATH for cycle 017 if the
    operator's repo path differs).
    """

    extras = extra_defaults or {}

    framework_root = _resolve(
        "framework_root", extras.get("framework_root", _DEFAULT_FRAMEWORK_ROOT)
    )
    src = f"{framework_root}/src"
    if src not in sys.path:
        sys.path.insert(0, src)

    from customer_retention.analysis.auto_explorer import RunNamespace

    run_id = _resolve("run_id") or os.environ.get("CR_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "probe_session: run_id is not bound. Run "
            "debug/engagement_e4ad6e1b/cycles/00_probe_config.ipynb first, "
            "or set the CR_RUN_ID environment variable."
        )

    experiments_root = Path(
        _resolve(
            "experiments_root",
            extras.get("experiments_root", _DEFAULT_EXPERIMENTS_ROOT),
        )
    )
    catalog = _resolve("catalog", extras.get("catalog", _DEFAULT_CATALOG))
    schema = _resolve("schema", extras.get("schema", _DEFAULT_SCHEMA))
    nb10_raw = _resolve("nb10_export_path", extras.get("nb10_export_path"))
    nb00 = Path(
        _resolve(
            "nb00_path",
            extras.get(
                "nb00_path",
                f"{framework_root}/exploration_notebooks/00_start_here.ipynb",
            ),
        )
    )
    nb06 = Path(
        _resolve(
            "nb06_path",
            extras.get(
                "nb06_path",
                f"{framework_root}/exploration_notebooks/06_feature_opportunities.ipynb",
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
    experiments_root: str = _DEFAULT_EXPERIMENTS_ROOT,
    catalog: str = _DEFAULT_CATALOG,
    schema: str = _DEFAULT_SCHEMA,
    framework_root: str = _DEFAULT_FRAMEWORK_ROOT,
    nb10_export_path: str = "",
    nb00_path: str = "",
    nb06_path: str = "",
    session_label: str = "",
) -> Dict[str, str]:
    """Publish task values (when running inside a Databricks job) AND set env
    vars (so the same config survives interactive single-notebook execution).
    Called from 00_probe_config's Cell 1.
    """

    payload = {
        "run_id": run_id,
        "experiments_root": experiments_root,
        "catalog": catalog,
        "schema": schema,
        "framework_root": framework_root,
    }
    if nb10_export_path:
        payload["nb10_export_path"] = nb10_export_path
    if nb00_path:
        payload["nb00_path"] = nb00_path
    if nb06_path:
        payload["nb06_path"] = nb06_path
    if session_label:
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
