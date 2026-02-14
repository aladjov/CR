from __future__ import annotations

import getpass
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .run_namespace import RunNamespace


@dataclass
class SessionState:
    active_dataset: Optional[str]
    active_run_id: str
    last_notebook: Optional[str] = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_dataset": self.active_dataset,
            "active_run_id": self.active_run_id,
            "last_notebook": self.last_notebook,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional[SessionState]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(
                active_dataset=data.get("active_dataset"),
                active_run_id=data["active_run_id"],
                last_notebook=data.get("last_notebook"),
            )
        except (json.JSONDecodeError, KeyError):
            return None


def sanitize_username(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw.strip())


def get_current_username() -> str:
    cr_username = os.environ.get("CR_USERNAME")
    if cr_username:
        return cr_username
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        db_user = os.environ.get("DATABRICKS_USERNAME")
        if db_user:
            return sanitize_username(db_user)
        from customer_retention.core.compat.detection import get_databricks_username
        detected = get_databricks_username()
        if detected:
            return sanitize_username(detected)
    return getpass.getuser()


def resolve_active_dataset(
    namespace: RunNamespace, username: Optional[str] = None
) -> Optional[str]:
    env_dataset = os.environ.get("CR_DATASET_ID")
    if env_dataset:
        return env_dataset

    if username is None:
        username = get_current_username()

    session_path = namespace.user_session_path(username)
    state = SessionState.load(session_path)
    if state and state.active_dataset:
        return state.active_dataset

    datasets = namespace.list_datasets()
    return datasets[0] if datasets else None


def set_active_dataset(
    namespace: RunNamespace, dataset_name: str, username: Optional[str] = None
) -> None:
    if username is None:
        username = get_current_username()

    session_path = namespace.user_session_path(username)
    state = SessionState.load(session_path)
    if state:
        state.active_dataset = dataset_name
    else:
        state = SessionState(active_dataset=dataset_name, active_run_id=namespace.run_id)
    state.save(session_path)


def initialize_run(
    root: Path, project_name: str, username: Optional[str] = None
) -> RunNamespace:
    namespace = RunNamespace.create(root=root, project_name=project_name)
    os.environ["CR_RUN_ID"] = namespace.run_id
    if username is None:
        username = get_current_username()
    state = SessionState(active_dataset=None, active_run_id=namespace.run_id)
    state.save(namespace.user_session_path(username))
    return namespace


def mark_notebook(
    namespace: RunNamespace, notebook_name: str, username: Optional[str] = None
) -> None:
    if username is None:
        username = get_current_username()
    session_path = namespace.user_session_path(username)
    state = SessionState.load(session_path)
    if state:
        state.last_notebook = notebook_name
    else:
        state = SessionState(
            active_dataset=None,
            active_run_id=namespace.run_id,
            last_notebook=notebook_name,
        )
    state.save(session_path)


def resolve_findings_path(
    namespace: RunNamespace,
    dataset_name: str,
    prefer_aggregated: bool = True,
) -> Optional[Path]:
    findings_dir = namespace.dataset_findings_dir(dataset_name)
    if not findings_dir.is_dir():
        return None

    findings_files = [
        f for f in findings_dir.glob("*_findings.yaml")
        if "multi_dataset" not in f.name
    ]
    if not findings_files:
        return None

    aggregated = [f for f in findings_files if "_aggregated" in f.name]
    non_aggregated = [f for f in findings_files if "_aggregated" not in f.name]

    if prefer_aggregated and aggregated:
        return sorted(aggregated)[0]
    if non_aggregated:
        return sorted(non_aggregated)[0]
    return sorted(findings_files)[0]


def resolve_target_column(namespace: Optional[RunNamespace], findings) -> Optional[str]:
    if namespace:
        ctx_path = namespace.project_context_path
        if ctx_path.exists():
            from .project_context import ProjectContext

            try:
                ctx = ProjectContext.load(ctx_path)
                if ctx.target_column:
                    return ctx.target_column
            except Exception:
                pass
    return findings.target_column


def load_notebook_findings(
    notebook_name: str,
    *,
    prefer_aggregated: bool = True,
    exclude_aggregated: bool = False,
    prefer_merged: bool = False,
    root: Optional[Path] = None,
    findings_dir: Optional[Path] = None,
) -> tuple[str, Optional[RunNamespace], Optional[str]]:
    namespace = RunNamespace.from_env(root=root) or RunNamespace.from_latest(root=root)

    if namespace and prefer_merged:
        if namespace.merged_findings_path.exists():
            mark_notebook(namespace, notebook_name)
            return str(namespace.merged_findings_path), namespace, None

    if namespace:
        dataset_name = resolve_active_dataset(namespace)
        if dataset_name:
            if exclude_aggregated:
                resolved = resolve_findings_path(
                    namespace, dataset_name, prefer_aggregated=False
                )
            else:
                resolved = resolve_findings_path(
                    namespace, dataset_name, prefer_aggregated=prefer_aggregated
                )
            if resolved:
                if exclude_aggregated and "_aggregated" in resolved.name:
                    non_agg = [
                        f
                        for f in namespace.dataset_findings_dir(dataset_name).glob(
                            "*_findings.yaml"
                        )
                        if "_aggregated" not in f.name and "multi_dataset" not in f.name
                    ]
                    if non_agg:
                        resolved = sorted(non_agg)[0]
                    else:
                        resolved = None
                if resolved:
                    mark_notebook(namespace, notebook_name)
                    return str(resolved), namespace, dataset_name

    if findings_dir is None:
        from customer_retention.core.config.experiments import FINDINGS_DIR

        findings_dir = FINDINGS_DIR

    all_files = [
        f
        for f in findings_dir.glob("*_findings.yaml")
        if "multi_dataset" not in f.name
    ]
    if exclude_aggregated:
        all_files = [f for f in all_files if "_aggregated" not in f.name]

    if not all_files:
        raise FileNotFoundError(
            f"No findings files found in {findings_dir}. Run notebook 01 first."
        )

    aggregated = sorted(f for f in all_files if "_aggregated" in f.name)
    non_aggregated = sorted(f for f in all_files if "_aggregated" not in f.name)

    if prefer_aggregated and aggregated:
        return str(aggregated[0]), None, None
    if non_aggregated:
        return str(non_aggregated[0]), None, None
    return str(sorted(all_files)[0]), None, None
