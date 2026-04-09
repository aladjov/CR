"""YAML → row loader for ``playbook_catalog`` and ``playbook_steps``.

Reads ``*.yaml`` files from the playbooks directory (typically a Databricks
volume, but ``Path`` works equally well for local development) and converts
them to row dictionaries that match the ``StructType`` definitions in
``schemas.py``.

This module does NOT write to Delta — that is the responsibility of
``delta_writer.py``. It also does not upload files into the volume — that
is a one-shot bootstrap concern handled by ``scripts/seed_playbooks_volume.py``.

Each YAML file under ``PLAYBOOKS_DIR/`` represents one playbook and contains
two top-level keys:

```yaml
catalog:                          # → one playbook_catalog row
  playbook_id: low_nps
  version: "1.0.0"
  ...
steps:                            # → N playbook_steps rows
  - step_id: nps_issue_review
    playbook_id: low_nps
    playbook_version: "1.0.0"
    step_sequence: 1
    ...
```

The loader is permissive about missing optional fields and serializes
``skip_conditions`` / ``stop_conditions`` to JSON strings (the schema stores
them as opaque JSON blobs to accommodate future operator/condition
extensions).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Union

import yaml

from customer_retention.core.compat.remote_path import RemotePath

logger = logging.getLogger(__name__)

PathLike = Union["Path", RemotePath]  # noqa: F821 — Path imported lazily


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_playbooks_from_dir(
    playbooks_dir: PathLike,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load every playbook YAML in ``playbooks_dir`` (top level only).

    Returns ``(catalog_rows, step_rows)`` ready for Delta write. Subdirectories
    (e.g., ``policies/``) are intentionally ignored — they belong to
    ``policy_loader``. Files that fail to parse are logged and skipped so a
    single bad file does not block the rest.
    """
    yaml_files = _list_top_level_yaml_files(playbooks_dir)
    catalog_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for yaml_file in sorted(yaml_files, key=lambda p: p.name):
        try:
            doc = _load_yaml(yaml_file)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to parse playbook YAML %s: %s", yaml_file, exc)
            continue
        if not isinstance(doc, dict):
            logger.warning("Playbook YAML %s did not parse to a mapping; skipping", yaml_file)
            continue
        try:
            catalog_row = _parse_catalog(doc.get("catalog"))
            steps = _parse_steps(doc.get("steps"), catalog_row)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Schema mismatch in playbook YAML %s: %s", yaml_file, exc)
            continue
        catalog_rows.append(catalog_row)
        step_rows.extend(steps)
    logger.info(
        "Loaded %d playbooks (%d step rows) from %s",
        len(catalog_rows),
        len(step_rows),
        playbooks_dir,
    )
    return catalog_rows, step_rows


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------


def _parse_catalog(catalog: Any) -> Dict[str, Any]:
    if not isinstance(catalog, dict):
        raise ValueError("'catalog' block must be a mapping")
    required = ("playbook_id", "version", "name")
    for field in required:
        if field not in catalog:
            raise ValueError(f"playbook 'catalog' missing required field {field!r}")

    return {
        "playbook_id": _as_str(catalog.get("playbook_id")),
        "version": _as_str(catalog.get("version")),
        "name": _as_str(catalog.get("name")),
        "description": _as_optional_str(catalog.get("description")),
        "cost_per_customer_default": _as_optional_float(catalog.get("cost_per_customer_default")),
        "expected_uplift_pct_default": _as_optional_float(catalog.get("expected_uplift_pct_default")),
        "outcome_windows_days": _as_int_array(catalog.get("outcome_windows_days")),
        "outcome_definition_version": _as_optional_str(catalog.get("outcome_definition_version")),
        "time_zero_definition": _as_optional_str(catalog.get("time_zero_definition")),
        "grace_period_days": _as_optional_int(catalog.get("grace_period_days")),
        "followup_start_rule": _as_optional_str(catalog.get("followup_start_rule")),
        "followup_end_rule": _as_optional_str(catalog.get("followup_end_rule")),
        "default_estimand": _as_optional_str(catalog.get("default_estimand")),
        "analysis_population_rule": _as_optional_str(catalog.get("analysis_population_rule")),
        "active_from": _as_optional_timestamp(catalog.get("active_from")),
        "active_to": _as_optional_timestamp(catalog.get("active_to")),
    }


def _parse_steps(steps: Any, catalog_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if steps is None:
        return []
    if not isinstance(steps, list):
        raise ValueError("'steps' block must be a list")
    parsed: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {idx} is not a mapping")
        if "step_id" not in step:
            raise ValueError(f"step {idx} missing 'step_id'")
        # Default playbook ref to the parent catalog if the step omits it
        playbook_id = _as_str(step.get("playbook_id") or catalog_row["playbook_id"])
        playbook_version = _as_str(step.get("playbook_version") or catalog_row["version"])
        parsed.append(
            {
                "step_id": _as_str(step.get("step_id")),
                "playbook_id": playbook_id,
                "playbook_version": playbook_version,
                "step_sequence": _as_int(step.get("step_sequence", idx + 1)),
                "step_name": _as_str(step.get("step_name", step.get("step_id"))),
                "action_type": _as_str(step.get("action_type")),
                "automation_level": _as_str(step.get("automation_level")),
                "owner_role": _as_optional_str(step.get("owner_role")),
                "cadence_trigger": _as_optional_str(step.get("cadence_trigger")),
                "cadence_relative_to": _as_optional_str(step.get("cadence_relative_to")),
                "cadence_offset_days": _as_optional_int(step.get("cadence_offset_days")),
                "cadence_condition": _as_optional_str(step.get("cadence_condition")),
                "timeout_days": _as_optional_int(step.get("timeout_days")),
                "response_schema_id": _as_optional_str(step.get("response_schema_id")),
                "intensity_param_name": _as_optional_str(step.get("intensity_param_name")),
                "skip_conditions": _as_json_string(step.get("skip_conditions")),
                "stop_conditions": _as_json_string(step.get("stop_conditions")),
            }
        )
    return parsed


# ---------------------------------------------------------------------------
# Filesystem helpers (Path / RemotePath dual-mode)
# ---------------------------------------------------------------------------


def _list_top_level_yaml_files(playbooks_dir: PathLike) -> List[PathLike]:
    """Return YAML files at the top level of ``playbooks_dir`` only.

    Subdirectories (notably ``policies/``) are excluded — they are owned by
    ``policy_loader``.
    """
    if not _exists(playbooks_dir):
        logger.warning("Playbooks directory does not exist: %s", playbooks_dir)
        return []
    yaml_files: List[PathLike] = []
    for pattern in ("*.yaml", "*.yml"):
        for entry in playbooks_dir.glob(pattern):
            if _is_file(entry):
                yaml_files.append(entry)
    return yaml_files


def _is_file(entry: PathLike) -> bool:
    """Return True when ``entry`` is a regular file (or has no is_file check)."""
    if not hasattr(entry, "is_file"):
        return True
    try:
        return bool(entry.is_file())
    except OSError:
        return True


def _exists(path: PathLike) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _load_yaml(path: PathLike) -> Any:
    text = path.read_text() if hasattr(path, "read_text") else open(path).read()
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Field coercion helpers
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    if value is None:
        raise ValueError("expected non-null string")
    return str(value)


def _as_optional_str(value: Any) -> Union[str, None]:
    if value is None:
        return None
    return str(value).strip() or None


def _as_int(value: Any) -> int:
    return int(value)


def _as_optional_int(value: Any) -> Union[int, None]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Union[float, None]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_array(value: Any) -> Union[List[int], None]:
    if value is None:
        return None
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return None
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _as_optional_timestamp(value: Any) -> Union[datetime, None]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    # YAML may already deserialize ISO8601, but stringified inputs land here
    # Accept both Z-suffix and offset suffix
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_json_string(value: Any) -> Union[str, None]:
    """Serialize structured values to a JSON string for Delta storage.

    ``skip_conditions`` and ``stop_conditions`` in the playbook YAMLs may be
    nested operator/condition trees. Storing them as JSON strings keeps the
    Delta schema flexible while remaining queryable via ``from_json``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return None
