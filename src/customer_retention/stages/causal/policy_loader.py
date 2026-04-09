"""YAML → row loader for the policy / vocabulary / response-schema tables.

Reads the three seed YAMLs from ``PLAYBOOKS_DIR/policies/`` and converts them
to row sets matching the schemas in ``schemas.py``:

- ``policies/decision_policy.yaml``  → ``decision_policy`` rows
- ``policies/response_schemas.yaml`` → ``response_schemas`` rows
- ``policies/vocabularies.yaml``     → ``vocabularies`` rows

Like ``playbook_loader``, this module does NOT write to Delta — that is the
responsibility of ``delta_writer.py``. The framework defaults that ship at
``src/customer_retention/stages/causal/seed_yamls/`` are copied to
``policies/`` once at bootstrap time and then customized per deployment; the
loader only reads from the customized copies in ``PLAYBOOKS_DIR/policies/``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

import yaml

from customer_retention.core.compat.remote_path import RemotePath

logger = logging.getLogger(__name__)

PathLike = Union["Path", RemotePath]  # noqa: F821


_DECISION_POLICY_FILENAME = "decision_policy.yaml"
_RESPONSE_SCHEMAS_FILENAME = "response_schemas.yaml"
_VOCABULARIES_FILENAME = "vocabularies.yaml"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_policies_from_dir(
    playbooks_dir: PathLike,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load all three policy YAMLs from ``{playbooks_dir}/policies/``.

    Returns a dict keyed by table name with the row sets:

    ```
    {
      "decision_policy":  [ {decision_policy row}, ... ],
      "response_schemas": [ {response_schemas row}, ... ],
      "vocabularies":     [ {vocabularies row}, ... ],
    }
    ```

    Missing files produce empty row lists with a warning. The caller decides
    whether that is fatal (publishing job) or acceptable (smoke test).
    """
    policies_dir = playbooks_dir / "policies"
    return {
        "decision_policy": load_decision_policy(policies_dir / _DECISION_POLICY_FILENAME),
        "response_schemas": load_response_schemas(policies_dir / _RESPONSE_SCHEMAS_FILENAME),
        "vocabularies": load_vocabularies(policies_dir / _VOCABULARIES_FILENAME),
    }


def load_decision_policy(path: PathLike) -> List[Dict[str, Any]]:
    """Parse ``decision_policy.yaml`` into ``decision_policy`` rows.

    The YAML may contain either a single policy mapping or a list of
    policies. Each becomes one row.
    """
    doc = _load_yaml_file(path)
    if doc is None:
        return []
    if isinstance(doc, dict):
        policies: List[Any] = [doc]
    elif isinstance(doc, list):
        policies = doc
    else:
        logger.warning("decision_policy.yaml must be a mapping or list; got %s", type(doc).__name__)
        return []
    rows: List[Dict[str, Any]] = []
    for entry in policies:
        if not isinstance(entry, dict):
            continue
        try:
            rows.append(_parse_decision_policy_row(entry))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed decision_policy entry: %s", exc)
            continue
    return rows


def load_response_schemas(path: PathLike) -> List[Dict[str, Any]]:
    """Parse ``response_schemas.yaml`` into ``response_schemas`` rows.

    The YAML is expected to be a list of named schemas, each with a
    ``schema_id`` and a ``fields`` block. The ``fields`` block is serialized
    to JSON for Delta storage.
    """
    doc = _load_yaml_file(path)
    if doc is None:
        return []
    schemas = doc if isinstance(doc, list) else doc.get("schemas") if isinstance(doc, dict) else None
    if not isinstance(schemas, list):
        logger.warning("response_schemas.yaml must be a list (or contain a 'schemas' list)")
        return []
    rows: List[Dict[str, Any]] = []
    for entry in schemas:
        if not isinstance(entry, dict) or "schema_id" not in entry:
            continue
        rows.append(
            {
                "schema_id": str(entry["schema_id"]),
                "version": str(entry.get("version", "1.0.0")),
                "name": _as_optional_str(entry.get("name")),
                "description": _as_optional_str(entry.get("description")),
                "fields": _as_json_string(entry.get("fields")),
                "valid_from": _as_optional_timestamp(entry.get("valid_from")) or _now_utc(),
                "valid_to": _as_optional_timestamp(entry.get("valid_to")),
            }
        )
    return rows


def load_vocabularies(path: PathLike) -> List[Dict[str, Any]]:
    """Parse ``vocabularies.yaml`` into long-format ``vocabularies`` rows.

    Two accepted shapes:

    ```yaml
    # Shape A — dict of vocabulary_name → list of values
    sentiment: [positive, neutral, negative, mixed]
    engagement_level: [engaged, polite_but_disengaged, hostile, unreachable]
    ```

    ```yaml
    # Shape B — explicit list of rows (allows descriptions and validity windows)
    - vocabulary_name: sentiment
      value: positive
      description: "Customer expressed satisfaction"
      valid_from: "2026-04-08T00:00:00Z"
    ```

    Shape A is the convenient default for the seed YAML; Shape B is what
    deployments use when they need richer metadata.
    """
    doc = _load_yaml_file(path)
    if doc is None:
        return []
    rows: List[Dict[str, Any]] = []
    now = _now_utc()
    if isinstance(doc, dict):
        for vocab_name, values in doc.items():
            if not isinstance(values, list):
                continue
            for value in values:
                rows.append(
                    {
                        "vocabulary_name": str(vocab_name),
                        "value": str(value),
                        "description": None,
                        "valid_from": now,
                        "valid_to": None,
                    }
                )
    elif isinstance(doc, list):
        for entry in doc:
            if not isinstance(entry, dict):
                continue
            if "vocabulary_name" not in entry or "value" not in entry:
                continue
            rows.append(
                {
                    "vocabulary_name": str(entry["vocabulary_name"]),
                    "value": str(entry["value"]),
                    "description": _as_optional_str(entry.get("description")),
                    "valid_from": _as_optional_timestamp(entry.get("valid_from")) or now,
                    "valid_to": _as_optional_timestamp(entry.get("valid_to")),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Decision policy row builder
# ---------------------------------------------------------------------------


def _parse_decision_policy_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    if "decision_policy_id" not in entry:
        raise ValueError("decision_policy entry missing 'decision_policy_id'")
    return {
        "decision_policy_id": str(entry["decision_policy_id"]),
        "version": str(entry.get("version", "1.0.0")),
        "valid_from": _as_optional_timestamp(entry.get("valid_from")) or _now_utc(),
        "valid_to": _as_optional_timestamp(entry.get("valid_to")),
        "eligibility_logic_version": _as_optional_str(entry.get("eligibility_logic_version")),
        "prioritization_logic_version": _as_optional_str(entry.get("prioritization_logic_version")),
        "capacity_rule_version": _as_optional_str(entry.get("capacity_rule_version")),
        "holdout_rule_version": _as_optional_str(entry.get("holdout_rule_version")),
        "holdout_stratification_version": _as_optional_str(entry.get("holdout_stratification_version")),
        "holdout_stratification_fields": _as_str_array(entry.get("holdout_stratification_fields")),
        "holdout_seed_recipe": _as_optional_str(entry.get("holdout_seed_recipe")),
        "holdout_fractions_by_playbook": _as_str_double_map(entry.get("holdout_fractions_by_playbook")),
        "capacity_by_playbook": _as_str_int_map(entry.get("capacity_by_playbook")),
        "cooldown_days_by_playbook": _as_str_int_map(entry.get("cooldown_days_by_playbook")),
        "suppression_rules": _as_json_string(entry.get("suppression_rules")),
        "channel_availability_rules": _as_json_string(entry.get("channel_availability_rules")),
        "risk_tier_high_threshold": _as_optional_float(entry.get("risk_tier_high_threshold")),
        "risk_tier_medium_threshold": _as_optional_float(entry.get("risk_tier_medium_threshold")),
        "description": _as_optional_str(entry.get("description")),
        "changed_by": _as_optional_str(entry.get("changed_by")),
    }


# ---------------------------------------------------------------------------
# Filesystem + coercion helpers
# ---------------------------------------------------------------------------


def _load_yaml_file(path: PathLike) -> Any:
    if not _exists(path):
        logger.warning("Policy YAML not found: %s", path)
        return None
    try:
        text = path.read_text() if hasattr(path, "read_text") else open(path).read()
    except OSError as exc:
        logger.warning("Failed to read policy YAML %s: %s", path, exc)
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse policy YAML %s: %s", path, exc)
        return None


def _exists(path: PathLike) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_optional_str(value: Any) -> Union[str, None]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_float(value: Any) -> Union[float, None]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str_array(value: Any) -> Union[List[str], None]:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [str(v) for v in value]


def _as_str_double_map(value: Any) -> Union[Dict[str, float], None]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in value.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _as_str_int_map(value: Any) -> Union[Dict[str, int], None]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    out: Dict[str, int] = {}
    for k, v in value.items():
        try:
            out[str(k)] = int(v)
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
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_json_string(value: Any) -> Union[str, None]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return None
