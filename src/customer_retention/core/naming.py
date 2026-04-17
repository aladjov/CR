from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_LAYER_DATASET_TEMPLATES = {
    "silver": "silver_featureset_{name}",
    "gold": "gold_features_{name}",
}

_LAYER_SCRIPT_TEMPLATES = {
    "landing": "landing_{name}",
    "bronze_event": "bronze_event_{name}",
    "bronze_entity": "bronze_entity_{name}",
    "silver": "silver_featureset_{name}",
    "gold": "gold_features_{name}",
}

_PREFIX_CHAR_LIMIT = 4

_COLUMN_TOKEN_BAD_CHARS = re.compile(r"[^\w]+", re.UNICODE)


def sanitize_column_token(value: object) -> str:
    """Collapse a categorical value into a Delta-safe column-name token.

    Delta rejects ' ,;{}()\\n\\t=' in column names. Dots, slashes, quotes and
    hyphens are additionally problematic in Spark SQL without backticks. Any
    run of such chars collapses to a single '_'; leading/trailing '_' are
    stripped; empty input returns '_' so the token is never empty.
    """
    text = str(value)
    cleaned = _COLUMN_TOKEN_BAD_CHARS.sub("_", text).strip("_")
    return cleaned or "_"


def composite_name(sources: list[str]) -> str:
    if not sources:
        raise ValueError("sources must not be empty")
    canonical = sorted(sources)
    prefix = _readable_prefix(canonical)
    hash_suffix = _short_hash(canonical)
    return f"{prefix}__{hash_suffix}"


def dataset_name(layer: str, name_or_cn: str) -> str:
    template = _LAYER_DATASET_TEMPLATES.get(layer)
    if template:
        return template.format(name=name_or_cn)
    return name_or_cn


def script_name(layer: str, name_or_cn: str) -> str:
    template = _LAYER_SCRIPT_TEMPLATES.get(layer)
    if template:
        return template.format(name=name_or_cn)
    return name_or_cn


def _readable_prefix(sorted_sources: list[str]) -> str:
    parts: list[str] = []
    for source in sorted_sources:
        tokens = source.split("_")
        truncated = [t[:_PREFIX_CHAR_LIMIT].lower() for t in tokens]
        parts.append("_".join(truncated))
    return "_".join(parts)


def _short_hash(sorted_sources: list[str]) -> str:
    joined = "|".join(sorted_sources)
    return hashlib.sha256(joined.encode()).hexdigest()[:7]


@dataclass
class Manifest:
    pipeline_name: str
    composite_name: str
    sources: List[str]
    recommendations_hash: str
    datasets: Dict[str, str]
    scripts: Dict[str, object]
    feast: Dict[str, str]
    mlflow: Dict[str, str]

    @classmethod
    def from_sources(cls, sources: list[str], pipeline_name: str, recommendations_hash: str = "") -> Manifest:
        cn = composite_name(sources)
        landing_scripts = {s: script_name("landing", s) for s in sources}
        return cls(
            pipeline_name=pipeline_name,
            composite_name=cn,
            sources=sorted(sources),
            recommendations_hash=recommendations_hash,
            datasets={
                "silver": dataset_name("silver", cn),
                "gold": dataset_name("gold", cn),
            },
            scripts={
                "landing": landing_scripts,
                "silver": script_name("silver", cn),
                "gold": script_name("gold", cn),
            },
            feast={
                "feature_view": f"featureset_{cn}",
                "entity_key": "",
                "timestamp_column": "event_timestamp",
            },
            mlflow={"experiment_name": pipeline_name},
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Manifest:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        data = json.loads(path.read_text())
        return cls(
            pipeline_name=data["pipeline_name"],
            composite_name=data["composite_name"],
            sources=data["sources"],
            recommendations_hash=data.get("recommendations_hash", ""),
            datasets=data["datasets"],
            scripts=data["scripts"],
            feast=data.get("feast", {}),
            mlflow=data.get("mlflow", {}),
        )

    def _to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "composite_name": self.composite_name,
            "sources": self.sources,
            "recommendations_hash": self.recommendations_hash,
            "datasets": self.datasets,
            "scripts": self.scripts,
            "feast": self.feast,
            "mlflow": self.mlflow,
        }
