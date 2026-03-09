from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

NOTEBOOKS_ORDER = [
    "00_start_here",
    "01_data_discovery",
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
    "02_source_integrity",
    "03_dataset_merge",
    "04_column_deep_dive",
    "04a_text_columns_deep_dive",
    "05_relationship_analysis",
    "06_feature_opportunities",
    "07_modeling_readiness",
    "08_baseline_experiments",
    "09_business_alignment",
    "10_spec_generation",
    "11_scoring_validation",
    "12_view_documentation",
]

TEMPORAL_NOTEBOOKS = {
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
}

TEXT_NOTEBOOKS = {
    "01a_a_temporal_text_deep_dive",
    "04a_text_columns_deep_dive",
}

SETUP_NOTEBOOKS = {"00_start_here"}

PER_DATASET_STEMS = {
    "01_data_discovery",
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
    "02_source_integrity",
    "04a_text_columns_deep_dive",
}

CRITICAL_GLOBAL = {"03_dataset_merge"}

_GLOBAL_TEXT_NOTEBOOKS: Set[str] = set()

_FINDINGS_EXTENSIONS = ("*.yaml", "*.yml", "*.json")


def find_dataset_findings(
    findings_dir: Path, dataset_name: str, namespace=None
) -> Optional[Path]:
    if namespace is not None:
        ns_dir = namespace.dataset_findings_dir(dataset_name)
        if ns_dir.is_dir():
            exact = ns_dir / f"{dataset_name}_findings.yaml"
            if exact.exists():
                return exact
            aggregated = ns_dir / f"{dataset_name}_aggregated_findings.yaml"
            if aggregated.exists():
                return aggregated
            candidates = sorted(ns_dir.glob(f"{dataset_name}*_findings.yaml"))
            if candidates:
                return candidates[0]

    if not findings_dir.is_dir():
        return None
    exact = findings_dir / f"{dataset_name}_findings.yaml"
    if exact.exists():
        return exact
    aggregated = findings_dir / f"{dataset_name}_aggregated_findings.yaml"
    if aggregated.exists():
        return aggregated
    candidates = sorted(findings_dir.glob(f"{dataset_name}*_findings.yaml"))
    return candidates[0] if candidates else None


def is_event_level_dataset(
    findings_dir: Path,
    dataset_name: str,
    context=None,
    dataset_path: Optional[str] = None,
    namespace=None,
) -> bool:
    from customer_retention.core.config.column_config import DatasetGranularity

    findings_path = find_dataset_findings(findings_dir, dataset_name, namespace=namespace)
    if findings_path:
        try:
            from customer_retention.analysis.auto_explorer.findings import ExplorationFindings

            findings = ExplorationFindings.load(str(findings_path))
            if findings.time_series_metadata:
                return findings.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            logger.debug("Failed to load findings from %s: %s", findings_path, exc)

    if context is not None:
        entry = context.datasets.get(dataset_name)
        if entry and getattr(entry, "relationship", None) == "many_to_one":
            return True

    if dataset_path:
        try:
            from customer_retention.core.compat import native_pd
            from customer_retention.stages.profiling import TypeDetector

            raw = (
                native_pd.read_csv(dataset_path)
                if str(dataset_path).endswith(".csv")
                else native_pd.read_parquet(dataset_path)
            )
            detector = TypeDetector()
            result = detector.detect_granularity(raw)
            return result.granularity == DatasetGranularity.EVENT_LEVEL
        except (FileNotFoundError, ValueError, TypeError) as exc:
            logger.debug("Failed to detect granularity from %s: %s", dataset_path, exc)

    return False


def has_text_columns_for_dataset(
    findings_dir: Path,
    dataset_name: str,
    dataset_path: Optional[str] = None,
    namespace=None,
) -> bool:
    from customer_retention.core.config.column_config import ColumnType

    findings_path = find_dataset_findings(findings_dir, dataset_name, namespace=namespace)
    if findings_path:
        try:
            from customer_retention.analysis.auto_explorer.findings import ExplorationFindings

            findings = ExplorationFindings.load(str(findings_path))
            if findings.metadata.get("auto_drop_text_columns"):
                return False
            if findings.text_processing:
                return True
            if ColumnType.TEXT in findings.column_types.values():
                return True
            return False
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            logger.debug("Failed to check text columns in %s: %s", findings_path, exc)

    if dataset_path:
        try:
            from customer_retention.core.compat import native_pd
            from customer_retention.stages.profiling import TypeDetector

            raw = (
                native_pd.read_csv(dataset_path)
                if str(dataset_path).endswith(".csv")
                else native_pd.read_parquet(dataset_path)
            )
            detector = TypeDetector()
            for col in raw.select_dtypes(include=["object"]).columns:
                result = detector.detect_type(raw[col], col)
                if result.inferred_type == ColumnType.TEXT:
                    return True
        except (FileNotFoundError, ValueError, TypeError) as exc:
            logger.debug("Failed to detect text columns from %s: %s", dataset_path, exc)

    return False


def detect_skip_set_for_dataset(
    findings_dir: Path,
    dataset_name: str,
    context=None,
    dataset_path: Optional[str] = None,
    namespace=None,
) -> Tuple[Set[str], Dict[str, str]]:
    skip: Set[str] = set()
    reasons: Dict[str, str] = {}

    if not is_event_level_dataset(
        findings_dir, dataset_name, context=context,
        dataset_path=dataset_path, namespace=namespace,
    ):
        for nb in TEMPORAL_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = f"no event-level data ({dataset_name})"

    if not has_text_columns_for_dataset(
        findings_dir, dataset_name, dataset_path=dataset_path, namespace=namespace,
    ):
        for nb in TEXT_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = f"no TEXT columns ({dataset_name})"

    return skip, reasons


def detect_global_skip_set(
    findings_dir: Path, context, namespace=None
) -> Tuple[Set[str], Dict[str, str]]:
    skip: Set[str] = set()
    reasons: Dict[str, str] = {}

    any_text = False
    for ds_name in context.datasets:
        ds_entry = context.datasets[ds_name]
        ds_path = getattr(ds_entry, "path", None)
        if has_text_columns_for_dataset(
            findings_dir, ds_name, dataset_path=ds_path, namespace=namespace,
        ):
            any_text = True
            break

    if not any_text:
        for nb in _GLOBAL_TEXT_NOTEBOOKS:
            skip.add(nb)
            reasons[nb] = "no TEXT columns in any dataset"

    return skip, reasons
