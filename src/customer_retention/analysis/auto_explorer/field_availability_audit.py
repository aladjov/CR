from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

from customer_retention.core.compat import DataFrame, _is_spark_pandas, native_pd

from .service_unit_detector import (
    DatasetLinkage,
    ServiceUnitConfig,
    classify_dataset_linkage,
    detect_service_unit,
)

if TYPE_CHECKING:
    from .project_context import ProjectContext
    from .run_namespace import RunNamespace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FieldLeadLagProfile:
    field_name: str
    source_dataset: str
    analysis_tier: str
    anchor_type: str
    pct_before_90d: Optional[float] = None
    pct_30_to_90d: Optional[float] = None
    pct_0_to_30d: Optional[float] = None
    pct_after_anchor: Optional[float] = None
    median_lead_days: Optional[float] = None
    pct_non_null_terminated: Optional[float] = None
    pct_non_null_active: Optional[float] = None
    population_ratio: Optional[float] = None
    coverage: float = 0.0
    suspicion_score: float = 0.0
    recommendation: str = "safe"
    evidence: list[str] = field(default_factory=list)
    cardinality: Optional[int] = None
    value_suspicion_score: Optional[float] = None
    terminated_exclusive_values: list[str] = field(default_factory=list)


@dataclass
class DatasetDateInfo:
    """Date column and range info for a single dataset."""

    dataset_name: str
    time_column: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    linkage_tier: str = ""
    record_count: int = 0


@dataclass
class FieldAvailabilityAuditConfig:
    service_unit: ServiceUnitConfig
    probe_datasets: list[str] = field(default_factory=list)
    lead_buckets_days: list[int] = field(default_factory=lambda: [0, 30, 90])
    min_terminated_units: int = 50
    suspicion_threshold: float = 0.5
    target_columns: list[str] = field(default_factory=list)


@dataclass
class FieldAvailabilityAuditResult:
    config: FieldAvailabilityAuditConfig
    total_accounts: int = 0
    fully_terminated_accounts: int = 0
    partially_terminated_accounts: int = 0
    active_accounts: int = 0
    total_terminated_units: int = 0
    field_profiles: list[FieldLeadLagProfile] = field(default_factory=list)
    suspicious_fields: list[str] = field(default_factory=list)
    recommended_exclusions: list[str] = field(default_factory=list)
    dataset_linkage: dict[str, Any] = field(default_factory=dict)
    dataset_date_info: list[DatasetDateInfo] = field(default_factory=list)
    audit_timestamp: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    def save(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["audit_config"] = _save_yaml(output_dir / "audit_config.yaml", self._build_config_dict())
        paths["column_profiles"] = _save_yaml(output_dir / "column_profiles.yaml", self._build_profiles_dict())
        paths["audit_suggestions"] = _save_yaml(output_dir / "audit_suggestions.yaml", self._build_suggestions_dict())
        return paths

    @classmethod
    def load(cls, output_dir: Path) -> FieldAvailabilityAuditResult:
        config_data = _load_yaml(output_dir / "audit_config.yaml")
        profiles_data = _load_yaml(output_dir / "column_profiles.yaml")
        recs_data = _load_yaml(output_dir / "audit_suggestions.yaml")

        su = ServiceUnitConfig(**config_data["service_unit"])
        audit_cfg = FieldAvailabilityAuditConfig(
            service_unit=su,
            probe_datasets=list(config_data.get("dataset_linkage", {}).keys()),
            lead_buckets_days=config_data.get("parameters", {}).get("lead_buckets_days", [0, 30, 90]),
            min_terminated_units=config_data.get("parameters", {}).get("min_terminated_units", 50),
            suspicion_threshold=config_data.get("parameters", {}).get("suspicion_threshold", 0.5),
        )
        pop = config_data.get("population_summary", {})
        profiles = [FieldLeadLagProfile(**p) for p in profiles_data.get("columns", [])]
        return cls(
            config=audit_cfg,
            total_accounts=pop.get("total_accounts", 0),
            fully_terminated_accounts=pop.get("fully_terminated_accounts", 0),
            partially_terminated_accounts=pop.get("partially_terminated_accounts", 0),
            active_accounts=pop.get("active_accounts", 0),
            total_terminated_units=pop.get("terminated_service_units", 0),
            field_profiles=profiles,
            suspicious_fields=recs_data.get("investigate_fields", []),
            recommended_exclusions=recs_data.get("exclude_fields", []),
            dataset_linkage=config_data.get("dataset_linkage", {}),
            audit_timestamp=recs_data.get("audit_timestamp", ""),
        )

    def _build_config_dict(self) -> dict:
        return {
            "service_unit": asdict(self.config.service_unit),
            "dataset_linkage": self.dataset_linkage,
            "parameters": {
                "lead_buckets_days": self.config.lead_buckets_days,
                "min_terminated_units": self.config.min_terminated_units,
                "suspicion_threshold": self.config.suspicion_threshold,
            },
            "population_summary": {
                "total_accounts": self.total_accounts,
                "fully_terminated_accounts": self.fully_terminated_accounts,
                "partially_terminated_accounts": self.partially_terminated_accounts,
                "active_accounts": self.active_accounts,
                "terminated_service_units": self.total_terminated_units,
            },
            "dataset_dates": [asdict(d) for d in self.dataset_date_info],
        }

    def _build_profiles_dict(self) -> dict:
        return {"columns": [_profile_to_dict(p) for p in self.field_profiles]}

    def _build_suggestions_dict(self) -> dict:
        excludes = [p for p in self.field_profiles if p.recommendation == "exclude"]
        investigates = [p for p in self.field_profiles if p.recommendation == "investigate"]
        return {
            "audit_timestamp": self.audit_timestamp,
            "summary": (
                f"Audited {len(self.field_profiles)} fields. "
                f"Found {len(excludes)} recommended exclusions, "
                f"{len(investigates)} fields for investigation."
            ),
            "recommended_exclusions": [
                {
                    "field_name": p.field_name,
                    "source_dataset": p.source_dataset,
                    "rationale": "; ".join(p.evidence) if p.evidence else "",
                    "action": "exclude_from_features",
                    "target_config_field": "excluded_leaking_features",
                }
                for p in excludes
            ],
            "investigate": [
                {
                    "field_name": p.field_name,
                    "source_dataset": p.source_dataset,
                    "rationale": "; ".join(p.evidence) if p.evidence else "",
                    "action": "review_manually",
                }
                for p in investigates
            ],
            "exclude_fields": [p.field_name for p in excludes],
            "investigate_fields": [p.field_name for p in investigates],
            "safe_fields_count": sum(1 for p in self.field_profiles if p.recommendation == "safe"),
        }


def _profile_to_dict(p: FieldLeadLagProfile) -> dict:
    d = asdict(p)
    for k, v in d.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            d[k] = None
    return d


def _save_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


# ---------------------------------------------------------------------------
# Account anchor builder
# ---------------------------------------------------------------------------


def build_account_anchors(
    service_unit_df: DataFrame, config: ServiceUnitConfig
) -> DataFrame:
    if _is_spark_pandas(service_unit_df):
        return _build_account_anchors_spark(service_unit_df, config)
    return _build_account_anchors_pandas(service_unit_df, config)


def _build_account_anchors_pandas(
    su_df: DataFrame, cfg: ServiceUnitConfig
) -> DataFrame:
    entity_col, anchor_col, status_col = cfg.entity_column, cfg.anchor_date_column, cfg.status_column

    total_per_account = su_df.groupby(entity_col).size().reset_index(name="total_units")

    if status_col and cfg.terminated_statuses:
        terminated = su_df[su_df[status_col].isin(cfg.terminated_statuses)]
    else:
        terminated = su_df[su_df[anchor_col].notna()]

    if len(terminated) == 0:
        return native_pd.DataFrame(columns=[
            entity_col, "first_partial_cancel_date", "full_termination_date",
            "total_units", "terminated_units", "is_fully_terminated",
        ])

    term_agg = terminated.groupby(entity_col).agg(
        first_partial_cancel_date=(anchor_col, "min"),
        full_termination_date=(anchor_col, "max"),
        terminated_units=(anchor_col, "count"),
    ).reset_index()

    result = term_agg.merge(total_per_account, on=entity_col, how="left")
    result["is_fully_terminated"] = result["terminated_units"] == result["total_units"]
    result.loc[~result["is_fully_terminated"], "full_termination_date"] = None
    return result


def _build_account_anchors_spark(su_df: DataFrame, cfg: ServiceUnitConfig) -> DataFrame:  # pragma: no cover
    import pyspark.sql.functions as F  # noqa: N812

    from customer_retention.core.compat import as_pandas_api, as_spark_df

    spark_df = as_spark_df(su_df)
    entity_col, anchor_col, status_col = cfg.entity_column, cfg.anchor_date_column, cfg.status_column

    total_per_account = spark_df.groupBy(entity_col).agg(F.count("*").alias("total_units"))

    if status_col and cfg.terminated_statuses:
        terminated = spark_df.filter(F.col(status_col).isin(cfg.terminated_statuses))
    else:
        terminated = spark_df.filter(F.col(anchor_col).isNotNull())

    term_agg = terminated.groupBy(entity_col).agg(
        F.min(anchor_col).alias("first_partial_cancel_date"),
        F.max(anchor_col).alias("full_termination_date"),
        F.count("*").alias("terminated_units"),
    )

    joined = term_agg.join(F.broadcast(total_per_account), on=entity_col, how="left")
    joined = joined.withColumn("is_fully_terminated", F.col("terminated_units") == F.col("total_units"))
    joined = joined.withColumn(
        "full_termination_date",
        F.when(F.col("is_fully_terminated"), F.col("full_termination_date")).otherwise(F.lit(None)),
    )
    return as_pandas_api(joined)


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

_SKIP_COLUMNS = frozenset({"entity_id"})
_MAX_VALUE_DISTRIBUTION_CARDINALITY = 50
_TARGET_LIKE_PATTERNS = frozenset({
    "churned", "churn_date", "is_churned", "churn_flag", "churn_label",
    "target", "label_timestamp",
})
_LIFECYCLE_END_SUFFIXES = (
    "_end_date", "_termination_date", "_cancellation_date",
    "_expiry_date", "_expiration_date",
)


class FieldAvailabilityAuditor:
    def __init__(self, config: FieldAvailabilityAuditConfig):
        self.config = config

    def run(
        self,
        service_unit_df: DataFrame,
        probe_dfs: dict[str, DataFrame],
        dataset_linkage: dict[str, DatasetLinkage],
        dataset_granularities: Optional[dict[str, str]] = None,
        dataset_time_columns: Optional[dict[str, str]] = None,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> FieldAvailabilityAuditResult:
        cfg = self.config
        su_cfg = cfg.service_unit

        anchors = build_account_anchors(service_unit_df, su_cfg)
        if len(anchors) == 0:
            raise ValueError(f"No terminated service units found in '{su_cfg.dataset_name}'")

        total_terminated = int(anchors["terminated_units"].sum())
        if total_terminated < cfg.min_terminated_units:
            raise ValueError(
                f"Only {total_terminated} terminated units found, "
                f"minimum is {cfg.min_terminated_units}"
            )

        fully_terminated = int(anchors["is_fully_terminated"].sum())
        partially_terminated = int((~anchors["is_fully_terminated"]).sum())
        all_entities_in_su = int(service_unit_df[su_cfg.entity_column].nunique())
        active_accounts = all_entities_in_su - len(anchors)

        profiles: list[FieldLeadLagProfile] = []

        self_probe_profiles = self._probe_self(service_unit_df, su_cfg, progress_fn)
        profiles.extend(self_probe_profiles)

        for ds_name, df in probe_dfs.items():
            if ds_name == su_cfg.dataset_name:
                continue
            linkage = dataset_linkage.get(ds_name)
            if not linkage:
                continue

            gran = (dataset_granularities or {}).get(ds_name, "entity_level")
            time_col = (dataset_time_columns or {}).get(ds_name)

            if progress_fn:
                progress_fn(f"Probing {ds_name} ({linkage.tier})...")

            if gran == "event_level" and time_col:
                ds_profiles = self._probe_event_dataset(df, ds_name, time_col, anchors, su_cfg, linkage)
            else:
                ds_profiles = self._probe_entity_dataset(df, ds_name, anchors, su_cfg)
            profiles.extend(ds_profiles)

        threshold = self.config.suspicion_threshold
        for p in profiles:
            if any(p.field_name.lower().endswith(s) for s in _LIFECYCLE_END_SUFFIXES):
                if p.suspicion_score < threshold:
                    p.suspicion_score = round(threshold + 0.01, 4)
                    p.evidence.append(
                        f"{p.field_name}: lifecycle end-date column (name pattern)"
                    )

        for p in profiles:
            p.recommendation = self._classify(p)

        suspicious = [p.field_name for p in profiles if p.recommendation in ("investigate", "exclude")]
        exclusions = [p.field_name for p in profiles if p.recommendation == "exclude"]

        return FieldAvailabilityAuditResult(
            config=cfg,
            total_accounts=fully_terminated + partially_terminated + active_accounts,
            fully_terminated_accounts=fully_terminated,
            partially_terminated_accounts=partially_terminated,
            active_accounts=active_accounts,
            total_terminated_units=total_terminated,
            field_profiles=profiles,
            suspicious_fields=suspicious,
            recommended_exclusions=exclusions,
            dataset_linkage={
                ds_name: asdict(lnk)
                for ds_name, lnk in dataset_linkage.items()
            },
        )

    def _classify(self, p: FieldLeadLagProfile) -> str:
        threshold = self.config.suspicion_threshold
        if p.suspicion_score > threshold + 0.2:
            return "exclude"
        if p.suspicion_score > threshold:
            return "investigate"
        return "safe"

    def run_value_distribution_pass(
        self, result: FieldAvailabilityAuditResult,
        service_unit_df: DataFrame, probe_dfs: dict[str, DataFrame],
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Second pass: value-distribution on low-cardinality fields. Updates result in-place."""
        su_cfg = self.config.service_unit
        anchors = build_account_anchors(service_unit_df, su_cfg)
        self._augment_with_value_distributions(
            result.field_profiles, service_unit_df, probe_dfs, anchors, su_cfg, progress_fn,
        )
        for p in result.field_profiles:
            p.recommendation = self._classify(p)
            if p.recommendation == "exclude" and p.field_name not in result.recommended_exclusions:
                result.recommended_exclusions.append(p.field_name)
            if p.recommendation in ("investigate", "exclude") and p.field_name not in result.suspicious_fields:
                result.suspicious_fields.append(p.field_name)

    def _probe_self(
        self, su_df: DataFrame, cfg: ServiceUnitConfig, progress_fn: Optional[Callable] = None,
    ) -> list[FieldLeadLagProfile]:
        if progress_fn:
            progress_fn(f"Self-probing {cfg.dataset_name}...")
        skip = {
            cfg.unit_id_column.lower(), cfg.entity_column.lower(),
            cfg.anchor_date_column.lower(),
        }
        if cfg.status_column:
            skip.add(cfg.status_column.lower())
        if cfg.start_date_column:
            skip.add(cfg.start_date_column.lower())
        for s in cfg.secondary_anchor_columns:
            skip.add(s.lower())
        skip.update(c.lower() for c in _SKIP_COLUMNS)
        skip.update(c.lower() for c in self.config.target_columns)

        probe_cols = [c for c in su_df.columns if c.lower() not in skip]
        if not probe_cols:
            return []
        return self._population_rate_on_units(su_df, probe_cols, cfg)

    def _population_rate_on_units(
        self, su_df: DataFrame, probe_cols: list[str], cfg: ServiceUnitConfig,
    ) -> list[FieldLeadLagProfile]:
        if _is_spark_pandas(su_df):
            return self._population_rate_on_units_spark(su_df, probe_cols, cfg)
        return self._population_rate_on_units_pandas(su_df, probe_cols, cfg)

    def _population_rate_on_units_pandas(
        self, su_df: DataFrame, probe_cols: list[str], cfg: ServiceUnitConfig,
    ) -> list[FieldLeadLagProfile]:
        status_col = cfg.status_column
        if status_col and cfg.terminated_statuses:
            is_term = su_df[status_col].isin(cfg.terminated_statuses)
        else:
            is_term = su_df[cfg.anchor_date_column].notna()

        term_df = su_df[is_term]
        active_df = su_df[~is_term]
        n_term = len(term_df)
        n_active = len(active_df)

        profiles = []
        for col in probe_cols:
            pct_term = float(term_df[col].notna().sum() / n_term) if n_term > 0 else 0.0
            pct_act = float(active_df[col].notna().sum() / n_active) if n_active > 0 else 0.0
            ratio = (pct_term / pct_act) if pct_act > 0 else (float("inf") if pct_term > 0 else 0.0)
            score = _population_suspicion_score(ratio, pct_term, pct_act)
            evidence = _population_evidence(col, pct_term, pct_act, ratio, score)
            profiles.append(FieldLeadLagProfile(
                field_name=col, source_dataset=cfg.dataset_name,
                analysis_tier="contract_level", anchor_type="contract_termination",
                pct_non_null_terminated=round(pct_term, 4),
                pct_non_null_active=round(pct_act, 4),
                population_ratio=round(ratio, 4) if not math.isinf(ratio) else None,
                coverage=round(pct_term, 4),
                suspicion_score=round(score, 4),
                evidence=evidence,
            ))
        return profiles

    def _population_rate_on_units_spark(  # pragma: no cover
        self, su_df: DataFrame, probe_cols: list[str], cfg: ServiceUnitConfig,
    ) -> list[FieldLeadLagProfile]:
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat import as_spark_df

        spark_df = as_spark_df(su_df)
        status_col = cfg.status_column
        if status_col and cfg.terminated_statuses:
            is_term = F.col(status_col).isin(cfg.terminated_statuses)
        else:
            is_term = F.col(cfg.anchor_date_column).isNotNull()

        term_exprs = [F.count(F.when(is_term & F.col(c).isNotNull(), True)).alias(f"term_{c}") for c in probe_cols]
        active_exprs = [F.count(F.when(~is_term & F.col(c).isNotNull(), True)).alias(f"act_{c}") for c in probe_cols]
        count_exprs = [
            F.count(F.when(is_term, True)).alias("n_term"),
            F.count(F.when(~is_term, True)).alias("n_active"),
        ]
        row = spark_df.agg(*(count_exprs + term_exprs + active_exprs)).collect()[0]

        n_term = row["n_term"] or 0
        n_active = row["n_active"] or 0
        profiles = []
        for col in probe_cols:
            t_count = row[f"term_{col}"] or 0
            a_count = row[f"act_{col}"] or 0
            pct_term = t_count / n_term if n_term > 0 else 0.0
            pct_act = a_count / n_active if n_active > 0 else 0.0
            ratio = (pct_term / pct_act) if pct_act > 0 else (float("inf") if pct_term > 0 else 0.0)
            score = _population_suspicion_score(ratio, pct_term, pct_act)
            evidence = _population_evidence(col, pct_term, pct_act, ratio, score)
            profiles.append(FieldLeadLagProfile(
                field_name=col, source_dataset=cfg.dataset_name,
                analysis_tier="contract_level", anchor_type="contract_termination",
                pct_non_null_terminated=round(pct_term, 4),
                pct_non_null_active=round(pct_act, 4),
                population_ratio=round(ratio, 4) if not math.isinf(ratio) else None,
                coverage=round(pct_term, 4),
                suspicion_score=round(score, 4),
                evidence=evidence,
            ))
        return profiles

    def _probe_entity_dataset(
        self, df: DataFrame, ds_name: str, anchors: DataFrame, su_cfg: ServiceUnitConfig,
    ) -> list[FieldLeadLagProfile]:
        entity_col = su_cfg.entity_column
        if entity_col not in df.columns:
            return []

        skip = {entity_col.lower()} | {c.lower() for c in _SKIP_COLUMNS}
        skip.update(c.lower() for c in self.config.target_columns)
        probe_cols = [c for c in df.columns if c.lower() not in skip]
        if not probe_cols:
            return []

        if _is_spark_pandas(df) or _is_spark_pandas(anchors):
            return self._probe_entity_spark(df, ds_name, probe_cols, anchors, entity_col)
        return self._probe_entity_pandas(df, ds_name, probe_cols, anchors, entity_col)

    def _probe_entity_pandas(
        self, df: DataFrame, ds_name: str, probe_cols: list[str],
        anchors: DataFrame, entity_col: str,
    ) -> list[FieldLeadLagProfile]:
        merged = df.merge(anchors[[entity_col, "is_fully_terminated"]], on=entity_col, how="left")
        fully_term = merged[merged["is_fully_terminated"] == True]  # noqa: E712
        active = merged[merged["is_fully_terminated"].isna()]
        n_term = len(fully_term)
        n_active = len(active)

        profiles = []
        for col in probe_cols:
            pct_term = float(fully_term[col].notna().sum() / n_term) if n_term > 0 else 0.0
            pct_act = float(active[col].notna().sum() / n_active) if n_active > 0 else 0.0
            ratio = (pct_term / pct_act) if pct_act > 0 else (float("inf") if pct_term > 0 else 0.0)
            score = _population_suspicion_score(ratio, pct_term, pct_act)
            evidence = _population_evidence(col, pct_term, pct_act, ratio, score)
            profiles.append(FieldLeadLagProfile(
                field_name=col, source_dataset=ds_name,
                analysis_tier="account_level", anchor_type="full_termination",
                pct_non_null_terminated=round(pct_term, 4),
                pct_non_null_active=round(pct_act, 4),
                population_ratio=round(ratio, 4) if not math.isinf(ratio) else None,
                coverage=round(pct_term, 4),
                suspicion_score=round(score, 4),
                evidence=evidence,
            ))
        return profiles

    def _probe_entity_spark(  # pragma: no cover
        self, df: DataFrame, ds_name: str, probe_cols: list[str],
        anchors: DataFrame, entity_col: str,
    ) -> list[FieldLeadLagProfile]:
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat import as_spark_df

        spark_df = as_spark_df(df)
        anchors_spark = F.broadcast(as_spark_df(anchors[[entity_col, "is_fully_terminated"]]))
        joined = spark_df.join(anchors_spark, on=entity_col, how="left")

        is_term = F.col("is_fully_terminated") == True  # noqa: E712
        is_active = F.col("is_fully_terminated").isNull()

        exprs = [
            F.count(F.when(is_term, True)).alias("n_term"),
            F.count(F.when(is_active, True)).alias("n_active"),
        ]
        for col in probe_cols:
            exprs.append(F.count(F.when(is_term & F.col(col).isNotNull(), True)).alias(f"term_{col}"))
            exprs.append(F.count(F.when(is_active & F.col(col).isNotNull(), True)).alias(f"act_{col}"))

        row = joined.agg(*exprs).collect()[0]
        n_term = row["n_term"] or 0
        n_active = row["n_active"] or 0

        profiles = []
        for col in probe_cols:
            t_count = row[f"term_{col}"] or 0
            a_count = row[f"act_{col}"] or 0
            pct_term = t_count / n_term if n_term > 0 else 0.0
            pct_act = a_count / n_active if n_active > 0 else 0.0
            ratio = (pct_term / pct_act) if pct_act > 0 else (float("inf") if pct_term > 0 else 0.0)
            score = _population_suspicion_score(ratio, pct_term, pct_act)
            evidence = _population_evidence(col, pct_term, pct_act, ratio, score)
            profiles.append(FieldLeadLagProfile(
                field_name=col, source_dataset=ds_name,
                analysis_tier="account_level", anchor_type="full_termination",
                pct_non_null_terminated=round(pct_term, 4),
                pct_non_null_active=round(pct_act, 4),
                population_ratio=round(ratio, 4) if not math.isinf(ratio) else None,
                coverage=round(pct_term, 4),
                suspicion_score=round(score, 4),
                evidence=evidence,
            ))
        return profiles

    def _probe_event_dataset(
        self, df: DataFrame, ds_name: str, time_col: str,
        anchors: DataFrame, su_cfg: ServiceUnitConfig,
        linkage: DatasetLinkage,
    ) -> list[FieldLeadLagProfile]:
        entity_col = su_cfg.entity_column
        if entity_col not in df.columns:
            return []
        skip = {entity_col.lower(), time_col.lower()} | {c.lower() for c in _SKIP_COLUMNS}
        skip.update(c.lower() for c in self.config.target_columns)
        probe_cols = [c for c in df.columns if c.lower() not in skip]
        if not probe_cols:
            return []

        anchor_col_name = "full_termination_date"
        if _is_spark_pandas(df) or _is_spark_pandas(anchors):
            return self._probe_event_spark(df, ds_name, probe_cols, time_col, anchors, entity_col, anchor_col_name)
        return self._probe_event_pandas(df, ds_name, probe_cols, time_col, anchors, entity_col, anchor_col_name)

    def _probe_event_pandas(
        self, df: DataFrame, ds_name: str, probe_cols: list[str], time_col: str,
        anchors: DataFrame, entity_col: str, anchor_col_name: str,
    ) -> list[FieldLeadLagProfile]:
        from customer_retention.core.compat import safe_to_datetime, timedelta_to_days

        anchor_subset = anchors[anchors[anchor_col_name].notna()][[entity_col, anchor_col_name]]
        merged = df.merge(anchor_subset, on=entity_col, how="inner")
        if len(merged) == 0:
            return []

        n_entities = int(anchor_subset[entity_col].nunique())
        profiles = []
        for col in probe_cols:
            non_null = merged[merged[col].notna()]
            if len(non_null) == 0:
                profiles.append(self._empty_event_profile(col, ds_name, anchor_col_name))
                continue
            first_pop = non_null.groupby(entity_col)[time_col].min().reset_index()
            first_pop.columns = [entity_col, "first_pop_date"]
            with_anchor = first_pop.merge(anchor_subset, on=entity_col, how="left")
            with_anchor["first_pop_date"] = safe_to_datetime(with_anchor["first_pop_date"])
            with_anchor[anchor_col_name] = safe_to_datetime(with_anchor[anchor_col_name])
            lead = timedelta_to_days(with_anchor[anchor_col_name] - with_anchor["first_pop_date"])
            profiles.append(self._build_event_profile(col, ds_name, anchor_col_name, lead, n_entities))
        return profiles

    def _probe_event_spark(  # pragma: no cover
        self, df: DataFrame, ds_name: str, probe_cols: list[str], time_col: str,
        anchors: DataFrame, entity_col: str, anchor_col_name: str,
    ) -> list[FieldLeadLagProfile]:
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat import as_spark_df

        spark_df = as_spark_df(df)
        anchor_subset = anchors[anchors[anchor_col_name].notna()][[entity_col, anchor_col_name]]
        anchors_spark = F.broadcast(as_spark_df(anchor_subset))

        joined = spark_df.join(anchors_spark, on=entity_col, how="inner")
        n_entities = int(anchor_subset[entity_col].nunique())

        first_pop_exprs = [
            F.min(F.when(F.col(c).isNotNull(), F.col(time_col))).alias(f"first_{c}")
            for c in probe_cols
        ]
        first_pop_df = joined.groupBy(entity_col).agg(*first_pop_exprs)
        first_pop_with_anchor = first_pop_df.join(anchors_spark, on=entity_col, how="left")

        lead_exprs = []
        for c in probe_cols:
            lead_col = F.datediff(F.col(anchor_col_name), F.col(f"first_{c}")).alias(f"lead_{c}")
            lead_exprs.append(lead_col)

        lead_df = first_pop_with_anchor.select(entity_col, *lead_exprs)

        stats_exprs = []
        for c in probe_cols:
            lc = f"lead_{c}"
            stats_exprs.extend([
                F.count(F.when(F.col(lc).isNotNull(), True)).alias(f"n_{c}"),
                F.count(F.when(F.col(lc) > 90, True)).alias(f"b90_{c}"),
                F.count(F.when((F.col(lc) >= 30) & (F.col(lc) <= 90), True)).alias(f"b30_{c}"),
                F.count(F.when((F.col(lc) >= 0) & (F.col(lc) < 30), True)).alias(f"b0_{c}"),
                F.count(F.when(F.col(lc) < 0, True)).alias(f"baft_{c}"),
                F.percentile_approx(F.col(lc).cast("double"), 0.5).alias(f"med_{c}"),
            ])
        stats_row = lead_df.agg(*stats_exprs).collect()[0]

        profiles = []
        for c in probe_cols:
            n = stats_row[f"n_{c}"] or 0
            if n == 0:
                profiles.append(self._empty_event_profile(c, ds_name, anchor_col_name))
                continue
            pct_90 = (stats_row[f"b90_{c}"] or 0) / n
            pct_30 = (stats_row[f"b30_{c}"] or 0) / n
            pct_0 = (stats_row[f"b0_{c}"] or 0) / n
            pct_aft = (stats_row[f"baft_{c}"] or 0) / n
            median = stats_row[f"med_{c}"]
            score = round(pct_0 + pct_aft, 4)
            coverage = round(n / n_entities, 4) if n_entities > 0 else 0.0
            evidence = _event_evidence(c, pct_90, pct_30, pct_0, pct_aft, median, score)
            profiles.append(FieldLeadLagProfile(
                field_name=c, source_dataset=ds_name,
                analysis_tier="account_level", anchor_type=anchor_col_name,
                pct_before_90d=round(pct_90, 4), pct_30_to_90d=round(pct_30, 4),
                pct_0_to_30d=round(pct_0, 4), pct_after_anchor=round(pct_aft, 4),
                median_lead_days=float(median) if median is not None else None,
                coverage=coverage, suspicion_score=score, evidence=evidence,
            ))
        return profiles

    def _build_event_profile(
        self, col: str, ds_name: str, anchor_col_name: str,
        lead_series: native_pd.Series, n_entities: int,
    ) -> FieldLeadLagProfile:
        valid = lead_series.dropna()
        n = len(valid)
        if n == 0:
            return self._empty_event_profile(col, ds_name, anchor_col_name)

        pct_90 = float((valid > 90).sum() / n)
        pct_30 = float(((valid >= 30) & (valid <= 90)).sum() / n)
        pct_0 = float(((valid >= 0) & (valid < 30)).sum() / n)
        pct_aft = float((valid < 0).sum() / n)
        median = float(valid.median())
        score = round(pct_0 + pct_aft, 4)
        coverage = round(n / n_entities, 4) if n_entities > 0 else 0.0
        evidence = _event_evidence(col, pct_90, pct_30, pct_0, pct_aft, median, score)
        return FieldLeadLagProfile(
            field_name=col, source_dataset=ds_name,
            analysis_tier="account_level", anchor_type=anchor_col_name,
            pct_before_90d=round(pct_90, 4), pct_30_to_90d=round(pct_30, 4),
            pct_0_to_30d=round(pct_0, 4), pct_after_anchor=round(pct_aft, 4),
            median_lead_days=median, coverage=coverage,
            suspicion_score=score, evidence=evidence,
        )

    def _empty_event_profile(self, col: str, ds_name: str, anchor_col_name: str) -> FieldLeadLagProfile:
        return FieldLeadLagProfile(
            field_name=col, source_dataset=ds_name,
            analysis_tier="account_level", anchor_type=anchor_col_name,
            coverage=0.0, suspicion_score=0.0,
            evidence=[f"{col}: no non-null values found for terminated entities"],
        )

    # -----------------------------------------------------------------------
    # Second pass: value-distribution analysis for low-cardinality fields
    # -----------------------------------------------------------------------

    def _augment_with_value_distributions(
        self,
        profiles: list[FieldLeadLagProfile],
        service_unit_df: DataFrame,
        probe_dfs: dict[str, DataFrame],
        anchors: DataFrame,
        su_cfg: ServiceUnitConfig,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        by_dataset: dict[str, list[FieldLeadLagProfile]] = {}
        for p in profiles:
            by_dataset.setdefault(p.source_dataset, []).append(p)

        for ds_name, ds_profiles in by_dataset.items():
            df = service_unit_df if ds_name == su_cfg.dataset_name else probe_dfs.get(ds_name)
            if df is None:
                continue
            col_names = [p.field_name for p in ds_profiles if p.field_name in df.columns]
            if not col_names:
                continue
            is_self = ds_name == su_cfg.dataset_name
            if _is_spark_pandas(df):
                self._value_dist_pass_spark(
                    ds_profiles, df, anchors, su_cfg, is_self, col_names, ds_name, progress_fn,
                )
            else:
                self._value_dist_pass_pandas(
                    ds_profiles, df, anchors, su_cfg, is_self, col_names, ds_name, progress_fn,
                )

    def _value_dist_pass_pandas(
        self,
        ds_profiles: list[FieldLeadLagProfile],
        df: DataFrame, anchors: DataFrame, su_cfg: ServiceUnitConfig,
        is_self: bool, col_names: list[str], ds_name: str,
        progress_fn: Optional[Callable[[str], None]],
    ) -> None:
        profile_map = {p.field_name: p for p in ds_profiles}
        present = [c for c in col_names if c in df.columns]
        if not present:
            return

        if progress_fn:
            progress_fn(f"  Value-distribution: computing cardinality for {ds_name} ({len(present)} columns)...")
        cards = df[present].nunique()
        cards = {c: int(cards[c]) for c in present}
        for c in present:
            if c in profile_map:
                profile_map[c].cardinality = cards[c]

        low_card = [c for c, n in cards.items() if n <= _MAX_VALUE_DISTRIBUTION_CARDINALITY]
        if not low_card:
            return
        if progress_fn:
            progress_fn(f"  Value-distribution: {ds_name} ({len(low_card)} low-cardinality fields)")

        if is_self:
            if su_cfg.status_column and su_cfg.terminated_statuses:
                term_mask = df[su_cfg.status_column].isin(su_cfg.terminated_statuses)
            else:
                term_mask = df[su_cfg.anchor_date_column].notna()
            active_mask = ~term_mask
            src = df
        else:
            entity_col = su_cfg.entity_column
            if entity_col not in df.columns:
                return
            src = df[[entity_col] + low_card].merge(
                anchors[[entity_col, "is_fully_terminated"]], on=entity_col, how="left",
            )
            term_mask = src["is_fully_terminated"] == True  # noqa: E712
            active_mask = src["is_fully_terminated"].isna()

        n_term, n_active = int(term_mask.sum()), int(active_mask.sum())
        if n_term == 0:
            return

        for col in low_card:
            p = profile_map.get(col)
            if p is None:
                continue
            term_vc = src.loc[term_mask, col].value_counts(dropna=True).to_dict()
            active_vc = src.loc[active_mask, col].value_counts(dropna=True).to_dict()
            excl, score, evidence = _value_distribution_score(
                col, term_vc, active_vc, n_term, n_active,
            )
            p.value_suspicion_score = round(score, 4)
            p.terminated_exclusive_values = excl
            p.evidence.extend(evidence)
            if score > p.suspicion_score:
                p.suspicion_score = round(score, 4)

    def _value_dist_pass_spark(  # pragma: no cover
        self,
        ds_profiles: list[FieldLeadLagProfile],
        df: DataFrame, anchors: DataFrame, su_cfg: ServiceUnitConfig,
        is_self: bool, col_names: list[str], ds_name: str,
        progress_fn: Optional[Callable[[str], None]],
    ) -> None:
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat import as_spark_df

        spark_df = as_spark_df(df)
        profile_map = {p.field_name: p for p in ds_profiles}
        schema_cols = {f.name for f in spark_df.schema.fields}
        present = [c for c in col_names if c in schema_cols]
        if not present:
            return

        # Group marker: "T" = terminated, "A" = active, NULL = excluded
        if is_self:
            status_col = su_cfg.status_column
            if status_col and su_cfg.terminated_statuses:
                grp = F.when(F.col(status_col).isin(su_cfg.terminated_statuses), F.lit("T")).otherwise(F.lit("A"))
            else:
                grp = F.when(F.col(su_cfg.anchor_date_column).isNotNull(), F.lit("T")).otherwise(F.lit("A"))
            base = spark_df.withColumn("_grp", grp)
        else:
            entity_col = su_cfg.entity_column
            anchors_spark = F.broadcast(as_spark_df(anchors[[entity_col, "is_fully_terminated"]]))
            base = spark_df.join(anchors_spark, on=entity_col, how="left").withColumn(
                "_grp",
                F.when(F.col("is_fully_terminated") == True, F.lit("T"))  # noqa: E712
                .when(F.col("is_fully_terminated").isNull(), F.lit("A")),
            )
        base = base.filter(F.col("_grp").isNotNull()).cache()

        # Batch cardinality via unpivot+groupBy (safe for 100+ columns)
        stack_args = ", ".join(f"'{c}', cast(`{c}` as string)" for c in present)
        stacked_all = base.select(F.expr(f"stack({len(present)}, {stack_args}) as (__cn, __cv)"))
        cd_rows = stacked_all.groupBy("__cn").agg(
            F.approx_count_distinct("__cv").alias("cd"),
        ).collect()
        cards = {r["__cn"]: r["cd"] for r in cd_rows}
        for c, n in cards.items():
            if c in profile_map:
                profile_map[c].cardinality = n

        low_card = [c for c, n in cards.items() if n <= _MAX_VALUE_DISTRIBUTION_CARDINALITY]
        if not low_card:
            base.unpersist()
            return

        if progress_fn:
            progress_fn(f"  Value-distribution: {ds_name} ({len(low_card)} low-cardinality fields)")

        # Group counts (2 expressions — fine as wide agg)
        counts_row = base.groupBy("_grp").count().collect()
        grp_counts = {r["_grp"]: r["count"] for r in counts_row}
        n_term, n_active = grp_counts.get("T", 0), grp_counts.get("A", 0)
        if n_term == 0:
            base.unpersist()
            return

        # Unpivot low-cardinality columns + single groupBy for all value distributions
        lc_args = ", ".join(f"'{c}', cast(`{c}` as string)" for c in low_card)
        stacked = base.select("_grp", F.expr(f"stack({len(low_card)}, {lc_args}) as (__cn, __cv)"))
        rows = (
            stacked.filter(F.col("__cv").isNotNull())
            .groupBy("__cn", "__cv", "_grp").count()
            .collect()
        )
        base.unpersist()

        # Reconstruct per-column value counts
        term_vcs: dict[str, dict] = {c: {} for c in low_card}
        active_vcs: dict[str, dict] = {c: {} for c in low_card}
        for row in rows:
            target = term_vcs if row["_grp"] == "T" else active_vcs
            target[row["__cn"]][row["__cv"]] = row["count"]

        for col in low_card:
            p = profile_map.get(col)
            if p is None:
                continue
            excl, score, evidence = _value_distribution_score(
                col, term_vcs[col], active_vcs[col], n_term, n_active,
            )
            p.value_suspicion_score = round(score, 4)
            p.terminated_exclusive_values = excl
            p.evidence.extend(evidence)
            if score > p.suspicion_score:
                p.suspicion_score = round(score, 4)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _population_suspicion_score(ratio: float, pct_term: float, pct_act: float) -> float:
    if pct_term == 0.0 and pct_act == 0.0:
        return 0.0
    if math.isinf(ratio):
        return min(1.0, pct_term + 0.5)
    if ratio > 2.0:
        return min(1.0, 0.5 + (ratio - 2.0) * 0.1)
    if ratio > 1.5:
        return 0.3 + (ratio - 1.5) * 0.4
    return max(0.0, (ratio - 1.0) * 0.3)


def _population_evidence(
    col: str, pct_term: float, pct_act: float, ratio: float, score: float,
) -> list[str]:
    lines = [
        f"{col}: {pct_term:.1%} non-null for terminated, {pct_act:.1%} for active",
    ]
    if math.isinf(ratio):
        lines.append("Population ratio: Inf (only populated for terminated)")
    else:
        lines.append(f"Population ratio: {ratio:.2f}")
    if score > 0.7:
        lines.append(f"Suspicion score {score:.2f} exceeds exclude threshold")
    elif score > 0.5:
        lines.append(f"Suspicion score {score:.2f} flagged for investigation")
    return lines


def _event_evidence(
    col: str, pct_90: float, pct_30: float, pct_0: float, pct_aft: float,
    median: Optional[float], score: float,
) -> list[str]:
    lines = [
        f"{col}: {pct_90:.1%} >90d before, {pct_30:.1%} 30-90d, {pct_0:.1%} 0-30d, {pct_aft:.1%} after anchor",
    ]
    if median is not None:
        lines.append(f"Median lead days: {median:.1f}")
    if score > 0.7:
        lines.append(f"Suspicion score {score:.2f} exceeds exclude threshold")
    elif score > 0.5:
        lines.append(f"Suspicion score {score:.2f} flagged for investigation")
    return lines


def _value_distribution_score(
    col: str, term_counts: dict, active_counts: dict, n_term: int, n_active: int,
    min_count: int = 10,
) -> tuple[list[str], float, list[str]]:
    """Score a field by value distribution between terminated and active groups.

    Returns ``(terminated_exclusive_values, score, evidence_lines)``.
    """
    all_values = set(term_counts.keys()) | set(active_counts.keys())
    if not all_values or n_term == 0:
        return [], 0.0, []

    total = n_term + n_active
    base_rate = n_term / total if total > 0 else 0.5
    terminated_exclusive: list[str] = []
    max_excess = 0.0

    for val in all_values:
        ct, ca = term_counts.get(val, 0), active_counts.get(val, 0)
        if ct + ca < min_count:
            continue
        term_share = ct / (ct + ca)
        if ca == 0 and ct >= min_count:
            terminated_exclusive.append(str(val))
        if base_rate < 1.0:
            max_excess = max(max_excess, max(0.0, (term_share - base_rate) / (1.0 - base_rate)))

    if terminated_exclusive:
        excl_coverage = sum(term_counts.get(v, 0) for v in terminated_exclusive) / n_term
        score = min(1.0, 0.5 + excl_coverage)
    elif max_excess > 0.3:
        score = min(1.0, max_excess * 0.6)
    else:
        score = 0.0

    evidence: list[str] = []
    if terminated_exclusive:
        vals_str = ", ".join(terminated_exclusive[:5])
        if len(terminated_exclusive) > 5:
            vals_str += f" (+{len(terminated_exclusive) - 5} more)"
        evidence.append(f"{col}: {len(terminated_exclusive)} terminated-exclusive value(s): {vals_str}")
    elif max_excess > 0.3:
        evidence.append(f"{col}: value-distribution skew toward terminated = {max_excess:.2f}")
    return terminated_exclusive, round(score, 4), evidence


def _compute_date_range(df: DataFrame, col: str) -> tuple[Optional[str], Optional[str]]:
    """Return (min_date_str, max_date_str) for a date/timestamp column."""
    try:
        s = df[col].dropna()
        if len(s) == 0:
            return None, None
        lo, hi = s.min(), s.max()
        return (str(lo)[:10] if lo is not None else None, str(hi)[:10] if hi is not None else None)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_audit_results(result: FieldAvailabilityAuditResult) -> None:
    print(f"\n{'='*70}")
    print("Field Availability Audit Results")
    print(f"{'='*70}")

    su = result.config.service_unit
    print(f"\nService unit: {su.dataset_name}")
    print(f"  Anchor column: {su.anchor_date_column}")
    if su.status_column:
        print(f"  Status column: {su.status_column} (terminated: {', '.join(su.terminated_statuses)})")
    if su.start_date_column:
        print(f"  Start column:  {su.start_date_column}")
    if result.config.target_columns:
        print(f"  Target/label (excluded from audit): {', '.join(result.config.target_columns)}")

    print(f"\nAccounts: {result.total_accounts} total | "
          f"{result.fully_terminated_accounts} fully terminated | "
          f"{result.partially_terminated_accounts} partially terminated | "
          f"{result.active_accounts} active")
    print(f"Service units terminated: {result.total_terminated_units}")

    if result.dataset_date_info:
        print("\nDataset date columns:")
        for d in result.dataset_date_info:
            tc = d.time_column or "—"
            dates = f"{d.min_date} -> {d.max_date}" if d.min_date else "—"
            print(f"  {d.dataset_name:<24} {tc:<28} {dates:<26} {d.record_count:>8} rows  ({d.linkage_tier})")

    if result.field_profiles:
        total = len(result.field_profiles)
        n_always_pop = sum(
            1 for p in result.field_profiles
            if (p.pct_non_null_terminated or 0) > 0.95 and (p.pct_non_null_active or 0) > 0.95
        )
        print(f"\nNull-rate analysis: {total} fields")
        if n_always_pop > 0:
            print(f"  Always-populated (>=95% both groups): {n_always_pop}")
    print()

    if not result.field_profiles:
        print("No fields audited.")
        return

    sorted_profiles = sorted(result.field_profiles, key=lambda p: -p.suspicion_score)
    header = f"{'Field':<35} {'Dataset':<20} {'Tier':<15} {'Score':>6} {'Rec':<12}"
    print(header)
    print("-" * len(header))
    for p in sorted_profiles:
        print(f"{p.field_name:<35} {p.source_dataset:<20} {p.analysis_tier:<15} "
              f"{p.suspicion_score:>6.2f} {p.recommendation:<12}")

    if result.recommended_exclusions:
        print(f"\nRecommended exclusions ({len(result.recommended_exclusions)}):")
        for f_name in result.recommended_exclusions:
            print(f"  - {f_name}")
    if result.suspicious_fields:
        investigate_only = [f for f in result.suspicious_fields if f not in result.recommended_exclusions]
        if investigate_only:
            print(f"\nFields to investigate ({len(investigate_only)}):")
            for f_name in investigate_only:
                print(f"  - {f_name}")


def _display_value_distribution_results(result: FieldAvailabilityAuditResult) -> None:
    n_vd = sum(1 for p in result.field_profiles if p.value_suspicion_score is not None)
    if n_vd == 0:
        return

    print(f"\n{'='*70}")
    print("Value-Distribution Analysis")
    print(f"{'='*70}")
    print(f"Fields analyzed: {n_vd} (cardinality <= {_MAX_VALUE_DISTRIBUTION_CARDINALITY})")

    vd_flagged = [
        p for p in result.field_profiles
        if p.value_suspicion_score is not None
        and p.value_suspicion_score > result.config.suspicion_threshold
    ]
    if vd_flagged:
        print(f"\nSuspicious value patterns ({len(vd_flagged)} fields):")
        for p in sorted(vd_flagged, key=lambda x: -(x.value_suspicion_score or 0)):
            detail = ""
            if p.terminated_exclusive_values:
                vals = ", ".join(p.terminated_exclusive_values[:5])
                detail = f" -- terminated-exclusive: {vals}"
            print(f"  {p.field_name:<35} {p.source_dataset:<20} "
                  f"vd_score={p.value_suspicion_score:.2f} -> {p.recommendation}{detail}")
    else:
        print("\nNo suspicious value patterns found.")

    if result.recommended_exclusions:
        print(f"\nFinal exclusions ({len(result.recommended_exclusions)}):")
        for f_name in result.recommended_exclusions:
            print(f"  - {f_name}")
    invest = [f for f in result.suspicious_fields if f not in result.recommended_exclusions]
    if invest:
        print(f"\nFinal investigation list ({len(invest)}):")
        for f_name in invest:
            print(f"  - {f_name}")


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def run_field_availability_audit(
    context: "ProjectContext",
    loaded_frames: dict[str, DataFrame],
    namespace: "Optional[RunNamespace]" = None,
    service_unit_dataset: Optional[str] = None,
    service_unit_id_column: Optional[str] = None,
    service_unit_anchor_column: Optional[str] = None,
    service_unit_status_column: Optional[str] = None,
    service_unit_terminated_statuses: Optional[list[str]] = None,
    service_unit_start_column: Optional[str] = None,
    min_terminated_units: int = 50,
    suspicion_threshold: float = 0.5,
    lead_buckets_days: Optional[list[int]] = None,
    additional_exclusions: Optional[list[str]] = None,
) -> FieldAvailabilityAuditResult:
    if service_unit_dataset and service_unit_id_column and service_unit_anchor_column:
        su_config = ServiceUnitConfig(
            dataset_name=service_unit_dataset,
            unit_id_column=service_unit_id_column,
            entity_column=context.entity_column or "",
            anchor_date_column=service_unit_anchor_column,
            status_column=service_unit_status_column,
            terminated_statuses=service_unit_terminated_statuses or [],
            start_date_column=service_unit_start_column,
        )
    else:
        su_config = detect_service_unit(context, loaded_frames)
        if su_config is None:
            raise ValueError("No service unit dataset detected. Provide explicit overrides.")

    if service_unit_dataset and not service_unit_id_column:
        detected = detect_service_unit(context, loaded_frames)
        if detected and detected.dataset_name == service_unit_dataset:
            su_config = detected

    print(f"Service unit: {su_config.dataset_name} "
          f"(id={su_config.unit_id_column}, anchor={su_config.anchor_date_column})")

    linkage = classify_dataset_linkage(context, su_config, loaded_frames)
    print(f"Dataset linkage: {', '.join(f'{k}={v.tier}' for k, v in linkage.items())}")

    granularities = {}
    time_columns = {}
    for ds_name, entry in context.datasets.items():
        if entry.granularity:
            granularities[ds_name] = entry.granularity.value
        if entry.time_column:
            time_columns[ds_name] = entry.time_column

    target_cols = sorted({
        col for df in loaded_frames.values() for col in df.columns
        if col.lower() in _TARGET_LIKE_PATTERNS
    })
    if target_cols:
        print(f"Auto-detected target/label columns (excluded from audit): {', '.join(target_cols)}")

    audit_config = FieldAvailabilityAuditConfig(
        service_unit=su_config,
        probe_datasets=list(loaded_frames.keys()),
        lead_buckets_days=lead_buckets_days or [0, 30, 90],
        min_terminated_units=min_terminated_units,
        suspicion_threshold=suspicion_threshold,
        target_columns=target_cols,
    )

    auditor = FieldAvailabilityAuditor(audit_config)
    result = auditor.run(
        service_unit_df=loaded_frames[su_config.dataset_name],
        probe_dfs=loaded_frames,
        dataset_linkage=linkage,
        dataset_granularities=granularities,
        dataset_time_columns=time_columns,
        progress_fn=print,
    )

    # Compute dataset date info
    date_infos = []
    for ds_name, df in loaded_frames.items():
        lnk = linkage.get(ds_name)
        tier = lnk.tier if lnk else ("self" if ds_name == su_config.dataset_name else "")
        time_col = time_columns.get(ds_name)
        if ds_name == su_config.dataset_name:
            time_col = su_config.anchor_date_column
        if time_col and time_col in df.columns:
            min_d, max_d = _compute_date_range(df, time_col)
        else:
            time_col, min_d, max_d = None, None, None
        date_infos.append(DatasetDateInfo(
            dataset_name=ds_name, time_column=time_col,
            min_date=min_d, max_date=max_d,
            linkage_tier=tier, record_count=len(df),
        ))
    result.dataset_date_info = date_infos

    if additional_exclusions:
        for col in additional_exclusions:
            if col not in result.recommended_exclusions:
                result.recommended_exclusions.append(col)
            if col not in result.suspicious_fields:
                result.suspicious_fields.append(col)

    _display_audit_results(result)

    # Value-distribution pass (runs after main report is visible)
    auditor.run_value_distribution_pass(
        result, loaded_frames[su_config.dataset_name], loaded_frames, progress_fn=print,
    )
    _display_value_distribution_results(result)

    if namespace:
        output_dir = namespace.field_availability_audit_dir
        paths = result.save(output_dir)
        print(f"\nAudit saved to: {output_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    return result
