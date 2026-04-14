from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from customer_retention.analysis.diagnostics.feature_provenance import parse_feature_provenance
from customer_retention.core.compat import native_pd


def attribute_feature_to_source(
    feature_name: str,
    source_columns: Dict[str, Iterable[str]],
) -> Tuple[str, str, str]:
    parsed = parse_feature_provenance(feature_name, source_columns)
    if parsed.is_resolved:
        return (
            parsed.source,
            parsed.base_column,
            parsed.family or "raw",
        )
    return ("unknown", feature_name, "raw")


def _match_opt_in(column: str, opt_in_bases: List[str]) -> bool:
    from customer_retention.generators.pipeline_generator.findings_parser import _matches_any_prefix

    prefixes: List[str] = []
    for base in opt_in_bases:
        prefixes.append(base)
        prefixes.append(f"{base}_")
    return _matches_any_prefix(column, prefixes)


def _match_excluded_leaking(column: str, leaking_bases: List[str]) -> bool:
    from customer_retention.generators.pipeline_generator.findings_parser import _matches_any_prefix

    prefixes = [f"{base}_" for base in leaking_bases]
    return _matches_any_prefix(column, prefixes)


def build_recommendation_audit(
    registry,
    opt_in: Dict[str, List[str]],
    excluded_leaking: Dict[str, List[str]],
    source_columns: Dict[str, Iterable[str]],
) -> native_pd.DataFrame:
    gold = getattr(registry, "gold", None)
    if gold is None:
        return native_pd.DataFrame()

    all_opt_in_bases = [b for cols in opt_in.values() for b in cols]
    all_excluded_bases = [b for cols in excluded_leaking.values() for b in cols]

    rows: List[Dict[str, object]] = []
    for rec in getattr(gold, "transformations", []) or []:
        target = rec.target_column
        source_ds, base, derivation = attribute_feature_to_source(target, source_columns)

        if rec.action == "zero_inflation_handling" and not _match_opt_in(target, all_opt_in_bases):
            status, reason = "gated_opt_in", "not in ZERO_INFLATION_OPT_IN"
        elif _match_excluded_leaking(target, all_excluded_bases):
            status, reason = "gated_excluded_leaking", "matched EXCLUDED_LEAKING_FEATURES"
        else:
            status, reason = "applied", ""

        rows.append({
            "action": rec.action,
            "target_column": target,
            "source_dataset": source_ds,
            "base_column": base,
            "derivation": derivation,
            "status": status,
            "reason": reason,
        })
    return native_pd.DataFrame(rows)


def build_recommendation_summary_by_action(audit: native_pd.DataFrame) -> native_pd.DataFrame:
    if len(audit) == 0:
        return native_pd.DataFrame(columns=["action", "total", "applied", "gated_opt_in", "gated_excluded_leaking"])

    grouped = audit.groupby("action")
    summary = grouped["status"].value_counts().unstack().fillna(0).astype(int)
    for col in ("applied", "gated_opt_in", "gated_excluded_leaking"):
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary[["applied", "gated_opt_in", "gated_excluded_leaking"]].sum(axis=1)
    summary = summary.reset_index()
    return summary[["action", "total", "applied", "gated_opt_in", "gated_excluded_leaking"]]


def build_dataset_column_ledger(
    findings_dict,
    raw_columns_by_dataset: Dict[str, List[str]],
    drop_columns_by_dataset: Dict[str, List[str]],
    auto_drop_text_by_dataset: Dict[str, List[str]],
) -> native_pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in findings_dict:
        raw = list(raw_columns_by_dataset.get(name, []))
        dropped = list(drop_columns_by_dataset.get(name, []))
        auto_drop = list(auto_drop_text_by_dataset.get(name, []))
        survived = len(set(raw) - set(dropped) - set(auto_drop))
        rows.append({
            "dataset": name,
            "raw_cols": len(raw),
            "dropped_DROP_COLUMNS": len(dropped),
            "dropped_AUTO_TEXT": len(auto_drop),
            "survived_to_bronze": survived,
        })
    return native_pd.DataFrame(rows)


def build_column_drop_register(
    drop_columns_by_dataset: Dict[str, List[str]],
    auto_drop_text_by_dataset: Dict[str, List[str]],
    audit_scores: Dict[str, Dict[str, float]] | None = None,
) -> native_pd.DataFrame:
    rows: List[Dict[str, object]] = []
    scores = audit_scores or {}
    for ds, cols in drop_columns_by_dataset.items():
        for col in cols:
            rows.append({
                "dataset": ds,
                "column": col,
                "reason": "DROP_COLUMNS",
                "audit_score": scores.get(ds, {}).get(col),
            })
    for ds, cols in auto_drop_text_by_dataset.items():
        for col in cols:
            rows.append({
                "dataset": ds,
                "column": col,
                "reason": "AUTO_DROP_TEXT",
                "audit_score": scores.get(ds, {}).get(col),
            })
    return native_pd.DataFrame(rows)


def build_nb05_drop_ledger(
    feature_profile: Dict[str, object],
    source_columns: Dict[str, Iterable[str]],
) -> native_pd.DataFrame:
    buckets = {
        "drop_zero_variance": "drop_zero_var",
        "drop_weak": "drop_weak",
        "drop_multicollinear": "drop_multicollinear",
        "drop_excluded_leaking": "drop_excluded_leaking",
        "kept": "survives",
    }
    per_source: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, Dict[str, int]] = {}
    for bucket_key, column_label in buckets.items():
        for feature in feature_profile.get(bucket_key, []) or []:
            source_ds, _, _ = attribute_feature_to_source(feature, source_columns)
            per_source.setdefault(source_ds, {}).setdefault(column_label, 0)
            per_source[source_ds][column_label] += 1
            totals.setdefault(source_ds, {}).setdefault("gold_features_in", 0)
            totals[source_ds]["gold_features_in"] += 1

    ordered_cols = ["dataset", "gold_features_in"] + list(buckets.values())
    rows: List[Dict[str, object]] = []
    for source_ds, counts in per_source.items():
        row: Dict[str, object] = {"dataset": source_ds}
        row["gold_features_in"] = totals[source_ds]["gold_features_in"]
        for label in buckets.values():
            row[label] = counts.get(label, 0)
        rows.append(row)
    df = native_pd.DataFrame(rows, columns=ordered_cols) if rows else native_pd.DataFrame(columns=ordered_cols)
    return df


def build_nb08_per_source_survival(
    stage_features: Dict[str, List[str]],
    source_columns: Dict[str, Iterable[str]],
) -> native_pd.DataFrame:
    stage_order = list(stage_features.keys())
    counts: Dict[str, Dict[str, int]] = {}
    for stage, features in stage_features.items():
        for feature in features or []:
            source_ds, _, _ = attribute_feature_to_source(feature, source_columns)
            counts.setdefault(source_ds, {}).setdefault(stage, 0)
            counts[source_ds][stage] += 1
    rows: List[Dict[str, object]] = []
    for source_ds, stage_counts in counts.items():
        row: Dict[str, object] = {"dataset": source_ds}
        for stage in stage_order:
            row[stage] = stage_counts.get(stage, 0)
        rows.append(row)
    return native_pd.DataFrame(rows, columns=["dataset"] + stage_order) if rows else native_pd.DataFrame(columns=["dataset"] + stage_order)


def build_nb08_top30_attribution(
    importance_scores: Dict[str, Dict[str, float]],
    source_columns: Dict[str, Iterable[str]],
    top_n: int = 30,
) -> native_pd.DataFrame:
    primary = next(iter(importance_scores.values()), {}) if importance_scores else {}
    ranked = sorted(primary.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows: List[Dict[str, object]] = []
    for feature, _ in ranked:
        source_ds, base, derivation = attribute_feature_to_source(feature, source_columns)
        row: Dict[str, object] = {
            "feature": feature,
            "source_dataset": source_ds,
            "base_column": base,
            "derivation": derivation,
        }
        for model_name, scores in importance_scores.items():
            row[f"{model_name}_importance"] = float(scores.get(feature, 0.0))
        rows.append(row)
    return native_pd.DataFrame(rows)


def write_diagnostic_yaml(df: native_pd.DataFrame, path: Path | str) -> None:
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records") if len(df) else []
    with p.open("w") as f:
        yaml.safe_dump(records, f, default_flow_style=False, sort_keys=False)
