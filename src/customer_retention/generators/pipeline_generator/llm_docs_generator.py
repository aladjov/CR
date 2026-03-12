from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
from customer_retention.analysis.auto_explorer.project_context import ProjectContext
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.naming import composite_name

from .findings_parser import FindingsParser


def _enum_val(obj: Any) -> str:
    return obj.value if hasattr(obj, "value") else str(obj)


def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    return " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


def _write_md(path: Path, lines: List[str]) -> Path:
    path.write_text("\n".join(lines))
    return path


def _append_rec_sections(lines: List[str], sections: list[tuple[str, list]]) -> None:
    for heading, recs in sections:
        if not recs:
            continue
        lines.append(f"## {heading}\n")
        for rec in recs:
            lines.append(f"- `{rec.target_column}`: **{rec.action}**{_format_params(rec.parameters)} — {rec.rationale}")
        lines.append("")


class LLMDocsGenerator:

    def __init__(
        self, findings_dir: str, output_dir: str, pipeline_name: str,
        namespace: RunNamespace | None = None, intent: Any = None,
    ):
        self._output_dir = Path(output_dir) / "docs"
        self._pipeline_name = pipeline_name
        self._namespace = namespace
        self._parser = FindingsParser(findings_dir, namespace=namespace, intent=intent)

        self._config = None
        self._project_context: ProjectContext | None = None
        self._registry: RecommendationRegistry | None = None
        self._multi_dataset: dict | None = None
        self._per_dataset: Dict[str, ExplorationFindings] = {}
        self._event_sources: List[str] = []
        self._entity_sources: List[str] = []
        self._cn: str = pipeline_name

    def generate(self) -> List[Path]:
        self._load_context()
        for subdir in ("landing", "bronze", "silver", "gold", "training"):
            (self._output_dir / subdir).mkdir(parents=True, exist_ok=True)

        generated: List[Path] = [self._write_overview(), self._write_config_doc()]
        for name, findings in self._per_dataset.items():
            generated.append(self._write_landing(name, findings))
            generated.append(self._write_bronze(name, findings))
        generated.extend([self._write_silver(), self._write_gold(), self._write_training()])
        return generated

    def _load_context(self) -> None:
        self._config = self._parser.parse()
        self._config.name = self._pipeline_name

        source_names = [
            f"{s.name}_aggregated" if s.is_event_level else s.name
            for s in self._config.sources if not s.excluded
        ]
        self._cn = composite_name(source_names) if source_names else self._pipeline_name
        self._config.composite_name = self._cn

        if self._namespace and self._namespace.project_context_path.exists():
            self._project_context = ProjectContext.load(self._namespace.project_context_path)

        if self._namespace:
            for fp in self._namespace.discover_all_findings(prefer_aggregated=False):
                self._per_dataset[fp.parent.parent.name] = ExplorationFindings.load(str(fp))

        if self._namespace and self._namespace.merged_recommendations_path.exists():
            with open(self._namespace.merged_recommendations_path) as f:
                self._registry = RecommendationRegistry.from_dict(yaml.safe_load(f))

        if self._namespace and self._namespace.multi_dataset_findings_path.exists():
            with open(self._namespace.multi_dataset_findings_path) as f:
                self._multi_dataset = yaml.safe_load(f)

        self._classify_sources()

    def _classify_sources(self) -> None:
        if self._multi_dataset:
            self._event_sources = self._multi_dataset.get("event_datasets", [])
            excluded = set(self._multi_dataset.get("excluded_datasets", []))
            self._entity_sources = [
                s for s in self._multi_dataset.get("datasets", {})
                if s not in self._event_sources and s not in excluded
            ]
        elif self._per_dataset:
            for name, f in self._per_dataset.items():
                (self._event_sources if f.is_time_series else self._entity_sources).append(name)

    def _write_overview(self) -> Path:
        lines = [f"# {self._pipeline_name} — Pipeline Overview\n"]
        self._append_objectives(lines)
        self._append_datasets_table(lines)
        self._append_relationships(lines)
        lines.append(f"**Composite name:** `{self._cn}`\n")
        lines.append(f"**Target column:** `{self._config.target_column or '(not set)'}`\n")
        return _write_md(self._output_dir / "overview.md", lines)

    def _write_config_doc(self) -> Path:
        lines = [f"# {self._pipeline_name} — Configuration\n"]
        lines.append("## Naming\n")
        lines.append(f"- **Composite name (CN):** `{self._cn}`")
        lines.append(f"- **Pipeline name:** `{self._pipeline_name}`")
        lines.append(f"- **Silver table:** `silver_featureset_{self._cn}`")
        lines.append(f"- **Gold table:** `gold_features_{self._cn}`")
        lines.append("")
        self._append_temporal_config(lines)
        self._append_aggregation_windows(lines)
        lines.append("## Source Classification\n")
        lines.append(f"- **Entity sources:** {self._entity_sources or ['(none)']}")
        lines.append(f"- **Event sources:** {self._event_sources or ['(none)']}")
        lines.append("")
        return _write_md(self._output_dir / "config.md", lines)

    def _write_landing(self, name: str, findings: ExplorationFindings) -> Path:
        lines = [f"# Landing — {name}\n"]
        lines.append(f"**Source:** `{findings.source_path}`  ")
        lines.append(f"**Format:** {findings.source_format}  ")
        lines.append(f"**Rows:** {findings.row_count:,} | **Columns:** {findings.column_count}  ")
        lines.append(f"**Quality score:** {findings.overall_quality_score:.1f}/100\n")
        self._append_temporal_profile(lines, findings)
        self._append_column_schema(lines, findings)
        self._append_issues(lines, findings)
        return _write_md(self._output_dir / "landing" / f"landing_{name.replace(' ', '_').lower()}.md", lines)

    def _write_bronze(self, name: str, findings: ExplorationFindings) -> Path:
        is_event = name in self._event_sources
        prefix = "bronze_event" if is_event else "bronze_entity"
        lines = [f"# Bronze — {prefix}_{name}\n"]
        if is_event:
            lines.append("**Type:** Event source (produces two scripts: event + entity aggregated)\n")
            if findings.time_series_metadata and findings.time_series_metadata.suggested_aggregations:
                lines.append("## Aggregation Windows\n")
                for agg in findings.time_series_metadata.suggested_aggregations:
                    lines.append(f"- `{agg}`")
                lines.append("")
        else:
            lines.append("**Type:** Entity source (single bronze script)\n")
        self._append_source_recommendations(lines, name)
        self._append_column_cleaning(lines, findings)
        return _write_md(self._output_dir / "bronze" / f"{prefix}_{name.replace(' ', '_').lower()}.md", lines)

    def _write_silver(self) -> Path:
        lines = [f"# Silver — silver_featureset_{self._cn}\n"]
        if self._multi_dataset:
            primary = self._multi_dataset.get("primary_entity_dataset")
            if primary:
                lines.append(f"**Primary entity dataset:** `{primary}`\n")
        lines.append("All entity-grain and aggregated event datasets are joined into a single")
        lines.append(f"silver table `silver_featureset_{self._cn}` keyed by `entity_id`.\n")
        self._append_silver_recommendations(lines)
        if self._multi_dataset and self._multi_dataset.get("relationships"):
            lines.append("## Dataset Relationships\n")
            for rel in self._multi_dataset["relationships"]:
                lines.append(
                    f"- {rel.get('left_dataset')} → {rel.get('right_dataset')}: "
                    f"`{rel.get('left_columns')}` = `{rel.get('right_columns')}` "
                    f"({rel.get('relationship_type', '?')})"
                )
            lines.append("")
        return _write_md(self._output_dir / "silver" / f"silver_featureset_{self._cn}.md", lines)

    def _write_gold(self) -> Path:
        lines = [f"# Gold — gold_features_{self._cn}\n"]
        lines.append(f"**Target column:** `{self._config.target_column or '(not set)'}`\n")
        self._append_gold_recommendations(lines)
        self._append_feature_opportunities(lines)
        return _write_md(self._output_dir / "gold" / f"gold_features_{self._cn}.md", lines)

    def _write_training(self) -> Path:
        lines = [f"# Training — {self._pipeline_name}\n"]
        target_type = next((f.target_type for f in self._per_dataset.values() if f.target_type), "unknown")
        lines.append(f"**Target:** `{self._config.target_column}` ({target_type})")
        modeling_ready = all(not f.blocking_issues for f in self._per_dataset.values())
        lines.append(f"**Modeling ready:** {modeling_ready}\n")
        self._append_training_config(lines)
        lines.append("## Evaluation Metrics\n")
        for metric in ("ROC-AUC (primary)", "PR-AUC", "Accuracy", "F1 Score", "Weighted Precision / Recall"):
            lines.append(f"- **{metric}**")
        lines.append("")
        self._append_modeling_recommendations(lines)
        self._append_blocking_issues(lines)
        return _write_md(self._output_dir / "training" / "ml_experiment.md", lines)

    def _append_objectives(self, lines: List[str]) -> None:
        if not self._project_context:
            return
        lines.append("## Prediction Objective\n")
        for obj in self._project_context.active_objectives:
            anchor = _enum_val(obj.effective_anchor) if obj.effective_anchor else "auto"
            lines.append(f"- **{_enum_val(obj.objective)}** (priority: {_enum_val(obj.priority)}, anchor: {anchor})")
            for k, v in (obj.parameters or {}).items():
                lines.append(f"  - {k}: {v}")
        lines.append("")
        if self._project_context.temporal_posture:
            lines.append(f"**Temporal posture:** {_enum_val(self._project_context.temporal_posture)}\n")

    def _append_datasets_table(self, lines: List[str]) -> None:
        lines.append("## Datasets\n")
        lines.append("| Dataset | Granularity | Rows | Columns | Entity Column | Time Column |")
        lines.append("|---------|-------------|------|---------|---------------|-------------|")
        if self._multi_dataset and self._multi_dataset.get("datasets"):
            for ds_name, info in self._multi_dataset["datasets"].items():
                rows = f'{info["row_count"]:,}' if isinstance(info.get("row_count"), int) else "?"
                ent = info.get("entity_column") or "—"
                tc = info.get("time_column") or "—"
                lines.append(f"| {ds_name} | {info.get('granularity', 'unknown')} | {rows} | {info.get('column_count', '?')} | `{ent}` | `{tc}` |")
        else:
            first = next(iter(self._per_dataset.values()), None)
            if first:
                lines.append(f"| {self._pipeline_name} | — | {first.row_count:,} | {first.column_count} | — | — |")
        lines.append("")

    def _append_relationships(self, lines: List[str]) -> None:
        if not (self._multi_dataset and self._multi_dataset.get("relationships")):
            return
        lines.append("## Relationships\n")
        for rel in self._multi_dataset["relationships"]:
            conf = rel.get("confidence", 0)
            lines.append(
                f"- **{rel.get('left_dataset', '?')}** ↔ **{rel.get('right_dataset', '?')}** "
                f"on `{rel.get('left_columns', [])}` = `{rel.get('right_columns', [])}` "
                f"({rel.get('relationship_type', '?')}, confidence: {conf:.0%})"
            )
        lines.append("")

    def _append_temporal_config(self, lines: List[str]) -> None:
        if not (self._project_context and self._project_context.intent):
            return
        intent = self._project_context.intent
        lines.append("## Temporal Configuration\n")
        lines.append(f"- **Prediction horizons:** {intent.prediction_horizons} days")
        lines.append(f"- **Observation window:** {intent.observation_window_days} days")
        lines.append(f"- **Label window:** {intent.label_window_days} days")
        lines.append(f"- **Purge gap:** {intent.purge_gap_days} days")
        lines.append(f"- **Recent window:** {intent.recent_window_days} days")
        lines.append(f"- **Cadence:** {_enum_val(intent.cadence_interval)}")
        lines.append(f"- **Split strategy:** {_enum_val(intent.split_strategy)}")
        lines.append(f"- **Temporal split:** {intent.temporal_split}")
        if intent.history_upper_limit:
            lines.append(f"- **History upper limit:** {intent.history_upper_limit}")
        lines.append("")

    def _append_aggregation_windows(self, lines: List[str]) -> None:
        if not (self._multi_dataset and self._multi_dataset.get("aggregation_windows")):
            return
        lines.append("## Aggregation Windows\n")
        for w in self._multi_dataset["aggregation_windows"]:
            lines.append(f"- `{w}`")
        lines.append("")

    def _append_temporal_profile(self, lines: List[str], findings: ExplorationFindings) -> None:
        ts = findings.time_series_metadata
        if not ts:
            return
        lines.append("## Temporal Profile\n")
        lines.append(f"- **Granularity:** {_enum_val(ts.granularity)}")
        for label, val in [
            ("Entity column", ts.entity_column), ("Time column", ts.time_column),
        ]:
            if val:
                lines.append(f"- **{label}:** `{val}`")
        if ts.unique_entities:
            lines.append(f"- **Unique entities:** {ts.unique_entities:,}")
        if ts.avg_events_per_entity:
            lines.append(f"- **Avg events/entity:** {ts.avg_events_per_entity:.1f}")
        if ts.time_span_days:
            lines.append(f"- **Time span:** {ts.time_span_days} days")
        for label, val in [
            ("Drift risk", ts.drift_risk_level),
            ("Segmentation advisory", ts.temporal_segmentation_advisory),
        ]:
            if val:
                lines.append(f"- **{label}:** {val}")
        lines.append("")

    def _append_column_schema(self, lines: List[str], findings: ExplorationFindings) -> None:
        lines.append("## Column Schema\n")
        lines.append("| Column | Type | Nulls % | Unique | Quality |")
        lines.append("|--------|------|---------|--------|---------|")
        for col_name, col in findings.columns.items():
            null_pct = col.universal_metrics.get("null_percentage", 0)
            uniq = col.universal_metrics.get("distinct_count", "?")
            lines.append(f"| `{col_name}` | {col.inferred_type.value} | {null_pct:.1f}% | {uniq} | {col.quality_score:.0f} |")
        lines.append("")

    def _append_issues(self, lines: List[str], findings: ExplorationFindings) -> None:
        if findings.critical_issues:
            lines.append("## Critical Issues\n")
            for issue in findings.critical_issues:
                lines.append(f"- {issue}")
            lines.append("")
        if findings.warnings:
            lines.append("## Warnings\n")
            for w in findings.warnings[:10]:
                lines.append(f"- {w}")
            if len(findings.warnings) > 10:
                lines.append(f"- ... and {len(findings.warnings) - 10} more")
            lines.append("")
        if findings.excluded_leaking_features:
            lines.append("## Excluded Features (Leakage)\n")
            for feat in findings.excluded_leaking_features:
                lines.append(f"- `{feat}`")
            lines.append("")

    def _append_source_recommendations(self, lines: List[str], name: str) -> None:
        if not (self._registry and self._registry.sources and name in self._registry.sources):
            return
        src = self._registry.sources[name]
        _append_rec_sections(lines, [
            ("Null Handling", src.null_handling), ("Outlier Handling", src.outlier_handling),
            ("Type Conversions", src.type_conversions), ("Deduplication", src.deduplication),
            ("Filtering", src.filtering), ("Text Processing", src.text_processing),
            ("Modeling Strategy", src.modeling_strategy),
        ])

    def _append_column_cleaning(self, lines: List[str], findings: ExplorationFindings) -> None:
        cleaning_cols = [c for c in findings.columns.values() if c.cleaning_needed and c.cleaning_recommendations]
        if not cleaning_cols:
            return
        lines.append("## Column-Level Cleaning\n")
        for col in cleaning_cols:
            lines.append(f"### `{col.name}`")
            for rec in col.cleaning_recommendations:
                lines.append(f"- {rec}")
            lines.append("")

    def _append_silver_recommendations(self, lines: List[str]) -> None:
        if not (self._registry and self._registry.silver):
            return
        silver = self._registry.silver
        if silver.entity_column:
            lines.append(f"**Entity column:** `{silver.entity_column}`")
        if silver.time_column:
            lines.append(f"**Time column:** `{silver.time_column}`")
        lines.append("")
        if silver.joins:
            lines.append("## Joins\n")
            for rec in silver.joins:
                left = rec.parameters.get("left_source", "?")
                right = rec.parameters.get("right_source", "?")
                lines.append(f"- **{left}** ⟷ **{right}** on `{rec.parameters.get('join_keys', [])}` — {rec.rationale}")
            lines.append("")
        if silver.aggregations:
            lines.append("## Aggregations\n")
            for rec in silver.aggregations:
                lines.append(f"- `{rec.target_column}`: {rec.action} (windows: {rec.parameters.get('windows', [])}) — {rec.rationale}")
            lines.append("")
        if silver.derived_columns:
            lines.append("## Derived Columns\n")
            for rec in silver.derived_columns:
                lines.append(f"- `{rec.target_column}`: `{rec.parameters.get('expression', rec.action)}` — {rec.rationale}")
            lines.append("")

    def _append_gold_recommendations(self, lines: List[str]) -> None:
        if not (self._registry and self._registry.gold):
            return
        gold = self._registry.gold
        for heading, recs in [("Encoding", gold.encoding), ("Scaling", gold.scaling),
                              ("Feature Selection", gold.feature_selection), ("Transformations", gold.transformations)]:
            if not recs:
                continue
            lines.append(f"## {heading}\n")
            for rec in recs:
                lines.append(f"- `{rec.target_column}`: **{rec.parameters.get('method', rec.action)}** — {rec.rationale}")
            lines.append("")

    def _append_feature_opportunities(self, lines: List[str]) -> None:
        findings = next(iter(self._per_dataset.values()), None)
        if not findings:
            return
        transform_cols = [(n, c) for n, c in findings.columns.items() if c.transformation_recommendations]
        if not transform_cols:
            return
        lines.append("## Feature Engineering Opportunities\n")
        for col_name, col in transform_cols[:20]:
            lines.append(f"### `{col_name}` ({col.inferred_type.value})")
            for rec in col.transformation_recommendations:
                lines.append(f"- {rec}")
            lines.append("")

    def _append_training_config(self, lines: List[str]) -> None:
        if not (self._project_context and self._project_context.intent):
            return
        intent = self._project_context.intent
        lines.append("## Temporal Split Configuration\n")
        lines.append(f"- **Strategy:** {_enum_val(intent.split_strategy)}")
        lines.append(f"- **Prediction horizons:** {intent.prediction_horizons}")
        lines.append(f"- **Observation window:** {intent.observation_window_days} days")
        lines.append(f"- **Label window:** {intent.label_window_days} days")
        lines.append(f"- **Purge gap:** {intent.purge_gap_days} days")
        lines.append("")
        lines.append("Split is **entity-aware + temporally purged** (never row-random).")
        lines.append("CV uses `TemporalEntitySplit` to prevent information leakage.\n")

    def _append_modeling_recommendations(self, lines: List[str]) -> None:
        if not self._registry:
            return
        model_recs = [
            r for r in self._registry.all_recommendations
            if r.category in ("model_type", "imbalance_strategy", "segmentation_strategy")
        ]
        if not model_recs:
            return
        lines.append("## Modeling Recommendations\n")
        for rec in model_recs:
            lines.append(f"- **{rec.category}**: {rec.action} — {rec.rationale}")
        lines.append("")

    def _append_blocking_issues(self, lines: List[str]) -> None:
        for f in self._per_dataset.values():
            if f.blocking_issues:
                lines.append("## Blocking Issues\n")
                for issue in f.blocking_issues:
                    lines.append(f"- {issue}")
                lines.append("")
                break
