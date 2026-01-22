from dataclasses import dataclass, field
from typing import Any, Dict, List

from customer_retention.core.config.column_config import ColumnType

from .findings import ExplorationFindings


@dataclass
class TargetRecommendation:
    column_name: str
    confidence: float
    rationale: str
    alternatives: List[str] = field(default_factory=list)
    target_type: str = "binary"


@dataclass
class FeatureRecommendation:
    source_column: str
    feature_name: str
    feature_type: str
    description: str
    priority: str = "medium"
    implementation_hint: str = ""


@dataclass
class CleaningRecommendation:
    column_name: str
    issue_type: str
    severity: str
    strategy: str
    description: str
    affected_rows: int = 0
    strategy_label: str = ""
    problem_impact: str = ""
    action_steps: List[str] = field(default_factory=list)


@dataclass
class TransformRecommendation:
    column_name: str
    transform_type: str
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"


class RecommendationEngine:
    TARGET_PATTERNS = ["target", "label", "churn", "churned", "outcome", "class", "flag"]
    SKEWNESS_THRESHOLD = 1.0
    OUTLIER_THRESHOLD = 5.0
    NULL_WARNING_THRESHOLD = 5.0
    NULL_CRITICAL_THRESHOLD = 20.0

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def recommend_target(self, findings: ExplorationFindings) -> TargetRecommendation:
        if findings.target_column:
            target_finding = findings.columns.get(findings.target_column)
            return TargetRecommendation(
                column_name=findings.target_column,
                confidence=target_finding.confidence if target_finding else 0.9,
                rationale=f"Target already detected as {findings.target_type}",
                alternatives=self._find_alternative_targets(findings),
                target_type=findings.target_type or "binary"
            )
        return self._infer_target(findings)

    def _infer_target(self, findings: ExplorationFindings) -> TargetRecommendation:
        candidates = []
        for name, col in findings.columns.items():
            if col.inferred_type == ColumnType.IDENTIFIER:
                continue
            score = 0.0
            rationale_parts = []
            if col.inferred_type == ColumnType.BINARY:
                score += 0.4
                rationale_parts.append("Binary column")
            if col.inferred_type == ColumnType.TARGET:
                score += 0.5
                rationale_parts.append("Detected as target type")
            name_lower = name.lower()
            for pattern in self.TARGET_PATTERNS:
                if pattern in name_lower:
                    score += 0.3
                    rationale_parts.append(f"Name contains '{pattern}'")
                    break
            distinct = col.universal_metrics.get("distinct_count", 0)
            if 2 <= distinct <= 10:
                score += 0.2
                rationale_parts.append(f"Few distinct values ({distinct})")
            if score > 0:
                candidates.append((name, score, rationale_parts, col))
        if not candidates:
            return TargetRecommendation(
                column_name="",
                confidence=0.0,
                rationale="No suitable target column found",
                alternatives=[],
                target_type="unknown"
            )
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        target_type = "binary" if best[3].universal_metrics.get("distinct_count", 0) == 2 else "multiclass"
        return TargetRecommendation(
            column_name=best[0],
            confidence=min(best[1], 1.0),
            rationale="; ".join(best[2]),
            alternatives=[c[0] for c in candidates[1:4]],
            target_type=target_type
        )

    def _find_alternative_targets(self, findings: ExplorationFindings) -> List[str]:
        alternatives = []
        for name, col in findings.columns.items():
            if name == findings.target_column:
                continue
            if col.inferred_type in [ColumnType.BINARY, ColumnType.TARGET]:
                alternatives.append(name)
            elif any(p in name.lower() for p in self.TARGET_PATTERNS):
                alternatives.append(name)
        return alternatives[:3]

    def recommend_features(self, findings: ExplorationFindings) -> List[FeatureRecommendation]:
        recommendations = []
        for name, col in findings.columns.items():
            if col.inferred_type == ColumnType.IDENTIFIER:
                continue
            if col.inferred_type == ColumnType.TARGET:
                continue
            recommendations.extend(self._feature_recs_for_column(name, col))
        return recommendations

    def _feature_recs_for_column(self, name: str, col) -> List[FeatureRecommendation]:
        recs = []
        if col.inferred_type == ColumnType.DATETIME:
            recs.extend([
                FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_year",
                    feature_type="temporal",
                    description=f"Extract year from {name}",
                    priority="medium",
                    implementation_hint="DatetimeTransformer.extract_year()"
                ),
                FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_month",
                    feature_type="temporal",
                    description=f"Extract month from {name}",
                    priority="medium",
                    implementation_hint="DatetimeTransformer.extract_month()"
                ),
                FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_dayofweek",
                    feature_type="temporal",
                    description=f"Extract day of week from {name}",
                    priority="medium",
                    implementation_hint="DatetimeTransformer.extract_dayofweek()"
                ),
                FeatureRecommendation(
                    source_column=name,
                    feature_name=f"days_since_{name}",
                    feature_type="datetime",
                    description=f"Days since {name} until today",
                    priority="high",
                    implementation_hint="DatetimeTransformer.days_since()"
                )
            ])
        elif col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            recs.append(FeatureRecommendation(
                source_column=name,
                feature_name=f"{name}_binned",
                feature_type="numeric",
                description=f"Binned version of {name}",
                priority="low",
                implementation_hint="NumericTransformer.bin()"
            ))
            if col.type_metrics.get("skewness", 0) and abs(col.type_metrics.get("skewness", 0)) > self.SKEWNESS_THRESHOLD:
                recs.append(FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_log",
                    feature_type="numeric",
                    description=f"Log transform of {name} (high skewness)",
                    priority="high",
                    implementation_hint="NumericTransformer.log_transform()"
                ))
        elif col.inferred_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
            cardinality = col.type_metrics.get("cardinality", 0)
            if cardinality <= 10:
                recs.append(FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_encoded",
                    feature_type="categorical",
                    description=f"One-hot encoded {name}",
                    priority="high",
                    implementation_hint="CategoricalEncoder.one_hot()"
                ))
            else:
                recs.append(FeatureRecommendation(
                    source_column=name,
                    feature_name=f"{name}_target_encoded",
                    feature_type="categorical",
                    description=f"Target encoded {name}",
                    priority="medium",
                    implementation_hint="CategoricalEncoder.target_encode()"
                ))
        elif col.inferred_type == ColumnType.CATEGORICAL_CYCLICAL:
            recs.append(FeatureRecommendation(
                source_column=name,
                feature_name=f"{name}_sin_cos",
                feature_type="cyclical",
                description=f"Cyclical encoding (sin/cos) for {name}",
                priority="high",
                implementation_hint="CategoricalEncoder.cyclical_encode()"
            ))
        return recs

    def recommend_cleaning(self, findings: ExplorationFindings) -> List[CleaningRecommendation]:
        recommendations = []
        for name, col in findings.columns.items():
            null_pct = col.universal_metrics.get("null_percentage", 0)
            null_count = col.universal_metrics.get("null_count", 0)
            if null_pct > self.NULL_CRITICAL_THRESHOLD:
                recommendations.append(CleaningRecommendation(
                    column_name=name,
                    issue_type="missing_values",
                    severity="high",
                    strategy="drop_column_or_impute_indicator",
                    description=f"{null_pct:.1f}% missing values (critical)",
                    affected_rows=null_count,
                    strategy_label="Drop Column or Create Missing Indicator",
                    problem_impact="Models will fail or lose significant data. High missingness often indicates systematic data collection issues.",
                    action_steps=[
                        "Investigate why so much data is missing (data collection issue?)",
                        "If pattern-based: create binary indicator column for 'is_missing'",
                        "If random: consider dropping column if not critical",
                        "If critical: use advanced imputation (KNN, iterative)"
                    ]
                ))
            elif null_pct > self.NULL_WARNING_THRESHOLD:
                is_numeric = col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]
                strategy = "impute_median" if is_numeric else "impute_mode"
                strategy_label = "Impute with Median" if is_numeric else "Impute with Mode"
                recommendations.append(CleaningRecommendation(
                    column_name=name,
                    issue_type="missing_values",
                    severity="medium",
                    strategy=strategy,
                    description=f"{null_pct:.1f}% missing values",
                    affected_rows=null_count,
                    strategy_label=strategy_label,
                    problem_impact="May introduce bias if missing values are not random (MAR/MNAR). Model performance degradation possible.",
                    action_steps=[
                        "Check if missingness correlates with other columns (MAR pattern)",
                        f"{'Use median (robust to outliers)' if is_numeric else 'Use mode (most frequent value)'}",
                        "Consider creating additional 'is_missing' indicator feature",
                        "Validate imputation doesn't distort distributions"
                    ]
                ))
            elif null_count > 0:
                is_numeric = col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]
                strategy = "impute_median" if is_numeric else "impute_mode"
                strategy_label = "Impute with Median" if is_numeric else "Impute with Mode"
                recommendations.append(CleaningRecommendation(
                    column_name=name,
                    issue_type="null_values",
                    severity="low",
                    strategy=strategy,
                    description=f"{null_count} null values ({null_pct:.1f}%)",
                    affected_rows=null_count,
                    strategy_label=strategy_label,
                    problem_impact="Minor impact. Some models (XGBoost, LightGBM) handle nulls natively. Others will fail.",
                    action_steps=[
                        f"{'Impute with median for robustness' if is_numeric else 'Impute with most frequent value'}",
                        "Alternatively: drop rows if very few affected",
                        "For tree-based models: can leave as-is"
                    ]
                ))
            outlier_pct = col.type_metrics.get("outlier_percentage", 0)
            if outlier_pct > self.OUTLIER_THRESHOLD:
                recommendations.append(CleaningRecommendation(
                    column_name=name,
                    issue_type="outliers",
                    severity="medium",
                    strategy="clip_or_winsorize",
                    description=f"{outlier_pct:.1f}% outliers detected",
                    affected_rows=int(outlier_pct * findings.row_count / 100),
                    strategy_label="Clip to Bounds or Winsorize",
                    problem_impact="Outliers skew mean/std calculations, affect scaling, and can dominate model training. May cause unstable predictions.",
                    action_steps=[
                        "First verify if outliers are valid (high-value customers) or errors",
                        "If errors: remove or cap at reasonable bounds",
                        "If valid: clip to 1st/99th percentile (Winsorization)",
                        "Consider log transform if right-skewed",
                        "Use RobustScaler instead of StandardScaler"
                    ]
                ))
        return recommendations

    def recommend_transformations(self, findings: ExplorationFindings) -> List[TransformRecommendation]:
        recommendations = []
        for name, col in findings.columns.items():
            if col.inferred_type == ColumnType.IDENTIFIER:
                continue
            if col.inferred_type == ColumnType.TARGET:
                continue
            recommendations.extend(self._transform_recs_for_column(name, col))
        return recommendations

    def _transform_recs_for_column(self, name: str, col) -> List[TransformRecommendation]:
        recs = []
        if col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            skewness = col.type_metrics.get("skewness", 0)
            if skewness and abs(skewness) > self.SKEWNESS_THRESHOLD:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="log_transform",
                    reason=f"High skewness ({skewness:.2f})",
                    parameters={"base": "natural"},
                    priority="high"
                ))
            outlier_pct = col.type_metrics.get("outlier_percentage", 0)
            if outlier_pct > self.OUTLIER_THRESHOLD:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="robust_scaling",
                    reason=f"High outlier percentage ({outlier_pct:.1f}%)",
                    parameters={"method": "robust_scaler"},
                    priority="high"
                ))
            else:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="standard_scaling",
                    reason="Standard normalization for numeric column",
                    parameters={"method": "standard_scaler"},
                    priority="medium"
                ))
        elif col.inferred_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
            cardinality = col.type_metrics.get("cardinality", 0)
            if cardinality <= 5:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="one_hot_encoding",
                    reason=f"Low cardinality ({cardinality})",
                    parameters={"drop_first": True},
                    priority="high"
                ))
            elif cardinality <= 20:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="target_encoding",
                    reason=f"Medium cardinality ({cardinality})",
                    parameters={"smoothing": 1.0},
                    priority="medium"
                ))
            else:
                recs.append(TransformRecommendation(
                    column_name=name,
                    transform_type="hashing_encoding",
                    reason=f"High cardinality ({cardinality})",
                    parameters={"n_components": 8},
                    priority="medium"
                ))
        elif col.inferred_type == ColumnType.DATETIME:
            recs.append(TransformRecommendation(
                column_name=name,
                transform_type="datetime_extraction",
                reason="Extract temporal features from datetime",
                parameters={"features": ["year", "month", "day", "dayofweek"]},
                priority="high"
            ))
        elif col.inferred_type == ColumnType.BINARY:
            recs.append(TransformRecommendation(
                column_name=name,
                transform_type="binary_encoding",
                reason="Ensure binary column is 0/1",
                parameters={"true_value": 1, "false_value": 0},
                priority="low"
            ))
        return recs

    def generate_summary(self, findings: ExplorationFindings) -> Dict[str, Any]:
        return {
            "target": self.recommend_target(findings),
            "features": self.recommend_features(findings),
            "cleaning": self.recommend_cleaning(findings),
            "transformations": self.recommend_transformations(findings)
        }

    def to_markdown(self, findings: ExplorationFindings) -> str:
        summary = self.generate_summary(findings)
        lines = ["# Recommendations Report", ""]
        lines.append("## Target Column")
        target = summary["target"]
        lines.append(f"**Recommended:** {target.column_name}")
        lines.append(f"**Confidence:** {target.confidence:.0%}")
        lines.append(f"**Rationale:** {target.rationale}")
        if target.alternatives:
            lines.append(f"**Alternatives:** {', '.join(target.alternatives)}")
        lines.append("")
        lines.append("## Feature Engineering Recommendations")
        for rec in summary["features"][:10]:
            lines.append(f"- **{rec.feature_name}** ({rec.priority}): {rec.description}")
        lines.append("")
        lines.append("## Data Cleaning Recommendations")
        for rec in summary["cleaning"]:
            lines.append(f"- **{rec.column_name}** [{rec.severity}]: {rec.description} → {rec.strategy}")
        lines.append("")
        lines.append("## Transformation Recommendations")
        for rec in summary["transformations"][:10]:
            lines.append(f"- **{rec.column_name}**: {rec.transform_type} ({rec.reason})")
        return "\n".join(lines)
