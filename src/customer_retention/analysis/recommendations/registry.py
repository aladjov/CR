import re
from typing import TYPE_CHECKING, List, Optional

from .base import BaseRecommendation
from .cleaning import ImputeRecommendation, OutlierCapRecommendation
from .datetime import DaysSinceRecommendation, ExtractDayOfWeekRecommendation, ExtractMonthRecommendation
from .encoding import LabelEncodeRecommendation, OneHotEncodeRecommendation
from .transform import LogTransformRecommendation, MinMaxScaleRecommendation, StandardScaleRecommendation

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings


class RecommendationRegistry:
    @classmethod
    def create_cleaning(cls, rec_str: str, columns: List[str], finding: Optional["ColumnFinding"]) -> Optional[BaseRecommendation]:
        if rec_str == "impute_median":
            return ImputeRecommendation(columns, strategy="median", source_finding=finding)
        if rec_str == "impute_mean":
            return ImputeRecommendation(columns, strategy="mean", source_finding=finding)
        if rec_str == "impute_mode":
            return ImputeRecommendation(columns, strategy="mode", source_finding=finding)
        if rec_str == "impute_zero":
            return ImputeRecommendation(columns, strategy="constant", fill_value=0, source_finding=finding)
        if match := re.match(r"cap_outliers_(\d+)", rec_str):
            return OutlierCapRecommendation(columns, percentile=int(match.group(1)), source_finding=finding)
        return None

    @classmethod
    def create_transform(cls, rec_str: str, columns: List[str], finding: Optional["ColumnFinding"]) -> Optional[BaseRecommendation]:
        if rec_str == "standard_scale":
            return StandardScaleRecommendation(columns, source_finding=finding)
        if rec_str == "minmax_scale":
            return MinMaxScaleRecommendation(columns, source_finding=finding)
        if rec_str == "log_transform":
            return LogTransformRecommendation(columns, source_finding=finding)
        return None

    @classmethod
    def create_encoding(cls, rec_str: str, columns: List[str], finding: Optional["ColumnFinding"]) -> Optional[BaseRecommendation]:
        if rec_str == "onehot_encode":
            return OneHotEncodeRecommendation(columns, source_finding=finding)
        if rec_str == "label_encode":
            return LabelEncodeRecommendation(columns, source_finding=finding)
        return None

    @classmethod
    def create_datetime(cls, rec_str: str, columns: List[str], finding: Optional["ColumnFinding"]) -> Optional[BaseRecommendation]:
        if rec_str == "extract_month":
            return ExtractMonthRecommendation(columns, source_finding=finding)
        if rec_str == "extract_dayofweek":
            return ExtractDayOfWeekRecommendation(columns, source_finding=finding)
        if rec_str == "days_since":
            return DaysSinceRecommendation(columns, source_finding=finding)
        return None

    @classmethod
    def from_findings(cls, findings: "ExplorationFindings") -> List[BaseRecommendation]:
        from customer_retention.core.config.column_config import ColumnType
        recommendations = []
        for col_name, col_finding in findings.columns.items():
            if col_finding.inferred_type in (ColumnType.IDENTIFIER, ColumnType.TARGET):
                continue
            cleaning_recs = getattr(col_finding, "cleaning_recommendations", []) or []
            for rec_str in cleaning_recs:
                rec = cls.create_cleaning(rec_str, [col_name], col_finding)
                if rec:
                    recommendations.append(rec)
            transform_recs = getattr(col_finding, "transformation_recommendations", []) or []
            for rec_str in transform_recs:
                rec = cls.create_transform(rec_str, [col_name], col_finding) or \
                      cls.create_encoding(rec_str, [col_name], col_finding) or \
                      cls.create_datetime(rec_str, [col_name], col_finding)
                if rec:
                    recommendations.append(rec)
        return recommendations
