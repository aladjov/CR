"""
Customer segmentation module for feature engineering.

This module provides functions for creating customer segments based on
value, engagement, recency, and other behavioral patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum

from customer_retention.core.compat import pd, DataFrame, Series


class SegmentationType(Enum):
    """Types of customer segmentation."""
    VALUE_FREQUENCY = "value_frequency"
    RECENCY = "recency"
    ENGAGEMENT = "engagement"
    LIFECYCLE = "lifecycle"
    RFM = "rfm"


@dataclass
class SegmentDefinition:
    """Definition of a customer segment."""
    name: str
    segment_type: SegmentationType
    description: str
    criteria: Dict[str, Any] = field(default_factory=dict)
    count: int = 0
    percentage: float = 0.0


@dataclass
class SegmentationResult:
    """Result of customer segmentation."""
    segment_column: str
    segment_type: SegmentationType
    total_customers: int
    segments: List[SegmentDefinition] = field(default_factory=list)
    segment_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "segment_column": self.segment_column,
            "segment_type": self.segment_type.value,
            "total_customers": self.total_customers,
            "segment_distribution": self.segment_distribution,
            "segments": [
                {
                    "name": s.name,
                    "description": s.description,
                    "count": s.count,
                    "percentage": round(s.percentage, 2)
                }
                for s in self.segments
            ]
        }


class CustomerSegmenter:
    """
    Creates customer segments based on various behavioral patterns.

    Provides methods for value-based, recency-based, engagement-based,
    and RFM segmentation.
    """

    def segment_by_value_frequency(
        self,
        df: DataFrame,
        value_column: str,
        frequency_column: str,
        value_threshold: Optional[float] = None,
        frequency_threshold: Optional[float] = None,
        output_column: str = "customer_segment"
    ) -> tuple[DataFrame, SegmentationResult]:
        """
        Segment customers by value and purchase frequency.

        Creates 4 segments:
        - High_Value_Frequent: High value + high frequency
        - High_Value_Infrequent: High value + low frequency
        - Low_Value_Frequent: Low value + high frequency
        - Low_Value_Infrequent: Low value + low frequency

        Parameters
        ----------
        df : DataFrame
            Data to segment
        value_column : str
            Column representing customer value (e.g., total revenue)
        frequency_column : str
            Column representing purchase frequency
        value_threshold : float, optional
            Threshold for high value. Default: median
        frequency_threshold : float, optional
            Threshold for high frequency. Default: median
        output_column : str
            Name of the output segment column

        Returns
        -------
        tuple[DataFrame, SegmentationResult]
            DataFrame with segment column and segmentation results
        """
        df_result = df.copy()

        # Calculate thresholds if not provided
        if value_threshold is None:
            value_threshold = df[value_column].median()
        if frequency_threshold is None:
            frequency_threshold = df[frequency_column].median()

        def assign_segment(row):
            high_value = row[value_column] >= value_threshold
            high_freq = row[frequency_column] >= frequency_threshold

            if high_value and high_freq:
                return "High_Value_Frequent"
            elif high_value and not high_freq:
                return "High_Value_Infrequent"
            elif not high_value and high_freq:
                return "Low_Value_Frequent"
            else:
                return "Low_Value_Infrequent"

        df_result[output_column] = df_result.apply(assign_segment, axis=1)

        # Build result
        distribution = df_result[output_column].value_counts().to_dict()
        total = len(df_result)

        segments = [
            SegmentDefinition(
                name="High_Value_Frequent",
                segment_type=SegmentationType.VALUE_FREQUENCY,
                description="Best customers - high value and frequent purchases",
                criteria={"value": f">= {value_threshold:.2f}", "frequency": f">= {frequency_threshold:.2f}"},
                count=distribution.get("High_Value_Frequent", 0),
                percentage=(distribution.get("High_Value_Frequent", 0) / total * 100) if total > 0 else 0
            ),
            SegmentDefinition(
                name="High_Value_Infrequent",
                segment_type=SegmentationType.VALUE_FREQUENCY,
                description="Potential for increased frequency - high value but low frequency",
                criteria={"value": f">= {value_threshold:.2f}", "frequency": f"< {frequency_threshold:.2f}"},
                count=distribution.get("High_Value_Infrequent", 0),
                percentage=(distribution.get("High_Value_Infrequent", 0) / total * 100) if total > 0 else 0
            ),
            SegmentDefinition(
                name="Low_Value_Frequent",
                segment_type=SegmentationType.VALUE_FREQUENCY,
                description="Potential for upselling - frequent but low value",
                criteria={"value": f"< {value_threshold:.2f}", "frequency": f">= {frequency_threshold:.2f}"},
                count=distribution.get("Low_Value_Frequent", 0),
                percentage=(distribution.get("Low_Value_Frequent", 0) / total * 100) if total > 0 else 0
            ),
            SegmentDefinition(
                name="Low_Value_Infrequent",
                segment_type=SegmentationType.VALUE_FREQUENCY,
                description="Needs activation - low value and low frequency",
                criteria={"value": f"< {value_threshold:.2f}", "frequency": f"< {frequency_threshold:.2f}"},
                count=distribution.get("Low_Value_Infrequent", 0),
                percentage=(distribution.get("Low_Value_Infrequent", 0) / total * 100) if total > 0 else 0
            )
        ]

        result = SegmentationResult(
            segment_column=output_column,
            segment_type=SegmentationType.VALUE_FREQUENCY,
            total_customers=total,
            segments=segments,
            segment_distribution=distribution
        )

        return df_result, result

    def segment_by_recency(
        self,
        df: DataFrame,
        days_since_column: str,
        thresholds: Optional[Dict[str, int]] = None,
        output_column: str = "recency_segment"
    ) -> tuple[DataFrame, SegmentationResult]:
        """
        Segment customers by recency (days since last activity).

        Default segments:
        - Active_30d: Active within 30 days
        - Recent_90d: Active 31-90 days ago
        - Lapsing_180d: Active 91-180 days ago
        - Dormant_180d+: Inactive for 180+ days

        Parameters
        ----------
        df : DataFrame
            Data to segment
        days_since_column : str
            Column with days since last activity
        thresholds : Dict[str, int], optional
            Custom thresholds {"active": 30, "recent": 90, "lapsing": 180}
        output_column : str
            Name of the output segment column

        Returns
        -------
        tuple[DataFrame, SegmentationResult]
            DataFrame with segment column and segmentation results
        """
        df_result = df.copy()

        if thresholds is None:
            thresholds = {"active": 30, "recent": 90, "lapsing": 180}

        active_days = thresholds.get("active", 30)
        recent_days = thresholds.get("recent", 90)
        lapsing_days = thresholds.get("lapsing", 180)

        def assign_recency_bucket(days):
            if pd.isna(days):
                return "Unknown"
            days = int(days)
            if days <= active_days:
                return f"Active_{active_days}d"
            elif days <= recent_days:
                return f"Recent_{recent_days}d"
            elif days <= lapsing_days:
                return f"Lapsing_{lapsing_days}d"
            else:
                return f"Dormant_{lapsing_days}d+"

        df_result[output_column] = df_result[days_since_column].apply(assign_recency_bucket)

        # Build result
        distribution = df_result[output_column].value_counts().to_dict()
        total = len(df_result)

        segment_names = [f"Active_{active_days}d", f"Recent_{recent_days}d",
                        f"Lapsing_{lapsing_days}d", f"Dormant_{lapsing_days}d+"]
        segment_descriptions = [
            "Recently active customers",
            "Customers with recent activity",
            "Customers at risk of churning",
            "Inactive customers needing re-engagement"
        ]

        segments = []
        for name, desc in zip(segment_names, segment_descriptions):
            count = distribution.get(name, 0)
            segments.append(SegmentDefinition(
                name=name,
                segment_type=SegmentationType.RECENCY,
                description=desc,
                count=count,
                percentage=(count / total * 100) if total > 0 else 0
            ))

        result = SegmentationResult(
            segment_column=output_column,
            segment_type=SegmentationType.RECENCY,
            total_customers=total,
            segments=segments,
            segment_distribution=distribution
        )

        return df_result, result

    def segment_by_engagement(
        self,
        df: DataFrame,
        engagement_column: str,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
        output_column: str = "engagement_segment"
    ) -> tuple[DataFrame, SegmentationResult]:
        """
        Segment customers by engagement score.

        Parameters
        ----------
        df : DataFrame
            Data to segment
        engagement_column : str
            Column with engagement score (0-1 scale)
        low_threshold : float
            Threshold below which engagement is considered low
        high_threshold : float
            Threshold above which engagement is considered high
        output_column : str
            Name of the output segment column

        Returns
        -------
        tuple[DataFrame, SegmentationResult]
            DataFrame with segment column and segmentation results
        """
        df_result = df.copy()

        def assign_engagement(score):
            if pd.isna(score):
                return "Unknown"
            if score >= high_threshold:
                return "High_Engagement"
            elif score >= low_threshold:
                return "Medium_Engagement"
            else:
                return "Low_Engagement"

        df_result[output_column] = df_result[engagement_column].apply(assign_engagement)

        # Build result
        distribution = df_result[output_column].value_counts().to_dict()
        total = len(df_result)

        segments = [
            SegmentDefinition(
                name="High_Engagement",
                segment_type=SegmentationType.ENGAGEMENT,
                description=f"Highly engaged customers (score >= {high_threshold})",
                criteria={"score": f">= {high_threshold}"},
                count=distribution.get("High_Engagement", 0),
                percentage=(distribution.get("High_Engagement", 0) / total * 100) if total > 0 else 0
            ),
            SegmentDefinition(
                name="Medium_Engagement",
                segment_type=SegmentationType.ENGAGEMENT,
                description=f"Moderately engaged customers ({low_threshold} <= score < {high_threshold})",
                criteria={"score": f"{low_threshold} - {high_threshold}"},
                count=distribution.get("Medium_Engagement", 0),
                percentage=(distribution.get("Medium_Engagement", 0) / total * 100) if total > 0 else 0
            ),
            SegmentDefinition(
                name="Low_Engagement",
                segment_type=SegmentationType.ENGAGEMENT,
                description=f"Low engagement customers (score < {low_threshold})",
                criteria={"score": f"< {low_threshold}"},
                count=distribution.get("Low_Engagement", 0),
                percentage=(distribution.get("Low_Engagement", 0) / total * 100) if total > 0 else 0
            )
        ]

        result = SegmentationResult(
            segment_column=output_column,
            segment_type=SegmentationType.ENGAGEMENT,
            total_customers=total,
            segments=segments,
            segment_distribution=distribution
        )

        return df_result, result

    def create_engagement_score(
        self,
        df: DataFrame,
        open_rate_column: str,
        click_rate_column: str,
        open_weight: float = 0.6,
        click_weight: float = 0.4,
        output_column: str = "engagement_score"
    ) -> DataFrame:
        """
        Create a composite email engagement score.

        Parameters
        ----------
        df : DataFrame
            Data to process
        open_rate_column : str
            Column with email open rate (0-100 scale)
        click_rate_column : str
            Column with email click rate (0-100 scale)
        open_weight : float
            Weight for open rate (default: 0.6)
        click_weight : float
            Weight for click rate (default: 0.4)
        output_column : str
            Name of the output column

        Returns
        -------
        DataFrame
            DataFrame with engagement score column
        """
        df_result = df.copy()

        # Normalize to 0-1 scale if needed
        open_rate = df_result[open_rate_column]
        click_rate = df_result[click_rate_column]

        if open_rate.max() > 1:
            open_rate = open_rate / 100
        if click_rate.max() > 1:
            click_rate = click_rate / 100

        df_result[output_column] = (open_weight * open_rate + click_weight * click_rate)

        return df_result

    def create_tenure_features(
        self,
        df: DataFrame,
        created_column: str,
        reference_date: Optional[Any] = None,
        output_prefix: str = ""
    ) -> DataFrame:
        """
        Create tenure-based features from account creation date.

        Parameters
        ----------
        df : DataFrame
            Data to process
        created_column : str
            Column with account creation date
        reference_date : datetime-like, optional
            Reference date for calculations. Default: max date in data
        output_prefix : str
            Prefix for output column names

        Returns
        -------
        DataFrame
            DataFrame with tenure features
        """
        df_result = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_result[created_column]):
            df_result[created_column] = pd.to_datetime(df_result[created_column], errors='coerce', format='mixed')

        # Set reference date
        if reference_date is None:
            reference_date = df_result[created_column].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        prefix = f"{output_prefix}_" if output_prefix else ""

        # Tenure in days
        df_result[f"{prefix}tenure_days"] = (reference_date - df_result[created_column]).dt.days

        # Tenure in months
        df_result[f"{prefix}tenure_months"] = df_result[f"{prefix}tenure_days"] / 30.44

        # Tenure bucket
        def tenure_bucket(days):
            if pd.isna(days) or days < 0:
                return "Unknown"
            if days <= 90:
                return "New_0_3m"
            elif days <= 180:
                return "Growing_3_6m"
            elif days <= 365:
                return "Established_6_12m"
            else:
                return "Mature_12m+"

        df_result[f"{prefix}tenure_bucket"] = df_result[f"{prefix}tenure_days"].apply(tenure_bucket)

        return df_result

    def create_recency_features(
        self,
        df: DataFrame,
        last_activity_column: str,
        reference_date: Optional[Any] = None,
        output_column: str = "days_since_last_activity"
    ) -> DataFrame:
        """
        Create recency features from last activity date.

        Parameters
        ----------
        df : DataFrame
            Data to process
        last_activity_column : str
            Column with last activity date
        reference_date : datetime-like, optional
            Reference date for calculations. Default: max date in data
        output_column : str
            Name of the output column

        Returns
        -------
        DataFrame
            DataFrame with recency feature
        """
        df_result = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_result[last_activity_column]):
            df_result[last_activity_column] = pd.to_datetime(df_result[last_activity_column], errors='coerce', format='mixed')

        # Set reference date
        if reference_date is None:
            reference_date = df_result[last_activity_column].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        df_result[output_column] = (reference_date - df_result[last_activity_column]).dt.days

        return df_result
