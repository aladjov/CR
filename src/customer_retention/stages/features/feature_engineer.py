"""
Feature engineering pipeline for customer retention analysis.

This module provides the FeatureEngineer class that orchestrates
all feature generation components.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from customer_retention.core.compat import DataFrame, Timestamp

if TYPE_CHECKING:
    from customer_retention.integrations.feature_store.registry import FeatureRegistry
from customer_retention.stages.features.behavioral_features import BehavioralFeatureGenerator
from customer_retention.stages.features.feature_definitions import (
    FeatureCatalog,
    FeatureCategory,
    FeatureDefinition,
    LeakageRisk,
)
from customer_retention.stages.features.interaction_features import InteractionFeatureGenerator
from customer_retention.stages.features.temporal_features import (
    ReferenceDateSource,
    TemporalFeatureGenerator,
)
from customer_retention.stages.temporal.point_in_time_join import PointInTimeJoiner


@dataclass
class FeatureEngineerConfig:
    """
    Configuration for the feature engineering pipeline.

    Parameters
    ----------
    reference_date : Timestamp, optional
        Reference date for temporal calculations.
    generate_temporal : bool, default True
        Whether to generate temporal features.
    generate_behavioral : bool, default True
        Whether to generate behavioral features.
    generate_interaction : bool, default True
        Whether to generate interaction features.
    created_column : str, optional
        Column name for account creation date.
    first_order_column : str, optional
        Column name for first order date.
    last_order_column : str, optional
        Column name for last order date.
    tenure_months_column : str, optional
        Column name for tenure in months (if pre-computed).
    total_orders_column : str, optional
        Column name for total orders.
    emails_sent_column : str, optional
        Column name for emails sent.
    open_rate_column : str, optional
        Column name for email open rate.
    click_rate_column : str, optional
        Column name for email click rate.
    service_columns : List[str], optional
        List of binary service adoption columns.
    interaction_combinations : List[Tuple], optional
        List of feature combinations for interaction features.
    interaction_ratios : List[Tuple], optional
        List of ratio features for interaction features.
    populate_catalog : bool, default False
        Whether to populate feature catalog with definitions.
    preserve_original : bool, default True
        Whether to preserve original columns.
    id_column : str, optional
        Column name for customer ID (always preserved).
    enforce_point_in_time : bool, default True
        Whether to enforce point-in-time validation.
    feature_timestamp_column : str, optional
        Column name for feature observation timestamp.
    """
    reference_date: Optional[Timestamp] = None
    generate_temporal: bool = True
    generate_behavioral: bool = True
    generate_interaction: bool = True
    created_column: Optional[str] = None
    first_order_column: Optional[str] = None
    last_order_column: Optional[str] = None
    tenure_months_column: Optional[str] = None
    total_orders_column: Optional[str] = None
    emails_sent_column: Optional[str] = None
    open_rate_column: Optional[str] = None
    click_rate_column: Optional[str] = None
    service_columns: Optional[List[str]] = None
    interaction_combinations: Optional[List[Tuple[str, str, str, str]]] = None
    interaction_ratios: Optional[List[Tuple[str, str, str]]] = None
    populate_catalog: bool = False
    preserve_original: bool = True
    id_column: Optional[str] = None
    enforce_point_in_time: bool = True
    feature_timestamp_column: Optional[str] = None


@dataclass
class FeatureEngineerResult:
    """Result of feature engineering pipeline."""
    df: DataFrame
    generated_features: List[str]
    feature_categories: Dict[str, List[str]]
    config: FeatureEngineerConfig
    pit_validation: Optional[Dict[str, Any]] = None


class FeatureEngineer:
    """
    Feature engineering pipeline that orchestrates feature generation.

    This class combines temporal, behavioral, and interaction feature
    generators into a single pipeline.

    Parameters
    ----------
    config : FeatureEngineerConfig
        Pipeline configuration.

    Attributes
    ----------
    catalog : FeatureCatalog
        Catalog of generated feature definitions.
    generated_features : List[str]
        List of all generated feature names.
    """

    def __init__(self, config: FeatureEngineerConfig):
        self.config = config
        self.catalog = FeatureCatalog()
        self.generated_features: List[str] = []
        self._feature_categories: Dict[str, List[str]] = {
            "temporal": [],
            "behavioral": [],
            "interaction": [],
        }
        self._is_fitted = False

        # Initialize generators
        self._init_generators()

    def _init_generators(self) -> None:
        """Initialize feature generators based on config."""
        # Temporal generator
        if self.config.generate_temporal and self.config.reference_date:
            self._temporal_generator = TemporalFeatureGenerator(
                reference_date=self.config.reference_date,
                reference_date_source=ReferenceDateSource.CONFIG,
                created_column=self.config.created_column,
                first_order_column=self.config.first_order_column,
                last_order_column=self.config.last_order_column,
            )
        else:
            self._temporal_generator = None

        # Behavioral generator
        if self.config.generate_behavioral:
            self._behavioral_generator = BehavioralFeatureGenerator(
                tenure_months_column=self.config.tenure_months_column,
                total_orders_column=self.config.total_orders_column,
                emails_sent_column=self.config.emails_sent_column,
                open_rate_column=self.config.open_rate_column,
                click_rate_column=self.config.click_rate_column,
                service_columns=self.config.service_columns,
            )
        else:
            self._behavioral_generator = None

        # Interaction generator
        if self.config.generate_interaction:
            self._interaction_generator = InteractionFeatureGenerator(
                combinations=self.config.interaction_combinations,
                ratios=self.config.interaction_ratios,
            )
        else:
            self._interaction_generator = None

    def fit(self, df: DataFrame) -> "FeatureEngineer":
        """
        Fit the feature engineering pipeline.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        self
        """
        if self._temporal_generator:
            self._temporal_generator.fit(df)
        if self._behavioral_generator:
            self._behavioral_generator.fit(df)
        if self._interaction_generator:
            self._interaction_generator.fit(df)

        self._is_fitted = True
        return self

    def transform(self, df: DataFrame) -> FeatureEngineerResult:
        """
        Generate features for the input DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        FeatureEngineerResult
            Result containing DataFrame with features and metadata.
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer not fitted. Call fit() first.")

        result_df = df.copy()
        self.generated_features = []
        self._feature_categories = {
            "temporal": [],
            "behavioral": [],
            "interaction": [],
        }
        pit_validation = None

        # Run point-in-time validation if enabled and feature_timestamp exists
        if self.config.enforce_point_in_time:
            pit_validation = self._validate_point_in_time(result_df)

        # Apply temporal features
        if self._temporal_generator:
            result_df = self._temporal_generator.transform(result_df)
            temporal_features = self._temporal_generator.generated_features
            self.generated_features.extend(temporal_features)
            self._feature_categories["temporal"] = temporal_features
            if self.config.populate_catalog:
                self._add_temporal_definitions(temporal_features)

        # Apply behavioral features
        if self._behavioral_generator:
            result_df = self._behavioral_generator.transform(result_df)
            behavioral_features = self._behavioral_generator.generated_features
            self.generated_features.extend(behavioral_features)
            self._feature_categories["behavioral"] = behavioral_features
            if self.config.populate_catalog:
                self._add_behavioral_definitions(behavioral_features)

        # Apply interaction features (needs computed features)
        if self._interaction_generator:
            result_df = self._interaction_generator.transform(result_df)
            interaction_features = self._interaction_generator.generated_features
            self.generated_features.extend(interaction_features)
            self._feature_categories["interaction"] = interaction_features

        return FeatureEngineerResult(
            df=result_df,
            generated_features=self.generated_features.copy(),
            feature_categories=self._feature_categories.copy(),
            config=self.config,
            pit_validation=pit_validation,
        )

    def fit_transform(self, df: DataFrame) -> FeatureEngineerResult:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        FeatureEngineerResult
            Result containing DataFrame with features and metadata.
        """
        self.fit(df)
        return self.transform(df)

    def _validate_point_in_time(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate point-in-time correctness of the DataFrame.

        Returns validation report with any issues found.
        """
        ts_col = self.config.feature_timestamp_column or "feature_timestamp"

        if ts_col not in df.columns:
            return {"validated": False, "reason": f"No {ts_col} column found"}

        report = PointInTimeJoiner.validate_temporal_integrity(df)
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        future_issues = PointInTimeJoiner.validate_no_future_data(
            df, ts_col, [c for c in datetime_cols if c != ts_col]
        )

        report["future_data_issues"] = future_issues
        report["validated"] = True
        return report

    def _add_temporal_definitions(self, features: List[str]) -> None:
        """Add temporal feature definitions to catalog."""
        definitions = {
            "tenure_days": FeatureDefinition(
                name="tenure_days",
                description="Customer lifetime in days since account creation",
                category=FeatureCategory.TEMPORAL,
                derivation="reference_date - created_date",
                source_columns=[self.config.created_column or "created"],
                data_type="float",
                business_meaning="How long customer has been with us",
                leakage_risk=LeakageRisk.LOW,
            ),
            "account_age_months": FeatureDefinition(
                name="account_age_months",
                description="Customer tenure in months",
                category=FeatureCategory.TEMPORAL,
                derivation="tenure_days / 30.44",
                source_columns=["tenure_days"],
                data_type="float",
                business_meaning="Customer tenure normalized to months",
                leakage_risk=LeakageRisk.LOW,
            ),
            "days_since_last_order": FeatureDefinition(
                name="days_since_last_order",
                description="Days between reference date and last order",
                category=FeatureCategory.TEMPORAL,
                derivation="reference_date - last_order_date",
                source_columns=[self.config.last_order_column or "lastorder"],
                data_type="float",
                business_meaning="Customer recency - higher values indicate dormant customers",
                leakage_risk=LeakageRisk.MEDIUM,
            ),
            "days_to_first_order": FeatureDefinition(
                name="days_to_first_order",
                description="Days between account creation and first order",
                category=FeatureCategory.TEMPORAL,
                derivation="first_order_date - created_date",
                source_columns=[
                    self.config.created_column or "created",
                    self.config.first_order_column or "firstorder"
                ],
                data_type="float",
                business_meaning="Activation time - how quickly customer made first purchase",
                leakage_risk=LeakageRisk.LOW,
            ),
            "active_period_days": FeatureDefinition(
                name="active_period_days",
                description="Days between first and last order",
                category=FeatureCategory.TEMPORAL,
                derivation="last_order_date - first_order_date",
                source_columns=[
                    self.config.first_order_column or "firstorder",
                    self.config.last_order_column or "lastorder"
                ],
                data_type="float",
                business_meaning="Active purchasing span",
                leakage_risk=LeakageRisk.LOW,
            ),
        }

        for feature_name in features:
            if feature_name in definitions:
                self.catalog.add(definitions[feature_name], overwrite=True)

    def _add_behavioral_definitions(self, features: List[str]) -> None:
        """Add behavioral feature definitions to catalog."""
        definitions = {
            "email_engagement_score": FeatureDefinition(
                name="email_engagement_score",
                description="Combined email engagement metric",
                category=FeatureCategory.ENGAGEMENT,
                derivation="(open_rate + click_rate) / 2",
                source_columns=[
                    self.config.open_rate_column or "eopenrate",
                    self.config.click_rate_column or "eclickrate"
                ],
                data_type="float",
                business_meaning="Overall email engagement level",
                leakage_risk=LeakageRisk.LOW,
            ),
            "click_to_open_rate": FeatureDefinition(
                name="click_to_open_rate",
                description="Click rate relative to open rate",
                category=FeatureCategory.ENGAGEMENT,
                derivation="click_rate / open_rate",
                source_columns=[
                    self.config.open_rate_column or "eopenrate",
                    self.config.click_rate_column or "eclickrate"
                ],
                data_type="float",
                business_meaning="Email quality - how engaging emails are to openers",
                leakage_risk=LeakageRisk.LOW,
            ),
            "service_adoption_score": FeatureDefinition(
                name="service_adoption_score",
                description="Count of services adopted",
                category=FeatureCategory.ADOPTION,
                derivation="sum(service_flags)",
                source_columns=self.config.service_columns or [],
                data_type="float",
                business_meaning="Customer investment in platform services",
                leakage_risk=LeakageRisk.LOW,
            ),
            "service_adoption_pct": FeatureDefinition(
                name="service_adoption_pct",
                description="Percentage of available services adopted",
                category=FeatureCategory.ADOPTION,
                derivation="services_used / total_services",
                source_columns=self.config.service_columns or [],
                data_type="float",
                business_meaning="Relative service adoption level",
                leakage_risk=LeakageRisk.LOW,
            ),
        }

        for feature_name in features:
            if feature_name in definitions:
                self.catalog.add(definitions[feature_name], overwrite=True)

    def to_feature_registry(self) -> "FeatureRegistry":
        """Convert generated features to a FeatureRegistry for the feature store.

        This creates temporal feature definitions that can be used with
        the FeatureStoreManager for publishing and retrieval.

        Returns
        -------
        FeatureRegistry
            Registry containing all generated features
        """
        from customer_retention.integrations.feature_store import (
            FeatureComputationType,
            FeatureRegistry,
            TemporalFeatureDefinition,
        )

        registry = FeatureRegistry()
        entity_key = self.config.id_column or "entity_id"
        timestamp_col = self.config.feature_timestamp_column or "feature_timestamp"

        # Map FeatureCategory to leakage risk
        category_to_risk = {
            FeatureCategory.TEMPORAL: "low",
            FeatureCategory.BEHAVIORAL: "low",
            FeatureCategory.ENGAGEMENT: "low",
            FeatureCategory.ADOPTION: "low",
            FeatureCategory.DEMOGRAPHIC: "low",
            FeatureCategory.AGGREGATE: "low",
            FeatureCategory.RATIO: "low",
            FeatureCategory.TREND: "medium",
            FeatureCategory.INTERACTION: "low",
            FeatureCategory.MONETARY: "low",
        }

        # Convert catalog entries to temporal feature definitions
        for name in self.catalog.list_names():
            old_def = self.catalog.get(name)
            if old_def is None:
                continue

            # Determine computation type
            if "interaction" in name.lower() or "_x_" in name:
                comp_type = FeatureComputationType.INTERACTION
            elif "ratio" in name.lower() or "_per_" in name:
                comp_type = FeatureComputationType.RATIO
            elif old_def.category in {FeatureCategory.AGGREGATE, FeatureCategory.TREND}:
                comp_type = FeatureComputationType.AGGREGATION
            else:
                comp_type = FeatureComputationType.DERIVED

            # For DERIVED type, we need a formula - fall back to PASSTHROUGH if none
            if comp_type == FeatureComputationType.DERIVED and not old_def.derivation:
                comp_type = FeatureComputationType.PASSTHROUGH

            registry.register(TemporalFeatureDefinition(
                name=old_def.name,
                description=old_def.description,
                entity_key=entity_key,
                timestamp_column=timestamp_col,
                source_columns=old_def.source_columns,
                computation_type=comp_type,
                derivation_formula=old_def.derivation if comp_type == FeatureComputationType.DERIVED else None,
                data_type=old_def.data_type,
                leakage_risk=category_to_risk.get(old_def.category, "low"),
                leakage_notes=f"Category: {old_def.category.value}",
            ))

        # Add any generated features not in catalog
        for feature_name in self.generated_features:
            if feature_name not in registry:
                registry.register(TemporalFeatureDefinition(
                    name=feature_name,
                    description=f"Generated feature: {feature_name}",
                    entity_key=entity_key,
                    timestamp_column=timestamp_col,
                    source_columns=[],
                    computation_type=FeatureComputationType.PASSTHROUGH,
                    data_type="float64",
                    leakage_risk="low",
                ))

        return registry

    def publish_to_feature_store(
        self,
        df: DataFrame,
        table_name: str = "customer_features",
        backend: str = "feast",
        repo_path: str = "./feature_store/feature_repo",
    ) -> str:
        """Publish features to the feature store.

        Parameters
        ----------
        df : DataFrame
            DataFrame with features to publish
        table_name : str
            Name of the feature table
        backend : str
            Feature store backend ("feast" or "databricks")
        repo_path : str
            Path to feature store repo (for Feast)

        Returns
        -------
        str
            Name of the created feature table
        """
        from customer_retention.integrations.feature_store import FeatureStoreManager

        registry = self.to_feature_registry()

        manager = FeatureStoreManager.create(
            backend=backend,
            repo_path=repo_path,
        )

        entity_key = self.config.id_column or "entity_id"
        timestamp_col = self.config.feature_timestamp_column or "feature_timestamp"

        return manager.publish_features(
            df=df,
            registry=registry,
            table_name=table_name,
            entity_key=entity_key,
            timestamp_column=timestamp_col,
        )
