from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from customer_retention.core.components.enums import Platform


class OutputFormat(str, Enum):
    NOTEBOOK = "notebook"
    SCRIPT = "script"


@dataclass
class MLflowConfig:
    tracking_uri: str = "./experiments/mlruns"
    registry_uri: Optional[str] = None
    experiment_name: str = "customer_retention"
    model_name: str = "churn_model"
    track_data_quality: bool = True
    track_transformations: bool = True
    track_pipeline_stages: bool = True


@dataclass
class FeatureStoreConfig:
    base_path: str = "./experiments/feature_store"
    catalog: str = "main"
    schema: str = "default"
    table_name: str = "customer_features"


@dataclass
class NotebookConfig:
    project_name: str = "customer_retention"
    platform: Platform = Platform.LOCAL
    output_format: OutputFormat = OutputFormat.NOTEBOOK
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    model_type: str = "xgboost"
    test_size: float = 0.2
    threshold: float = 0.5
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95

    @property
    def use_framework(self) -> bool:
        return self.platform == Platform.LOCAL
