from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ModelType(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class RiskSegment(Enum):
    """Customer risk segmentation levels."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


class Platform(str, Enum):
    """Deployment platform options."""
    LOCAL = "local"
    DATABRICKS = "databricks"
