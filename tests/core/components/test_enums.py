

class TestSeverity:
    def test_severity_has_required_values(self):
        from customer_retention.core.components.enums import Severity
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.WARNING.value == "warning"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_severity_is_string_enum(self):
        from customer_retention.core.components.enums import Severity
        assert isinstance(Severity.CRITICAL.value, str)
        assert Severity.CRITICAL.value == "critical"

    def test_severity_comparison(self):
        from customer_retention.core.components.enums import Severity
        assert Severity.CRITICAL == Severity.CRITICAL
        assert Severity.CRITICAL != Severity.HIGH


class TestModelType:
    def test_model_type_has_required_values(self):
        from customer_retention.core.components.enums import ModelType
        assert ModelType.LOGISTIC_REGRESSION.value == "logistic_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.LIGHTGBM.value == "lightgbm"
        assert ModelType.CATBOOST.value == "catboost"


class TestModelTypeExport:
    def test_modeling_module_exports_model_type(self):
        from customer_retention.core.components.enums import ModelType as CoreModelType
        from customer_retention.stages.modeling import ModelType
        assert ModelType is CoreModelType
