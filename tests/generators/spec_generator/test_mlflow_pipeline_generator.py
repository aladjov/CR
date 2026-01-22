import pytest
from customer_retention.core.config.column_config import ColumnType
from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.generators.spec_generator.mlflow_pipeline_generator import (
    MLflowPipelineGenerator,
    MLflowConfig,
    RecommendationParser,
    TransformAction,
    CleanAction,
)


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric"],
            universal_metrics={"null_count": 50, "null_percentage": 2.5},
            cleaning_recommendations=["impute_median", "cap_outliers_99"],
            transformation_recommendations=["standard_scale"],
        ),
        "monthly_charges": ColumnFinding(
            name="monthly_charges",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9,
            evidence=["Numeric"],
            universal_metrics={"null_count": 0, "null_percentage": 0},
            type_metrics={"skewness": 1.8},
            cleaning_recommendations=[],
            transformation_recommendations=["log_transform", "standard_scale"],
        ),
        "tenure_months": ColumnFinding(
            name="tenure_months",
            inferred_type=ColumnType.NUMERIC_DISCRETE,
            confidence=0.85,
            evidence=["Integer values"],
            cleaning_recommendations=["impute_zero"],
            transformation_recommendations=["minmax_scale"],
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["Categorical"],
            cleaning_recommendations=["impute_mode", "drop_rare_5"],
            transformation_recommendations=["onehot_encode"],
        ),
        "payment_method": ColumnFinding(
            name="payment_method",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["Categorical"],
            transformation_recommendations=["label_encode"],
        ),
        "signup_date": ColumnFinding(
            name="signup_date",
            inferred_type=ColumnType.DATETIME,
            confidence=0.95,
            evidence=["Date format"],
            transformation_recommendations=["extract_month", "extract_dayofweek", "days_since"],
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"],
        ),
    }
    return ExplorationFindings(
        source_path="data/customers.csv",
        source_format="csv",
        row_count=10000,
        column_count=8,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"],
        datetime_columns=["signup_date"],
    )


@pytest.fixture
def mlflow_config() -> MLflowConfig:
    return MLflowConfig(
        tracking_uri="./mlruns",
        experiment_name="customer_churn",
        run_name="baseline_v1",
    )


class TestMLflowConfig:
    def test_default_values(self):
        config = MLflowConfig()
        assert config.tracking_uri == "./mlruns"
        assert config.experiment_name is not None
        assert config.log_data_quality is True
        assert config.log_transformations is True

    def test_custom_values(self):
        config = MLflowConfig(
            tracking_uri="http://mlflow:5000",
            experiment_name="test_exp",
            run_name="run1",
            log_data_quality=False,
        )
        assert config.tracking_uri == "http://mlflow:5000"
        assert config.experiment_name == "test_exp"
        assert config.log_data_quality is False


class TestRecommendationParser:
    def test_parse_impute_median(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("impute_median")
        assert isinstance(action, CleanAction)
        assert action.action_type == "impute"
        assert action.strategy == "median"

    def test_parse_impute_mode(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("impute_mode")
        assert action.action_type == "impute"
        assert action.strategy == "mode"

    def test_parse_impute_zero(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("impute_zero")
        assert action.action_type == "impute"
        assert action.strategy == "constant"
        assert action.params.get("fill_value") == 0

    def test_parse_cap_outliers(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("cap_outliers_99")
        assert action.action_type == "cap_outliers"
        assert action.params.get("percentile") == 99

    def test_parse_drop_rare(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("drop_rare_5")
        assert action.action_type == "drop_rare"
        assert action.params.get("threshold_percent") == 5

    def test_parse_standard_scale(self):
        parser = RecommendationParser()
        action = parser.parse_transform("standard_scale")
        assert isinstance(action, TransformAction)
        assert action.action_type == "scale"
        assert action.method == "standard"

    def test_parse_minmax_scale(self):
        parser = RecommendationParser()
        action = parser.parse_transform("minmax_scale")
        assert action.action_type == "scale"
        assert action.method == "minmax"

    def test_parse_log_transform(self):
        parser = RecommendationParser()
        action = parser.parse_transform("log_transform")
        assert action.action_type == "transform"
        assert action.method == "log1p"

    def test_parse_onehot_encode(self):
        parser = RecommendationParser()
        action = parser.parse_transform("onehot_encode")
        assert action.action_type == "encode"
        assert action.method == "onehot"

    def test_parse_label_encode(self):
        parser = RecommendationParser()
        action = parser.parse_transform("label_encode")
        assert action.action_type == "encode"
        assert action.method == "label"

    def test_parse_extract_datetime_components(self):
        parser = RecommendationParser()
        action = parser.parse_transform("extract_month")
        assert action.action_type == "datetime_extract"
        assert action.method == "month"

        action = parser.parse_transform("extract_dayofweek")
        assert action.method == "dayofweek"

    def test_parse_days_since(self):
        parser = RecommendationParser()
        action = parser.parse_transform("days_since")
        assert action.action_type == "datetime_extract"
        assert action.method == "days_since"

    def test_parse_unknown_returns_none(self):
        parser = RecommendationParser()
        action = parser.parse_cleaning("unknown_action")
        assert action is None

        action = parser.parse_transform("unknown_transform")
        assert action is None


class TestMLflowPipelineGeneratorInit:
    def test_default_init(self):
        generator = MLflowPipelineGenerator()
        assert generator.mlflow_config is not None

    def test_custom_config(self, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        assert generator.mlflow_config.experiment_name == "customer_churn"


class TestGeneratePipelineCode:
    def test_generates_valid_python(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert isinstance(code, str)
        assert len(code) > 500
        compile(code, "<string>", "exec")

    def test_includes_mlflow_imports(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "import mlflow" in code
        assert "mlflow.start_run" in code or "mlflow.set_experiment" in code

    def test_includes_experiment_setup(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "customer_churn" in code
        assert "set_tracking_uri" in code or "tracking_uri" in code

    def test_logs_data_quality_metrics(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "log_metric" in code or "log_metrics" in code
        assert "null" in code.lower() or "missing" in code.lower()

    def test_applies_cleaning_recommendations(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "fillna" in code or "SimpleImputer" in code
        assert "median" in code.lower()

    def test_applies_transformation_recommendations(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "StandardScaler" in code
        assert "log1p" in code.lower() or "log_transform" in code.lower()

    def test_includes_categorical_encoding(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "OneHotEncoder" in code or "LabelEncoder" in code

    def test_includes_datetime_features(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "dt.month" in code or "month" in code
        assert "days_since" in code.lower() or "day" in code

    def test_logs_transformation_parameters(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "log_param" in code or "log_params" in code

    def test_includes_model_training(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "fit" in code
        assert "predict" in code
        assert "train_test_split" in code

    def test_logs_model_metrics(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "accuracy" in code.lower() or "roc_auc" in code.lower()
        assert "log_metric" in code

    def test_logs_model_artifact(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "log_model" in code or "sklearn.log_model" in code

    def test_includes_validation_set(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "X_val" in code or "validation" in code.lower()

    def test_excludes_identifier_from_features(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_pipeline(sample_findings)

        assert "customer_id" not in code or "drop" in code


class TestGenerateWithDifferentConfigs:
    def test_disabled_quality_logging(self, sample_findings):
        config = MLflowConfig(log_data_quality=False)
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(sample_findings)

        assert "log_data_quality" not in code or "data_quality" not in code.lower()

    def test_disabled_transformation_logging(self, sample_findings):
        config = MLflowConfig(log_transformations=False)
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(sample_findings)

        compile(code, "<string>", "exec")

    def test_remote_tracking_uri(self, sample_findings):
        config = MLflowConfig(tracking_uri="http://mlflow-server:5000")
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(sample_findings)

        assert "mlflow-server:5000" in code


class TestGenerateCleaningFunctions:
    def test_generates_cleaning_function(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_cleaning_functions(sample_findings)

        assert "def clean_data" in code
        compile(code, "<string>", "exec")

    def test_handles_missing_values(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_cleaning_functions(sample_findings)

        assert "fillna" in code or "impute" in code.lower()

    def test_handles_outliers(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_cleaning_functions(sample_findings)

        assert "clip" in code or "quantile" in code or "percentile" in code.lower()


class TestGenerateTransformFunctions:
    def test_generates_transform_function(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_transform_functions(sample_findings)

        assert "def transform_data" in code or "def apply_transforms" in code
        compile(code, "<string>", "exec")

    def test_includes_scalers(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_transform_functions(sample_findings)

        assert "StandardScaler" in code or "MinMaxScaler" in code

    def test_includes_encoders(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_transform_functions(sample_findings)

        assert "OneHotEncoder" in code or "LabelEncoder" in code


class TestGenerateFeatureEngineering:
    def test_generates_feature_function(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_feature_engineering(sample_findings)

        assert "def engineer_features" in code
        compile(code, "<string>", "exec")

    def test_extracts_datetime_features(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_feature_engineering(sample_findings)

        assert "month" in code or "dayofweek" in code


class TestGenerateModelTraining:
    def test_generates_training_function(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_model_training(sample_findings)

        assert "def train_model" in code
        compile(code, "<string>", "exec")

    def test_includes_cross_validation(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_model_training(sample_findings)

        assert "cross_val" in code or "cv=" in code or "validation" in code.lower()

    def test_includes_multiple_metrics(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_model_training(sample_findings)

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        found = sum(1 for m in metrics if m in code.lower())
        assert found >= 2


class TestGenerateMonitoring:
    def test_generates_monitoring_function(self, sample_findings, mlflow_config):
        generator = MLflowPipelineGenerator(mlflow_config=mlflow_config)
        code = generator.generate_monitoring(sample_findings)

        assert "def monitor" in code or "def evaluate" in code
        compile(code, "<string>", "exec")


class TestSaveAll:
    def test_saves_pipeline_files(self, sample_findings, mlflow_config, tmp_path):
        generator = MLflowPipelineGenerator(
            mlflow_config=mlflow_config,
            output_dir=str(tmp_path),
        )
        saved = generator.save_all(sample_findings)

        assert len(saved) > 0
        assert (tmp_path / "pipeline.py").exists()
        pipeline_code = (tmp_path / "pipeline.py").read_text()
        assert "mlflow" in pipeline_code

    def test_saves_all_modules(self, sample_findings, mlflow_config, tmp_path):
        generator = MLflowPipelineGenerator(
            mlflow_config=mlflow_config,
            output_dir=str(tmp_path),
        )
        saved = generator.save_all(sample_findings)

        expected_files = ["pipeline.py", "requirements.txt"]
        for f in expected_files:
            assert f in saved or (tmp_path / f).exists()


class TestEdgeCases:
    def test_no_numeric_columns(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["All unique"],
            ),
            "category": ColumnFinding(
                name="category",
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence=0.9,
                evidence=["Categorical"],
                transformation_recommendations=["onehot_encode"],
            ),
            "target": ColumnFinding(
                name="target",
                inferred_type=ColumnType.TARGET,
                confidence=0.9,
                evidence=["Binary"],
            ),
        }
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=100,
            column_count=3,
            columns=columns,
            target_column="target",
            identifier_columns=["id"],
        )
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        compile(code, "<string>", "exec")

    def test_no_categorical_columns(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["All unique"],
            ),
            "value": ColumnFinding(
                name="value",
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                confidence=0.9,
                evidence=["Numeric"],
                transformation_recommendations=["standard_scale"],
            ),
            "target": ColumnFinding(
                name="target",
                inferred_type=ColumnType.TARGET,
                confidence=0.9,
                evidence=["Binary"],
            ),
        }
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=100,
            column_count=3,
            columns=columns,
            target_column="target",
            identifier_columns=["id"],
        )
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        compile(code, "<string>", "exec")

    def test_no_recommendations(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["All unique"],
            ),
            "value": ColumnFinding(
                name="value",
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                confidence=0.9,
                evidence=["Numeric"],
            ),
            "target": ColumnFinding(
                name="target",
                inferred_type=ColumnType.TARGET,
                confidence=0.9,
                evidence=["Binary"],
            ),
        }
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=100,
            column_count=3,
            columns=columns,
            target_column="target",
            identifier_columns=["id"],
        )
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        compile(code, "<string>", "exec")

    def test_empty_findings(self):
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=0,
            column_count=0,
            columns={},
        )
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        compile(code, "<string>", "exec")
