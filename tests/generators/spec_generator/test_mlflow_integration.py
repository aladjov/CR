import ast

import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.spec_generator.mlflow_pipeline_generator import (
    MLflowConfig,
    MLflowPipelineGenerator,
    RecommendationParser,
)


@pytest.fixture
def comprehensive_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.99,
            evidence=["All unique values", "Sequential pattern"],
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.92,
            evidence=["Continuous numeric range 18-85"],
            universal_metrics={
                "null_count": 150,
                "null_percentage": 1.5,
                "unique_count": 68,
            },
            type_metrics={
                "mean": 42.3,
                "median": 41.0,
                "std": 15.2,
                "min": 18,
                "max": 85,
                "skewness": 0.3,
            },
            cleaning_recommendations=["impute_median", "cap_outliers_99"],
            transformation_recommendations=["standard_scale"],
        ),
        "monthly_charges": ColumnFinding(
            name="monthly_charges",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.95,
            evidence=["Currency values"],
            universal_metrics={
                "null_count": 0,
                "null_percentage": 0,
            },
            type_metrics={
                "mean": 64.76,
                "median": 70.35,
                "skewness": 1.85,
                "kurtosis": 2.1,
            },
            cleaning_recommendations=[],
            transformation_recommendations=["log_transform", "standard_scale"],
        ),
        "total_charges": ColumnFinding(
            name="total_charges",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.88,
            evidence=["Cumulative amount"],
            universal_metrics={"null_count": 11, "null_percentage": 0.11},
            type_metrics={"skewness": 1.2},
            cleaning_recommendations=["impute_zero"],
            transformation_recommendations=["log_transform", "minmax_scale"],
        ),
        "tenure_months": ColumnFinding(
            name="tenure_months",
            inferred_type=ColumnType.NUMERIC_DISCRETE,
            confidence=0.91,
            evidence=["Integer values 0-72"],
            universal_metrics={"null_count": 0},
            type_metrics={"min": 0, "max": 72},
            cleaning_recommendations=[],
            transformation_recommendations=["minmax_scale"],
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.95,
            evidence=["3 unique categories"],
            universal_metrics={
                "null_count": 25,
                "null_percentage": 0.25,
                "unique_count": 3,
            },
            cleaning_recommendations=["impute_mode", "drop_rare_5"],
            transformation_recommendations=["onehot_encode"],
        ),
        "payment_method": ColumnFinding(
            name="payment_method",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.93,
            evidence=["4 unique categories"],
            universal_metrics={"unique_count": 4},
            cleaning_recommendations=["impute_mode"],
            transformation_recommendations=["label_encode"],
        ),
        "gender": ColumnFinding(
            name="gender",
            inferred_type=ColumnType.BINARY,
            confidence=0.99,
            evidence=["2 unique values: Male, Female"],
            transformation_recommendations=["label_encode"],
        ),
        "signup_date": ColumnFinding(
            name="signup_date",
            inferred_type=ColumnType.DATETIME,
            confidence=0.97,
            evidence=["ISO date format"],
            transformation_recommendations=[
                "extract_month",
                "extract_dayofweek",
                "days_since",
            ],
        ),
        "last_interaction_date": ColumnFinding(
            name="last_interaction_date",
            inferred_type=ColumnType.DATETIME,
            confidence=0.96,
            evidence=["Date values"],
            transformation_recommendations=["days_since"],
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.98,
            evidence=["Binary 0/1 target variable"],
        ),
    }

    return ExplorationFindings(
        source_path="data/telecom_customers.csv",
        source_format="csv",
        row_count=10000,
        column_count=11,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"],
        datetime_columns=["signup_date", "last_interaction_date"],
        overall_quality_score=85.5,
        modeling_ready=True,
    )


class TestMLflowPipelineIntegration:
    def test_full_pipeline_compilation(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_pipeline_contains_all_mlflow_stages(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        required_elements = [
            "mlflow.set_tracking_uri",
            "mlflow.set_experiment",
            "mlflow.start_run",
            "mlflow.log_param",
            "mlflow.log_metric",
            "mlflow.sklearn.log_model",
            "mlflow.end_run" if "end_run" in code else "with mlflow.start_run",
        ]

        for element in required_elements:
            assert element in code, f"Missing required MLflow element: {element}"

    def test_all_cleaning_recommendations_applied(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        assert "fillna" in code, "Missing null imputation code"
        assert "median" in code.lower(), "Missing median imputation"
        assert "mode" in code.lower(), "Missing mode imputation"
        assert "clip" in code or "quantile" in code, "Missing outlier handling"

    def test_all_transformation_recommendations_applied(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        assert "StandardScaler" in code, "Missing standard scaling"
        assert "log1p" in code.lower() or "log_transform" in code.lower(), "Missing log transform"
        assert "MinMaxScaler" in code, "Missing minmax scaling"
        assert "OneHotEncoder" in code or "get_dummies" in code, "Missing one-hot encoding"
        assert "LabelEncoder" in code, "Missing label encoding"

    def test_datetime_features_extracted(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        assert "dt.month" in code or "month" in code, "Missing month extraction"
        assert "dayofweek" in code or "weekday" in code, "Missing day of week extraction"
        assert "days_since" in code.lower(), "Missing days since calculation"

    def test_model_training_complete(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        assert "train_test_split" in code, "Missing train/test split"
        assert "X_train" in code, "Missing training data"
        assert "X_val" in code or "validation" in code.lower(), "Missing validation set"
        assert "fit" in code, "Missing model fitting"
        assert "predict" in code, "Missing prediction"
        assert "accuracy" in code.lower() or "roc_auc" in code.lower(), "Missing metrics"

    def test_identifier_columns_excluded(self, comprehensive_findings):
        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(comprehensive_findings)

        assert "exclude" in code.lower() or "customer_id" in code, "Should reference identifier column"
        assert "feature_cols" in code or "feature_columns" in code, "Should define feature columns"


class TestRecommendationParserComprehensive:
    def test_all_cleaning_patterns(self):
        parser = RecommendationParser()

        test_cases = [
            ("impute_median", "impute", "median"),
            ("impute_mean", "impute", "mean"),
            ("impute_mode", "impute", "mode"),
            ("impute_zero", "impute", "constant"),
            ("cap_outliers_95", "cap_outliers", ""),
            ("cap_outliers_99", "cap_outliers", ""),
            ("remove_outliers_iqr", "remove_outliers", "iqr"),
            ("drop_rare_5", "drop_rare", ""),
            ("drop_rare_10", "drop_rare", ""),
            ("drop_nulls", "drop_nulls", ""),
        ]

        for rec, expected_type, expected_strategy in test_cases:
            action = parser.parse_cleaning(rec)
            assert action is not None, f"Failed to parse: {rec}"
            assert action.action_type == expected_type, f"Wrong type for {rec}"
            if expected_strategy:
                assert action.strategy == expected_strategy, f"Wrong strategy for {rec}"

    def test_all_transform_patterns(self):
        parser = RecommendationParser()

        test_cases = [
            ("standard_scale", "scale", "standard"),
            ("minmax_scale", "scale", "minmax"),
            ("robust_scale", "scale", "robust"),
            ("log_transform", "transform", "log1p"),
            ("sqrt_transform", "transform", "sqrt"),
            ("power_transform", "transform", "yeo_johnson"),
            ("onehot_encode", "encode", "onehot"),
            ("label_encode", "encode", "label"),
            ("ordinal_encode", "encode", "ordinal"),
            ("extract_month", "datetime_extract", "month"),
            ("extract_dayofweek", "datetime_extract", "dayofweek"),
            ("extract_hour", "datetime_extract", "hour"),
            ("extract_year", "datetime_extract", "year"),
            ("days_since", "datetime_extract", "days_since"),
        ]

        for rec, expected_type, expected_method in test_cases:
            action = parser.parse_transform(rec)
            assert action is not None, f"Failed to parse: {rec}"
            assert action.action_type == expected_type, f"Wrong type for {rec}"
            assert action.method == expected_method, f"Wrong method for {rec}"


class TestMLflowConfigVariations:
    def test_remote_tracking_uri(self, comprehensive_findings):
        config = MLflowConfig(
            tracking_uri="http://mlflow-server:5000",
            experiment_name="production_churn_model",
        )
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(comprehensive_findings)

        assert "mlflow-server:5000" in code
        assert "production_churn_model" in code
        ast.parse(code)

    def test_disabled_quality_logging(self, comprehensive_findings):
        config = MLflowConfig(log_data_quality=False)
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(comprehensive_findings)

        assert "log_data_quality_metrics" not in code
        ast.parse(code)

    def test_custom_model_name(self, comprehensive_findings):
        config = MLflowConfig(
            model_name="telecom_churn_predictor",
            experiment_name="telecom_experiments",
        )
        generator = MLflowPipelineGenerator(mlflow_config=config)
        code = generator.generate_pipeline(comprehensive_findings)

        assert "telecom_experiments" in code
        ast.parse(code)


class TestSaveAllIntegration:
    def test_saves_complete_project(self, comprehensive_findings, tmp_path):
        config = MLflowConfig(experiment_name="test_experiment")
        generator = MLflowPipelineGenerator(
            mlflow_config=config,
            output_dir=str(tmp_path),
        )
        saved = generator.save_all(comprehensive_findings)

        assert "pipeline.py" in saved
        assert "requirements.txt" in saved

        pipeline_path = tmp_path / "pipeline.py"
        assert pipeline_path.exists()

        pipeline_code = pipeline_path.read_text()
        ast.parse(pipeline_code)

        requirements_path = tmp_path / "requirements.txt"
        assert requirements_path.exists()

        requirements = requirements_path.read_text()
        assert "mlflow" in requirements
        assert "scikit-learn" in requirements
        assert "pandas" in requirements

    def test_generated_pipeline_is_self_contained(self, comprehensive_findings, tmp_path):
        generator = MLflowPipelineGenerator(output_dir=str(tmp_path))
        generator.save_all(comprehensive_findings)

        pipeline_code = (tmp_path / "pipeline.py").read_text()

        required_imports = [
            "import pandas",
            "import numpy",
            "import mlflow",
            "from sklearn",
        ]
        for imp in required_imports:
            assert imp in pipeline_code, f"Missing import: {imp}"

        required_functions = [
            "def setup_mlflow",
            "def clean_data",
            "def apply_transforms",
            "def engineer_features",
            "def train_model",
            "def main",
        ]
        for func in required_functions:
            assert func in pipeline_code, f"Missing function: {func}"


class TestEdgeCasesIntegration:
    def test_minimal_findings(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["Unique"],
            ),
            "target": ColumnFinding(
                name="target",
                inferred_type=ColumnType.TARGET,
                confidence=0.95,
                evidence=["Binary"],
            ),
        }
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=100,
            column_count=2,
            columns=columns,
            target_column="target",
            identifier_columns=["id"],
        )

        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)

        ast.parse(code)
        assert "mlflow" in code

    def test_no_datetime_columns(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["Unique"],
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
                confidence=0.95,
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
            datetime_columns=[],
        )

        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        ast.parse(code)

    def test_all_categorical_features(self):
        columns = {
            "id": ColumnFinding(
                name="id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["Unique"],
            ),
            "cat1": ColumnFinding(
                name="cat1",
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence=0.9,
                evidence=["Categories"],
                transformation_recommendations=["onehot_encode"],
            ),
            "cat2": ColumnFinding(
                name="cat2",
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence=0.9,
                evidence=["Categories"],
                transformation_recommendations=["label_encode"],
            ),
            "target": ColumnFinding(
                name="target",
                inferred_type=ColumnType.TARGET,
                confidence=0.95,
                evidence=["Binary"],
            ),
        }
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            row_count=100,
            column_count=4,
            columns=columns,
            target_column="target",
            identifier_columns=["id"],
        )

        generator = MLflowPipelineGenerator()
        code = generator.generate_pipeline(findings)
        ast.parse(code)
        assert "OneHotEncoder" in code or "get_dummies" in code
        assert "LabelEncoder" in code
