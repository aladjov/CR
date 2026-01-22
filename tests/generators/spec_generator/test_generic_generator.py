import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"]
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric"]
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["Categorical"]
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"]
        )
    }
    return ExplorationFindings(
        source_path="data/customers.csv",
        source_format="csv",
        row_count=10000,
        column_count=4,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"]
    )


@pytest.fixture
def pipeline_spec(sample_findings) -> PipelineSpec:
    return PipelineSpec.from_findings(sample_findings, name="churn_pipeline")


class TestGenericSpecGeneratorInit:
    def test_default_init(self):
        generator = GenericSpecGenerator()
        assert generator is not None

    def test_custom_output_dir(self):
        generator = GenericSpecGenerator(output_dir="/custom/path")
        assert generator.output_dir == "/custom/path"


class TestGeneratePythonPipeline:
    def test_generates_valid_python(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(pipeline_spec)

        assert isinstance(code, str)
        assert len(code) > 100
        assert "def" in code or "class" in code
        assert "import" in code

    def test_contains_pipeline_name(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(pipeline_spec)

        assert "churn_pipeline" in code or "pipeline" in code.lower()

    def test_includes_data_loading(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(pipeline_spec)

        assert "load" in code.lower() or "read" in code.lower()

    def test_includes_target_column(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(pipeline_spec)

        assert "churned" in code or "target" in code.lower()


class TestGenerateAirflowDag:
    def test_generates_valid_dag(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_airflow_dag(pipeline_spec)

        assert isinstance(code, str)
        assert "DAG" in code
        assert "airflow" in code.lower()

    def test_contains_tasks(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_airflow_dag(pipeline_spec)

        assert "task" in code.lower() or "operator" in code.lower()

    def test_contains_dag_id(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_airflow_dag(pipeline_spec)

        assert "churn_pipeline" in code or "dag_id" in code


class TestGeneratePrefectFlow:
    def test_generates_valid_flow(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_prefect_flow(pipeline_spec)

        assert isinstance(code, str)
        assert "prefect" in code.lower()
        assert "@flow" in code or "Flow" in code

    def test_contains_tasks(self, pipeline_spec):
        generator = GenericSpecGenerator()
        code = generator.generate_prefect_flow(pipeline_spec)

        assert "@task" in code or "task" in code.lower()


class TestGenerateDockerCompose:
    def test_generates_valid_yaml(self, pipeline_spec):
        generator = GenericSpecGenerator()
        compose = generator.generate_docker_compose(pipeline_spec)

        assert isinstance(compose, str)
        assert "version" in compose or "services" in compose

    def test_contains_services(self, pipeline_spec):
        generator = GenericSpecGenerator()
        compose = generator.generate_docker_compose(pipeline_spec)

        assert "services" in compose


class TestGenerateRequirements:
    def test_generates_requirements(self, pipeline_spec):
        generator = GenericSpecGenerator()
        requirements = generator.generate_requirements(pipeline_spec)

        assert isinstance(requirements, str)
        assert "pandas" in requirements.lower()
        assert "scikit-learn" in requirements.lower() or "sklearn" in requirements.lower()

    def test_includes_version_pins(self, pipeline_spec):
        generator = GenericSpecGenerator()
        requirements = generator.generate_requirements(pipeline_spec)

        assert ">=" in requirements or "==" in requirements


class TestGenerateReadme:
    def test_generates_readme(self, pipeline_spec):
        generator = GenericSpecGenerator()
        readme = generator.generate_readme(pipeline_spec)

        assert isinstance(readme, str)
        assert "#" in readme
        assert "churn" in readme.lower() or "pipeline" in readme.lower()

    def test_contains_sections(self, pipeline_spec):
        generator = GenericSpecGenerator()
        readme = generator.generate_readme(pipeline_spec)

        assert "##" in readme


class TestGenerateAll:
    def test_generate_all_returns_dict(self, pipeline_spec):
        generator = GenericSpecGenerator()
        files = generator.generate_all(pipeline_spec)

        assert isinstance(files, dict)
        assert len(files) > 0

    def test_contains_expected_files(self, pipeline_spec):
        generator = GenericSpecGenerator()
        files = generator.generate_all(pipeline_spec)

        assert "pipeline.py" in files or any("pipeline" in k for k in files.keys())
        assert "requirements.txt" in files or any("requirements" in k for k in files.keys())

    def test_save_all(self, pipeline_spec, tmp_path):
        generator = GenericSpecGenerator(output_dir=str(tmp_path))
        saved_files = generator.save_all(pipeline_spec)

        assert len(saved_files) > 0
        for path in saved_files:
            assert (tmp_path / path).exists() or any(str(tmp_path) in str(p) for p in [path])
