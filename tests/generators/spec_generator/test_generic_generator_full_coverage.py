
import pytest


class TestGenericSpecGeneratorFullCoverage:
    @pytest.fixture
    def full_spec(self):
        from customer_retention.generators.spec_generator.pipeline_spec import (
            ColumnSpec,
            ModelSpec,
            PipelineSpec,
            SchemaSpec,
            SourceSpec,
            TransformSpec,
        )
        spec = PipelineSpec(
            name="test_pipeline",
            version="1.0.0",
            description="Test pipeline description"
        )
        spec.sources.append(SourceSpec(
            name="main_source",
            path="/data/customers.csv",
            format="csv"
        ))
        spec.schema = SchemaSpec(
            columns=[
                ColumnSpec("customer_id", "string", "identifier"),
                ColumnSpec("age", "float", "numeric_continuous"),
                ColumnSpec("gender", "string", "categorical_nominal"),
                ColumnSpec("tier", "string", "categorical_ordinal"),
                ColumnSpec("amount", "float", "numeric_discrete"),
                ColumnSpec("extra", "string", "other"),
            ],
            primary_key="customer_id"
        )
        spec.silver_transforms = [
            TransformSpec("scale_age", "standard_scaling", ["age"], ["age_scaled"]),
            TransformSpec("encode_gender", "one_hot_encoding", ["gender"], ["gender_encoded"]),
            TransformSpec("scale_amount", "standard_scaling", ["amount"], ["amount_scaled"]),
            TransformSpec("encode_tier", "one_hot_encoding", ["tier"], ["tier_encoded"]),
            TransformSpec("extra_transform", "other", ["extra"], ["extra_out"]),
            TransformSpec("sixth_transform", "other", ["x"], ["y"]),
        ]
        spec.model_config = ModelSpec(
            name="churn_model",
            model_type="gradient_boosting",
            target_column="churned",
            feature_columns=["age", "gender"]
        )
        return spec

    def test_generate_python_pipeline_with_csv_source(self, full_spec):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(full_spec)
        assert "pd.read_csv" in code
        assert "churned" in code
        assert "numeric_features" in code
        assert "categorical_features" in code

    def test_generate_python_pipeline_with_parquet_source(self, full_spec):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        full_spec.sources[0].format = "parquet"
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(full_spec)
        assert "pd.read_parquet" in code

    def test_generate_python_pipeline_with_unknown_source(self, full_spec):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        full_spec.sources[0].format = "json"
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(full_spec)
        assert "pd.read_csv" in code

    def test_generate_python_pipeline_without_schema(self):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_schema")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(spec)
        assert "def load_data" in code

    def test_generate_requirements(self, full_spec):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        generator = GenericSpecGenerator()
        reqs = generator.generate_requirements(full_spec)
        assert "pandas>=2.0.0" in reqs
        assert "scikit-learn>=1.3.0" in reqs

    def test_generate_readme_with_model_config(self, full_spec):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        generator = GenericSpecGenerator()
        readme = generator.generate_readme(full_spec)
        assert "gradient_boosting" in readme
        assert "churned" in readme
        assert "2 columns" in readme

    def test_generate_readme_without_model_config(self):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_model")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        generator = GenericSpecGenerator()
        readme = generator.generate_readme(spec)
        assert "## Model" in readme

    def test_generate_readme_without_schema(self):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_schema")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        generator = GenericSpecGenerator()
        readme = generator.generate_readme(spec)
        assert "## Schema" in readme

    def test_save_all_creates_files(self, full_spec, tmp_path):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        generator = GenericSpecGenerator(output_dir=str(tmp_path))
        files = generator.save_all(full_spec)
        assert "pipeline.py" in files
        assert "dag.py" in files
        assert "flow.py" in files
        assert "docker-compose.yml" in files
        assert "requirements.txt" in files
        assert "README.md" in files
        assert (tmp_path / "pipeline.py").exists()


class TestGenericSpecGeneratorEdgeCases:
    def test_generate_requirements_without_model_config(self):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        spec = PipelineSpec(name="no_model")
        spec.model_config = None
        generator = GenericSpecGenerator()
        reqs = generator.generate_requirements(spec)
        assert "pandas" in reqs

    def test_generate_python_pipeline_no_sources(self):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        spec = PipelineSpec(name="no_sources")
        generator = GenericSpecGenerator()
        code = generator.generate_python_pipeline(spec)
        assert "def load_data" in code
