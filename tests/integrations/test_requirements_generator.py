from customer_retention.integrations.requirements_generator import (
    DATABRICKS_RUNTIME_PACKAGES,
    _collect_unique_deps,
    _extract_package_name,
    _is_databricks_provided,
    _is_self_referential,
    _normalize_name,
    generate_requirements,
    write_requirements,
)

MINIMAL_PYPROJECT = """\
[project]
name = "mypackage"
dependencies = [
    "pydantic>=2.0.0",
    "pandas>=2.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
delta = ["deltalake>=0.17.0"]
ml = [
    "mypackage[delta]",
    "scikit-learn>=1.3.0",
    "feast>=0.40.0",
]
"""


class TestNormalizeName:
    def test_dashes_and_underscores(self):
        assert _normalize_name("scikit-learn") == "scikit-learn"
        assert _normalize_name("scikit_learn") == "scikit-learn"

    def test_dots(self):
        assert _normalize_name("my.package") == "my-package"

    def test_mixed(self):
        assert _normalize_name("My_Package.Name") == "my-package-name"


class TestExtractPackageName:
    def test_simple(self):
        assert _extract_package_name("pandas>=2.0.0") == "pandas"

    def test_with_extras(self):
        assert _extract_package_name("mypackage[delta]") == "mypackage"

    def test_comment(self):
        assert _extract_package_name("# comment") is None

    def test_empty(self):
        assert _extract_package_name("") is None

    def test_whitespace(self):
        assert _extract_package_name("  pandas>=2.0.0  ") == "pandas"


class TestIsSelfReferential:
    def test_self_ref(self):
        assert _is_self_referential("churnkit[delta]", "churnkit") is True

    def test_not_self_ref(self):
        assert _is_self_referential("pandas>=2.0.0", "churnkit") is False

    def test_normalized_match(self):
        assert _is_self_referential("my_package[ml]", "my-package") is True


class TestIsDatabricksProvided:
    def test_known_packages(self):
        assert _is_databricks_provided("pandas>=2.0.0") is True
        assert _is_databricks_provided("scikit-learn>=1.3.0") is True
        assert _is_databricks_provided("matplotlib>=3.7.0") is True
        assert _is_databricks_provided("mlflow>=2.10.0") is True

    def test_not_provided(self):
        assert _is_databricks_provided("pydantic>=2.0.0") is False
        assert _is_databricks_provided("feast>=0.40.0") is False
        assert _is_databricks_provided("rich>=13.0.0") is False

    def test_underscore_variant(self):
        assert _is_databricks_provided("scikit_learn>=1.3.0") is True


class TestCollectUniqueDeps:
    def test_deduplicates(self):
        deps = ["pandas>=2.0.0", "pandas>=1.0.0", "rich>=13.0.0"]
        result = _collect_unique_deps(deps)
        assert len(result) == 2
        assert result[0] == "pandas>=2.0.0"

    def test_skips_comments(self):
        deps = ["# a comment", "pandas>=2.0.0"]
        result = _collect_unique_deps(deps)
        assert result == ["pandas>=2.0.0"]


class TestGenerateRequirements:
    def test_generates_both(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        req_all, req_dbr = generate_requirements(tmp_path / "pyproject.toml")
        assert "pydantic>=2.0.0" in req_all
        assert "pandas>=2.0.0" in req_all
        assert "rich>=13.0.0" in req_all
        assert "deltalake>=0.17.0" in req_all
        assert "feast>=0.40.0" in req_all

    def test_databricks_filters_runtime_packages(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        _, req_dbr = generate_requirements(tmp_path / "pyproject.toml")
        assert "pydantic>=2.0.0" in req_dbr
        assert "rich>=13.0.0" in req_dbr
        assert "feast>=0.40.0" in req_dbr
        assert "pandas" not in req_dbr
        assert "scikit-learn" not in req_dbr

    def test_excludes_self_referential(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        req_all, _ = generate_requirements(tmp_path / "pyproject.toml")
        assert "mypackage" not in req_all

    def test_auto_generated_header(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        req_all, req_dbr = generate_requirements(tmp_path / "pyproject.toml")
        assert req_all.startswith("# Auto-generated")
        assert "do not edit manually" in req_dbr


class TestWriteRequirements:
    def test_writes_both_files(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        req_path, dbr_path = write_requirements(tmp_path)
        assert req_path.exists()
        assert dbr_path.exists()
        assert req_path.name == "requirements.txt"
        assert dbr_path.name == "requirements-databricks.txt"

    def test_content_matches_generate(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(MINIMAL_PYPROJECT)
        expected_all, expected_dbr = generate_requirements(tmp_path / "pyproject.toml")
        req_path, dbr_path = write_requirements(tmp_path)
        assert req_path.read_text() == expected_all
        assert dbr_path.read_text() == expected_dbr


class TestDatabricksRuntimePackages:
    def test_known_set_completeness(self):
        expected = {"pyspark", "delta-spark", "databricks-sdk", "py4j", "mlflow", "mlflow-skinny",
                    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "lightgbm",
                    "catboost", "hyperopt", "shap", "torch", "torchvision", "tensorflow",
                    "matplotlib", "seaborn", "plotly", "ipykernel", "ipywidgets"}
        assert DATABRICKS_RUNTIME_PACKAGES == expected
