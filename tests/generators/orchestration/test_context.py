import tempfile
from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.orchestration.context import ContextManager, PipelineContext, setup_notebook_context


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
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"]
        )
    }
    return ExplorationFindings(
        source_path="test_data.csv",
        source_format="csv",
        row_count=1000,
        column_count=3,
        columns=columns,
        target_column="churned",
        identifier_columns=["customer_id"]
    )


class TestPipelineContextCreation:
    def test_default_values(self):
        ctx = PipelineContext()
        assert ctx.project_name == "customer_retention"
        assert ctx.run_id is not None
        assert len(ctx.run_id) == 8
        assert ctx.current_stage == "raw"

    def test_custom_project_name(self):
        ctx = PipelineContext(project_name="my_project")
        assert ctx.project_name == "my_project"

    def test_has_started_at(self):
        ctx = PipelineContext()
        assert ctx.started_at is not None

    def test_default_delta_versions(self):
        ctx = PipelineContext()
        assert ctx.delta_versions == {}

    def test_default_run_type(self):
        ctx = PipelineContext()
        assert ctx.run_type == "exploration"

    def test_custom_run_type(self):
        ctx = PipelineContext(run_type="production")
        assert ctx.run_type == "production"


class TestPipelineContextProperties:
    def test_column_types_without_findings(self):
        ctx = PipelineContext()
        assert ctx.column_types == {}

    def test_column_configs_without_findings(self):
        ctx = PipelineContext()
        assert ctx.column_configs == {}

    def test_target_column_without_findings(self):
        ctx = PipelineContext()
        assert ctx.target_column is None

    def test_column_types_with_findings(self, sample_findings):
        ctx = PipelineContext(exploration_findings=sample_findings)
        assert ctx.column_types["churned"] == ColumnType.TARGET
        assert ctx.column_types["age"] == ColumnType.NUMERIC_CONTINUOUS

    def test_column_configs_with_findings(self, sample_findings):
        ctx = PipelineContext(exploration_findings=sample_findings)
        assert len(ctx.column_configs) == 3
        assert ctx.column_configs["age"].column_type == ColumnType.NUMERIC_CONTINUOUS

    def test_target_column_with_findings(self, sample_findings):
        ctx = PipelineContext(exploration_findings=sample_findings)
        assert ctx.target_column == "churned"


class TestBuildCommitMetadata:
    def test_returns_correct_keys(self):
        ctx = PipelineContext()
        meta = ctx.build_commit_metadata()
        assert set(meta.keys()) == {"run_id", "run_type", "pipeline_stage", "timestamp"}

    def test_reflects_context_values(self):
        ctx = PipelineContext(run_type="production", current_stage="gold")
        meta = ctx.build_commit_metadata()
        assert meta["run_id"] == ctx.run_id
        assert meta["run_type"] == "production"
        assert meta["pipeline_stage"] == "gold"
        assert meta["timestamp"] == ctx.started_at

    def test_all_values_are_strings(self):
        ctx = PipelineContext()
        meta = ctx.build_commit_metadata()
        for v in meta.values():
            assert isinstance(v, str)


class TestPipelineContextPersistence:
    def test_save_creates_file(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            ctx.save()

            expected_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            assert expected_path.exists()

    def test_save_and_load_roundtrip(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(
                project_name="test_project",
                exploration_findings=sample_findings
            )
            ctx._context_dir = tmpdir
            ctx.save()

            context_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            loaded = PipelineContext.load(str(context_path))

            assert loaded.run_id == ctx.run_id
            assert loaded.project_name == "test_project"
            assert loaded.target_column == "churned"

    def test_save_and_load_delta_versions(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            ctx.delta_versions = {"bronze/customers": 3, "silver/features": 1}
            ctx.run_type = "production"
            ctx.save()

            context_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            loaded = PipelineContext.load(str(context_path))

            assert loaded.delta_versions == {"bronze/customers": 3, "silver/features": 1}
            assert loaded.run_type == "production"

    def test_load_without_delta_fields_uses_defaults(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            ctx.save()

            context_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            loaded = PipelineContext.load(str(context_path))

            assert loaded.delta_versions == {}
            assert loaded.run_type == "exploration"

    def test_save_includes_exploration_findings(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            ctx.save()

            findings_path = Path(tmpdir) / f"{ctx.run_id}_findings.yaml"
            assert findings_path.exists()

    def test_load_restores_findings(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            ctx.save()

            context_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            loaded = PipelineContext.load(str(context_path))

            assert loaded.exploration_findings is not None
            assert loaded.column_types["age"] == ColumnType.NUMERIC_CONTINUOUS


class TestContextManager:
    def test_update_context(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            manager = ContextManager(ctx, auto_save=False)

            manager.update(current_stage="profiled")
            assert ctx.current_stage == "profiled"

    def test_auto_save_on_update(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            manager = ContextManager(ctx, auto_save=True)

            manager.update(current_stage="cleaned")
            expected_path = Path(tmpdir) / f"{ctx.run_id}_context.json"
            assert expected_path.exists()

    def test_add_artifact(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(exploration_findings=sample_findings)
            ctx._context_dir = tmpdir
            manager = ContextManager(ctx, auto_save=False)

            manager.add_artifact("model", "/path/to/model.pkl")
            assert ctx.artifacts["model"] == "/path/to/model.pkl"


class TestSetupNotebookContext:
    def test_creates_new_context(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            findings_path = Path(tmpdir) / "findings.yaml"
            sample_findings.save(str(findings_path))

            ctx, manager = setup_notebook_context(
                exploration_findings=str(findings_path),
                output_dir=tmpdir
            )

            assert ctx is not None
            assert manager is not None
            assert ctx.target_column == "churned"

    def test_uses_project_name(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            findings_path = Path(tmpdir) / "findings.yaml"
            sample_findings.save(str(findings_path))

            ctx, _ = setup_notebook_context(
                project_name="custom_project",
                exploration_findings=str(findings_path),
                output_dir=tmpdir
            )

            assert ctx.project_name == "custom_project"

    def test_sets_raw_data_path(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            findings_path = Path(tmpdir) / "findings.yaml"
            sample_findings.save(str(findings_path))

            ctx, _ = setup_notebook_context(
                exploration_findings=str(findings_path),
                output_dir=tmpdir
            )

            assert ctx.raw_data_path == "test_data.csv"

    def test_resume_existing_run(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PipelineContext(
                project_name="resumable",
                exploration_findings=sample_findings
            )
            ctx._context_dir = tmpdir
            ctx.current_stage = "transformed"
            ctx.save()

            resumed_ctx, _ = setup_notebook_context(
                resume_run=ctx.run_id,
                output_dir=tmpdir
            )

            assert resumed_ctx.run_id == ctx.run_id
            assert resumed_ctx.current_stage == "transformed"

    def test_accepts_findings_object(self, sample_findings):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx, _ = setup_notebook_context(
                exploration_findings=sample_findings,
                output_dir=tmpdir
            )

            assert ctx.target_column == "churned"
