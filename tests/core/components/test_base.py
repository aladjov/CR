from abc import ABC

import pytest


class TestComponentStatus:
    def test_status_enum_has_expected_values(self):
        from customer_retention.core.components.base import ComponentStatus
        assert ComponentStatus.PENDING.value == "pending"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.COMPLETED.value == "completed"
        assert ComponentStatus.FAILED.value == "failed"
        assert ComponentStatus.SKIPPED.value == "skipped"


class TestComponentResult:
    def test_result_creation_minimal(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(success=True, status=ComponentStatus.COMPLETED)
        assert result.success is True
        assert result.status == ComponentStatus.COMPLETED

    def test_result_has_empty_defaults(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(success=True, status=ComponentStatus.COMPLETED)
        assert result.artifacts == {}
        assert result.metrics == {}
        assert result.errors == []
        assert result.warnings == []
        assert result.duration_seconds == 0.0
        assert result.output_data is None

    def test_result_with_artifacts(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(
            success=True, status=ComponentStatus.COMPLETED,
            artifacts={"model": "/path/to/model", "data": "/path/to/data"}
        )
        assert result.artifacts["model"] == "/path/to/model"

    def test_result_with_metrics(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(
            success=True, status=ComponentStatus.COMPLETED,
            metrics={"accuracy": 0.95, "pr_auc": 0.88}
        )
        assert result.metrics["accuracy"] == 0.95

    def test_result_with_errors(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(
            success=False, status=ComponentStatus.FAILED,
            errors=["Data not found", "Invalid format"]
        )
        assert len(result.errors) == 2

    def test_result_get_summary(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(
            success=True, status=ComponentStatus.COMPLETED,
            duration_seconds=5.5
        )
        summary = result.get_summary()
        assert "COMPLETED" in summary
        assert "5.5" in summary

    def test_result_to_dict(self):
        from customer_retention.core.components.base import ComponentResult, ComponentStatus
        result = ComponentResult(success=True, status=ComponentStatus.COMPLETED)
        d = result.to_dict()
        assert d["success"] is True
        assert d["status"] == "completed"


class TestComponent:
    def test_component_is_abstract(self):
        from customer_retention.core.components.base import Component
        assert issubclass(Component, ABC)

    def test_component_cannot_instantiate_directly(self):
        from customer_retention.core.components.base import Component
        with pytest.raises(TypeError):
            Component(name="test", chapters=[1])

    def test_concrete_component_can_instantiate(self):
        from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus

        class TestComponent(Component):
            def validate_inputs(self, context):
                return []
            def run(self, context):
                return ComponentResult(success=True, status=ComponentStatus.COMPLETED)

        comp = TestComponent(name="Test", chapters=[1])
        assert comp.name == "Test"
        assert comp.chapters == [1]

    def test_component_validate_inputs(self):
        from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus
        from customer_retention.generators.orchestration.context import PipelineContext

        class ValidatingComponent(Component):
            def validate_inputs(self, context):
                errors = []
                if not context.raw_data_path:
                    errors.append("raw_data_path required")
                return errors
            def run(self, context):
                return ComponentResult(success=True, status=ComponentStatus.COMPLETED)

        comp = ValidatingComponent(name="Validator", chapters=[1])
        context = PipelineContext()
        errors = comp.validate_inputs(context)
        assert "raw_data_path required" in errors

    def test_component_run_returns_result(self):
        from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus
        from customer_retention.generators.orchestration.context import PipelineContext

        class RunningComponent(Component):
            def validate_inputs(self, context):
                return []
            def run(self, context):
                return ComponentResult(
                    success=True, status=ComponentStatus.COMPLETED,
                    artifacts={"output": "/path/to/output"},
                    metrics={"rows_processed": 100}
                )

        comp = RunningComponent(name="Runner", chapters=[1, 2])
        context = PipelineContext()
        result = comp.run(context)
        assert result.success is True
        assert result.artifacts["output"] == "/path/to/output"
        assert result.metrics["rows_processed"] == 100

    def test_component_should_skip_default(self):
        from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus
        from customer_retention.generators.orchestration.context import PipelineContext

        class SkippableComponent(Component):
            def validate_inputs(self, context):
                return []
            def run(self, context):
                return ComponentResult(success=True, status=ComponentStatus.COMPLETED)

        comp = SkippableComponent(name="Skipper", chapters=[1])
        context = PipelineContext()
        assert comp.should_skip(context) is False

    def test_component_create_result_helper(self):
        from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus
        from customer_retention.generators.orchestration.context import PipelineContext

        class HelperComponent(Component):
            def validate_inputs(self, context):
                return []
            def run(self, context):
                return self.create_result(
                    success=True,
                    artifacts={"data": "/path"},
                    metrics={"count": 10}
                )

        comp = HelperComponent(name="Helper", chapters=[1])
        context = PipelineContext()
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.status == ComponentStatus.COMPLETED
