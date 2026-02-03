
import pandas as pd

from customer_retention.core.components.base import Component, ComponentResult, ComponentStatus
from customer_retention.core.components.orchestrator import Orchestrator, OrchestratorResult
from customer_retention.core.components.registry import ComponentRegistration, ComponentRegistry


class _SuccessComponent(Component):
    """Component that always succeeds."""

    def __init__(self):
        super().__init__(name="success", chapters=[1])

    def validate_inputs(self, context):
        return []

    def run(self, context):
        return ComponentResult(success=True, status=ComponentStatus.COMPLETED)


class _FailComponent(Component):
    """Component that always fails validation."""

    def __init__(self):
        super().__init__(name="fail", chapters=[1, 2])

    def validate_inputs(self, context):
        return ["missing required input"]

    def run(self, context):
        return ComponentResult(success=True, status=ComponentStatus.COMPLETED)


class _SkipComponent(Component):
    """Component that always skips."""

    def __init__(self):
        super().__init__(name="skip", chapters=[1])

    def validate_inputs(self, context):
        return []

    def should_skip(self, context):
        return True

    def run(self, context):
        return ComponentResult(success=True, status=ComponentStatus.COMPLETED)


class _RunFailComponent(Component):
    """Component that passes validation but fails during run."""

    def __init__(self):
        super().__init__(name="runfail", chapters=[1])

    def validate_inputs(self, context):
        return []

    def run(self, context):
        return ComponentResult(success=False, status=ComponentStatus.FAILED, errors=["run error"])


class TestOrchestrator:
    def test_orchestrator_creation(self):
        from customer_retention.core.components.orchestrator import Orchestrator
        from customer_retention.core.components.registry import get_default_registry
        from customer_retention.generators.orchestration.context import PipelineContext
        registry = get_default_registry()
        context = PipelineContext()
        orchestrator = Orchestrator(registry, context)
        assert orchestrator is not None

    def test_run_single_component(self, tmp_path):
        from customer_retention.core.components.orchestrator import Orchestrator
        from customer_retention.core.components.registry import get_default_registry
        from customer_retention.generators.orchestration.context import PipelineContext
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(csv_path, index=False)
        registry = get_default_registry()
        context = PipelineContext(raw_data_path=str(csv_path))
        orchestrator = Orchestrator(registry, context)
        result = orchestrator.run_single("ingester")
        assert result.success is True

    def test_run_chapters(self, tmp_path):
        from customer_retention.core.components.orchestrator import Orchestrator, OrchestratorResult
        from customer_retention.core.components.registry import get_default_registry
        from customer_retention.generators.orchestration.context import PipelineContext
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(csv_path, index=False)
        registry = get_default_registry()
        context = PipelineContext(raw_data_path=str(csv_path))
        orchestrator = Orchestrator(registry, context)
        result = orchestrator.run_chapters([1])
        assert isinstance(result, OrchestratorResult)
        assert "ingester" in result.components_run


class TestOrchestratorResult:
    def test_result_creation(self):
        from customer_retention.core.components.orchestrator import OrchestratorResult
        result = OrchestratorResult(success=True, components_run=["ingester"], results={}, total_duration_seconds=1.0)
        assert result.success is True

    def test_result_get_summary(self):
        from customer_retention.core.components.orchestrator import OrchestratorResult
        result = OrchestratorResult(success=True, components_run=["a", "b"], results={}, total_duration_seconds=2.5)
        summary = result.get_summary()
        assert "2" in summary
        assert "2.5" in summary

    def test_result_get_summary_failed(self):
        result = OrchestratorResult(success=False, components_run=["a"], results={}, total_duration_seconds=0.1)
        summary = result.get_summary()
        assert "FAILED" in summary


class TestOrchestratorMocked:
    """Tests using mock components to cover all orchestrator branches."""

    def _make_orchestrator(self, components_dict):
        """Create an orchestrator with a custom registry from a dict of {name: (cls, phase)}."""
        from customer_retention.generators.orchestration.context import PipelineContext
        registry = ComponentRegistry()
        for comp_name, (comp_class, phase) in components_dict.items():
            registry.register(comp_name, comp_class, phase)
        context = PipelineContext()
        return Orchestrator(registry, context)

    # --- run_training ---
    def test_run_training(self):
        orchestrator = self._make_orchestrator({"comp": (_SuccessComponent, "data_preparation")})
        result = orchestrator.run_training()
        assert isinstance(result, OrchestratorResult)
        assert result.success is True

    # --- run_phase ---
    def test_run_phase_success(self):
        orchestrator = self._make_orchestrator({"comp": (_SuccessComponent, "data_preparation")})
        result = orchestrator.run_phase("data_preparation")
        assert result.success is True
        assert "comp" in result.components_run

    def test_run_phase_empty(self):
        orchestrator = self._make_orchestrator({"comp": (_SuccessComponent, "data_preparation")})
        result = orchestrator.run_phase("production")
        assert result.success is True
        assert result.components_run == []

    def test_run_phase_failure_stops(self):
        """When a component fails in run_phase, it stops and marks success=False."""
        orchestrator = self._make_orchestrator({
            "first": (_RunFailComponent, "data_preparation"),
            "second": (_SuccessComponent, "data_preparation"),
        })
        result = orchestrator.run_phase("data_preparation")
        assert result.success is False
        assert len(result.components_run) == 1

    # --- _run_component: validation errors ---
    def test_run_component_validation_errors(self):
        orchestrator = self._make_orchestrator({"fail_comp": (_FailComponent, "data_preparation")})
        result = orchestrator.run_single("fail_comp")
        assert result.success is False
        assert result.status == ComponentStatus.FAILED
        assert len(result.errors) > 0

    # --- _run_component: should_skip ---
    def test_run_component_skip(self):
        orchestrator = self._make_orchestrator({"skip_comp": (_SkipComponent, "data_preparation")})
        result = orchestrator.run_single("skip_comp")
        assert result.success is True
        assert result.status == ComponentStatus.SKIPPED

    # --- _run_component: run failure ---
    def test_run_component_run_failure(self):
        orchestrator = self._make_orchestrator({"runfail": (_RunFailComponent, "data_preparation")})
        result = orchestrator.run_single("runfail")
        assert result.success is False
        assert result.status == ComponentStatus.FAILED

    # --- run_chapters: failure stops iteration ---
    def test_run_chapters_failure_stops(self):
        orchestrator = self._make_orchestrator({
            "first": (_RunFailComponent, "data_preparation"),
            "second": (_SuccessComponent, "data_preparation"),
        })
        result = orchestrator.run_chapters([1])
        assert result.success is False
        # Only the first component should have run before stopping
        assert len(result.components_run) == 1

    # --- _get_name_for_registration: fallback ---
    def test_get_name_for_registration_fallback(self):
        """When registration is not found in _components dict, use class name."""
        from customer_retention.generators.orchestration.context import PipelineContext
        registry = ComponentRegistry()
        context = PipelineContext()
        orchestrator = Orchestrator(registry, context)

        # Create an unregistered ComponentRegistration
        fake_reg = ComponentRegistration(
            component_class=_SuccessComponent, phase="data_preparation"
        )
        name = orchestrator._get_name_for_registration(fake_reg)
        assert name == "_successcomponent"

    # --- run_phase: with multiple successful components ---
    def test_run_phase_multiple_success(self):
        orchestrator = self._make_orchestrator({
            "first": (_SuccessComponent, "data_preparation"),
            "second": (_SuccessComponent, "data_preparation"),
        })
        result = orchestrator.run_phase("data_preparation")
        assert result.success is True
        assert len(result.components_run) == 2

    # --- run_chapters: with validation-failing component ---
    def test_run_chapters_validation_failure_stops(self):
        orchestrator = self._make_orchestrator({
            "fail_comp": (_FailComponent, "data_preparation"),
            "ok_comp": (_SuccessComponent, "data_preparation"),
        })
        result = orchestrator.run_chapters([1])
        assert result.success is False
        assert len(result.components_run) == 1
