import pytest
import pandas as pd
from pathlib import Path


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
