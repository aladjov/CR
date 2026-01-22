

class TestComponentRegistry:
    def test_registry_creation(self):
        from customer_retention.core.components.registry import ComponentRegistry
        registry = ComponentRegistry()
        assert registry is not None

    def test_registry_has_phases(self):
        from customer_retention.core.components.registry import ComponentRegistry
        registry = ComponentRegistry()
        assert "data_preparation" in registry.PHASES
        assert "model_development" in registry.PHASES

    def test_register_component(self):
        from customer_retention.core.components.components import Ingester
        from customer_retention.core.components.registry import ComponentRegistry
        registry = ComponentRegistry()
        registry.register("ingester", Ingester, phase="data_preparation")
        assert "ingester" in registry.list_components()

    def test_get_component(self):
        from customer_retention.core.components.components import Ingester
        from customer_retention.core.components.registry import ComponentRegistration, ComponentRegistry
        registry = ComponentRegistry()
        registry.register("ingester", Ingester, phase="data_preparation")
        reg = registry.get_component("ingester")
        assert isinstance(reg, ComponentRegistration)
        assert reg.component_class == Ingester

    def test_get_phase_components(self):
        from customer_retention.core.components.components import Ingester, Profiler
        from customer_retention.core.components.registry import ComponentRegistry
        registry = ComponentRegistry()
        registry.register("ingester", Ingester, phase="data_preparation")
        registry.register("profiler", Profiler, phase="data_preparation")
        components = registry.get_phase_components("data_preparation")
        assert len(components) == 2

    def test_get_chapters_components(self):
        from customer_retention.core.components.components import Ingester, Profiler, Trainer
        from customer_retention.core.components.registry import ComponentRegistry
        registry = ComponentRegistry()
        registry.register("ingester", Ingester, phase="data_preparation")
        registry.register("profiler", Profiler, phase="data_preparation")
        registry.register("trainer", Trainer, phase="model_development")
        components = registry.get_chapters_components([1, 2])
        assert len(components) == 2

    def test_default_registry_has_all_components(self):
        from customer_retention.core.components.registry import get_default_registry
        registry = get_default_registry()
        assert len(registry.list_components()) >= 8
