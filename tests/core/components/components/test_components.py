import pandas as pd

from customer_retention.core.components.base import Component, ComponentResult
from customer_retention.generators.orchestration.context import PipelineContext


class TestIngester:
    def test_ingester_is_component(self):
        from customer_retention.core.components.components import Ingester
        assert issubclass(Ingester, Component)

    def test_ingester_has_chapter_1(self):
        from customer_retention.core.components.components import Ingester
        comp = Ingester()
        assert 1 in comp.chapters

    def test_ingester_validate_requires_raw_data_path(self):
        from customer_retention.core.components.components import Ingester
        comp = Ingester()
        context = PipelineContext()
        errors = comp.validate_inputs(context)
        assert any("raw_data_path" in e for e in errors)

    def test_ingester_run_returns_result(self, tmp_path):
        from customer_retention.core.components.components import Ingester
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(csv_path, index=False)
        comp = Ingester()
        context = PipelineContext(raw_data_path=str(csv_path))
        context.bronze_path = str(tmp_path / "bronze")
        result = comp.run(context)
        assert isinstance(result, ComponentResult)


class TestProfiler:
    def test_profiler_is_component(self):
        from customer_retention.core.components.components import Profiler
        assert issubclass(Profiler, Component)

    def test_profiler_has_chapter_2(self):
        from customer_retention.core.components.components import Profiler
        comp = Profiler()
        assert 2 in comp.chapters

    def test_profiler_validate_requires_dataframe(self):
        from customer_retention.core.components.components import Profiler
        comp = Profiler()
        context = PipelineContext()
        errors = comp.validate_inputs(context)
        assert len(errors) > 0


class TestTransformer:
    def test_transformer_is_component(self):
        from customer_retention.core.components.components import Transformer
        assert issubclass(Transformer, Component)

    def test_transformer_has_chapter_3(self):
        from customer_retention.core.components.components import Transformer
        comp = Transformer()
        assert 3 in comp.chapters


class TestFeatureEngineer:
    def test_feature_eng_is_component(self):
        from customer_retention.core.components.components import FeatureEngineer
        assert issubclass(FeatureEngineer, Component)

    def test_feature_eng_has_chapter_4(self):
        from customer_retention.core.components.components import FeatureEngineer
        comp = FeatureEngineer()
        assert 4 in comp.chapters


class TestTrainer:
    def test_trainer_is_component(self):
        from customer_retention.core.components.components import Trainer
        assert issubclass(Trainer, Component)

    def test_trainer_has_chapter_5(self):
        from customer_retention.core.components.components import Trainer
        comp = Trainer()
        assert 5 in comp.chapters


class TestValidator:
    def test_validator_is_component(self):
        from customer_retention.core.components.components import Validator
        assert issubclass(Validator, Component)

    def test_validator_has_chapter_6(self):
        from customer_retention.core.components.components import Validator
        comp = Validator()
        assert 6 in comp.chapters


class TestExplainer:
    def test_explainer_is_component(self):
        from customer_retention.core.components.components import Explainer
        assert issubclass(Explainer, Component)

    def test_explainer_has_chapter_7(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        assert 7 in comp.chapters


class TestDeployer:
    def test_deployer_is_component(self):
        from customer_retention.core.components.components import Deployer
        assert issubclass(Deployer, Component)

    def test_deployer_has_chapter_8(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        assert 8 in comp.chapters
