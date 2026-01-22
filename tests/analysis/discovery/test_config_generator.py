import pandas as pd


class TestConfigGenerator:
    def test_generator_creation(self):
        from customer_retention.analysis.discovery.config_generator import ConfigGenerator
        gen = ConfigGenerator()
        assert gen is not None

    def test_from_inference_returns_pipeline_config(self):
        from customer_retention.analysis.discovery.config_generator import ConfigGenerator
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        from customer_retention.core.config.pipeline_config import PipelineConfig
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5]})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        gen = ConfigGenerator()
        config = gen.from_inference(result)
        assert isinstance(config, PipelineConfig)

    def test_config_has_data_sources_with_columns(self):
        from customer_retention.analysis.discovery.config_generator import ConfigGenerator
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5]})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        gen = ConfigGenerator()
        config = gen.from_inference(result)
        assert len(config.data_sources) >= 1
        assert len(config.data_sources[0].columns) >= 2

    def test_save_config(self, tmp_path):
        from customer_retention.analysis.discovery.config_generator import ConfigGenerator
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5]})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        gen = ConfigGenerator()
        config = gen.from_inference(result)
        config_path = tmp_path / "config.json"
        gen.save(config, str(config_path))
        assert config_path.exists()
