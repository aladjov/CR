import pandas as pd


class TestDiscoverAndConfigure:
    def test_discover_and_configure_from_dataframe(self):
        from customer_retention.analysis.discovery import discover_and_configure
        from customer_retention.core.config.pipeline_config import PipelineConfig
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
            "amount": [100.5, 200.3, 300.7, 400.1, 500.9],
            "churn": [0, 1, 0, 1, 0]
        })
        config = discover_and_configure(df)
        assert isinstance(config, PipelineConfig)

    def test_discover_and_configure_from_path(self, tmp_path):
        from customer_retention.analysis.discovery import discover_and_configure
        from customer_retention.core.config.pipeline_config import PipelineConfig
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "target": [0, 1, 0]
        }).to_csv(csv_path, index=False)
        config = discover_and_configure(str(csv_path))
        assert isinstance(config, PipelineConfig)

    def test_discover_with_target_hint(self):
        from customer_retention.analysis.discovery import discover_and_configure
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "outcome": [0, 1, 0],
            "value": [10, 20, 30]
        })
        config = discover_and_configure(df, target_hint="outcome")
        assert config.modeling.target_column == "outcome"

    def test_discover_with_project_name(self):
        from customer_retention.analysis.discovery import discover_and_configure
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        config = discover_and_configure(df, project_name="my_project")
        assert config.project_name == "my_project"
