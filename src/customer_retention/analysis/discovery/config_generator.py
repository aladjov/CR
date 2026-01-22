import json
from typing import Optional

from customer_retention.core.config.column_config import ColumnConfig
from customer_retention.core.config.pipeline_config import (
    BronzeConfig,
    GoldConfig,
    ModelingConfig,
    PipelineConfig,
    SilverConfig,
)
from customer_retention.core.config.source_config import DataSourceConfig, FileFormat, SourceType

from .type_inferencer import InferenceResult


class ConfigGenerator:
    def from_inference(self, result: InferenceResult, project_name: str = "customer_retention",
                       source_path: Optional[str] = None) -> PipelineConfig:
        column_configs = []
        for col, inf in result.inferences.items():
            cc = ColumnConfig(name=col, column_type=inf.inferred_type)
            column_configs.append(cc)
        primary_key = result.identifier_columns[0] if result.identifier_columns else "id"
        data_source = DataSourceConfig(
            name="main_source",
            source_type=SourceType.BATCH_FILE,
            primary_key=primary_key,
            path=source_path or "./data.csv",
            file_format=FileFormat.CSV,
            columns=column_configs
        )
        target_col = result.target_column or "target"
        modeling = ModelingConfig(target_column=target_col)
        bronze = BronzeConfig(dedup_keys=[primary_key])
        silver = SilverConfig(entity_key=primary_key)
        return PipelineConfig(
            project_name=project_name,
            data_sources=[data_source],
            bronze=bronze,
            silver=silver,
            gold=GoldConfig(),
            modeling=modeling
        )

    def save(self, config: PipelineConfig, path: str) -> None:
        data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
