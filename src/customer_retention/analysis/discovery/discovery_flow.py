from typing import Optional, Union
import pandas as pd
from .type_inferencer import TypeInferencer
from .config_generator import ConfigGenerator
from customer_retention.core.config.pipeline_config import PipelineConfig


def discover_and_configure(source: Union[str, pd.DataFrame], project_name: str = "customer_retention",
                           target_hint: Optional[str] = None) -> PipelineConfig:
    inferencer = TypeInferencer()
    result = inferencer.infer(source)
    if target_hint:
        result.target_column = target_hint
    generator = ConfigGenerator()
    config = generator.from_inference(result, project_name=project_name)
    return config
