from typing import List
from ..base import Component, ComponentResult
from customer_retention.generators.orchestration.context import PipelineContext


class FeatureEngineer(Component):
    def __init__(self):
        super().__init__(name="FeatureEngineer", chapters=[4])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if context.current_df is None:
            errors.append("No DataFrame available for feature engineering")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            from customer_retention.stages.features.feature_engineer import FeatureEngineer as FE
            df = context.current_df
            fe = FE()
            df = fe.engineer_all(df, context.column_configs)
            context.current_df = df
            context.current_stage = "gold"
            return self.create_result(
                success=True,
                artifacts={"gold_data": context.gold_path} if context.gold_path else {},
                metrics={"feature_count": len(df.columns)}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
