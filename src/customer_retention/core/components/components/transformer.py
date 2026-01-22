from typing import List

from customer_retention.generators.orchestration.context import PipelineContext

from ..base import Component, ComponentResult


class Transformer(Component):
    def __init__(self):
        super().__init__(name="Transformer", chapters=[3])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if context.current_df is None:
            errors.append("No DataFrame available for transformation")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            from customer_retention.stages.cleaning.missing_handler import MissingHandler
            from customer_retention.stages.cleaning.outlier_handler import OutlierHandler
            df = context.current_df
            missing_handler = MissingHandler()
            df = missing_handler.handle(df)
            outlier_handler = OutlierHandler()
            df = outlier_handler.handle(df)
            context.current_df = df
            context.current_stage = "silver"
            return self.create_result(
                success=True,
                artifacts={"silver_data": context.silver_path} if context.silver_path else {},
                metrics={"row_count": len(df)}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
