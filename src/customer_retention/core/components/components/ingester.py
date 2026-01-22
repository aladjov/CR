from typing import List

from customer_retention.core.compat import ops
from customer_retention.generators.orchestration.context import PipelineContext

from ..base import Component, ComponentResult


class Ingester(Component):
    def __init__(self):
        super().__init__(name="Ingester", chapters=[1])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if not context.raw_data_path:
            errors.append("raw_data_path is required")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            path = context.raw_data_path
            df = ops.read_csv(path)
            context.current_df = df
            context.current_stage = "bronze"
            row_count = len(df)
            col_count = len(df.columns)
            return self.create_result(
                success=True,
                artifacts={"bronze_data": context.bronze_path} if context.bronze_path else {},
                metrics={"row_count": row_count, "column_count": col_count}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
