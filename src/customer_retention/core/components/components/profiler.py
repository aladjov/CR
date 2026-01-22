from typing import List

from customer_retention.generators.orchestration.context import PipelineContext

from ..base import Component, ComponentResult


class Profiler(Component):
    def __init__(self):
        super().__init__(name="Profiler", chapters=[2])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if context.current_df is None:
            errors.append("No DataFrame available for profiling")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            from customer_retention.stages.profiling.column_profiler import ColumnProfiler
            from customer_retention.stages.profiling.type_detector import TypeDetector
            df = context.current_df
            type_detector = TypeDetector()
            type_results = type_detector.detect_all(df)
            profiler = ColumnProfiler()
            profile = profiler.profile_all(df)
            context.profiling_results = {"types": type_results, "profile": profile}
            return self.create_result(
                success=True,
                metrics={"columns_profiled": len(df.columns)}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
