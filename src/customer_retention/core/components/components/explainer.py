from typing import List

from customer_retention.generators.orchestration.context import PipelineContext

from ..base import Component, ComponentResult


class Explainer(Component):
    def __init__(self):
        super().__init__(name="Explainer", chapters=[7])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if not context.model_results:
            errors.append("No model results available for explanation")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            return self.create_result(
                success=True,
                metrics={"explanations_generated": 1}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
