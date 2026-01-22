from typing import List

from customer_retention.generators.orchestration.context import PipelineContext

from ..base import Component, ComponentResult


class Trainer(Component):
    def __init__(self):
        super().__init__(name="Trainer", chapters=[5])

    def validate_inputs(self, context: PipelineContext) -> List[str]:
        errors = []
        if context.current_df is None:
            errors.append("No DataFrame available for training")
        if not context.target_column:
            errors.append("target_column is required for training")
        return errors

    def run(self, context: PipelineContext) -> ComponentResult:
        self._start_timer()
        try:
            from customer_retention.stages.modeling.baseline_trainer import BaselineTrainer
            from customer_retention.stages.modeling.data_splitter import DataSplitter
            df = context.current_df
            target = context.target_column
            splitter = DataSplitter()
            X_train, X_test, y_train, y_test = splitter.split(df, target)
            trainer = BaselineTrainer()
            results = trainer.train_all(X_train, y_train, X_test, y_test)
            context.model_results = results
            best_model = max(results, key=lambda x: results[x].get("pr_auc", 0))
            return self.create_result(
                success=True,
                metrics={"best_model": best_model, "pr_auc": results[best_model].get("pr_auc", 0)}
            )
        except Exception as e:
            return self.create_result(success=False, errors=[str(e)])
