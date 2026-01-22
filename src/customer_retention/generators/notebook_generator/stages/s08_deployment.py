from typing import List
import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class DeploymentStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.DEPLOYMENT

    @property
    def title(self) -> str:
        return "08 - Model Deployment"

    @property
    def description(self) -> str:
        return "Register model to registry and promote to production."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        tracking_uri = self.config.mlflow.tracking_uri
        model_name = self.config.mlflow.model_name
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.deployment": ["ModelRegistry", "ModelStage"],
                "customer_retention.integrations.adapters": ["get_mlflow"],
            }),
            self.cb.section("Initialize Registry"),
            self.cb.code(f'''mlflow_adapter = get_mlflow(tracking_uri="{tracking_uri}", force_local=True)
registry = ModelRegistry(tracking_uri="{tracking_uri}")
model_name = "{model_name}"'''),
            self.cb.section("List Model Versions"),
            self.cb.code('''versions = registry.list_versions(model_name)
for v in versions:
    print(f"Version {v.version}: Stage={v.current_stage}, Run={v.run_id}")'''),
            self.cb.section("Validate for Promotion"),
            self.cb.code('''latest_version = max(versions, key=lambda v: int(v.version)).version if versions else "1"
validation = registry.validate_for_promotion(
    model_name=model_name,
    version=latest_version,
    required_metrics={"roc_auc": 0.6},
)
print(f"Validation passed: {validation.is_valid}")
if not validation.is_valid:
    print(f"Errors: {validation.errors}")'''),
            self.cb.section("Promote to Production"),
            self.cb.code('''if validation.is_valid:
    registry.transition_stage(model_name, latest_version, ModelStage.PRODUCTION)
    print(f"Model {model_name} v{latest_version} promoted to Production")
else:
    print("Model not promoted due to validation failure")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        model_name = self.config.mlflow.model_name
        return self.header_cells() + [
            self.cb.section("Initialize MLflow Client"),
            self.cb.code('''import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()'''),
            self.cb.section("Get Model Versions"),
            self.cb.code(f'''model_full_name = "{catalog}.{schema}.{model_name}"
versions = client.search_model_versions(f"name='{{model_full_name}}'")
for v in versions:
    print(f"Version {{v.version}}: Status={{v.status}}")'''),
            self.cb.section("Get Latest Version"),
            self.cb.code('''latest = max(versions, key=lambda v: int(v.version))
print(f"Latest version: {latest.version}")'''),
            self.cb.section("Set Production Alias"),
            self.cb.code(f'''client.set_registered_model_alias(model_full_name, "production", latest.version)
print(f"Model {{model_full_name}} v{{latest.version}} aliased as 'production'")'''),
            self.cb.section("Verify Production Model"),
            self.cb.code(f'''prod_version = client.get_model_version_by_alias(model_full_name, "production")
print(f"Production model version: {{prod_version.version}}")'''),
        ]
