import ast

import pytest

from customer_retention.generators.notebook_generator.stages.causal_setup_cell import (
    _SETUP_BODY_NEEDS_MODEL,
    _SETUP_BODY_PUBLISH_ONLY,
    C01_RUN_PIPELINE_BODY,
)


class TestSetupBodiesAreValidPython:
    @pytest.mark.parametrize(
        "body",
        [_SETUP_BODY_NEEDS_MODEL, _SETUP_BODY_PUBLISH_ONLY, C01_RUN_PIPELINE_BODY],
    )
    def test_body_parses(self, body):
        ast.parse(body)


class TestSetupBodyNeedsModelUcAddressing:
    """`registered_model_name` from training metadata is already a 3-part UC FQN
    (rendered as ``f"{CATALOG}.{SCHEMA}.model_{COMPOSITE_NAME}"`` in
    ``databricks_renderer._log_best_model``). The c02 setup cell must use it
    AS the FQN — never prepend CATALOG.SCHEMA again, which produces a 5-part
    name and trips Databricks' ``INVALID_PARAMETER_VALUE: Bad model name``.
    """

    def test_does_not_double_prefix_model_name_in_alias_lookup(self):
        assert 'f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"' not in _SETUP_BODY_NEEDS_MODEL

    def test_does_not_double_prefix_model_name_in_model_uri(self):
        assert 'f"models:/{CATALOG}.{SCHEMA}.{MODEL_NAME}@production"' not in _SETUP_BODY_NEEDS_MODEL

    def test_alias_lookup_passes_model_name_directly(self):
        assert "get_model_version_by_alias(MODEL_NAME, " in _SETUP_BODY_NEEDS_MODEL

    def test_model_uri_built_from_model_name_only(self):
        assert 'MODEL_URI = f"models:/{MODEL_NAME}@production"' in _SETUP_BODY_NEEDS_MODEL

    def test_model_name_assigned_from_registered_model_name(self):
        assert "MODEL_NAME = scoring_config.registered_model_name" in _SETUP_BODY_NEEDS_MODEL


class TestSetupBodyNeedsModelExecutes:
    """End-to-end exec the body with a stub ScoringConfig where
    ``registered_model_name`` is a full UC FQN; assert MODEL_URI / MODEL_NAME
    end up shaped as Databricks expects (exactly 3 dot-segments, no duplication).
    """

    def _exec_body_with_stubs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        scoring_config = SimpleNamespace(
            catalog="analytics",
            schema="churn",
            registered_model_name="analytics.churn.model_cust_emai_aggr__26e8271",
            composite_name="cust_emai_aggr__26e8271",
            best_model_name="RandomForestClassifier",
        )
        mlflow_module = MagicMock()
        mlflow_module.tracking.MlflowClient.return_value.get_model_version_by_alias.return_value = (
            SimpleNamespace(version="3")
        )

        scoping = {
            "get_spark_session": lambda: MagicMock(),
            "is_databricks": lambda: True,
            "get_playbooks_dir": lambda: "/Volumes/analytics/churn/playbooks",
            "get_experiments_dir": lambda: "/Volumes/analytics/churn/experiments",
            "ScoringConfig": SimpleNamespace(
                from_databricks=lambda: scoring_config,
                from_local_config=lambda _: scoring_config,
            ),
        }
        body_with_mock_import = _SETUP_BODY_NEEDS_MODEL.replace(
            "import mlflow", "mlflow = _mlflow_stub",
        )
        scoping["_mlflow_stub"] = mlflow_module
        body_no_imports = "\n".join(
            line for line in body_with_mock_import.splitlines()
            if not line.startswith("from customer_retention")
        )
        exec(compile(body_no_imports, "<setup_body>", "exec"), scoping)
        return scoping, mlflow_module

    def test_model_name_is_three_part_uc_fqn(self):
        scoping, _ = self._exec_body_with_stubs()
        assert scoping["MODEL_NAME"].count(".") == 2
        assert scoping["MODEL_NAME"] == "analytics.churn.model_cust_emai_aggr__26e8271"

    def test_model_uri_has_three_part_name(self):
        scoping, _ = self._exec_body_with_stubs()
        assert scoping["MODEL_URI"] == "models:/analytics.churn.model_cust_emai_aggr__26e8271@production"

    def test_alias_lookup_called_with_three_part_fqn(self):
        _, mlflow_module = self._exec_body_with_stubs()
        client = mlflow_module.tracking.MlflowClient.return_value
        called_with = client.get_model_version_by_alias.call_args.args[0]
        assert called_with.count(".") == 2
        assert called_with == "analytics.churn.model_cust_emai_aggr__26e8271"

    def test_model_version_extracted_from_lookup(self):
        scoping, _ = self._exec_body_with_stubs()
        assert scoping["MODEL_VERSION"] == "3"

    def test_other_fqns_still_built_from_catalog_schema(self):
        scoping, _ = self._exec_body_with_stubs()
        assert scoping["GOLD_FEATURES_FQN"] == "analytics.churn.gold_features_cust_emai_aggr__26e8271"
        assert scoping["PREDICTIONS_FQN"] == "analytics.churn.predictions"
        assert scoping["ARCHETYPE_CATALOG_FQN"] == "analytics.churn.archetype_catalog"
