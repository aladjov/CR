import pytest
import pandas as pd


class TestInferenceConfidence:
    def test_confidence_has_levels(self):
        from customer_retention.analysis.discovery.type_inferencer import InferenceConfidence
        assert InferenceConfidence.HIGH.value == "high"
        assert InferenceConfidence.MEDIUM.value == "medium"
        assert InferenceConfidence.LOW.value == "low"


class TestColumnInference:
    def test_column_inference_creation(self):
        from customer_retention.analysis.discovery.type_inferencer import ColumnInference, InferenceConfidence
        from customer_retention.core.config.column_config import ColumnType
        inf = ColumnInference(
            column_name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=InferenceConfidence.HIGH,
            evidence=["unique values", "id in name"]
        )
        assert inf.column_name == "customer_id"
        assert inf.inferred_type == ColumnType.IDENTIFIER


class TestTypeInferencer:
    def test_inferencer_creation(self):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        inferencer = TypeInferencer()
        assert inferencer is not None

    def test_infer_returns_result(self):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer, InferenceResult
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.5, 2.5, 3.5]
        })
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        assert isinstance(result, InferenceResult)

    def test_infer_detects_identifier(self):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        from customer_retention.core.config.column_config import ColumnType
        df = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5]})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        assert result.inferences["customer_id"].inferred_type == ColumnType.IDENTIFIER

    def test_infer_detects_target(self):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        from customer_retention.core.config.column_config import ColumnType
        df = pd.DataFrame({"churn": [0, 1, 0, 1, 0]})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        assert result.target_column == "churn"

    def test_infer_detects_numeric(self):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        from customer_retention.core.config.column_config import ColumnType
        import numpy as np
        df = pd.DataFrame({"amount": np.random.uniform(0, 1000, 100)})
        inferencer = TypeInferencer()
        result = inferencer.infer(df)
        assert result.inferences["amount"].inferred_type == ColumnType.NUMERIC_CONTINUOUS

    def test_infer_from_csv_path(self, tmp_path):
        from customer_retention.analysis.discovery.type_inferencer import TypeInferencer
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).to_csv(csv_path, index=False)
        inferencer = TypeInferencer()
        result = inferencer.infer(str(csv_path))
        assert "id" in result.inferences
