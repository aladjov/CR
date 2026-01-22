"""End-to-end integration tests for training/scoring transformer consistency.

These tests verify that:
1. Transformations applied during training are exactly replicated during scoring
2. Same feature values produce identical transformed values in both phases
3. Using different transformers (re-fitting) produces incorrect results
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def training_data():
    """Training dataset with known categorical and numeric values."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(100),
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
        "income": [30000, 45000, 60000, 75000, 90000] * 20,
        "tenure_months": np.random.randint(1, 60, 100),
        "gender": ["M", "F", "M", "F", "O"] * 20,
        "region": ["North", "South", "East", "West"] * 25,
        "plan_type": ["Basic", "Standard", "Premium"] * 33 + ["Basic"],
        "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
    })


@pytest.fixture
def scoring_data():
    """Scoring dataset with overlapping values that must match training transformation."""
    return pd.DataFrame({
        "customer_id": range(100, 120),
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 2,
        "income": [30000, 45000, 60000, 75000, 90000] * 4,
        "tenure_months": [12, 24, 36, 48, 6] * 4,
        "gender": ["M", "F", "M", "F", "O"] * 4,
        "region": ["North", "South", "East", "West", "North"] * 4,
        "plan_type": ["Basic", "Standard", "Premium", "Basic", "Standard"] * 4,
    })


class TestTransformerConsistencyEndToEnd:
    """Verify transformations are consistent between training and scoring."""

    def test_numeric_scaling_produces_identical_values(self, training_data, scoring_data):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager(scaler_type="standard")
        train_transformed = manager.fit_transform(
            training_data,
            numeric_columns=["age", "income", "tenure_months"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "gender", "region", "plan_type"]
        )

        scoring_transformed = manager.transform(
            scoring_data,
            exclude_columns=["customer_id", "gender", "region", "plan_type"]
        )

        train_age_25 = train_transformed.loc[training_data["age"] == 25, "age"].iloc[0]
        score_age_25 = scoring_transformed.loc[scoring_data["age"] == 25, "age"].iloc[0]
        assert train_age_25 == pytest.approx(score_age_25, abs=1e-10)

        train_income_30k = train_transformed.loc[training_data["income"] == 30000, "income"].iloc[0]
        score_income_30k = scoring_transformed.loc[scoring_data["income"] == 30000, "income"].iloc[0]
        assert train_income_30k == pytest.approx(score_income_30k, abs=1e-10)

    def test_categorical_encoding_produces_identical_values(self, training_data, scoring_data):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        train_transformed = manager.fit_transform(
            training_data,
            numeric_columns=[],
            categorical_columns=["gender", "region", "plan_type"],
            exclude_columns=["customer_id", "target", "age", "income", "tenure_months"]
        )

        scoring_transformed = manager.transform(
            scoring_data,
            exclude_columns=["customer_id", "age", "income", "tenure_months"]
        )

        train_gender_m = train_transformed.loc[training_data["gender"] == "M", "gender"].iloc[0]
        score_gender_m = scoring_transformed.loc[scoring_data["gender"] == "M", "gender"].iloc[0]
        assert train_gender_m == score_gender_m

        train_region_north = train_transformed.loc[training_data["region"] == "North", "region"].iloc[0]
        score_region_north = scoring_transformed.loc[scoring_data["region"] == "North", "region"].iloc[0]
        assert train_region_north == score_region_north

    def test_full_pipeline_consistency(self, training_data, scoring_data, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager(scaler_type="standard")
        train_transformed = manager.fit_transform(
            training_data,
            numeric_columns=["age", "income", "tenure_months"],
            categorical_columns=["gender", "region", "plan_type"],
            exclude_columns=["customer_id", "target"]
        )

        save_path = tmp_path / "transformers.joblib"
        manager.save(save_path)

        loaded_manager = TransformerManager.load(save_path)
        scoring_transformed = loaded_manager.transform(
            scoring_data,
            exclude_columns=["customer_id"]
        )

        for col in ["age", "income", "tenure_months", "gender", "region", "plan_type"]:
            for value in training_data[col].unique()[:3]:
                if value in scoring_data[col].values:
                    train_val = train_transformed.loc[training_data[col] == value, col].iloc[0]
                    score_val = scoring_transformed.loc[scoring_data[col] == value, col].iloc[0]
                    assert train_val == pytest.approx(score_val, abs=1e-10), f"Mismatch for {col}={value}"


class TestRefitProducesDifferentResults:
    """Verify that re-fitting transformers produces different (incorrect) results."""

    def test_refit_scaler_produces_different_values(self):
        from sklearn.preprocessing import StandardScaler

        train_data = pd.DataFrame({"age": [20, 30, 40, 50, 60], "income": [25000, 50000, 75000, 100000, 125000]})
        score_data = pd.DataFrame({"age": [60, 70, 80, 90, 100], "income": [100000, 150000, 200000, 250000, 300000]})

        train_scaler = StandardScaler()
        train_scaler.fit(train_data)

        score_scaler = StandardScaler()
        score_scaler.fit(score_data)

        assert not np.allclose(train_scaler.mean_, score_scaler.mean_), "Means should differ"

    def test_refit_encoder_with_different_categories_produces_different_mappings(self):
        from sklearn.preprocessing import LabelEncoder

        data_train = pd.Series(["A", "B", "C", "A", "B"])
        data_score = pd.Series(["X", "Y", "Z", "X", "Y"])

        encoder_train = LabelEncoder()
        encoder_train.fit(data_train)

        encoder_score = LabelEncoder()
        encoder_score.fit(data_score)

        assert set(encoder_train.classes_) != set(encoder_score.classes_)

    def test_refit_with_different_distribution_produces_wrong_scaling(self):
        from sklearn.preprocessing import StandardScaler

        train_data = pd.DataFrame({"age": [20, 25, 30, 35, 40]})
        score_data = pd.DataFrame({"age": [60, 65, 70, 75, 80]})

        train_scaler = StandardScaler()
        train_scaler.fit(train_data)

        score_scaler = StandardScaler()
        score_scaler.fit(score_data)

        test_value = np.array([[50]])
        train_scaled = train_scaler.transform(test_value)[0, 0]
        score_scaled = score_scaler.transform(test_value)[0, 0]

        assert train_scaled != pytest.approx(score_scaled, abs=0.5), "Scaling should differ significantly"


class TestTransformerManagerPreservesFeatureOrder:
    """Verify feature order is preserved between training and scoring."""

    def test_feature_order_matches(self, training_data, scoring_data, tmp_path):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        train_transformed = manager.fit_transform(
            training_data,
            numeric_columns=["age", "income"],
            categorical_columns=["gender", "region"],
            exclude_columns=["customer_id", "target", "tenure_months", "plan_type"]
        )

        expected_order = list(manager.manifest.feature_order)

        manager.save(tmp_path / "transformers.joblib")
        loaded = TransformerManager.load(tmp_path / "transformers.joblib")

        scoring_transformed = loaded.transform(
            scoring_data,
            exclude_columns=["customer_id", "tenure_months", "plan_type"]
        )

        assert list(scoring_transformed.columns) == expected_order[:len(scoring_transformed.columns)]


class TestUnseenCategoriesHandling:
    """Test handling of unseen categories during scoring."""

    def test_unseen_category_gets_default_encoding(self, training_data):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            training_data,
            numeric_columns=[],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "age", "income", "tenure_months", "region", "plan_type"]
        )

        scoring_with_unseen = pd.DataFrame({
            "customer_id": [999],
            "gender": ["X"],  # Unseen category
        })

        result = manager.transform(scoring_with_unseen, exclude_columns=["customer_id"])
        assert result["gender"].iloc[0] == 0  # Default to 0 for unseen


class TestScoringPipelineIntegration:
    """End-to-end scoring pipeline integration tests."""

    def test_model_prediction_with_consistent_transformers(self, training_data, scoring_data, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        from customer_retention.stages.preprocessing import TransformerManager

        feature_cols = ["age", "income", "tenure_months", "gender", "region", "plan_type"]

        manager = TransformerManager(scaler_type="standard")
        train_transformed = manager.fit_transform(
            training_data[feature_cols].copy(),
            numeric_columns=["age", "income", "tenure_months"],
            categorical_columns=["gender", "region", "plan_type"],
            exclude_columns=[]
        )
        y_train = training_data["target"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(train_transformed, y_train)

        manager.save(tmp_path / "transformers.joblib")

        loaded_manager = TransformerManager.load(tmp_path / "transformers.joblib")
        X_score = loaded_manager.transform(scoring_data[feature_cols].copy(), exclude_columns=[])

        predictions = model.predict(X_score)
        probabilities = model.predict_proba(X_score)[:, 1]

        assert len(predictions) == len(scoring_data)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_shap_computation_with_consistent_transformers(self, training_data, scoring_data, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        from customer_retention.analysis.interpretability import ShapExplainer
        from customer_retention.stages.preprocessing import TransformerManager

        feature_cols = ["age", "income", "tenure_months", "gender", "region", "plan_type"]

        manager = TransformerManager(scaler_type="standard")
        train_transformed = manager.fit_transform(
            training_data[feature_cols].copy(),
            numeric_columns=["age", "income", "tenure_months"],
            categorical_columns=["gender", "region", "plan_type"],
            exclude_columns=[]
        )
        y_train = training_data["target"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(train_transformed, y_train)

        manager.save(tmp_path / "transformers.joblib")

        loaded_manager = TransformerManager.load(tmp_path / "transformers.joblib")
        X_score = loaded_manager.transform(scoring_data[feature_cols].copy(), exclude_columns=[])

        explainer = ShapExplainer(model, train_transformed.head(50))
        shap_values = explainer.get_shap_values(X_score)

        assert shap_values.shape == (len(X_score), X_score.shape[1])
        assert not np.isnan(shap_values).any()


class TestTransformerManagerCoverage:
    """Additional tests for high coverage."""

    def test_auto_detect_column_types(self, training_data):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager()
        result = manager.fit_transform(
            training_data,
            numeric_columns=None,
            categorical_columns=None,
            exclude_columns=["customer_id", "target"]
        )

        assert manager.is_fitted
        assert len(manager.manifest.numeric_columns) > 0
        assert len(manager.manifest.categorical_columns) > 0

    def test_manifest_serialization(self):
        from customer_retention.stages.preprocessing.transformer_manager import TransformerManifest

        manifest = TransformerManifest(
            numeric_columns=["a", "b"],
            categorical_columns=["c", "d"],
            scaler_type="robust",
            encoder_type="label",
            feature_order=["a", "b", "c", "d"],
            created_at="2024-01-01T00:00:00"
        )

        d = manifest.to_dict()
        loaded = TransformerManifest.from_dict(d)

        assert loaded.numeric_columns == manifest.numeric_columns
        assert loaded.categorical_columns == manifest.categorical_columns
        assert loaded.scaler_type == manifest.scaler_type
        assert loaded.encoder_type == manifest.encoder_type

    def test_bundle_serialization(self, training_data):
        from customer_retention.stages.preprocessing import TransformerBundle, TransformerManager

        manager = TransformerManager()
        manager.fit_transform(
            training_data,
            numeric_columns=["age"],
            categorical_columns=["gender"],
            exclude_columns=["customer_id", "target", "income", "tenure_months", "region", "plan_type"]
        )

        bundle_dict = manager._bundle.to_dict()
        loaded_bundle = TransformerBundle.from_dict(bundle_dict)

        assert loaded_bundle.scaler is not None
        assert "gender" in loaded_bundle.encoders
        assert loaded_bundle.manifest.scaler_type == "standard"

    @pytest.mark.parametrize("scaler_type", ["standard", "robust", "minmax"])
    def test_all_scaler_types_work(self, training_data, scoring_data, scaler_type):
        from customer_retention.stages.preprocessing import TransformerManager

        manager = TransformerManager(scaler_type=scaler_type)
        train_result = manager.fit_transform(
            training_data,
            numeric_columns=["age", "income"],
            categorical_columns=[],
            exclude_columns=["customer_id", "target", "tenure_months", "gender", "region", "plan_type"]
        )

        score_result = manager.transform(
            scoring_data,
            exclude_columns=["customer_id", "tenure_months", "gender", "region", "plan_type"]
        )

        assert not train_result["age"].isna().any()
        assert not score_result["age"].isna().any()

    def test_empty_columns_handled(self):
        from customer_retention.stages.preprocessing import TransformerManager

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })

        manager = TransformerManager()
        result = manager.fit_transform(
            df,
            numeric_columns=[],
            categorical_columns=[],
            exclude_columns=["id", "value"]
        )

        assert manager.is_fitted
