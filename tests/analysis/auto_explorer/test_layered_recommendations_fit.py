from dataclasses import asdict

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    LayeredRecommendation,
    RecommendationRegistry,
)


class TestLayeredRecommendationFitArtifactId:
    def test_default_fit_artifact_id_is_none(self):
        rec = LayeredRecommendation(
            id="gold_scale_income",
            layer="gold",
            category="scaling",
            action="standard",
            target_column="income",
            parameters={"method": "standard"},
            rationale="Normalize for modeling",
            source_notebook="06",
        )
        assert rec.fit_artifact_id is None

    def test_creates_with_fit_artifact_id(self):
        rec = LayeredRecommendation(
            id="gold_scale_income",
            layer="gold",
            category="scaling",
            action="standard",
            target_column="income",
            parameters={"method": "standard"},
            rationale="Normalize for modeling",
            source_notebook="06",
            fit_artifact_id="income_scaler",
        )
        assert rec.fit_artifact_id == "income_scaler"

    def test_serializes_fit_artifact_id_to_dict(self):
        rec = LayeredRecommendation(
            id="gold_encode_region",
            layer="gold",
            category="encoding",
            action="label",
            target_column="region",
            parameters={"method": "label"},
            rationale="Encode for modeling",
            source_notebook="06",
            fit_artifact_id="region_encoder",
        )
        d = asdict(rec)
        assert d["fit_artifact_id"] == "region_encoder"

    def test_serializes_none_fit_artifact_id(self):
        rec = LayeredRecommendation(
            id="gold_encode_region",
            layer="gold",
            category="encoding",
            action="label",
            target_column="region",
            parameters={},
            rationale="",
            source_notebook="",
        )
        d = asdict(rec)
        assert d["fit_artifact_id"] is None


class TestRecommendationRegistryFitArtifacts:
    def test_registry_has_fit_artifacts_dict(self):
        registry = RecommendationRegistry()
        assert hasattr(registry, "fit_artifacts")
        assert isinstance(registry.fit_artifacts, dict)

    def test_link_fit_artifact_stores_mapping(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "Normalize", "06")
        rec_id = registry.gold.scaling[0].id
        registry.link_fit_artifact(rec_id, "income_scaler")
        assert registry.fit_artifacts[rec_id] == "income_scaler"

    def test_get_fit_artifact_returns_artifact_id(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "", "")
        rec_id = registry.gold.scaling[0].id
        registry.link_fit_artifact(rec_id, "income_scaler")
        assert registry.get_fit_artifact(rec_id) == "income_scaler"

    def test_get_fit_artifact_returns_none_if_not_linked(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "", "")
        rec_id = registry.gold.scaling[0].id
        assert registry.get_fit_artifact(rec_id) is None

    def test_link_multiple_fit_artifacts(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "", "")
        registry.add_gold_scaling("age", "minmax", "", "")
        registry.add_gold_encoding("region", "label", "", "")
        income_id = registry.gold.scaling[0].id
        age_id = registry.gold.scaling[1].id
        region_id = registry.gold.encoding[0].id
        registry.link_fit_artifact(income_id, "income_scaler")
        registry.link_fit_artifact(age_id, "age_scaler")
        registry.link_fit_artifact(region_id, "region_encoder")
        assert len(registry.fit_artifacts) == 3


class TestRecommendationRegistryFitArtifactsSerialization:
    def test_to_dict_includes_fit_artifacts(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "", "")
        rec_id = registry.gold.scaling[0].id
        registry.link_fit_artifact(rec_id, "income_scaler")
        d = registry.to_dict()
        assert "fit_artifacts" in d
        assert d["fit_artifacts"][rec_id] == "income_scaler"

    def test_from_dict_restores_fit_artifacts(self):
        data = {
            "gold": {
                "target_column": "churned",
                "encoding": [],
                "scaling": [{
                    "id": "gold_scaling_income",
                    "layer": "gold",
                    "category": "scaling",
                    "action": "standard",
                    "target_column": "income",
                    "parameters": {"method": "standard"},
                    "rationale": "Normalize",
                    "source_notebook": "06",
                    "priority": 1,
                    "dependencies": [],
                    "fit_artifact_id": None,
                }],
                "transformations": [],
                "feature_selection": [],
            },
            "fit_artifacts": {"gold_scaling_income": "income_scaler"},
        }
        registry = RecommendationRegistry.from_dict(data)
        assert registry.get_fit_artifact("gold_scaling_income") == "income_scaler"

    def test_from_dict_handles_missing_fit_artifacts(self):
        data = {
            "gold": {
                "target_column": "churned",
                "encoding": [],
                "scaling": [],
                "transformations": [],
                "feature_selection": [],
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.fit_artifacts) == 0

    def test_roundtrip_preserves_fit_artifacts(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_scaling("income", "standard", "", "")
        registry.add_gold_encoding("region", "label", "", "")
        income_id = registry.gold.scaling[0].id
        region_id = registry.gold.encoding[0].id
        registry.link_fit_artifact(income_id, "income_scaler")
        registry.link_fit_artifact(region_id, "region_encoder")
        d = registry.to_dict()
        restored = RecommendationRegistry.from_dict(d)
        assert restored.get_fit_artifact(income_id) == "income_scaler"
        assert restored.get_fit_artifact(region_id) == "region_encoder"


class TestLayeredRecommendationFromDict:
    def test_from_dict_restores_fit_artifact_id(self):
        data = {
            "id": "gold_scale_income",
            "layer": "gold",
            "category": "scaling",
            "action": "standard",
            "target_column": "income",
            "parameters": {"method": "standard"},
            "rationale": "Normalize",
            "source_notebook": "06",
            "priority": 1,
            "dependencies": [],
            "fit_artifact_id": "income_scaler",
        }
        rec = LayeredRecommendation(**data)
        assert rec.fit_artifact_id == "income_scaler"

    def test_from_dict_handles_missing_fit_artifact_id(self):
        data = {
            "id": "gold_scale_income",
            "layer": "gold",
            "category": "scaling",
            "action": "standard",
            "target_column": "income",
            "parameters": {},
            "rationale": "",
            "source_notebook": "",
            "priority": 1,
            "dependencies": [],
        }
        rec = LayeredRecommendation(**data)
        assert rec.fit_artifact_id is None
