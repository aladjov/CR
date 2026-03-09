from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from customer_retention.analysis.auto_explorer.project_context import (
    CadenceInterval,
    DatasetProvenance,
    DatasetRegistryEntry,
    DatasetValidation,
    ExplorationContract,
    IntentConfig,
    KeyResolutionStep,
    MergeScaffoldEntry,
    ObjectiveAssessment,
    ObjectivePriority,
    ObjectiveSpec,
    ObjectiveSupport,
    ObjectiveSupportEntry,
    PredictionAnchor,
    PredictionObjective,
    ProjectContext,
    RawTimeColumnRole,
    SplitStrategy,
    TemporalPosture,
    mount_target_column,
)
from customer_retention.core.config.column_config import DatasetGranularity

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
TINY_PROFILES = FIXTURES / "3set_tiny_customer_profiles.csv"
TINY_TRANSACTIONS = FIXTURES / "3set_tiny_edi_transactions.csv"
TINY_TICKETS = FIXTURES / "3set_tiny_support_tickets.csv"


def _minimal_entry(**overrides):
    defaults = dict(name="profiles", path="data/profiles.csv", storage_format="csv")
    defaults.update(overrides)
    return DatasetRegistryEntry(**defaults)


def _immediate_spec(**overrides):
    defaults = dict(
        objective=PredictionObjective.IMMEDIATE_RISK,
        priority=ObjectivePriority.PRIMARY,
        anchor=PredictionAnchor.NOW,
    )
    defaults.update(overrides)
    return ObjectiveSpec(**defaults)


def _three_objectives(primary=PredictionObjective.IMMEDIATE_RISK):
    priorities = {
        PredictionObjective.IMMEDIATE_RISK: ObjectivePriority.SECONDARY,
        PredictionObjective.RENEWAL_RISK: ObjectivePriority.SECONDARY,
        PredictionObjective.DISENGAGEMENT: ObjectivePriority.EXPLORATORY,
    }
    priorities[primary] = ObjectivePriority.PRIMARY
    return [
        ObjectiveSpec(
            objective=PredictionObjective.IMMEDIATE_RISK,
            priority=priorities[PredictionObjective.IMMEDIATE_RISK],
            anchor=PredictionAnchor.NOW,
            assessment=ObjectiveAssessment(confidence=90, suggested_anchor=PredictionAnchor.NOW, rationale=["binary target"]),
        ),
        ObjectiveSpec(
            objective=PredictionObjective.RENEWAL_RISK,
            priority=priorities[PredictionObjective.RENEWAL_RISK],
            anchor=PredictionAnchor.CONTRACT,
            assessment=ObjectiveAssessment(confidence=50, suggested_anchor=PredictionAnchor.CONTRACT, rationale=["contract cols"]),
        ),
        ObjectiveSpec(
            objective=PredictionObjective.DISENGAGEMENT,
            priority=priorities[PredictionObjective.DISENGAGEMENT],
            anchor=PredictionAnchor.INACTIVITY,
            assessment=ObjectiveAssessment(confidence=40, suggested_anchor=PredictionAnchor.INACTIVITY, rationale=["event density"]),
        ),
    ]


def _minimal_context(**overrides):
    entry = _minimal_entry(
        entity_column="customer_id",
        target_candidates=["churned"],
        role="target",
    )
    defaults = dict(
        project_name="test_project",
        datasets={"profiles": entry},
        target_dataset="profiles",
        target_column="churned",
        entity_column="customer_id",
        objectives=[_immediate_spec()],
        primary_objective=PredictionObjective.IMMEDIATE_RISK,
    )
    defaults.update(overrides)
    return ProjectContext(**defaults)


class TestResolveDatasetName:
    def test_exact_path_match(self):
        ctx = _minimal_context(
            datasets={
                "support_tickets": _minimal_entry(
                    name="support_tickets",
                    path="../tests/fixtures/3set_support_tickets.csv",
                    entity_column="customer_id",
                    target_candidates=["churned"],
                    role="target",
                ),
            },
        )
        result = ctx.resolve_dataset_name("../tests/fixtures/3set_support_tickets.csv")
        assert result == "support_tickets"

    def test_stem_match_when_path_differs(self):
        ctx = _minimal_context(
            datasets={
                "support_tickets": _minimal_entry(
                    name="support_tickets",
                    path="../tests/fixtures/3set_support_tickets.csv",
                    entity_column="customer_id",
                    target_candidates=["churned"],
                    role="target",
                ),
            },
        )
        result = ctx.resolve_dataset_name("/abs/path/3set_support_tickets.csv")
        assert result == "support_tickets"

    def test_returns_none_when_no_match(self):
        ctx = _minimal_context()
        result = ctx.resolve_dataset_name("/some/unknown/file.csv")
        assert result is None

    def test_exact_path_takes_priority_over_stem(self):
        ctx = _minimal_context(
            datasets={
                "profiles": _minimal_entry(
                    name="profiles",
                    path="data/profiles.csv",
                    entity_column="customer_id",
                    target_candidates=["churned"],
                    role="target",
                ),
                "alt_profiles": _minimal_entry(
                    name="alt_profiles",
                    path="/other/dir/profiles.csv",
                ),
            },
        )
        result = ctx.resolve_dataset_name("data/profiles.csv")
        assert result == "profiles"

    def test_accepts_path_object(self):
        ctx = _minimal_context(
            datasets={
                "support_tickets": _minimal_entry(
                    name="support_tickets",
                    path="../tests/fixtures/3set_support_tickets.csv",
                    entity_column="customer_id",
                    target_candidates=["churned"],
                    role="target",
                ),
            },
        )
        result = ctx.resolve_dataset_name(Path("../tests/fixtures/3set_support_tickets.csv"))
        assert result == "support_tickets"


class TestEnumValues:
    def test_prediction_objective_values(self):
        assert PredictionObjective.IMMEDIATE_RISK == "immediate_risk"
        assert PredictionObjective.RENEWAL_RISK == "renewal_risk"
        assert PredictionObjective.DISENGAGEMENT == "disengagement"

    def test_prediction_anchor_values(self):
        assert PredictionAnchor.NOW == "now"
        assert PredictionAnchor.CONTRACT == "contract"
        assert PredictionAnchor.INACTIVITY == "inactivity"

    def test_temporal_posture_values(self):
        assert TemporalPosture.STABLE == "long_memory"
        assert TemporalPosture.REACTIVE == "short_memory"

    def test_raw_time_column_role_values(self):
        assert RawTimeColumnRole.EVENT_TIME == "event_time"
        assert RawTimeColumnRole.ENTITY_UPDATE_TIME == "entity_update_time"

    def test_enum_from_string(self):
        assert PredictionObjective("immediate_risk") is PredictionObjective.IMMEDIATE_RISK
        assert PredictionAnchor("contract") is PredictionAnchor.CONTRACT
        assert TemporalPosture("long_memory") is TemporalPosture.STABLE

    def test_objective_priority_values(self):
        assert ObjectivePriority.PRIMARY == "primary"
        assert ObjectivePriority.SECONDARY == "secondary"
        assert ObjectivePriority.EXPLORATORY == "exploratory"
        assert ObjectivePriority.DISABLED == "disabled"

    def test_objective_priority_from_string(self):
        assert ObjectivePriority("primary") is ObjectivePriority.PRIMARY
        assert ObjectivePriority("disabled") is ObjectivePriority.DISABLED


class TestObjectiveAssessment:
    def test_creation(self):
        a = ObjectiveAssessment(confidence=85, suggested_anchor=PredictionAnchor.NOW, rationale=["binary target"])
        assert a.confidence == 85
        assert a.suggested_anchor == PredictionAnchor.NOW
        assert a.rationale == ["binary target"]

    def test_defaults(self):
        a = ObjectiveAssessment(confidence=50, suggested_anchor=PredictionAnchor.NOW, rationale=[])
        assert a.feasibility is None

    def test_with_feasibility(self):
        a = ObjectiveAssessment(
            confidence=70, suggested_anchor=PredictionAnchor.CONTRACT,
            rationale=["contract cols"], feasibility={"span_days": 365},
        )
        assert a.feasibility["span_days"] == 365


class TestObjectiveSpec:
    def test_minimal_creation(self):
        spec = _immediate_spec()
        assert spec.objective == PredictionObjective.IMMEDIATE_RISK
        assert spec.priority == ObjectivePriority.PRIMARY
        assert spec.anchor == PredictionAnchor.NOW

    def test_defaults(self):
        spec = _immediate_spec()
        assert spec.parameters == {}
        assert spec.assessment is None

    def test_with_parameters(self):
        spec = _immediate_spec(parameters={"prediction_horizon_days": 90})
        assert spec.parameters["prediction_horizon_days"] == 90

    def test_with_assessment(self):
        assessment = ObjectiveAssessment(confidence=90, suggested_anchor=PredictionAnchor.NOW, rationale=["binary"])
        spec = _immediate_spec(assessment=assessment)
        assert spec.assessment.confidence == 90

    def test_anchor_defaults_to_none(self):
        spec = ObjectiveSpec(
            objective=PredictionObjective.RENEWAL_RISK,
            priority=ObjectivePriority.SECONDARY,
        )
        assert spec.anchor is None

    def test_effective_anchor_from_explicit(self):
        spec = ObjectiveSpec(
            objective=PredictionObjective.RENEWAL_RISK,
            priority=ObjectivePriority.SECONDARY,
            anchor=PredictionAnchor.CONTRACT,
            assessment=ObjectiveAssessment(confidence=50, suggested_anchor=PredictionAnchor.NOW, rationale=[]),
        )
        assert spec.effective_anchor == PredictionAnchor.CONTRACT

    def test_effective_anchor_falls_back_to_assessment(self):
        spec = ObjectiveSpec(
            objective=PredictionObjective.RENEWAL_RISK,
            priority=ObjectivePriority.SECONDARY,
            assessment=ObjectiveAssessment(confidence=50, suggested_anchor=PredictionAnchor.CONTRACT, rationale=[]),
        )
        assert spec.effective_anchor == PredictionAnchor.CONTRACT

    def test_effective_anchor_none_when_both_missing(self):
        spec = ObjectiveSpec(
            objective=PredictionObjective.RENEWAL_RISK,
            priority=ObjectivePriority.SECONDARY,
        )
        assert spec.effective_anchor is None


class TestDatasetRegistryEntry:
    def test_minimal_creation(self):
        entry = _minimal_entry()
        assert entry.name == "profiles"
        assert entry.path == "data/profiles.csv"
        assert entry.storage_format == "csv"

    def test_defaults(self):
        entry = _minimal_entry()
        assert entry.entity_column is None
        assert entry.time_column is None
        assert entry.raw_time_column_role is None
        assert entry.granularity is None
        assert entry.row_count is None
        assert entry.unique_entities is None
        assert entry.avg_rows_per_entity is None
        assert entry.target_candidates == []
        assert entry.role is None
        assert entry.join_keys == []
        assert entry.join_key is None
        assert entry.join_to is None
        assert entry.relationship is None
        assert entry.detected_schema == {}

    def test_full_creation(self):
        entry = DatasetRegistryEntry(
            name="transactions",
            path="data/tx.csv",
            storage_format="csv",
            entity_column="customer_id",
            time_column="event_timestamp",
            raw_time_column_role=RawTimeColumnRole.EVENT_TIME,
            granularity=DatasetGranularity.EVENT_LEVEL,
            row_count=50000,
            unique_entities=1000,
            avg_rows_per_entity=50.0,
            target_candidates=[],
            role="feature_source",
            join_keys=["customer_id"],
            join_to="profiles",
            relationship="many_to_one",
            detected_schema={"customer_id": "str", "amount": "float"},
        )
        assert entry.granularity == DatasetGranularity.EVENT_LEVEL
        assert entry.raw_time_column_role == RawTimeColumnRole.EVENT_TIME
        assert entry.row_count == 50000
        assert entry.join_keys == ["customer_id"]
        assert entry.join_key == "customer_id"

    def test_backward_compat_join_key_string(self):
        entry = DatasetRegistryEntry(
            name="transactions",
            path="data/tx.csv",
            join_key="customer_id",
        )
        assert entry.join_keys == ["customer_id"]
        assert entry.join_key == "customer_id"

    def test_provenance_initially_empty(self):
        entry = _minimal_entry()
        assert entry.provenance.landing_table is None
        assert entry.provenance.bronze_table is None
        assert entry.provenance.silver_table is None
        assert entry.provenance.gold_table is None

    def test_validation_initially_empty(self):
        entry = _minimal_entry()
        assert entry.validation.temporal_quality is None
        assert entry.validation.leakage_gates is None
        assert entry.validation.join_integrity is None


class TestDatasetProvenance:
    def test_defaults(self):
        prov = DatasetProvenance()
        assert prov.landing_table is None
        assert prov.bronze_table is None
        assert prov.recommendations == {}

    def test_update_individual_entry(self):
        prov = DatasetProvenance(
            bronze_table={"path": "bronze/tx", "delta_version": 1, "generated_at": "2025-01-01", "run_id": "abc"},
        )
        assert prov.bronze_table["path"] == "bronze/tx"
        assert prov.landing_table is None


class TestDatasetValidation:
    def test_defaults(self):
        val = DatasetValidation()
        assert val.temporal_quality is None
        assert val.leakage_gates is None
        assert val.join_integrity is None
        assert val.feature_timestamp_definition is None
        assert val.label_timestamp_definition is None

    def test_partial_population(self):
        val = DatasetValidation(temporal_quality={"coverage": 0.95})
        assert val.temporal_quality["coverage"] == 0.95
        assert val.leakage_gates is None


class TestProjectContextCreation:
    def test_minimal_creation(self):
        ctx = _minimal_context()
        assert ctx.project_name == "test_project"
        assert ctx.target_dataset == "profiles"
        assert ctx.target_column == "churned"
        assert ctx.entity_column == "customer_id"
        assert ctx.primary_objective == PredictionObjective.IMMEDIATE_RISK
        assert len(ctx.objectives) == 1
        assert ctx.objectives[0].objective == PredictionObjective.IMMEDIATE_RISK

    def test_defaults(self):
        ctx = _minimal_context()
        assert ctx.run_id is not None
        assert ctx.timezone == "UTC"
        assert ctx.storage_backend == "local"
        assert ctx.created_at is not None
        assert ctx.join_dataset_name is None
        assert ctx.temporal_posture == TemporalPosture.STABLE
        assert ctx.merge_scaffold == []

    def test_exploration_contract_defaults(self):
        ctx = _minimal_context()
        assert ctx.exploration_contract.views == ["full_history", "recent_behavior"]
        assert ctx.exploration_contract.insight_mapping_required is True

    def test_with_join_dataset_name(self):
        ctx = _minimal_context(join_dataset_name="retail_churn")
        assert ctx.join_dataset_name == "retail_churn"

    def test_with_multiple_datasets(self):
        profiles = _minimal_entry(
            name="profiles", entity_column="customer_id",
            target_candidates=["churned"], role="target",
        )
        transactions = DatasetRegistryEntry(
            name="transactions", path="data/tx.csv", storage_format="csv",
            entity_column="customer_id", time_column="event_timestamp",
            granularity=DatasetGranularity.EVENT_LEVEL,
            role="feature_source", join_key="customer_id", join_to="profiles",
            relationship="many_to_one",
        )
        ctx = _minimal_context(datasets={"profiles": profiles, "transactions": transactions})
        assert len(ctx.datasets) == 2

    def test_with_three_objectives(self):
        ctx = _minimal_context(objectives=_three_objectives())
        assert len(ctx.objectives) == 3
        assert ctx.primary_objective == PredictionObjective.IMMEDIATE_RISK

    def test_primary_spec_accessor(self):
        ctx = _minimal_context(objectives=_three_objectives())
        ps = ctx.primary_spec
        assert ps.objective == PredictionObjective.IMMEDIATE_RISK
        assert ps.priority == ObjectivePriority.PRIMARY

    def test_active_objectives(self):
        objs = _three_objectives()
        objs[2] = ObjectiveSpec(
            objective=PredictionObjective.DISENGAGEMENT,
            priority=ObjectivePriority.DISABLED,
            anchor=PredictionAnchor.INACTIVITY,
        )
        ctx = _minimal_context(objectives=objs, primary_objective=PredictionObjective.IMMEDIATE_RISK)
        active = ctx.active_objectives
        assert len(active) == 2
        assert all(o.priority != ObjectivePriority.DISABLED for o in active)


class TestProjectContextValidation:
    def test_no_objectives_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(objectives=[])

    def test_no_primary_objective_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(
                objectives=[_immediate_spec(priority=ObjectivePriority.SECONDARY)],
                primary_objective=PredictionObjective.IMMEDIATE_RISK,
            )

    def test_two_primaries_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(
                objectives=[
                    _immediate_spec(priority=ObjectivePriority.PRIMARY),
                    ObjectiveSpec(objective=PredictionObjective.RENEWAL_RISK, priority=ObjectivePriority.PRIMARY, anchor=PredictionAnchor.CONTRACT),
                ],
                primary_objective=PredictionObjective.IMMEDIATE_RISK,
            )

    def test_primary_objective_not_in_list_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(
                objectives=[_immediate_spec()],
                primary_objective=PredictionObjective.RENEWAL_RISK,
            )

    def test_primary_objective_mismatch_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(
                objectives=[
                    _immediate_spec(priority=ObjectivePriority.SECONDARY),
                    ObjectiveSpec(objective=PredictionObjective.RENEWAL_RISK, priority=ObjectivePriority.PRIMARY, anchor=PredictionAnchor.CONTRACT),
                ],
                primary_objective=PredictionObjective.IMMEDIATE_RISK,
            )

    def test_duplicate_objectives_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(
                objectives=[
                    _immediate_spec(),
                    _immediate_spec(priority=ObjectivePriority.SECONDARY),
                ],
                primary_objective=PredictionObjective.IMMEDIATE_RISK,
            )

    def test_invalid_temporal_posture_raises(self):
        with pytest.raises(ValueError):
            _minimal_context(temporal_posture="invalid_posture")


class TestProjectContextSerialization:
    def test_save_creates_file(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "project_context.yaml"
        ctx.save(out)
        assert out.exists()

    def test_save_yaml_structure(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "project_context.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert data["project_name"] == "test_project"
        assert data["target_dataset"] == "profiles"
        assert data["target_column"] == "churned"
        assert data["primary_objective"] == "immediate_risk"
        assert len(data["objectives"]) == 1
        assert data["objectives"][0]["objective"] == "immediate_risk"
        assert data["objectives"][0]["priority"] == "primary"
        assert "profiles" in data["datasets"]

    def test_load_from_yaml(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "project_context.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.project_name == "test_project"
        assert loaded.target_dataset == "profiles"
        assert loaded.primary_objective == PredictionObjective.IMMEDIATE_RISK
        assert loaded.objectives[0].anchor == PredictionAnchor.NOW

    def test_round_trip(self, tmp_path):
        profiles = _minimal_entry(
            name="profiles", entity_column="customer_id",
            target_candidates=["churned"], role="target",
        )
        transactions = DatasetRegistryEntry(
            name="transactions", path="data/tx.csv", storage_format="csv",
            entity_column="customer_id", time_column="event_timestamp",
            raw_time_column_role=RawTimeColumnRole.EVENT_TIME,
            granularity=DatasetGranularity.EVENT_LEVEL,
            role="feature_source", join_keys=["customer_id"], join_to="profiles",
            relationship="many_to_one",
        )
        ctx = ProjectContext(
            project_name="round_trip_test",
            datasets={"profiles": profiles, "transactions": transactions},
            target_dataset="profiles",
            target_column="churned",
            entity_column="customer_id",
            join_dataset_name="retail_churn",
            objectives=_three_objectives(primary=PredictionObjective.RENEWAL_RISK),
            primary_objective=PredictionObjective.RENEWAL_RISK,
            temporal_posture=TemporalPosture.REACTIVE,
            merge_scaffold=[
                MergeScaffoldEntry(
                    left_dataset="transactions", right_dataset="profiles",
                    join_keys=["customer_id"], relationship="many_to_one",
                ),
            ],
        )
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.project_name == "round_trip_test"
        assert loaded.join_dataset_name == "retail_churn"
        assert loaded.primary_objective == PredictionObjective.RENEWAL_RISK
        assert len(loaded.objectives) == 3
        renewal = [o for o in loaded.objectives if o.objective == PredictionObjective.RENEWAL_RISK][0]
        assert renewal.priority == ObjectivePriority.PRIMARY
        assert renewal.anchor == PredictionAnchor.CONTRACT
        assert renewal.assessment.confidence == 50
        assert loaded.temporal_posture == TemporalPosture.REACTIVE
        assert len(loaded.datasets) == 2
        assert loaded.datasets["transactions"].raw_time_column_role == RawTimeColumnRole.EVENT_TIME
        assert loaded.datasets["transactions"].granularity == DatasetGranularity.EVENT_LEVEL
        assert loaded.datasets["transactions"].join_keys == ["customer_id"]
        assert loaded.datasets["transactions"].join_key == "customer_id"
        assert len(loaded.merge_scaffold) == 1
        assert loaded.merge_scaffold[0].join_keys == ["customer_id"]
        assert loaded.merge_scaffold[0].join_key == "customer_id"

    def test_legacy_join_key_yaml_round_trip(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        raw = yaml.safe_load(out.read_text())
        raw["datasets"]["profiles"]["join_key"] = "customer_id"
        raw["datasets"]["profiles"].pop("join_keys", None)
        out.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].join_keys == ["customer_id"]
        assert loaded.datasets["profiles"].join_key == "customer_id"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProjectContext.load(tmp_path / "nope.yaml")

    def test_enum_serialized_as_strings(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        raw_text = out.read_text()
        assert "immediate_risk" in raw_text
        assert "primary" in raw_text

    def test_creates_parent_directories(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "nested" / "deep" / "project_context.yaml"
        ctx.save(out)
        assert out.exists()

    def test_objective_parameters_round_trip(self, tmp_path):
        spec = _immediate_spec(parameters={"prediction_horizon_days": 90})
        ctx = _minimal_context(objectives=[spec])
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.objectives[0].parameters["prediction_horizon_days"] == 90

    def test_assessment_round_trip(self, tmp_path):
        assessment = ObjectiveAssessment(
            confidence=85, suggested_anchor=PredictionAnchor.NOW,
            rationale=["binary target", "churn pattern"], feasibility={"positive_rate": 0.15},
        )
        spec = _immediate_spec(assessment=assessment)
        ctx = _minimal_context(objectives=[spec])
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        a = loaded.objectives[0].assessment
        assert a.confidence == 85
        assert a.rationale == ["binary target", "churn pattern"]
        assert a.feasibility["positive_rate"] == 0.15

    def test_legacy_posture_values_load_correctly(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        raw = yaml.safe_load(out.read_text())
        for old_key in ("training_posture", "temporal_posture"):
            for legacy_val, expected in [
                ("both", TemporalPosture.STABLE),
                ("stable", TemporalPosture.STABLE),
                ("reactive", TemporalPosture.REACTIVE),
            ]:
                patched = {**raw}
                patched.pop("temporal_posture", None)
                patched.pop("training_posture", None)
                patched[old_key] = legacy_val
                out.write_text(yaml.dump(patched, default_flow_style=False, sort_keys=False))
                loaded = ProjectContext.load(out)
                assert loaded.temporal_posture == expected, f"{old_key}={legacy_val}"


class TestProjectContextProvenanceRoundTrip:
    def test_initially_empty_provenance_round_trips(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        entry = loaded.datasets["profiles"]
        assert entry.provenance.bronze_table is None
        assert entry.provenance.silver_table is None

    def test_updated_provenance_round_trips(self, tmp_path):
        ctx = _minimal_context()
        ctx.datasets["profiles"].provenance.bronze_table = {
            "path": "bronze/profiles", "delta_version": 1,
            "generated_at": "2025-01-01", "run_id": "abc",
        }
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].provenance.bronze_table["path"] == "bronze/profiles"

    def test_per_dataset_provenance_isolation(self, tmp_path):
        profiles = _minimal_entry(
            name="profiles", entity_column="customer_id",
            target_candidates=["churned"], role="target",
        )
        transactions = DatasetRegistryEntry(
            name="transactions", path="data/tx.csv", storage_format="csv",
            role="feature_source",
        )
        ctx = _minimal_context(datasets={"profiles": profiles, "transactions": transactions})
        ctx.datasets["profiles"].provenance.bronze_table = {"path": "bronze/profiles"}
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].provenance.bronze_table is not None
        assert loaded.datasets["transactions"].provenance.bronze_table is None


class TestProjectContextValidationRoundTrip:
    def test_initially_empty_validation(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].validation.temporal_quality is None

    def test_updated_validation_round_trips(self, tmp_path):
        ctx = _minimal_context()
        ctx.datasets["profiles"].validation.temporal_quality = {"coverage": 0.95}
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].validation.temporal_quality["coverage"] == 0.95

    def test_per_dataset_validation_isolation(self, tmp_path):
        profiles = _minimal_entry(
            name="profiles", entity_column="customer_id",
            target_candidates=["churned"], role="target",
        )
        transactions = DatasetRegistryEntry(
            name="transactions", path="data/tx.csv", storage_format="csv",
            role="feature_source",
        )
        ctx = _minimal_context(datasets={"profiles": profiles, "transactions": transactions})
        ctx.datasets["profiles"].validation.leakage_gates = {"passed": True}
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.datasets["profiles"].validation.leakage_gates is not None
        assert loaded.datasets["transactions"].validation.leakage_gates is None


class TestExplorationContract:
    def test_default_views(self):
        contract = ExplorationContract()
        assert contract.views == ["full_history", "recent_behavior"]

    def test_insight_mapping_default(self):
        contract = ExplorationContract()
        assert contract.insight_mapping_required is True

    def test_round_trip(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.exploration_contract.views == ["full_history", "recent_behavior"]
        assert loaded.exploration_contract.insight_mapping_required is True


class TestForSingleDataset:
    def test_creates_single_entry(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
        )
        assert len(ctx.datasets) == 1
        assert "profiles" in ctx.datasets

    def test_sets_target_fields(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
        )
        assert ctx.target_dataset == "profiles"
        assert ctx.target_column == "churned"
        assert ctx.entity_column == "customer_id"

    def test_entry_role_is_target(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
        )
        assert ctx.datasets["profiles"].role == "target"

    def test_default_primary_objective(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
        )
        assert ctx.primary_objective == PredictionObjective.IMMEDIATE_RISK
        assert ctx.objectives[0].anchor == PredictionAnchor.NOW

    def test_custom_objective(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
            primary_objective=PredictionObjective.RENEWAL_RISK,
            primary_anchor=PredictionAnchor.CONTRACT,
        )
        assert ctx.primary_objective == PredictionObjective.RENEWAL_RISK
        assert ctx.objectives[0].anchor == PredictionAnchor.CONTRACT

    def test_with_join_dataset_name(self):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
            join_dataset_name="retail_churn",
        )
        assert ctx.join_dataset_name == "retail_churn"

    def test_round_trip(self, tmp_path):
        ctx = ProjectContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned", entity_column="customer_id",
            primary_objective=PredictionObjective.RENEWAL_RISK,
            primary_anchor=PredictionAnchor.CONTRACT,
        )
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.target_dataset == "profiles"
        assert loaded.primary_objective == PredictionObjective.RENEWAL_RISK
        assert loaded.objectives[0].anchor == PredictionAnchor.CONTRACT


class TestMergeScaffoldEntry:
    def test_creation(self):
        entry = MergeScaffoldEntry(
            left_dataset="transactions",
            right_dataset="profiles",
            join_keys=["customer_id"],
            relationship="many_to_one",
        )
        assert entry.left_dataset == "transactions"
        assert entry.join_keys == ["customer_id"]
        assert entry.join_key == "customer_id"

    def test_backward_compat_join_key_string(self):
        entry = MergeScaffoldEntry(
            left_dataset="transactions",
            right_dataset="profiles",
            join_key="customer_id",
            relationship="many_to_one",
        )
        assert entry.join_keys == ["customer_id"]
        assert entry.join_key == "customer_id"

    def test_composite_keys(self):
        entry = MergeScaffoldEntry(
            left_dataset="transactions",
            right_dataset="profiles",
            join_keys=["customer_id", "as_of_date"],
            relationship="many_to_one",
        )
        assert entry.join_keys == ["customer_id", "as_of_date"]
        assert entry.join_key == "customer_id"


class TestMountTargetColumn:
    def _make_context(self):
        profiles_entry = DatasetRegistryEntry(
            name="3set_tiny_customer_profiles",
            path=str(TINY_PROFILES),
            storage_format="csv",
            entity_column="customer_id",
            target_candidates=["churned"],
            role="target",
        )
        transactions_entry = DatasetRegistryEntry(
            name="3set_tiny_edi_transactions",
            path=str(TINY_TRANSACTIONS),
            storage_format="csv",
            entity_column="customer_id",
            role="feature_source",
            join_key="customer_id",
            join_to="3set_tiny_customer_profiles",
            relationship="many_to_one",
        )
        tickets_entry = DatasetRegistryEntry(
            name="3set_tiny_support_tickets",
            path=str(TINY_TICKETS),
            storage_format="csv",
            entity_column="customer_id",
            role="feature_source",
            join_key="customer_id",
            join_to="3set_tiny_customer_profiles",
            relationship="many_to_one",
        )
        return ProjectContext(
            project_name="test",
            datasets={
                "3set_tiny_customer_profiles": profiles_entry,
                "3set_tiny_edi_transactions": transactions_entry,
                "3set_tiny_support_tickets": tickets_entry,
            },
            target_dataset="3set_tiny_customer_profiles",
            target_column="churned",
            entity_column="customer_id",
            objectives=[_immediate_spec()],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )

    def test_mount_adds_target_column(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "churned" in result.columns

    def test_mount_returns_info(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        _, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert isinstance(info, dict)
        assert "target_mounted_from" in info
        assert "target_mount_key" in info

    def test_mount_preserves_original_columns(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        original_cols = list(df.columns)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        for col in original_cols:
            assert col in result.columns

    def test_mount_preserves_row_count(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert len(result) == len(df)

    def test_mount_target_values_correct(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        c1_row = result[result["customer_id"] == "C00001"].iloc[0]
        c2_row = result[result["customer_id"] == "C00002"].iloc[0]
        assert c1_row["churned"] == 0
        assert c2_row["churned"] == 1

    def test_mount_no_context_returns_none(self):
        import pandas as pd
        ctx = ProjectContext(
            project_name="empty",
            datasets={},
            objectives=[_immediate_spec()],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert info is None

    def test_mount_target_dataset_returns_none(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_PROFILES)
        result, info = mount_target_column(df, ctx, "3set_tiny_customer_profiles")
        assert info is None

    def test_mount_with_renamed_entity_column(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        df = df.rename(columns={"customer_id": "entity_id"})
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "churned" in result.columns
        assert info is not None

    def test_mount_with_missing_join_key_and_no_entity_id(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        df = df.rename(columns={"customer_id": "some_other_col"})
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert info is None

    def test_mount_no_join_key_returns_none(self):
        import pandas as pd
        ctx = self._make_context()
        ctx.datasets["3set_tiny_edi_transactions"].join_keys = []
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert info is None


class TestMountTargetWithLabelTimestamp:
    def _make_context(self):
        profiles_entry = DatasetRegistryEntry(
            name="3set_tiny_customer_profiles",
            path=str(TINY_PROFILES),
            storage_format="csv",
            entity_column="customer_id",
            target_candidates=["churned"],
            role="target",
        )
        transactions_entry = DatasetRegistryEntry(
            name="3set_tiny_edi_transactions",
            path=str(TINY_TRANSACTIONS),
            storage_format="csv",
            entity_column="customer_id",
            role="feature_source",
            join_key="customer_id",
            join_to="3set_tiny_customer_profiles",
            relationship="many_to_one",
        )
        return ProjectContext(
            project_name="test",
            datasets={
                "3set_tiny_customer_profiles": profiles_entry,
                "3set_tiny_edi_transactions": transactions_entry,
            },
            target_dataset="3set_tiny_customer_profiles",
            target_column="churned",
            entity_column="customer_id",
            objectives=[_immediate_spec()],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )

    def _make_target_label_df(self):
        import pandas as pd
        return pd.DataFrame({
            "entity_id": ["C00001", "C00002", "C00003"],
            "target": [0, 1, 0],
            "label_timestamp": pd.to_datetime(["2025-01-15", "2025-03-01", "2025-02-01"]),
            "label_available_flag": [True, True, False],
        })

    def test_mount_with_label_df_adds_target(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        target_df = self._make_target_label_df()
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions", target_label_df=target_df)
        assert ctx.target_column in result.columns

    def test_mount_with_label_df_adds_label_timestamp(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        target_df = self._make_target_label_df()
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions", target_label_df=target_df)
        assert "label_timestamp" in result.columns

    def test_mount_with_label_df_preserves_row_count(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        target_df = self._make_target_label_df()
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions", target_label_df=target_df)
        assert len(result) == len(df)

    def test_mount_info_records_label_mounted(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        target_df = self._make_target_label_df()
        _, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions", target_label_df=target_df)
        assert info["label_timestamp_mounted"] is True

    def test_mount_without_label_df_no_label_timestamp(self):
        import pandas as pd
        ctx = self._make_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "label_timestamp" not in result.columns
        assert info.get("label_timestamp_mounted") is not True


class TestObjectiveSupportEntry:
    def test_creation(self):
        entry = ObjectiveSupportEntry(
            signals={"entity_ratio": 0.8, "regime_shift": False},
            implications=["Recent data covers most entities"],
            strength="strong",
        )
        assert entry.signals["entity_ratio"] == 0.8
        assert len(entry.implications) == 1
        assert entry.strength == "strong"

    def test_empty_signals(self):
        entry = ObjectiveSupportEntry(signals={}, implications=[], strength="weak")
        assert entry.signals == {}
        assert entry.implications == []

    def test_multiple_implications(self):
        entry = ObjectiveSupportEntry(
            signals={"drift_pct": 0.5},
            implications=["High drift detected", "Short-window features unreliable"],
            strength="medium",
        )
        assert len(entry.implications) == 2


class TestObjectiveSupport:
    def test_creation(self):
        entries = {
            PredictionObjective.IMMEDIATE_RISK: ObjectiveSupportEntry(
                signals={"entity_ratio": 0.9},
                implications=["Good recent coverage"],
                strength="strong",
            ),
        }
        support = ObjectiveSupport(entries=entries)
        assert support.scope == "dataset"
        assert support.lifecycle_stage == "evidence"
        assert PredictionObjective.IMMEDIATE_RISK in support.entries

    def test_defaults(self):
        support = ObjectiveSupport(entries={})
        assert support.scope == "dataset"
        assert support.lifecycle_stage == "evidence"

    def test_multiple_objectives(self):
        entries = {
            PredictionObjective.IMMEDIATE_RISK: ObjectiveSupportEntry(
                signals={}, implications=[], strength="strong",
            ),
            PredictionObjective.DISENGAGEMENT: ObjectiveSupportEntry(
                signals={}, implications=[], strength="medium",
            ),
            PredictionObjective.RENEWAL_RISK: ObjectiveSupportEntry(
                signals={}, implications=[], strength="weak",
            ),
        }
        support = ObjectiveSupport(entries=entries)
        assert len(support.entries) == 3


class TestObjectiveSupportOnProjectContext:
    def test_default_is_none(self):
        ctx = _minimal_context()
        assert ctx.objective_support is None

    def test_set_objective_support(self):
        entries = {
            PredictionObjective.IMMEDIATE_RISK: ObjectiveSupportEntry(
                signals={"entity_ratio": 0.9},
                implications=["Good coverage"],
                strength="strong",
            ),
        }
        support = ObjectiveSupport(entries=entries)
        ctx = _minimal_context(objective_support=support)
        assert ctx.objective_support is not None
        assert ctx.objective_support.entries[PredictionObjective.IMMEDIATE_RISK].strength == "strong"

    def test_round_trip(self, tmp_path):
        entries = {
            PredictionObjective.IMMEDIATE_RISK: ObjectiveSupportEntry(
                signals={"entity_ratio": 0.9, "regime_shift": False},
                implications=["Good recent coverage", "No regime shift"],
                strength="strong",
            ),
            PredictionObjective.DISENGAGEMENT: ObjectiveSupportEntry(
                signals={"coverage_span_days": 365},
                implications=["Sufficient history"],
                strength="medium",
            ),
        }
        support = ObjectiveSupport(entries=entries)
        ctx = _minimal_context(objective_support=support)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.objective_support is not None
        assert loaded.objective_support.scope == "dataset"
        ir_entry = loaded.objective_support.entries[PredictionObjective.IMMEDIATE_RISK]
        assert ir_entry.strength == "strong"
        assert ir_entry.signals["entity_ratio"] == 0.9
        assert len(ir_entry.implications) == 2

    def test_round_trip_none(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.objective_support is None

    def test_yaml_structure(self, tmp_path):
        entries = {
            PredictionObjective.IMMEDIATE_RISK: ObjectiveSupportEntry(
                signals={"entity_ratio": 0.9},
                implications=["test"],
                strength="strong",
            ),
        }
        support = ObjectiveSupport(entries=entries)
        ctx = _minimal_context(objective_support=support)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert "objective_support" in data
        assert data["objective_support"]["scope"] == "dataset"
        assert "immediate_risk" in data["objective_support"]["entries"]


class TestIntentConfig:
    def test_default_values(self):
        ic = IntentConfig()
        assert ic.prediction_horizons == [30, 60, 90]
        assert ic.recent_window_days == 270
        assert ic.observation_window_days == 270
        assert ic.purge_gap_days == 104
        assert ic.label_window_days == 90
        assert ic.temporal_split is True

    def test_custom_values(self):
        ic = IntentConfig(
            prediction_horizons=[15, 30],
            recent_window_days=180,
            observation_window_days=180,
            purge_gap_days=44,
            label_window_days=30,
            temporal_split=False,
        )
        assert ic.prediction_horizons == [15, 30]
        assert ic.recent_window_days == 180
        assert ic.observation_window_days == 180
        assert ic.purge_gap_days == 44
        assert ic.label_window_days == 30
        assert ic.temporal_split is False

    def test_default_is_none_on_context(self):
        ctx = _minimal_context()
        assert ctx.intent is None

    def test_set_intent_on_context(self):
        intent = IntentConfig(recent_window_days=365)
        ctx = _minimal_context(intent=intent)
        assert ctx.intent is not None
        assert ctx.intent.recent_window_days == 365

    def test_yaml_round_trip_with_intent(self, tmp_path):
        intent = IntentConfig(
            prediction_horizons=[15, 30],
            recent_window_days=180,
            observation_window_days=180,
            purge_gap_days=44,
            label_window_days=30,
            temporal_split=False,
        )
        ctx = _minimal_context(intent=intent)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.intent is not None
        assert loaded.intent.prediction_horizons == [15, 30]
        assert loaded.intent.recent_window_days == 180
        assert loaded.intent.observation_window_days == 180
        assert loaded.intent.purge_gap_days == 44
        assert loaded.intent.label_window_days == 30
        assert loaded.intent.temporal_split is False

    def test_yaml_round_trip_none_intent(self, tmp_path):
        ctx = _minimal_context()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.intent is None

    def test_prediction_horizons_list(self):
        ic = IntentConfig(prediction_horizons=[30, 60, 90])
        assert ic.prediction_horizons == [30, 60, 90]
        assert isinstance(ic.prediction_horizons, list)

    def test_cadence_interval_default(self):
        ic = IntentConfig()
        assert ic.cadence_interval == CadenceInterval.WEEKLY

    def test_split_strategy_default(self):
        ic = IntentConfig()
        assert ic.split_strategy == SplitStrategy.TEMPORAL

    def test_custom_cadence_and_split(self):
        ic = IntentConfig(
            cadence_interval=CadenceInterval.DAILY,
            split_strategy=SplitStrategy.COHORT_BASED,
        )
        assert ic.cadence_interval == CadenceInterval.DAILY
        assert ic.split_strategy == SplitStrategy.COHORT_BASED

    def test_yaml_round_trip_cadence(self, tmp_path):
        intent = IntentConfig(
            cadence_interval=CadenceInterval.BIWEEKLY,
            split_strategy=SplitStrategy.COHORT_BASED,
        )
        ctx = _minimal_context(intent=intent)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.intent.cadence_interval == CadenceInterval.BIWEEKLY
        assert loaded.intent.split_strategy == SplitStrategy.COHORT_BASED

    def test_history_upper_limit_default_none(self):
        ic = IntentConfig()
        assert ic.history_upper_limit is None

    def test_lookback_periods_default_none(self):
        ic = IntentConfig()
        assert ic.lookback_periods is None

    def test_history_upper_limit_set(self):
        ic = IntentConfig(history_upper_limit="2024-12-31")
        assert ic.history_upper_limit == "2024-12-31"

    def test_lookback_periods_set(self):
        ic = IntentConfig(lookback_periods=52)
        assert ic.lookback_periods == 52

    def test_lookback_periods_rejects_zero(self):
        with pytest.raises(ValueError):
            IntentConfig(lookback_periods=0)

    def test_lookback_periods_rejects_negative(self):
        with pytest.raises(ValueError):
            IntentConfig(lookback_periods=-5)

    def test_history_window_yaml_round_trip(self, tmp_path):
        intent = IntentConfig(history_upper_limit="2024-06-30", lookback_periods=26)
        ctx = _minimal_context(intent=intent)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.intent.history_upper_limit == "2024-06-30"
        assert loaded.intent.lookback_periods == 26

    def test_history_window_none_yaml_round_trip(self, tmp_path):
        intent = IntentConfig()
        ctx = _minimal_context(intent=intent)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.intent.history_upper_limit is None
        assert loaded.intent.lookback_periods is None


class TestLightRunAndSampleFraction:
    def test_light_run_default_false(self):
        ctx = _minimal_context()
        assert ctx.light_run is False

    def test_light_run_true_roundtrip(self, tmp_path):
        ctx = _minimal_context(light_run=True)
        assert ctx.light_run is True
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.light_run is True

    def test_sample_fraction_default_none(self):
        ctx = _minimal_context()
        assert ctx.sample_fraction is None

    def test_sample_fraction_valid_roundtrip(self, tmp_path):
        ctx = _minimal_context(sample_fraction=0.1)
        assert ctx.sample_fraction == 0.1
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.sample_fraction == 0.1

    def test_sample_fraction_rejects_zero(self):
        with pytest.raises(ValueError):
            _minimal_context(sample_fraction=0.0)

    def test_sample_fraction_rejects_negative(self):
        with pytest.raises(ValueError):
            _minimal_context(sample_fraction=-0.5)

    def test_sample_fraction_rejects_over_one(self):
        with pytest.raises(ValueError):
            _minimal_context(sample_fraction=1.5)

    def test_sample_fraction_accepts_one(self):
        ctx = _minimal_context(sample_fraction=1.0)
        assert ctx.sample_fraction == 1.0

    def test_sample_filters_default_none(self):
        ctx = _minimal_context()
        assert ctx.sample_filters is None

    def test_sample_filters_roundtrip(self, tmp_path):
        filters = {"transactions": "amount > 100", "customers": "region == 'US'"}
        ctx = _minimal_context(sample_filters=filters)
        assert ctx.sample_filters == filters
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.sample_filters == filters

    def test_sample_filters_empty_dict_roundtrip(self, tmp_path):
        ctx = _minimal_context(sample_filters={})
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = ProjectContext.load(out)
        assert loaded.sample_filters == {}


class TestCadenceIntervalEnum:
    def test_values(self):
        assert CadenceInterval.DAILY == "daily"
        assert CadenceInterval.WEEKLY == "weekly"
        assert CadenceInterval.BIWEEKLY == "biweekly"
        assert CadenceInterval.MONTHLY == "monthly"

    def test_from_string(self):
        assert CadenceInterval("daily") is CadenceInterval.DAILY
        assert CadenceInterval("biweekly") is CadenceInterval.BIWEEKLY


class TestSplitStrategyEnum:
    def test_values(self):
        assert SplitStrategy.TEMPORAL == "temporal"
        assert SplitStrategy.COHORT_BASED == "cohort_based"

    def test_from_string(self):
        assert SplitStrategy("temporal") is SplitStrategy.TEMPORAL
        assert SplitStrategy("cohort_based") is SplitStrategy.COHORT_BASED


class TestKeyResolutionStep:
    def test_creation_with_all_fields(self):
        step = KeyResolutionStep(
            bridge_dataset="case",
            source_key="CASE_ID",
            bridge_key="CASE_ID",
            resolve_column="ACCOUNT_ID",
        )
        assert step.bridge_dataset == "case"
        assert step.source_key == "CASE_ID"
        assert step.bridge_key == "CASE_ID"
        assert step.resolve_column == "ACCOUNT_ID"

    def test_dataset_registry_entry_with_key_resolution(self):
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        entry = _minimal_entry(key_resolution=steps)
        assert len(entry.key_resolution) == 1
        assert entry.key_resolution[0].bridge_dataset == "case"

    def test_dataset_registry_entry_default_empty_key_resolution(self):
        entry = _minimal_entry()
        assert entry.key_resolution == []

    def test_yaml_roundtrip_with_key_resolution(self, tmp_path):
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
            KeyResolutionStep(
                bridge_dataset="account",
                source_key="ACCOUNT_ID",
                bridge_key="ACCOUNT_ID",
                resolve_column="PARENT_ACCOUNT_ID",
            ),
        ]
        entry = DatasetRegistryEntry(
            name="case_history",
            path="data/case_history.csv",
            storage_format="csv",
            key_resolution=steps,
        )
        ctx = ProjectContext(
            project_name="roundtrip_test",
            datasets={"case_history": entry},
            objectives=[_immediate_spec()],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        p = tmp_path / "ctx.yaml"
        ctx.save(p)
        loaded = ProjectContext.load(p)
        loaded_entry = loaded.datasets["case_history"]
        assert len(loaded_entry.key_resolution) == 2
        assert loaded_entry.key_resolution[0].bridge_dataset == "case"
        assert loaded_entry.key_resolution[0].source_key == "CASE_ID"
        assert loaded_entry.key_resolution[1].resolve_column == "PARENT_ACCOUNT_ID"
