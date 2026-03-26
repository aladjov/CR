from __future__ import annotations

from customer_retention.analysis.auto_explorer.project_context import (
    DatasetRegistryEntry,
    KeyResolutionStep,
    ObjectivePriority,
    ObjectiveSpec,
    PredictionObjective,
    ProjectContext,
)
from customer_retention.analysis.auto_explorer.service_unit_detector import (
    ServiceUnitConfig,
    classify_dataset_linkage,
    detect_service_unit,
)
from customer_retention.core.compat import pd


def _sps_context(entity_col="ACCOUNT_ID") -> ProjectContext:
    return ProjectContext(
        project_name="test",
        entity_column=entity_col,
        datasets={
            "account": DatasetRegistryEntry(name="account", path="/tmp/account"),
            "contract": DatasetRegistryEntry(name="contract", path="/tmp/contract"),
            "subscription": DatasetRegistryEntry(
                name="subscription", path="/tmp/subscription",
                key_resolution=[KeyResolutionStep(
                    bridge_dataset="contract", source_key="CONTRACT_ID",
                    bridge_key="CONTRACT_ID", resolve_column="ACCOUNT_ID",
                )],
            ),
            "case": DatasetRegistryEntry(name="case", path="/tmp/case"),
        },
        objectives=[ObjectiveSpec(
            objective=PredictionObjective.IMMEDIATE_RISK,
            priority=ObjectivePriority.PRIMARY,
        )],
        primary_objective=PredictionObjective.IMMEDIATE_RISK,
    )


def _sps_frames() -> dict[str, pd.DataFrame]:
    return {
        "account": pd.DataFrame({
            "ACCOUNT_ID": ["A1", "A2", "A3"],
            "ACCOUNT_NAME": ["Foo", "Bar", "Baz"],
            "INDUSTRY_SPS": ["Tech", "Retail", None],
        }),
        "contract": pd.DataFrame({
            "CONTRACT_ID": ["C1", "C2", "C3", "C4"],
            "ACCOUNT_ID": ["A1", "A1", "A2", "A3"],
            "CONTRACT_START_DATE": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-01-01", "2023-03-01"]),
            "CONTRACT_END_DATE": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01", "2024-03-01"]),
            "CONTRACT_STATUS": ["Cancelled", "Active", "Active", "Terminated"],
            "BILLING_TERMINATION_DATE": pd.to_datetime(["2024-01-15", None, None, "2024-03-15"]),
            "CANCELLATION_REASON": ["Price", None, None, "Service"],
        }),
        "subscription": pd.DataFrame({
            "SUBSCRIPTION_ID": ["S1", "S2"],
            "CONTRACT_ID": ["C1", "C2"],
            "SUBSCRIPTION_STATUS": ["Cancelled", "Active"],
        }),
        "case": pd.DataFrame({
            "CASE_ID": ["CS1", "CS2"],
            "ACCOUNT_ID": ["A1", "A2"],
            "CASE_STATUS": ["Closed", "Open"],
        }),
    }


class TestDetectServiceUnit:
    def test_detect_sps_contract_table(self):
        result = detect_service_unit(_sps_context(), _sps_frames())
        assert result is not None
        assert result.dataset_name == "contract"
        assert result.unit_id_column == "CONTRACT_ID"
        assert result.entity_column == "ACCOUNT_ID"
        assert result.anchor_date_column == "CONTRACT_END_DATE"
        assert result.start_date_column == "CONTRACT_START_DATE"

    def test_detect_terminated_statuses(self):
        result = detect_service_unit(_sps_context(), _sps_frames())
        assert result is not None
        assert "Cancelled" in result.terminated_statuses
        assert "Terminated" in result.terminated_statuses

    def test_detect_status_column(self):
        result = detect_service_unit(_sps_context(), _sps_frames())
        assert result is not None
        assert result.status_column == "CONTRACT_STATUS"

    def test_detect_secondary_anchors(self):
        result = detect_service_unit(_sps_context(), _sps_frames())
        assert result is not None
        assert "BILLING_TERMINATION_DATE" in result.secondary_anchor_columns

    def test_no_detection_for_entity_only(self):
        ctx = ProjectContext(
            project_name="test", entity_column="ACCOUNT_ID",
            datasets={"account": DatasetRegistryEntry(name="account", path="/tmp/a")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        frames = {"account": pd.DataFrame({
            "ACCOUNT_ID": ["A1"], "NAME": ["Foo"],
        })}
        assert detect_service_unit(ctx, frames) is None

    def test_no_entity_column_returns_none(self):
        ctx = ProjectContext(
            project_name="test",
            datasets={"x": DatasetRegistryEntry(name="x", path="/tmp/x")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        assert detect_service_unit(ctx, {"x": pd.DataFrame({"a": [1]})}) is None

    def test_subscription_naming(self):
        ctx = ProjectContext(
            project_name="test", entity_column="USER_ID",
            datasets={"subs": DatasetRegistryEntry(name="subs", path="/tmp/s")},
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
            primary_objective=PredictionObjective.IMMEDIATE_RISK,
        )
        frames = {"subs": pd.DataFrame({
            "SUBSCRIPTION_ID": ["S1", "S2"],
            "USER_ID": ["U1", "U2"],
            "SUBSCRIPTION_END_DATE": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "STATUS": ["Expired", "Active"],
        })}
        result = detect_service_unit(ctx, frames)
        assert result is not None
        assert result.unit_id_column == "SUBSCRIPTION_ID"
        assert result.anchor_date_column == "SUBSCRIPTION_END_DATE"
        assert "Expired" in result.terminated_statuses


class TestClassifyDatasetLinkage:
    def test_self_linkage(self):
        ctx = _sps_context()
        su = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        result = classify_dataset_linkage(ctx, su, _sps_frames())
        assert result["contract"].tier == "contract_linked"
        assert result["contract"].link_method == "self"

    def test_direct_fk_linkage(self):
        ctx = _sps_context()
        su = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        result = classify_dataset_linkage(ctx, su, _sps_frames())
        assert result["subscription"].tier == "contract_linked"
        assert result["subscription"].link_method == "direct_fk"
        assert result["subscription"].fk_column == "CONTRACT_ID"

    def test_account_only_linkage(self):
        ctx = _sps_context()
        su = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        result = classify_dataset_linkage(ctx, su, _sps_frames())
        assert result["case"].tier == "account_only"
        assert result["case"].link_method == "entity_column"

    def test_key_resolution_linkage(self):
        ctx = _sps_context()
        ctx.datasets["case"] = DatasetRegistryEntry(
            name="case", path="/tmp/case",
            key_resolution=[KeyResolutionStep(
                bridge_dataset="contract", source_key="CASE_ID",
                bridge_key="CASE_ID", resolve_column="ACCOUNT_ID",
            )],
        )
        su = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        result = classify_dataset_linkage(ctx, su, _sps_frames())
        assert result["case"].tier == "contract_linked"
        assert result["case"].link_method == "key_resolution"

    def test_missing_dataset_skipped(self):
        ctx = _sps_context()
        su = ServiceUnitConfig(
            dataset_name="contract", unit_id_column="CONTRACT_ID",
            entity_column="ACCOUNT_ID", anchor_date_column="CONTRACT_END_DATE",
        )
        frames = {k: v for k, v in _sps_frames().items() if k != "subscription"}
        result = classify_dataset_linkage(ctx, su, frames)
        assert "subscription" not in result
