"""PA-6 wiring lock: NB06 must register terminate/start event ratios via
``register_event_ratio_features`` (or its dataset-specific siblings) so
codegen materialises the ratio columns in production.

Background
----------
``add_silver_derived(expression=..., feature_type="ratio", source_columns=[...])``
registers a rec, but ``gold_transform_applicator._derived_source_columns``
reads only ``parameters["numerator"]`` / ``["denominator"]`` for the
ratio handler — it does not look at ``source_columns`` or parse the
``expression`` string. Recs registered the wrong way are silently
no-op'd at codegen, leaving production silver_merged without the
``*_terminate_to_start_ratio_*`` columns that NB08's FeatureSpec selects.

The framework helper ``register_event_ratio_features`` uses
``add_silver_ratio(numerator=..., denominator=...)`` which is the path
the codegen executor reads. These tests pin that NB06 uses the helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLORATION_NB06 = REPO_ROOT / "exploration_notebooks/06_feature_opportunities.ipynb"
FLAWLESS_NB06 = REPO_ROOT / "debug/flawless/06_feature_opportunities.ipynb"


def _all_user_code_sources(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ]


def _concatenated_code(nb_path: Path) -> str:
    return "\n".join(_all_user_code_sources(nb_path))


class TestExplorationNb06ContractRegistration:
    """Exploration NB06 carries the canonical contract terminate/start
    ratio cell — must call the framework register helper so the rec
    round-trips through `apply_derived_ratio` at codegen.
    """

    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _concatenated_code(EXPLORATION_NB06)

    def test_imports_register_helper(self, src: str):
        assert (
            "register_contract_ratio_features" in src
            or "register_event_ratio_features" in src
        ), (
            "exploration_notebooks/06_feature_opportunities.ipynb must import "
            "register_contract_ratio_features (or register_event_ratio_features) "
            "from customer_retention.analysis.business — without this call, no "
            "add_silver_ratio rec hits the registry and contract terminate-to-start "
            "ratio columns never materialise in production silver_merged."
        )

    def test_helper_is_actually_called(self, src: str):
        called = (
            "register_contract_ratio_features(" in src
            or "register_event_ratio_features(" in src
        )
        assert called, (
            "exploration NB06 imports the helper but does not call it — the "
            "import alone does not register anything."
        )

    def test_no_silent_add_silver_derived_for_term_ratio(self, src: str):
        """An ``add_silver_derived(expression=...)`` rec for a
        terminate-to-start ratio is the silently-broken path: it
        registers but is no-op'd at codegen because the executor's
        ratio handler reads numerator/denominator, not expression.
        """
        for token in (
            "contract_terminate_to_start_ratio_",
            "subscription_terminate_to_start_ratio_",
        ):
            if token in src:
                rec_block_idx = src.find(token)
                window = src[max(0, rec_block_idx - 600) : rec_block_idx + 200]
                if "add_silver_derived" in window and 'feature_type="ratio"' in window:
                    pytest.fail(
                        f"NB06 registers {token!r} via add_silver_derived "
                        "with feature_type='ratio' — codegen ignores the "
                        "expression string and the rec is silently no-op'd. "
                        "Use register_event_ratio_features(registry, dataset=...) "
                        "(or add_silver_ratio directly) so numerator/denominator "
                        "flow through to apply_derived_ratio."
                    )


class TestFlawlessNb06Registration:
    """Flawless NB06 is the SPS engagement reference. It must register
    BOTH contract AND subscription terminate-to-start ratios via the
    framework helper.
    """

    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _concatenated_code(FLAWLESS_NB06)

    def test_contract_ratio_helper_called(self, src: str):
        called = (
            "register_contract_ratio_features(" in src
            or 'register_event_ratio_features(' in src and 'dataset="contract"' in src
            or "register_event_ratio_features(" in src and "dataset='contract'" in src
        )
        assert called, (
            "flawless NB06 must call register_contract_ratio_features "
            "(or register_event_ratio_features(dataset='contract')) — "
            "without it, the SPS run's contract_terminate_to_start_ratio_* "
            "features land in exploration silver_merged but are missing from "
            "production silver_merged."
        )

    def test_subscription_ratio_helper_called(self, src: str):
        called = (
            "register_subscription_ratio_features(" in src
            or 'register_event_ratio_features(' in src and 'dataset="subscription"' in src
            or "register_event_ratio_features(" in src and "dataset='subscription'" in src
        )
        assert called, (
            "flawless NB06 must call register_subscription_ratio_features "
            "(or register_event_ratio_features(dataset='subscription'))."
        )

    def test_no_silent_add_silver_derived_for_term_ratio(self, src: str):
        """Same broken-path guard as the exploration test — flawless
        previously registered the ratio recs via ``add_silver_derived``
        with ``feature_type='ratio'``, which silently no-op'd at codegen.
        """
        for token in (
            "contract_terminate_to_start_ratio_",
            "subscription_terminate_to_start_ratio_",
        ):
            if token in src:
                rec_block_idx = src.find(token)
                window = src[max(0, rec_block_idx - 600) : rec_block_idx + 200]
                if "add_silver_derived" in window and 'feature_type="ratio"' in window:
                    pytest.fail(
                        f"flawless NB06 registers {token!r} via add_silver_derived "
                        "with feature_type='ratio' — codegen ignores the expression "
                        "string and the rec is silently no-op'd. Use "
                        "register_event_ratio_features(registry, dataset=...) instead."
                    )
