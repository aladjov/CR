from __future__ import annotations

from pathlib import Path

from customer_retention.parity.gaps import (
    GapKind,
    ParityGap,
    diff_manifests,
)
from customer_retention.parity.kinds import ApplyOpKind
from customer_retention.parity.manifest import (
    Manifest,
    ManifestEntry,
    SourceLocation,
    fingerprint_kwargs,
)


def _e(dataset, kind, order=0, file_="x.py", line=1, **kwargs) -> ManifestEntry:
    return ManifestEntry(
        dataset=dataset,
        kind=kind,
        kwargs_fingerprint=fingerprint_kwargs(kwargs),
        call_order=order,
        source_location=SourceLocation(file=Path(file_), line=line),
    )


class TestGapKindClosedSet:
    def test_all_kinds_present(self):
        present = {k.name for k in GapKind}
        required = {
            "PRODUCTION_ONLY",
            "EXPLORATION_ONLY",
            "KWARGS_MISMATCH",
            "ORDER_MISMATCH",
            "ORPHAN_NOTEBOOK",
            "ORPHAN_REGISTRATION",
            "ORPHAN_PRODUCTION",
            "SCHEDULE_TOPOLOGY",
            "RUNTIME_DRIFT",
        }
        assert required <= present


class TestDiffManifestsEqual:
    def test_empty_manifests_no_gaps(self):
        assert diff_manifests(Manifest(()), Manifest(())) == []

    def test_identical_single_entry_no_gap(self):
        m = Manifest((_e("a", ApplyOpKind.LIFECYCLE_ENRICH),))
        assert diff_manifests(m, m) == []

    def test_identical_multi_entry_no_gap(self):
        a = Manifest(
            (
                _e("c", ApplyOpKind.LIFECYCLE_ENRICH),
                _e("c", ApplyOpKind.TEMPORAL_LOOKBACK, order=1),
            )
        )
        b = Manifest(
            (
                _e("c", ApplyOpKind.LIFECYCLE_ENRICH, file_="render.py", line=99),
                _e(
                    "c",
                    ApplyOpKind.TEMPORAL_LOOKBACK,
                    order=2,
                    file_="render.py",
                    line=120,
                ),
            )
        )
        assert diff_manifests(a, b) == []


class TestProductionOnly:
    def test_production_has_extra_kind(self):
        exploration = Manifest(())
        production = Manifest(
            (_e("contract", ApplyOpKind.TEMPORAL_LOOKBACK, line=80),)
        )
        gaps = diff_manifests(exploration, production)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_kind is GapKind.PRODUCTION_ONLY
        assert g.dataset == "contract"
        assert g.op_kind is ApplyOpKind.TEMPORAL_LOOKBACK
        assert g.exploration_location is None
        assert g.production_location is not None
        assert g.production_location.line == 80


class TestExplorationOnly:
    def test_exploration_has_extra_kind(self):
        exploration = Manifest(
            (_e("contract", ApplyOpKind.DATETIME_DERIVE, line=42),)
        )
        production = Manifest(())
        gaps = diff_manifests(exploration, production)
        assert len(gaps) == 1
        assert gaps[0].gap_kind is GapKind.EXPLORATION_ONLY
        assert gaps[0].exploration_location.line == 42
        assert gaps[0].production_location is None


class TestKwargsMismatch:
    def test_same_kind_different_kwargs(self):
        exploration = Manifest(
            (_e("c", ApplyOpKind.TEMPORAL_LOOKBACK, lookback_periods=36),)
        )
        production = Manifest(
            (_e("c", ApplyOpKind.TEMPORAL_LOOKBACK, lookback_periods=12),)
        )
        gaps = diff_manifests(exploration, production)
        assert len(gaps) == 1
        assert gaps[0].gap_kind is GapKind.KWARGS_MISMATCH
        assert "lookback_periods" in gaps[0].detail

    def test_dynamic_kwargs_skip_comparison(self):
        # kwargs marked <dynamic> on either side suppress kwargs comparison
        # but kind comparison still fires
        exploration = Manifest(
            (_e("c", ApplyOpKind.TEMPORAL_LOOKBACK, lookback_periods="<dynamic>"),)
        )
        production = Manifest(
            (_e("c", ApplyOpKind.TEMPORAL_LOOKBACK, lookback_periods=36),)
        )
        gaps = diff_manifests(exploration, production)
        assert gaps == []


class TestOrderMismatch:
    def test_same_kinds_different_order(self):
        exploration = Manifest(
            (
                _e("c", ApplyOpKind.LIFECYCLE_ENRICH, order=0),
                _e("c", ApplyOpKind.DATETIME_DERIVE, order=1),
            )
        )
        production = Manifest(
            (
                _e("c", ApplyOpKind.DATETIME_DERIVE, order=0),
                _e("c", ApplyOpKind.LIFECYCLE_ENRICH, order=1),
            )
        )
        gaps = diff_manifests(exploration, production)
        assert any(g.gap_kind is GapKind.ORDER_MISMATCH for g in gaps)


class TestMultipleGaps:
    def test_history_window_bug_scenario(self):
        # Exploration: no temporal_lookback for contract (gate=False)
        exploration = Manifest(
            (_e("contract", ApplyOpKind.LIFECYCLE_ENRICH),)
        )
        # Production: emits temporal_lookback regardless
        production = Manifest(
            (
                _e("contract", ApplyOpKind.LIFECYCLE_ENRICH),
                _e(
                    "contract",
                    ApplyOpKind.TEMPORAL_LOOKBACK,
                    file_="generated/landing_contract.py",
                    line=80,
                ),
            )
        )
        gaps = diff_manifests(exploration, production)
        assert len(gaps) == 1
        assert gaps[0].gap_kind is GapKind.PRODUCTION_ONLY
        assert gaps[0].dataset == "contract"
        assert gaps[0].op_kind is ApplyOpKind.TEMPORAL_LOOKBACK


class TestParityGapFormat:
    def test_format_includes_both_sides(self):
        gap = ParityGap(
            gap_kind=GapKind.PRODUCTION_ONLY,
            dataset="contract",
            op_kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            exploration_location=None,
            production_location=SourceLocation(
                file=Path("renderer.py"), line=3371
            ),
            detail="Production emits TEMPORAL_LOOKBACK; exploration skips it",
        )
        s = gap.format()
        assert "PRODUCTION_ONLY" in s
        assert "contract" in s
        assert "TEMPORAL_LOOKBACK" in s
        assert "renderer.py:3371" in s

    def test_format_handles_missing_location(self):
        gap = ParityGap(
            gap_kind=GapKind.EXPLORATION_ONLY,
            dataset="account",
            op_kind=ApplyOpKind.SAMPLE_FILTER,
            exploration_location=SourceLocation(file=Path("nb.ipynb"), line=5),
            production_location=None,
            detail="Exploration applies SAMPLE_FILTER; production omits it",
        )
        s = gap.format()
        assert "nb.ipynb:5" in s
