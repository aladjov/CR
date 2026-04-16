"""Guard test: _chi_squared_primary skips scoring when top_k >= n_features.

When the user configures `STATISTICAL_MAX_FEATURES=1000` but the input pool
has fewer than 1000 features, every feature is guaranteed kept. Running the
full chi-squared scoring pass (which still allocates bucketizers + runs
Spark jobs) is wasted compute. The guard should:
  - return (all-kept, empty-drop) with NaN scores for all features
  - log a message indicating the skip
  - still emit trace-compatible stats so the downstream rescue flow works
"""
from __future__ import annotations

import math

from customer_retention.stages.features.feature_selector import _chi_squared_primary


class TestChiSquaredNoOpGuard:
    def test_top_k_equal_to_n_features_skips_scoring(self):
        """top_k == N — every feature will be kept; skip scoring."""
        captured = []
        keep, drop, stats = _chi_squared_primary(
            df=None, target_column="y",
            features=["a", "b", "c"], top_k=3,
            progress_fn=captured.append,
        )
        assert keep == {"a", "b", "c"}
        assert drop == set()
        assert set(stats.keys()) == {"a", "b", "c"}
        # Any skip-indicating log message is acceptable
        assert any("skip" in msg.lower() or "no-op" in msg.lower()
                   for msg in captured)

    def test_top_k_greater_than_n_features_skips_scoring(self):
        """top_k > N — same no-op semantics."""
        keep, drop, stats = _chi_squared_primary(
            df=None, target_column="y",
            features=["a", "b"], top_k=1000,
            progress_fn=lambda _msg: None,
        )
        assert keep == {"a", "b"}
        assert drop == set()
        assert len(stats) == 2

    def test_skip_emits_nan_scores(self):
        """NaN scores signal 'not evaluated' — distinguishes no-op from
        genuine zero chi² scoring."""
        _, _, stats = _chi_squared_primary(
            df=None, target_column="y",
            features=["a", "b"], top_k=10,
            progress_fn=lambda _msg: None,
        )
        for feat, entry in stats.items():
            assert math.isnan(entry["score"])

    def test_empty_features_returns_empty(self):
        keep, drop, stats = _chi_squared_primary(
            df=None, target_column="y",
            features=[], top_k=100,
        )
        assert keep == set()
        assert drop == set()
        assert stats == {}

    def test_rank_still_assigned_when_skipped(self):
        """Rank field present for downstream trace recording."""
        _, _, stats = _chi_squared_primary(
            df=None, target_column="y",
            features=["a", "b", "c"], top_k=5,
            progress_fn=lambda _msg: None,
        )
        for feat, entry in stats.items():
            assert "rank" in entry
            assert entry["rank"] >= 1
