"""Tests for `runtime.dataset_resolution.resolve_original_datasets`.

Closes the §7.3 framework gap: the prior NB10 cell `0f3cc762` if/elif
ladder silently dropped operator-supplied `DATASETS_ORIGINAL_FALLBACK`
overrides whenever `_ctx.original_datasets` was non-empty. The overlay
strategy here treats the operator dict as the authoritative override and
the ProjectContext as the gap-filler.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from customer_retention.runtime.dataset_resolution import (
    resolve_original_datasets,
)


def _ctx(original_datasets):
    return SimpleNamespace(original_datasets=dict(original_datasets))


class TestResolveOriginalDatasetsOverlay:
    def test_fallback_overrides_project_context_on_collision(self):
        """The §7.3 repro: operator pastes a fresh UC handle into
        DATASETS_ORIGINAL_FALLBACK to override a stale project-context
        entry. Pre-FW-8 the fallback was shadowed; post-FW-8 it wins."""
        ctx = _ctx({"account": "stale.global_temp.acct", "case": "raw.case"})
        fallback = {"account": "uc.cat.sch.account_with_churn"}
        merged, prov = resolve_original_datasets(ctx, fallback)
        assert merged["account"] == "uc.cat.sch.account_with_churn"
        assert merged["case"] == "raw.case"
        assert prov["account"] == "DATASETS_ORIGINAL_FALLBACK"
        assert prov["case"] == "project_context"

    def test_fallback_extends_project_context_with_new_keys(self):
        """A dataset that's missing from the project context can be added
        via fallback alone — useful when NB00 didn't see a derivable
        dataset and the operator wants to wire it in for codegen."""
        ctx = _ctx({"account": "raw.account"})
        fallback = {"contract": "raw.contract", "subscription": "raw.subscription"}
        merged, prov = resolve_original_datasets(ctx, fallback)
        assert set(merged.keys()) == {"account", "contract", "subscription"}
        assert prov["account"] == "project_context"
        assert prov["contract"] == "DATASETS_ORIGINAL_FALLBACK"
        assert prov["subscription"] == "DATASETS_ORIGINAL_FALLBACK"

    def test_empty_fallback_falls_back_to_project_context(self):
        """When operator hasn't pasted a fallback, the resolution still
        works — just from project_context alone."""
        ctx = _ctx({"account": "raw.account"})
        merged, prov = resolve_original_datasets(ctx, None)
        assert merged == {"account": "raw.account"}
        assert prov == {"account": "project_context"}

    def test_empty_project_context_uses_fallback_only(self):
        """When project_context is empty (e.g. NB00 didn't run), the
        operator's fallback dict carries the entire mapping."""
        ctx = _ctx({})
        fallback = {"account": "raw.account"}
        merged, prov = resolve_original_datasets(ctx, fallback)
        assert merged == {"account": "raw.account"}
        assert prov == {"account": "DATASETS_ORIGINAL_FALLBACK"}

    def test_both_empty_raises_actionable_error(self):
        """When neither input has any datasets, the resolver must surface
        a runnable error message naming both inputs the operator can
        edit. The message should also reference NB00 as the upstream
        source so a one-step diagnostic is possible."""
        ctx = _ctx({})
        with pytest.raises(RuntimeError, match="DATASETS_ORIGINAL_FALLBACK"):
            resolve_original_datasets(ctx, None)

    def test_none_project_context_is_tolerated(self):
        """`ProjectContext.load` returns an object with `.original_datasets`
        but a defensive caller could pass `None` (e.g. when the project
        context file is missing). Resolver must not AttributeError —
        treat `None` as an empty context."""
        merged, prov = resolve_original_datasets(None, {"account": "raw.account"})
        assert merged == {"account": "raw.account"}

    def test_overlay_does_not_mutate_inputs(self):
        """Resolver must not mutate the project context's dict or the
        operator's fallback dict in place — both should be safe to inspect
        after the call."""
        ctx_dict = {"account": "raw.account"}
        ctx = _ctx(ctx_dict)
        fallback = {"account": "uc.cat.sch.account_override"}
        original_ctx_snapshot = dict(ctx_dict)
        original_fallback_snapshot = dict(fallback)
        resolve_original_datasets(ctx, fallback)
        assert ctx_dict == original_ctx_snapshot
        assert fallback == original_fallback_snapshot
