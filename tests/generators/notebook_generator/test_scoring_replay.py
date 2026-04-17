from __future__ import annotations

from customer_retention.generators.notebook_generator.scoring_replay import (
    filter_replay_functions,
    render_scoring_replay_block,
)
from customer_retention.runtime.registry import RegisteredFunction


def _rf(name: str, *, replay: bool, scope: str = "dataset", **kw) -> RegisteredFunction:
    defaults = dict(
        name=name,
        source=f"def {name}(df): return df",
        scope=scope,
        dataset=kw.pop("dataset", "request"),
        datasets=kw.pop("datasets", None),
        primary=kw.pop("primary", None),
        replay_at_scoring=replay,
        expected_stage=kw.pop("expected_stage", None),
        inferred_stage=kw.pop("inferred_stage", None),
    )
    defaults.update(kw)
    return RegisteredFunction(**defaults)


class TestFilterReplayFunctions:
    def test_drops_non_replay(self):
        kept = filter_replay_functions([
            _rf("a", replay=True),
            _rf("b", replay=False),
        ])
        names = [rf.name for rf in kept]
        assert names == ["a"]

    def test_preserves_declaration_order(self):
        kept = filter_replay_functions([
            _rf("c", replay=True),
            _rf("a", replay=True),
            _rf("b", replay=True),
        ])
        assert [rf.name for rf in kept] == ["c", "a", "b"]


class TestRenderScoringReplayBlock:
    def test_empty_input_returns_empty_string(self):
        assert render_scoring_replay_block([]) == ""

    def test_all_opt_out_returns_empty_string(self):
        result = render_scoring_replay_block([
            _rf("a", replay=False),
            _rf("b", replay=False),
        ])
        assert result == ""

    def test_single_replay_uses_flat_import(self):
        result = render_scoring_replay_block([_rf("only", replay=True)])
        assert "from user_extensions import only" in result
        assert "TODO wire replay call-site for only" in result

    def test_multiple_replay_uses_parenthesized_import(self):
        result = render_scoring_replay_block([
            _rf("a", replay=True),
            _rf("b", replay=True),
        ])
        assert "from user_extensions import (" in result
        assert "    a," in result
        assert "    b," in result
        assert ")" in result

    def test_todo_includes_stage_and_scope(self):
        result = render_scoring_replay_block([
            _rf("filter_req", replay=True, scope="dataset",
                inferred_stage="landing_post"),
        ])
        assert "filter_req (scope=dataset, stage=landing_post)" in result

    def test_expected_stage_used_when_inferred_missing(self):
        result = render_scoring_replay_block([
            _rf("x", replay=True, expected_stage="target_derive"),
        ])
        assert "stage=target_derive" in result

    def test_mixed_replay_and_non_replay_only_emits_replay(self):
        result = render_scoring_replay_block([
            _rf("keep", replay=True),
            _rf("drop", replay=False),
        ])
        assert "keep" in result
        assert "drop" not in result

    def test_rendered_block_is_plain_text_no_surprise_chars(self):
        result = render_scoring_replay_block([_rf("only", replay=True)])
        assert isinstance(result, str)
        assert "\x00" not in result
