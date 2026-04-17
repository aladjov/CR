from __future__ import annotations

import pytest

from customer_retention.runtime import cr
from customer_retention.runtime.registry import registry


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


class TestDecoratorParamValidation:
    def test_dataset_xor_datasets_both_set_raises(self):
        with pytest.raises(TypeError, match="exactly one of dataset="):
            @cr.register(dataset="a", datasets=["a", "b"])
            def f():
                pass

    def test_dataset_xor_datasets_neither_set_raises(self):
        with pytest.raises(TypeError, match="exactly one of dataset="):
            @cr.register()
            def f():
                pass

    def test_datasets_string_must_be_wildcard(self):
        with pytest.raises(TypeError, match="must be exactly '\\*'"):
            @cr.register(datasets="request")
            def f():
                pass

    def test_datasets_wildcard_with_primary_raises(self):
        with pytest.raises(TypeError, match="primary= is forbidden when datasets='\\*'"):
            @cr.register(datasets="*", primary="x")
            def f():
                pass

    def test_dataset_with_primary_raises(self):
        with pytest.raises(TypeError, match="primary= is forbidden when dataset="):
            @cr.register(dataset="a", primary="a")
            def f():
                pass

    def test_multi_datasets_require_primary(self):
        with pytest.raises(TypeError, match="primary= is required"):
            @cr.register(datasets=["a", "b"])
            def f():
                pass

    def test_primary_must_appear_in_datasets(self):
        with pytest.raises(TypeError, match="primary='x' must appear in"):
            @cr.register(datasets=["a", "b"], primary="x")
            def f():
                pass

    def test_single_element_datasets_primary_optional(self):
        @cr.register(datasets=["solo"])
        def f():
            pass
        assert registry.get_registered()[0].scope == "datasets"

    def test_empty_datasets_list_raises(self):
        with pytest.raises(TypeError, match="list must not be empty"):
            @cr.register(datasets=[])
            def f():
                pass

    def test_name_must_be_valid_identifier(self):
        with pytest.raises(TypeError, match="not a valid Python identifier"):
            @cr.register(dataset="a", name="has spaces")
            def f():
                pass

    def test_datasets_wrong_type_raises(self):
        with pytest.raises(TypeError, match="must be a list of strings or the literal"):
            @cr.register(datasets=42)
            def f():
                pass


class TestDecoratorRegistersFunction:
    def test_dataset_scope_record(self):
        @cr.register(dataset="request")
        def filter_bad(df):
            return df
        recs = registry.get_registered()
        assert len(recs) == 1
        rec = recs[0]
        assert rec.name == "filter_bad"
        assert rec.scope == "dataset"
        assert rec.dataset == "request"
        assert rec.datasets is None

    def test_datasets_list_scope_record(self):
        @cr.register(datasets=["a", "b"], primary="a")
        def merge_ab(dfs):
            return dfs
        rec = registry.get_registered()[0]
        assert rec.scope == "datasets"
        assert rec.datasets == ["a", "b"]
        assert rec.primary == "a"
        assert rec.dataset is None

    def test_wildcard_scope_record(self):
        @cr.register(datasets="*")
        def all_of_them(dfs):
            return dfs
        rec = registry.get_registered()[0]
        assert rec.scope == "wildcard"
        assert rec.datasets is None
        assert rec.dataset is None
        assert rec.primary is None

    def test_custom_name_overrides_func_name(self):
        @cr.register(dataset="x", name="explicit_name")
        def some_func():
            pass
        assert registry.get_registered()[0].name == "explicit_name"

    def test_expected_stage_threaded_through(self):
        @cr.register(dataset="x", expected_stage="landing_post")
        def f():
            pass
        assert registry.get_registered()[0].expected_stage == "landing_post"

    def test_replay_at_scoring_default_false(self):
        @cr.register(dataset="x")
        def f():
            pass
        assert registry.get_registered()[0].replay_at_scoring is False

    def test_replay_at_scoring_true_preserved(self):
        @cr.register(dataset="x", replay_at_scoring=True)
        def f():
            pass
        assert registry.get_registered()[0].replay_at_scoring is True

    def test_decoration_returns_unchanged_function(self):
        def f():
            return "hello"
        decorated = cr.register(dataset="x")(f)
        assert decorated is f
        assert f() == "hello"


class TestSourceCapture:
    def test_free_function_source_captured_verbatim(self):
        @cr.register(dataset="x")
        def simple_body(df):
            return df[df.amount > 0]
        src = registry.get_registered()[0].source
        assert "def simple_body(df):" in src
        assert "return df[df.amount > 0]" in src
        assert "@cr.register(dataset=\"x\")" in src

    def test_class_method_source_captured(self):
        class Holder:
            @staticmethod
            @cr.register(dataset="y")
            def method_body(df):
                return df
        src = registry.get_registered()[0].source
        assert "def method_body(df):" in src


class TestRegistryDedup:
    def test_same_key_replaces_in_place(self):
        def make(val):
            @cr.register(dataset="x", name="stable_name")
            def f():
                return val
            return f
        make(1)
        make(2)
        recs = registry.get_registered()
        assert len(recs) == 1
        assert recs[0].name == "stable_name"

    def test_different_name_produces_two_records(self):
        @cr.register(dataset="x", name="a")
        def fa():
            pass
        @cr.register(dataset="x", name="b")
        def fb():
            pass
        assert len(registry.get_registered()) == 2


class TestDoubleDecoration:
    """Plan § 3.1.1 and § 13.2: stacking @cr.register on the same function
    must error at decoration time rather than silently re-register."""

    def test_stacked_decorator_raises_type_error(self):
        with pytest.raises(TypeError, match="already decorated with @cr.register"):
            @cr.register(dataset="a")
            @cr.register(dataset="a")
            def f():
                pass

    def test_stacked_decorator_includes_function_name(self):
        with pytest.raises(TypeError, match="my_fn"):
            @cr.register(dataset="a")
            @cr.register(dataset="a", name="my_fn")
            def my_fn():
                pass

    def test_marker_attribute_survives_on_wrapped_func(self):
        @cr.register(dataset="a")
        def original():
            return 1
        assert getattr(original, "__cr_registered__", False) is True


class TestCallerLocationCapture:
    """Plan § 3.1.3 + § 3.1.5: cell_id dedup only works if cell_id is
    stable. The IPython-input filename (`<ipython-input-3-...>`) changes
    every kernel restart — do not persist it as cell_id."""

    def test_ipython_input_filename_not_persisted_as_cell_id(self, monkeypatch):
        import inspect as _inspect

        from customer_retention.runtime import decorator as dec_module

        fake_frame = type("F", (), {"filename": "<ipython-input-42-deadbeef>"})()
        monkeypatch.setattr(_inspect, "stack", lambda: [fake_frame])
        monkeypatch.setattr(dec_module.inspect, "stack", lambda: [fake_frame])

        @cr.register(dataset="x")
        def f():
            pass

        rec = registry.get_registered()[0]
        assert rec.cell_id is None
        assert rec.notebook_path is None

    def test_notebook_path_extracted_when_present(self, monkeypatch):
        from customer_retention.runtime import decorator as dec_module

        fake_frame = type("F", (), {"filename": "/tmp/explore/00_start_here.ipynb"})()
        monkeypatch.setattr(dec_module.inspect, "stack", lambda: [fake_frame])

        @cr.register(dataset="x")
        def f():
            pass

        rec = registry.get_registered()[0]
        assert rec.notebook_path is not None
        assert rec.notebook_path.name == "00_start_here.ipynb"
