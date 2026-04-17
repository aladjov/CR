from __future__ import annotations

from customer_retention.runtime.registry import RegisteredFunction
from customer_retention.runtime.validation import (
    ValidationError,
    validate_registered_function,
)


def _rf(name: str, source: str, **kw) -> RegisteredFunction:
    defaults = dict(
        name=name,
        source=source,
        scope=kw.pop("scope", "dataset"),
        dataset=kw.pop("dataset", "request"),
    )
    defaults.update(kw)
    return RegisteredFunction(**defaults)


class TestCardinalRuleViolations:
    def test_registry_add_call_inside_register_raises_validation(self):
        src = """
def derive_ratio(df):
    registry.add_silver_ratio(column="x", numerator="a", denominator="b",
                              rationale="r", source_notebook="nb")
    return df
"""
        err = validate_registered_function(_rf("derive_ratio", src))
        assert err is not None
        assert "registry.add_silver_ratio" in err.reason
        assert "mixes declarative and imperative lanes" in err.reason
        assert "Fix: split into two cells" in err.reason

    def test_first_violation_line_number_reported(self):
        src = """
def f(df):
    df = df
    registry.add_bronze_filtering(column="amount", condition="x", action="drop",
                                    rationale="r", source_notebook="nb")
    return df
"""
        err = validate_registered_function(_rf("f", src))
        assert err is not None
        assert "line 4" in err.reason

    def test_multiple_registry_calls_reports_first_only(self):
        src = """
def f(df):
    registry.add_bronze_null(column="a", strategy="median", rationale="r", source_notebook="n")
    registry.add_silver_ratio(column="b", numerator="a", denominator="b", rationale="r",
                              source_notebook="n")
    return df
"""
        err = validate_registered_function(_rf("f", src))
        assert err is not None
        assert "registry.add_bronze_null" in err.reason


class TestFalsePositives:
    def test_no_registry_call_passes(self):
        src = """
def clean(df):
    df = df[df.amount > 0]
    return df
"""
        assert validate_registered_function(_rf("clean", src)) is None

    def test_cr_log_call_is_not_a_violation(self):
        src = """
def f(df):
    cr.log("hello")
    return df
"""
        assert validate_registered_function(_rf("f", src)) is None

    def test_registry_as_function_argument_is_not_a_violation(self):
        src = """
def f(registry, df):
    registry.add_silver_ratio(column="x", numerator="a", denominator="b",
                              rationale="r", source_notebook="nb")
    return df
"""
        assert validate_registered_function(_rf("f", src)) is None

    def test_registry_as_local_assignment_is_not_a_violation(self):
        src = """
def f(df):
    registry = SomeLocalObject()
    registry.add_widget("x")
    return df
"""
        assert validate_registered_function(_rf("f", src)) is None

    def test_registry_attribute_call_that_is_not_add_prefixed_passes(self):
        src = """
def f(df):
    registry.get_by_layer("landing")
    return df
"""
        assert validate_registered_function(_rf("f", src)) is None

    def test_some_registry_method_is_not_a_violation(self):
        """Bare method access outside add_ prefix is fine."""
        src = """
def f(df):
    registry.save("/tmp/x")
    return df
"""
        assert validate_registered_function(_rf("f", src)) is None


class TestSyntaxError:
    def test_unparseable_source_is_reported(self):
        src = "def f(df\n    return df"
        err = validate_registered_function(_rf("broken", src))
        assert err is not None
        assert "failed to parse" in err.reason


class TestValidationErrorStringFormat:
    def test_str_includes_function_name_and_reason(self):
        err = ValidationError(function_name="f", reason="something broke")
        assert "'f'" in str(err)
        assert "something broke" in str(err)

    def test_str_includes_notebook_and_cell(self):
        err = ValidationError(
            function_name="f",
            reason="bad",
            notebook_path="nb/00.ipynb",
            cell_id="cell-1",
        )
        s = str(err)
        assert "nb/00.ipynb" in s
        assert "cell-1" in s


class TestHarvestIntegration:
    """The harvester calls validate_registered_function on each record
    and routes violations to `HarvestResult.validation_errors`."""

    def test_violation_shows_up_in_harvest_result(self):
        from customer_retention.runtime.harvest import Harvester, PhaseMap
        from customer_retention.runtime.registry import Registry

        reg = Registry()
        reg.register(_rf(
            "bad",
            source=(
                "def bad(df):\n"
                "    registry.add_silver_ratio(column='x', numerator='a', denominator='b',\n"
                "                              rationale='r', source_notebook='nb')\n"
                "    return df\n"
            ),
            expected_stage="silver_post",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert len(result.validation_errors) == 1
        assert result.validation_errors[0].function_name == "bad"

    def test_clean_function_produces_no_errors(self):
        from customer_retention.runtime.harvest import Harvester, PhaseMap
        from customer_retention.runtime.registry import Registry

        reg = Registry()
        reg.register(_rf(
            "clean",
            source="def clean(df):\n    return df\n",
            expected_stage="landing_post",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.validation_errors == []
