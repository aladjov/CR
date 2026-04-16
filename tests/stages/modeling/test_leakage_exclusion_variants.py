"""Tests that LeakageExclusion accepts both bare-string and dict entries
when constructed via the pattern used in NB05's EXCLUDED_LEAKING_FEATURES
conversion loop. Dict form carries rationale/code/severity so future
maintainers can see WHY a feature was excluded.
"""
from __future__ import annotations

import pytest

from customer_retention.stages.modeling.feature_spec import LeakageExclusion


def _to_leakage_excl(entry):
    """Mirror of NB05 cell 44 conversion logic."""
    if isinstance(entry, str):
        return LeakageExclusion(column=entry)
    if isinstance(entry, dict):
        return LeakageExclusion(**entry)
    raise TypeError(
        f"EXCLUDED_LEAKING_FEATURES entries must be str or dict; got {type(entry).__name__}"
    )


class TestLeakageExclusionVariantConversion:
    def test_string_entry_yields_defaults(self):
        excl = _to_leakage_excl("NET_PRICE")
        assert excl.column == "NET_PRICE"
        assert excl.severity == "HIGH"
        assert excl.rationale == ""
        assert excl.code == ""

    def test_dict_entry_with_rationale(self):
        excl = _to_leakage_excl({
            "column": "NET_PRICE",
            "rationale": "Zero-on-termination leak fingerprint",
        })
        assert excl.column == "NET_PRICE"
        assert excl.rationale == "Zero-on-termination leak fingerprint"

    def test_dict_entry_full_fields(self):
        excl = _to_leakage_excl({
            "column": "NET_PRICE",
            "code": "SPS-SUB-LEAK-001",
            "severity": "HIGH",
            "rationale": "Documented leak source",
        })
        assert excl.code == "SPS-SUB-LEAK-001"
        assert excl.severity == "HIGH"
        assert excl.rationale == "Documented leak source"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match=r"str or dict"):
            _to_leakage_excl(42)

    def test_mixed_list_of_string_and_dict(self):
        """Config may mix legacy strings and dict entries — both work."""
        entries = [
            "LEGACY_COL",
            {"column": "NEW_COL", "rationale": "explicit reason"},
        ]
        result = [_to_leakage_excl(e) for e in entries]
        assert result[0].column == "LEGACY_COL"
        assert result[0].rationale == ""
        assert result[1].column == "NEW_COL"
        assert result[1].rationale == "explicit reason"

    def test_dict_rationale_flows_to_serialization(self):
        """Rationale survives roundtrip through to_dict/from_dict (used for
        feature_spec.yaml persistence)."""
        excl = _to_leakage_excl({
            "column": "NET_PRICE",
            "rationale": "Zero-on-termination",
        })
        d = excl.to_dict()
        assert d["rationale"] == "Zero-on-termination"
        restored = LeakageExclusion.from_dict(d)
        assert restored == excl
