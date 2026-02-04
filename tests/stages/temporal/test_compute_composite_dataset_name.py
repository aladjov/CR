import pytest

from customer_retention.stages.temporal.snapshot_manager import compute_composite_dataset_name


class TestComputeCompositeDatasetNameValidation:
    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="at least one"):
            compute_composite_dataset_name([])

    def test_raises_on_blank_strings_only(self):
        with pytest.raises(ValueError, match="at least one"):
            compute_composite_dataset_name(["", "  ", "\t"])


class TestComputeCompositeDatasetNameFormat:
    def test_single_name_produces_combined_prefix(self):
        result = compute_composite_dataset_name(["customers"])
        assert result.startswith("combined_")
        assert len(result) == len("combined_") + 6

    def test_two_names_produce_combined_prefix(self):
        result = compute_composite_dataset_name(["customers", "transactions"])
        assert result.startswith("combined_")
        assert len(result) == len("combined_") + 6


class TestComputeCompositeDatasetNameDeterminism:
    def test_same_order_same_result(self):
        a = compute_composite_dataset_name(["customers", "transactions"])
        b = compute_composite_dataset_name(["customers", "transactions"])
        assert a == b

    def test_different_order_same_result(self):
        a = compute_composite_dataset_name(["customers", "transactions"])
        b = compute_composite_dataset_name(["transactions", "customers"])
        assert a == b


class TestComputeCompositeDatasetNameUniqueness:
    def test_different_inputs_different_hashes(self):
        single = compute_composite_dataset_name(["customers"])
        multi = compute_composite_dataset_name(["customers", "transactions"])
        assert single != multi

    def test_overlapping_names_different_hashes(self):
        a = compute_composite_dataset_name(["a+b"])
        b = compute_composite_dataset_name(["a", "b"])
        assert a != b


class TestComputeCompositeDatasetNameWhitespace:
    def test_whitespace_stripped_from_names(self):
        a = compute_composite_dataset_name(["customers", "transactions"])
        b = compute_composite_dataset_name(["  customers  ", "  transactions  "])
        assert a == b
