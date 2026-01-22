"""Tests for RelationshipDetector - TDD approach."""
import pandas as pd
import pytest

from customer_retention.stages.profiling.relationship_detector import (
    DatasetRelationship,
    JoinSuggestion,
    RelationshipDetector,
    RelationshipType,
)


@pytest.fixture
def customers_df():
    """Entity-level customer data."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "city": ["NYC", "LA", "Chicago", "Houston", "Phoenix"],
        "signup_date": pd.to_datetime([
            "2023-01-01", "2023-02-15", "2023-03-20", "2023-04-10", "2023-05-05"
        ]),
    })


@pytest.fixture
def transactions_df():
    """Event-level transactions with foreign key to customers."""
    return pd.DataFrame({
        "transaction_id": range(1, 13),
        "customer_id": ["C001", "C001", "C001", "C002", "C002", "C003",
                        "C003", "C003", "C003", "C004", "C004", "C005"],
        "amount": [100, 150, 200, 50, 75, 300, 250, 180, 90, 500, 450, 60],
        "transaction_date": pd.to_datetime([
            "2023-01-15", "2023-02-10", "2023-03-05",
            "2023-03-01", "2023-04-15",
            "2023-04-01", "2023-04-20", "2023-05-10", "2023-06-01",
            "2023-05-15", "2023-06-10",
            "2023-06-01"
        ]),
    })


@pytest.fixture
def emails_df():
    """Event-level email interactions."""
    return pd.DataFrame({
        "email_id": range(1, 21),
        "customer_id": ["C001"] * 5 + ["C002"] * 4 + ["C003"] * 6 + ["C004"] * 3 + ["C005"] * 2,
        "email_type": ["promo", "newsletter", "promo", "newsletter", "promo"] * 4,
        "sent_date": pd.date_range("2023-01-01", periods=20, freq="W"),
        "opened": [1, 0, 1, 1, 0] * 4,
    })


@pytest.fixture
def products_df():
    """Reference data for products - no direct customer relationship."""
    return pd.DataFrame({
        "product_id": ["P01", "P02", "P03", "P04"],
        "product_name": ["Widget A", "Widget B", "Gadget X", "Gadget Y"],
        "category": ["widgets", "widgets", "gadgets", "gadgets"],
        "price": [29.99, 49.99, 99.99, 149.99],
    })


class TestRelationshipDetector:
    """Tests for basic relationship detection."""

    def test_detect_returns_relationship(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        assert isinstance(result, DatasetRelationship)

    def test_detects_common_column_names(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        # Should detect customer_id as common column
        assert result.join_columns is not None
        assert "customer_id" in result.join_columns

    def test_detects_one_to_many_relationship(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        # Customers (1) -> Transactions (many)
        assert result.relationship_type == RelationshipType.ONE_TO_MANY

    def test_detects_one_to_one_relationship(self):
        """Test 1:1 relationship when keys are unique in both."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        })
        df2 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        assert result.relationship_type == RelationshipType.ONE_TO_ONE

    def test_detects_many_to_many_relationship(self):
        """Test M:N relationship when keys are repeated in both."""
        df1 = pd.DataFrame({
            "tag_id": ["A", "A", "B", "B", "C"],
            "item1": [1, 2, 3, 4, 5],
        })
        df2 = pd.DataFrame({
            "tag_id": ["A", "B", "B", "C", "C", "C"],
            "item2": [10, 20, 30, 40, 50, 60],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        assert result.relationship_type == RelationshipType.MANY_TO_MANY


class TestJoinSuggestion:
    """Tests for join suggestion generation."""

    def test_suggests_join_column(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        assert result.suggested_join is not None
        assert isinstance(result.suggested_join, JoinSuggestion)

    def test_join_suggestion_has_columns(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        suggestion = result.suggested_join
        assert suggestion.left_column == "customer_id"
        assert suggestion.right_column == "customer_id"

    def test_join_suggestion_confidence(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        # High confidence when column names match and values overlap
        assert result.suggested_join.confidence >= 0.8

    def test_low_confidence_for_partial_overlap(self, customers_df):
        """Low confidence when many values don't match."""
        other_df = pd.DataFrame({
            "customer_id": ["C001", "C999", "C888", "C777"],  # Only C001 matches
            "value": [1, 2, 3, 4],
        })

        detector = RelationshipDetector()
        result = detector.detect(customers_df, other_df)

        # Lower confidence due to low overlap
        assert result.suggested_join.confidence < 0.5


class TestNameMatchDetection:
    """Tests for column name-based relationship detection."""

    def test_detects_id_suffix_match(self):
        """Should match 'id' to 'customer_id' via naming convention."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        })
        df2 = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2, df1_name="customer", df2_name="order")

        # When df1 is named "customer", should try to match "id" with "customer_id"
        assert result.join_columns is not None

    def test_detects_exact_name_match(self):
        """Columns with exact same name should be detected."""
        df1 = pd.DataFrame({
            "user_id": [1, 2, 3],
            "name": ["A", "B", "C"],
        })
        df2 = pd.DataFrame({
            "user_id": [1, 2, 2],
            "amount": [100, 200, 300],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        assert "user_id" in result.join_columns


class TestNoRelationship:
    """Tests for datasets with no relationship."""

    def test_no_common_columns(self, customers_df, products_df):
        """Should detect no relationship when no common columns exist."""
        detector = RelationshipDetector()
        result = detector.detect(customers_df, products_df)

        # No common identifier columns
        assert result.relationship_type == RelationshipType.NONE or result.join_columns is None

    def test_no_value_overlap(self):
        """Should not suggest join when values don't overlap."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        })
        df2 = pd.DataFrame({
            "id": [100, 200, 300],  # No overlap
            "value": [10, 20, 30],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        # Should have very low confidence or no suggestion
        if result.suggested_join:
            assert result.suggested_join.confidence < 0.1


class TestRelationshipStats:
    """Tests for relationship statistics."""

    def test_provides_coverage_stats(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        # Should report how many keys from each side match
        assert result.left_coverage is not None  # % of left keys found in right
        assert result.right_coverage is not None  # % of right keys found in left

    def test_coverage_calculation(self, customers_df, transactions_df):
        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_df)

        # All 5 customers have transactions
        assert result.left_coverage == 1.0

        # All transaction customer_ids exist in customers
        assert result.right_coverage == 1.0

    def test_partial_coverage(self, customers_df):
        """Test coverage when not all keys match."""
        partial_transactions = pd.DataFrame({
            "customer_id": ["C001", "C002", "C002"],  # Only 2 of 5 customers
            "amount": [100, 200, 300],
        })

        detector = RelationshipDetector()
        result = detector.detect(customers_df, partial_transactions)

        # Only 2/5 customers have transactions
        assert result.left_coverage == 0.4


class TestMultipleColumns:
    """Tests for composite key detection."""

    def test_detects_composite_key(self):
        """Should detect when multiple columns form the relationship."""
        df1 = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2", "C2"],
            "product_id": ["P1", "P2", "P1", "P2"],
            "quantity": [1, 2, 3, 4],
        })
        df2 = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2"],
            "product_id": ["P1", "P2", "P1"],
            "returns": [0, 1, 0],
        })

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        # Should detect both columns form the key
        assert len(result.join_columns) >= 2 or result.composite_key_detected


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, customers_df):
        empty_df = pd.DataFrame(columns=["customer_id", "value"])

        detector = RelationshipDetector()
        result = detector.detect(customers_df, empty_df)

        # Should handle gracefully
        assert result is not None

    def test_single_row_dataframes(self):
        df1 = pd.DataFrame({"id": [1], "name": ["A"]})
        df2 = pd.DataFrame({"id": [1], "value": [10]})

        detector = RelationshipDetector()
        result = detector.detect(df1, df2)

        assert result is not None

    def test_null_values_in_key(self, customers_df):
        """Handle nulls in potential key columns."""
        transactions_with_nulls = pd.DataFrame({
            "customer_id": ["C001", None, "C002", None, "C003"],
            "amount": [100, 200, 300, 400, 500],
        })

        detector = RelationshipDetector()
        result = detector.detect(customers_df, transactions_with_nulls)

        # Should still detect relationship, excluding nulls
        assert result.join_columns is not None
