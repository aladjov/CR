"""
Relationship detector for identifying join relationships between datasets.

Detects:
- Common columns suitable for joins
- Relationship types (1:1, 1:N, M:N)
- Key coverage statistics
- Composite key relationships
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

from customer_retention.core.compat import DataFrame


class RelationshipType(str, Enum):
    """Type of relationship between two datasets."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    NONE = "none"


@dataclass
class JoinSuggestion:
    """Suggested join configuration."""
    left_column: str
    right_column: str
    confidence: float
    join_type: str = "left"  # left, inner, outer


@dataclass
class DatasetRelationship:
    """Result of relationship detection between two datasets."""
    relationship_type: RelationshipType
    join_columns: Optional[List[str]] = None
    suggested_join: Optional[JoinSuggestion] = None
    left_coverage: Optional[float] = None  # % of left keys found in right
    right_coverage: Optional[float] = None  # % of right keys found in left
    composite_key_detected: bool = False
    notes: List[str] = field(default_factory=list)


class RelationshipDetector:
    """Detects relationships between two datasets."""

    # Column name patterns that suggest identifier columns
    ID_PATTERNS = ["_id", "id", "_key", "key", "_code", "code"]

    def __init__(self):
        pass

    def detect(self, df1: DataFrame, df2: DataFrame,
               df1_name: Optional[str] = None,
               df2_name: Optional[str] = None) -> DatasetRelationship:
        """
        Detect relationship between two dataframes.

        Args:
            df1: First dataframe (left side of join)
            df2: Second dataframe (right side of join)
            df1_name: Optional name for df1 (helps with name-based matching)
            df2_name: Optional name for df2 (helps with name-based matching)

        Returns:
            DatasetRelationship with detected relationship info
        """
        if len(df1) == 0 or len(df2) == 0:
            return DatasetRelationship(
                relationship_type=RelationshipType.NONE,
                notes=["One or both dataframes are empty"]
            )

        # Find candidate join columns
        candidates = self._find_candidate_columns(df1, df2, df1_name, df2_name)

        if not candidates:
            return DatasetRelationship(
                relationship_type=RelationshipType.NONE,
                notes=["No common columns found for joining"]
            )

        # Evaluate each candidate and find best match
        best_match = None
        best_score = 0

        for left_col, right_col in candidates:
            score, coverage_left, coverage_right = self._evaluate_join(
                df1, df2, left_col, right_col
            )
            if score > best_score:
                best_score = score
                best_match = (left_col, right_col, coverage_left, coverage_right)

        if best_match is None or best_score < 0.01:
            return DatasetRelationship(
                relationship_type=RelationshipType.NONE,
                notes=["No columns with sufficient value overlap found"]
            )

        left_col, right_col, coverage_left, coverage_right = best_match

        # Determine relationship type
        rel_type = self._determine_relationship_type(df1, df2, left_col, right_col)

        # Check for composite keys
        composite_detected = self._check_composite_key(df1, df2, candidates)

        # Build suggestion
        suggestion = JoinSuggestion(
            left_column=left_col,
            right_column=right_col,
            confidence=best_score,
            join_type="left" if rel_type == RelationshipType.ONE_TO_MANY else "inner"
        )

        return DatasetRelationship(
            relationship_type=rel_type,
            join_columns=[left_col] if left_col == right_col else [left_col, right_col],
            suggested_join=suggestion,
            left_coverage=coverage_left,
            right_coverage=coverage_right,
            composite_key_detected=composite_detected,
        )

    def _find_candidate_columns(self, df1: DataFrame, df2: DataFrame,
                                df1_name: Optional[str],
                                df2_name: Optional[str]) -> List[tuple]:
        """Find columns that could be used for joining."""
        candidates = []

        # Get identifier-like columns from each dataframe
        id_cols1 = self._get_id_columns(df1)
        id_cols2 = self._get_id_columns(df2)

        # 1. Exact name matches (highest priority)
        common_cols = set(df1.columns) & set(df2.columns)
        for col in common_cols:
            if col in id_cols1 or col in id_cols2:
                candidates.append((col, col))

        # 2. Pattern-based matches (e.g., "id" in df1 matches "customer_id" in df2)
        if df1_name:
            # If df1 is named "customer", look for "customer_id" in df2
            expected_id = f"{df1_name.lower()}_id"
            if expected_id in df2.columns and "id" in df1.columns:
                candidates.append(("id", expected_id))

        if df2_name:
            expected_id = f"{df2_name.lower()}_id"
            if expected_id in df1.columns and "id" in df2.columns:
                candidates.append((expected_id, "id"))

        # 3. Suffix matching for ID columns
        for col1 in id_cols1:
            for col2 in id_cols2:
                if col1 != col2:
                    # Check if one is a suffix of the other
                    if col1.endswith("_id") and col2.endswith("_id"):
                        base1 = col1[:-3]
                        base2 = col2[:-3]
                        if base1 in base2 or base2 in base1:
                            candidates.append((col1, col2))

        return list(set(candidates))  # Deduplicate

    def _get_id_columns(self, df: DataFrame) -> Set[str]:
        """Get columns that look like identifiers."""
        id_cols = set()
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.ID_PATTERNS:
                if pattern in col_lower:
                    id_cols.add(col)
                    break
        return id_cols

    def _evaluate_join(self, df1: DataFrame, df2: DataFrame,
                       left_col: str, right_col: str) -> tuple:
        """
        Evaluate quality of a join on given columns.

        Returns:
            Tuple of (score, left_coverage, right_coverage)
        """
        if left_col not in df1.columns or right_col not in df2.columns:
            return 0.0, 0.0, 0.0

        left_values = set(df1[left_col].dropna().unique())
        right_values = set(df2[right_col].dropna().unique())

        if not left_values or not right_values:
            return 0.0, 0.0, 0.0

        # Calculate coverage
        overlap = left_values & right_values
        left_coverage = len(overlap) / len(left_values) if left_values else 0
        right_coverage = len(overlap) / len(right_values) if right_values else 0

        # Score based on coverage (harmonic mean for balance)
        if left_coverage + right_coverage > 0:
            score = 2 * left_coverage * right_coverage / (left_coverage + right_coverage)
        else:
            score = 0

        # Boost score if column names match exactly
        if left_col == right_col:
            score = min(1.0, score * 1.2)

        return score, left_coverage, right_coverage

    def _determine_relationship_type(self, df1: DataFrame, df2: DataFrame,
                                     left_col: str, right_col: str) -> RelationshipType:
        """Determine the type of relationship based on key uniqueness."""
        left_unique = df1[left_col].nunique() == len(df1[left_col].dropna())
        right_unique = df2[right_col].nunique() == len(df2[right_col].dropna())

        if left_unique and right_unique:
            return RelationshipType.ONE_TO_ONE
        elif left_unique and not right_unique:
            return RelationshipType.ONE_TO_MANY
        elif not left_unique and right_unique:
            return RelationshipType.MANY_TO_ONE
        else:
            return RelationshipType.MANY_TO_MANY

    def _check_composite_key(self, df1: DataFrame, df2: DataFrame,
                            candidates: List[tuple]) -> bool:
        """Check if multiple columns together form a composite key."""
        if len(candidates) < 2:
            return False

        # Get common columns that could be part of composite key
        common_cols = []
        for left_col, right_col in candidates:
            if left_col == right_col:
                common_cols.append(left_col)

        if len(common_cols) < 2:
            return False

        # Check if combining columns creates unique keys where individual columns don't
        for i, col1 in enumerate(common_cols):
            for col2 in common_cols[i+1:]:
                # Check if (col1, col2) is more unique than col1 or col2 alone
                single_unique_1 = df1[col1].nunique()
                single_unique_2 = df1[col2].nunique()
                combo_unique = df1.groupby([col1, col2]).ngroups

                if combo_unique > max(single_unique_1, single_unique_2):
                    return True

        return False
