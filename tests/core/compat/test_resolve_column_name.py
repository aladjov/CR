import pandas as pd
import pytest

from customer_retention.core.compat import resolve_column_name


class TestResolveColumnName:
    def test_exact_match(self):
        columns = pd.Index(["ACCOUNT_ID", "Name", "age"])
        assert resolve_column_name(columns, "ACCOUNT_ID") == "ACCOUNT_ID"

    def test_lowercase_match(self):
        columns = pd.Index(["account_id", "name", "age"])
        assert resolve_column_name(columns, "ACCOUNT_ID") == "account_id"

    def test_uppercase_match(self):
        columns = pd.Index(["ACCOUNT_ID", "NAME", "AGE"])
        assert resolve_column_name(columns, "account_id") == "ACCOUNT_ID"

    def test_mixed_case_match(self):
        columns = pd.Index(["Account_Id", "Name", "Age"])
        assert resolve_column_name(columns, "account_id") == "Account_Id"

    def test_missing_column_raises(self):
        columns = pd.Index(["account_id", "name"])
        with pytest.raises(KeyError, match="not_here"):
            resolve_column_name(columns, "not_here")

    def test_prefers_exact_match_over_case_insensitive(self):
        columns = pd.Index(["account_id", "Account_Id"])
        assert resolve_column_name(columns, "account_id") == "account_id"
