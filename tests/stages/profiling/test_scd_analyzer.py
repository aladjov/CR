import pandas as pd
import pytest

from customer_retention.core.config import ColumnConfig, ColumnType, DataSourceConfig, FileFormat, SourceType
from customer_retention.stages.profiling import SCDAnalyzer


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data with changing attributes."""
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "order_date": pd.date_range("2024-01-01", periods=9),
        "city": ["NYC", "NYC", "LA", "SF", "SF", "Boston", "Chicago", "Chicago", "Chicago"],
        "status": ["active", "active", "active", "active", "inactive", "active", "active", "active", "inactive"],
        "order_count": [1, 2, 3, 1, 2, 1, 2, 3, 4]
    })


@pytest.fixture
def sample_config():
    return DataSourceConfig(
        name="test_data",
        source_type=SourceType.BATCH_FILE,
        path="/tmp/test.csv",
        file_format=FileFormat.CSV,
        primary_key="customer_id",
        columns=[
            ColumnConfig(name="customer_id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="order_date", column_type=ColumnType.DATETIME),
            ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL),
            ColumnConfig(name="status", column_type=ColumnType.CATEGORICAL_NOMINAL),
            ColumnConfig(name="order_count", column_type=ColumnType.NUMERIC_DISCRETE),
        ]
    )


class TestSCDAnalyzerBasic:
    def test_analyzer_initialization(self):
        analyzer = SCDAnalyzer()
        assert analyzer is not None

    def test_analyze_with_entity_key(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(sample_transaction_data)

        assert result is not None
        assert isinstance(result, dict)


class TestSCDDetection:
    def test_detect_changing_column(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(sample_transaction_data)

        # city changes for customer 1 and customer 3
        assert "city" in result
        assert result["city"]["changes_detected"] is True
        assert result["city"]["entities_with_change"] > 0

    def test_detect_non_changing_column(self, sample_transaction_data):
        # Add a constant column
        df = sample_transaction_data.copy()
        df["country"] = "USA"  # Never changes

        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(df)

        assert "country" in result
        assert result["country"]["changes_detected"] is False

    def test_change_percentage_calculation(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(sample_transaction_data)

        # city changes for 2 out of 3 customers
        assert "city" in result
        assert result["city"]["change_percentage"] > 50


class TestSCDTypeRecommendation:
    def test_recommend_type_0_for_static(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        # Simulate static attribute
        scd_metrics = {
            "changes_detected": False,
            "change_percentage": 0.0
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 0" in recommendation and "Static" in recommendation

    def test_recommend_type_1_for_rare_changes(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        # Simulate rare changes
        scd_metrics = {
            "changes_detected": True,
            "change_percentage": 5.0,
            "avg_changes_per_entity": 1.1
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 1" in recommendation

    def test_recommend_type_2_for_frequent_changes(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        # Simulate frequent changes
        scd_metrics = {
            "changes_detected": True,
            "change_percentage": 60.0,
            "avg_changes_per_entity": 3.5
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 2" in recommendation


class TestSCDMetrics:
    def test_metrics_contain_required_fields(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(sample_transaction_data)

        for col_name, metrics in result.items():
            if col_name != "customer_id":  # Skip entity key
                assert "changes_detected" in metrics
                assert "entities_with_change" in metrics
                assert "change_percentage" in metrics
                assert "max_changes" in metrics
                assert "avg_changes_per_entity" in metrics
                assert "scd_type_recommendation" in metrics

    def test_max_changes_calculation(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        result = analyzer.analyze(sample_transaction_data)

        # Customer 3 has 2 city changes (Boston -> Chicago)
        assert result["city"]["max_changes"] >= 1


class TestSCDResultToDict:
    def test_to_dict_returns_all_fields(self):
        from customer_retention.stages.profiling.scd_analyzer import SCDResult

        result = SCDResult(
            column_name="city",
            changes_detected=True,
            entities_with_change=5,
            change_percentage=33.33,
            max_changes=3,
            avg_changes_per_entity=1.5,
            scd_type_recommendation="Type 2 (Track History - Frequent changes, full history needed)"
        )
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["column_name"] == "city"
        assert d["changes_detected"] is True
        assert d["entities_with_change"] == 5
        assert d["change_percentage"] == 33.33
        assert d["max_changes"] == 3
        assert d["avg_changes_per_entity"] == 1.5
        assert "Type 2" in d["scd_type_recommendation"]


class TestSCDAnalyzerValidation:
    def test_analyze_with_entity_key_none_raises(self):
        analyzer = SCDAnalyzer(entity_key=None)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="entity_key must be set"):
            analyzer.analyze(df)

    def test_analyze_with_entity_key_not_in_df_raises(self):
        analyzer = SCDAnalyzer(entity_key="nonexistent_col")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="not found in dataframe"):
            analyzer.analyze(df)


class TestSCDType3Recommendation:
    def test_recommend_type_3_for_moderate_changes(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        # Moderate changes: 10-30% change, avg_changes 1-3
        scd_metrics = {
            "changes_detected": True,
            "change_percentage": 20.0,
            "avg_changes_per_entity": 2.0
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 3" in recommendation
        assert "Previous" in recommendation

    def test_recommend_type_3_boundary_low(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        scd_metrics = {
            "changes_detected": True,
            "change_percentage": 10.0,
            "avg_changes_per_entity": 2.0
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 3" in recommendation

    def test_recommend_type_3_boundary_high(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")

        scd_metrics = {
            "changes_detected": True,
            "change_percentage": 29.99,
            "avg_changes_per_entity": 2.99
        }

        recommendation = analyzer.recommend_scd_type(scd_metrics)
        assert "Type 3" in recommendation


class TestSCDToDataframe:
    def test_to_dataframe_returns_dataframe(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        results = analyzer.analyze(sample_transaction_data)
        df = analyzer.to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(results)
        assert "column" in df.columns
        assert "changes_detected" in df.columns
        assert "scd_type_recommendation" in df.columns

    def test_to_dataframe_column_values(self, sample_transaction_data):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        results = analyzer.analyze(sample_transaction_data)
        df = analyzer.to_dataframe(results)

        column_names = df["column"].tolist()
        for col_name in results:
            assert col_name in column_names

    def test_to_dataframe_empty_results(self):
        analyzer = SCDAnalyzer(entity_key="customer_id")
        df = analyzer.to_dataframe({})

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestSCDWithConfig:
    def test_analyze_with_config(self, sample_transaction_data, sample_config):
        analyzer = SCDAnalyzer()
        result = analyzer.analyze_with_config(sample_transaction_data, sample_config)

        assert result is not None
        # Should analyze all columns except the primary key
        assert len(result) <= len(sample_config.columns) - 1
