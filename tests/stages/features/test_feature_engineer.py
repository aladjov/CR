
import pandas as pd
import pytest

from customer_retention.stages.features import FeatureEngineer, FeatureEngineerConfig


class TestFeatureEngineerConfig:
    def test_default_config(self):
        config = FeatureEngineerConfig()

        assert config.reference_date is None
        assert config.generate_temporal is True
        assert config.generate_behavioral is True
        assert config.generate_interaction is True

    def test_custom_config(self):
        ref_date = pd.Timestamp("2024-07-01")
        config = FeatureEngineerConfig(
            reference_date=ref_date,
            generate_temporal=True,
            generate_behavioral=False,
            generate_interaction=True,
            created_column="created",
            last_order_column="lastorder"
        )

        assert config.reference_date == ref_date
        assert config.generate_behavioral is False


class TestFeatureEngineerBasic:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "created": pd.to_datetime(["2024-01-01", "2024-03-15", "2024-02-01"]),
            "firstorder": pd.to_datetime(["2024-01-15", "2024-03-20", "2024-02-10"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-06-15", "2024-05-20"]),
            "esent": [100, 50, 75],
            "eopenrate": [0.4, 0.6, 0.5],
            "eclickrate": [0.2, 0.3, 0.25],
            "avgorder": [50.0, 100.0, 75.0],
            "ordfreq": [2.0, 1.0, 1.5],
            "paperless": [1, 0, 1],
            "refill": [1, 1, 0],
            "doorstep": [0, 1, 1]
        })

    def test_fit_transform_basic(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert result.df is not None
        assert len(result.df) == len(sample_df)

    def test_generates_temporal_features(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder",
            first_order_column="firstorder",
            generate_temporal=True,
            generate_behavioral=False,
            generate_interaction=False
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert "tenure_days" in result.df.columns
        assert "days_since_last_order" in result.df.columns

    def test_generates_behavioral_features(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            generate_temporal=True,
            generate_behavioral=True,
            generate_interaction=False,
            open_rate_column="eopenrate",
            click_rate_column="eclickrate",
            service_columns=["paperless", "refill", "doorstep"]
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert "email_engagement_score" in result.df.columns
        assert "service_adoption_score" in result.df.columns


class TestFeatureEngineerInteraction:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002"],
            "created": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "avgorder": [50.0, 100.0],
            "ordfreq": [2.0, 1.0],
            "eopenrate": [0.4, 0.6],
            "eclickrate": [0.2, 0.3]
        })

    def test_generates_interaction_features(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            generate_temporal=True,
            generate_behavioral=False,
            generate_interaction=True,
            interaction_combinations=[
                ("avgorder", "ordfreq", "value_x_frequency", "multiply")
            ]
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert "value_x_frequency" in result.df.columns


class TestFeatureEngineerResult:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"]),
            "lastorder": pd.to_datetime(["2024-06-01"])
        })

    def test_result_contains_generated_features_list(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert hasattr(result, "generated_features")
        assert isinstance(result.generated_features, list)
        assert len(result.generated_features) > 0

    def test_result_contains_feature_categories(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created"
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert hasattr(result, "feature_categories")
        assert isinstance(result.feature_categories, dict)


class TestFeatureEngineerFitTransformSeparation:
    def test_fit_then_transform(self):
        train = pd.DataFrame({
            "custid": ["C001", "C002"],
            "created": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-06-15"])
        })
        test = pd.DataFrame({
            "custid": ["C003"],
            "created": pd.to_datetime(["2024-03-01"]),
            "lastorder": pd.to_datetime(["2024-06-20"])
        })

        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        engineer = FeatureEngineer(config)
        engineer.fit(train)
        result = engineer.transform(test)

        assert result.df is not None
        assert len(result.df) == 1


class TestFeatureEngineerNullHandling:
    def test_handles_null_input_columns(self):
        df = pd.DataFrame({
            "custid": ["C001", "C002"],
            "created": pd.to_datetime(["2024-01-01", None]),
            "lastorder": pd.to_datetime([None, "2024-06-15"])
        })

        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        # Should not raise error
        assert result.df is not None


class TestFeatureEngineerCatalogIntegration:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"]),
            "lastorder": pd.to_datetime(["2024-06-01"])
        })

    def test_populates_feature_catalog(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder",
            populate_catalog=True
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        assert hasattr(engineer, "catalog")
        assert len(engineer.catalog) > 0

    def test_catalog_contains_feature_definitions(self, sample_df):
        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            populate_catalog=True
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(sample_df)

        if "tenure_days" in result.df.columns:
            feature_def = engineer.catalog.get("tenure_days")
            assert feature_def is not None
            assert feature_def.name == "tenure_days"


class TestFeatureEngineerPreserveColumns:
    def test_preserves_original_columns(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"]),
            "extra_column": [100]
        })

        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            preserve_original=True
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        assert "custid" in result.df.columns
        assert "extra_column" in result.df.columns

    def test_drop_original_columns(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "created": pd.to_datetime(["2024-01-01"])
        })

        config = FeatureEngineerConfig(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            preserve_original=False,
            id_column="custid"
        )
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        # Should keep id column even when not preserving original
        assert "custid" in result.df.columns
