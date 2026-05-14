"""Tests for the refactored callable batch inference framework module."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    BatchInferenceResult,
    apply_risk_tiers_pandas,
    run_batch_inference,
)


class TestBatchInferenceConfig:
    def test_default_thresholds(self):
        config = BatchInferenceConfig()
        assert config.threshold == 0.5
        assert config.risk_tier_high == 0.6
        assert config.risk_tier_medium == 0.3
        assert config.target_table == "predictions"
        assert config.audit_table == "inference_audit_log"

    def test_custom_thresholds(self):
        config = BatchInferenceConfig(
            threshold=0.4, risk_tier_high=0.75, risk_tier_medium=0.4
        )
        assert config.threshold == 0.4
        assert config.risk_tier_high == 0.75
        assert config.risk_tier_medium == 0.4

    def test_filter_expression_default_none(self):
        assert BatchInferenceConfig().filter_expression is None

    def test_filter_expression_is_passthrough_string(self):
        config = BatchInferenceConfig(filter_expression="plan_type == 'enterprise'")
        assert config.filter_expression == "plan_type == 'enterprise'"

    def test_customer_table_default_none(self):
        assert BatchInferenceConfig().customer_table is None

    def test_customer_table_override_is_passthrough(self):
        cfg = BatchInferenceConfig(customer_table="cat.sch.gold_features_abc")
        assert cfg.customer_table == "cat.sch.gold_features_abc"

    def test_timestamp_column_default_matches_training(self):
        """Must match ``databricks_renderer.py`` training template's
        ``TIMESTAMP_COLUMN = "event_timestamp"`` — a mismatch with the
        registered FE lookup spec raises
        ``"Unable to join feature table ... because timestamp lookup key
        'event_timestamp' not found in DataFrame"``."""
        assert BatchInferenceConfig().timestamp_column == "event_timestamp"

    def test_timestamp_column_override_is_passthrough(self):
        cfg = BatchInferenceConfig(timestamp_column="ts")
        assert cfg.timestamp_column == "ts"


class TestResolveModelVersion:
    def test_resolves_alias_uri(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference

        mv = MagicMock(version="47", run_id="abc123")
        client = MagicMock()
        client.get_model_version_by_alias = MagicMock(return_value=mv)

        fake_tracking = type(sys)("mlflow.tracking")
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = type(sys)("mlflow")
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        version, run_id = batch_inference._resolve_model_version(
            "models:/c.s.model_abc@production"
        )
        assert (version, run_id) == ("47", "abc123")
        client.get_model_version_by_alias.assert_called_once_with("c.s.model_abc", "production")

    def test_resolves_numeric_version_uri(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference

        mv = MagicMock(version="3", run_id="def456")
        client = MagicMock()
        client.get_model_version = MagicMock(return_value=mv)

        fake_tracking = type(sys)("mlflow.tracking")
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = type(sys)("mlflow")
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        version, run_id = batch_inference._resolve_model_version("models:/c.s.m/3")
        assert (version, run_id) == ("3", "def456")

    def test_non_models_uri_returns_none_tuple(self):
        from customer_retention.stages.scoring import batch_inference

        assert batch_inference._resolve_model_version("runs:/abc/model") == (None, None)

    def test_mlflow_failure_is_swallowed_as_diagnostic(self, monkeypatch):
        """Resolution is *diagnostic enrichment*. Never block scoring over it."""
        import sys
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference

        client = MagicMock()
        client.get_model_version_by_alias = MagicMock(side_effect=RuntimeError("API down"))

        fake_tracking = type(sys)("mlflow.tracking")
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = type(sys)("mlflow")
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        assert batch_inference._resolve_model_version(
            "models:/c.s.m@production"
        ) == (None, None)


class TestProgressReporting:
    """Structured ``[batch_inference]`` prints + populated diagnostic fields
    on ``BatchInferenceResult`` so operators can grep cell output and
    downstream code can persist provenance without re-calling MLflow."""

    def test_provenance_block_prints_identity_lines(self, monkeypatch, capsys):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        class _FakeSparkDF:
            def filter(self, _expr): return self
            def select(self, *_a, **_kw): return self
            def distinct(self): return self
            def withColumn(self, *_a, **_kw): return self  # noqa: N802
            def count(self): return 77255

        class _BoomError(RuntimeError):
            pass

        monkeypatch.setattr(
            batch_inference,
            "_resolve_model_version",
            lambda _uri: ("48", "run_xyz"),
        )
        monkeypatch.setattr(
            batch_inference,
            "_score_with_feature_store",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomError()),
        )
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s",
            model_uri="models:/c.s.model_abc@production",
            customer_table="c.s.gold_features_abc",
            filter_expression="region == 'EU'",
        )
        with pytest.raises(_BoomError):
            _run_databricks(cfg)

        out = capsys.readouterr().out
        # Every diagnostic line is grep-friendly with the fixed prefix.
        assert "[batch_inference] model_uri=models:/c.s.model_abc@production" in out
        assert "[batch_inference] model_version=v48 run_id=run_xyz" in out
        assert "[batch_inference] customer_table=c.s.gold_features_abc" in out
        assert "[batch_inference] timestamp_column=event_timestamp" in out
        assert "[batch_inference] scope_filter=region == 'EU'" in out
        assert "[batch_inference] target_table=c.s.predictions" in out
        assert "[batch_inference] 77,255 entities after filter/dedup" in out

    def test_zero_entity_population_fails_fast(self, monkeypatch):
        """A filter/dedup that collapses to zero entities would silently write
        an empty predictions table without this guard. Fail fast with a clear
        message so operators see the actual cohort problem, not a downstream
        NullPointerException in the dashboard."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        class _FakeSparkDF:
            def filter(self, _expr): return self
            def select(self, *_a, **_kw): return self
            def distinct(self): return self
            def withColumn(self, *_a, **_kw): return self  # noqa: N802
            def count(self): return 0

        monkeypatch.setattr(
            batch_inference, "_resolve_model_version", lambda _u: (None, None)
        )
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s",
            model_uri="models:/c.s.m@production",
            customer_table="c.s.gold_features_abc",
        )
        with pytest.raises(ValueError, match="0 entities"):
            _run_databricks(cfg)


class TestBatchInferenceResultDiagnosticFields:
    def test_long_summary_includes_resolved_version_and_run_id(self):
        result = BatchInferenceResult(
            inference_id="batch_20260419_120000",
            inference_timestamp=datetime(2026, 4, 19, 12, 0, 0),
            total_scored=100,
            predicted_churners=20,
            avg_probability=0.2,
            risk_distribution={"High": 5, "Medium": 15, "Low": 80},
            model_uri="models:/c.s.m@production",
            resolved_model_version="48",
            resolved_run_id="abc123",
        )
        text = result.long_summary()
        assert "Resolved version:" in text
        assert "v48" in text
        assert "abc123" in text

    def test_long_summary_includes_phase_timings(self):
        result = BatchInferenceResult(
            inference_id="x",
            inference_timestamp=datetime(2026, 4, 19),
            total_scored=1,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="u",
            phase_seconds={"prep": 0.5, "score": 12.3, "write": 2.1, "total": 14.9},
        )
        text = result.long_summary()
        assert "Wall-clock phases:" in text
        assert "prep" in text and "12.3" in text
        assert "total" in text and "14.9" in text

    def test_long_summary_surfaces_entity_drop_warning(self):
        """If entity_count > total_scored it means fe.score_batch silently
        dropped rows — surface that in the summary so operators investigate
        missing features rather than trusting a shrunken population."""
        result = BatchInferenceResult(
            inference_id="x",
            inference_timestamp=datetime(2026, 4, 19),
            total_scored=75_000,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="u",
            entity_count=77_255,
        )
        text = result.long_summary()
        assert "Entity population (input)" in text
        assert "77,255" in text
        assert "dropped" in text.lower() or "missing features" in text


class TestScopeFilterApplication:
    """Scope filter (NB00 ``ProjectContext.sample_filters``) must narrow the
    customer population before the entity projection on both execution paths.
    Applied as a narrow projection (``df.filter`` / ``safe_query``) — no
    shuffle, no extra Spark jobs, stays distributed."""

    def test_databricks_path_calls_df_filter_when_expression_set(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        filter_calls: list[str] = []

        class _FakeSparkDF:
            def __init__(self, name="root"):
                self._name = name

            def filter(self, expr):
                filter_calls.append(expr)
                return _FakeSparkDF(name="filtered")

            def select(self, *_a, **_kw):
                return self

            def distinct(self):
                return self

            def count(self): return 100

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

            def agg(self, *_a, **_kw):
                r = MagicMock()
                r.collect.return_value = [{"total": 0, "churners": 0, "avg_prob": 0.0}]
                return r

            def groupBy(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                r = MagicMock()
                cnt = MagicMock()
                cnt.collect.return_value = []
                r.count.return_value = cnt
                return r

            def write(self, *_a, **_kw):
                return MagicMock()

        # Short-circuit at the first point past filter application so we can
        # assert the filter was invoked without wiring the full Spark plan.
        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        def stub_score(spark, entity_df, feature_table, model_uri, **_kw):  # noqa: ARG001
            raise _Boom("stop after filter")

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)

        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s", model_name="m",
            filter_expression="plan_type == 'enterprise'",
        )
        with pytest.raises(_Boom):
            _run_databricks(cfg)
        assert filter_calls == ["plan_type == 'enterprise'"]

    def test_databricks_path_translates_pandas_in_to_sql_tuple(self, monkeypatch):
        """NB00's ``sample_filter`` predicate is stored in pandas/Python syntax
        (``column in ['a', 'b']``) because exploration filters via ``df.query()``.
        Spark ``df.filter()`` needs SQL-tuple syntax (``column IN ('a', 'b')``);
        without translation it raises ``[PARSE_SYNTAX_ERROR] Syntax error at or
        near '['``. The same translator the landing-script generator already
        applies (``_spark_safe_query_expr``) must also fire at scoring time so
        one predicate string drives both stages."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        filter_calls: list[str] = []

        class _FakeSparkDF:
            def filter(self, expr):
                filter_calls.append(expr)
                return self

            def select(self, *_a, **_kw): return self
            def distinct(self): return self
            def count(self): return 100
            def withColumn(self, *_a, **_kw): return self  # noqa: N802

            def agg(self, *_a, **_kw):
                r = MagicMock()
                r.collect.return_value = [{"total": 0, "churners": 0, "avg_prob": 0.0}]
                return r

            def groupBy(self, *_a, **_kw):  # noqa: N802
                r = MagicMock()
                cnt = MagicMock()
                cnt.collect.return_value = []
                r.count.return_value = cnt
                return r

            def write(self, *_a, **_kw): return MagicMock()

        class _Boom2Error(RuntimeError):
            pass

        def stub_score(spark, entity_df, feature_table, model_uri, **_kw):  # noqa: ARG001
            raise _Boom2Error("stop after filter")

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s", model_name="m",
            filter_expression=(
                "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small'] and "
                "ACCOUNT_ID in (select ACCOUNT_ID from contract where event_type = 'start')"
            ),
        )
        with pytest.raises(_Boom2Error):
            _run_databricks(cfg)
        assert filter_calls == [
            "REVENUE_MARKET_SEGMENT in ('Emerging', 'Small') and "
            "ACCOUNT_ID in (select ACCOUNT_ID from contract where event_type = 'start')"
        ], (
            "Pandas-style 'in [..]' must be translated to SQL-style 'in (..)' "
            "before df.filter(); the embedded SQL subquery '(select ... from ...)' "
            "must pass through verbatim."
        )

    def test_databricks_path_routes_filter_via_landing_table(self, monkeypatch):
        """When the scope filter references a raw landing column that has been
        one-hot encoded in gold (typical for a string categorical like
        ``REVENUE_MARKET_SEGMENT``), Spark raises [UNRESOLVED_COLUMN] on direct
        ``df.filter()`` against the customer table. ``filter_via_table`` routes
        the filter through the source landing table and inner-joins the
        surviving entity_id set with the customer table."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        table_calls: list[str] = []
        filter_calls: list[str] = []
        join_calls: list[tuple] = []

        class _FakeSparkDF:
            def __init__(self, label="root"):
                self._label = label

            def filter(self, expr):
                filter_calls.append((self._label, expr))
                return _FakeSparkDF(label=f"{self._label}.filter")

            def select(self, *_a, **_kw):
                return _FakeSparkDF(label=f"{self._label}.select")

            def distinct(self):
                return _FakeSparkDF(label=f"{self._label}.distinct")

            def join(self, other, on=None, how=None):
                join_calls.append((self._label, getattr(other, "_label", "?"), on, how))
                return _FakeSparkDF(label=f"{self._label}.join")

            def count(self): return 100
            def withColumn(self, *_a, **_kw): return self  # noqa: N802

            def agg(self, *_a, **_kw):
                r = MagicMock()
                r.collect.return_value = [{"total": 0, "churners": 0, "avg_prob": 0.0}]
                return r

            def groupBy(self, *_a, **_kw):  # noqa: N802
                r = MagicMock()
                cnt = MagicMock()
                cnt.collect.return_value = []
                r.count.return_value = cnt
                return r

            def write(self, *_a, **_kw): return MagicMock()

        class _BoomError(RuntimeError):
            pass

        def stub_score(spark, entity_df, feature_table, model_uri, **_kw):  # noqa: ARG001
            raise _BoomError("stop after filter+join")

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)

        fake_spark = MagicMock()
        def _fake_table(name):
            table_calls.append(name)
            return _FakeSparkDF(label=name)
        fake_spark.table.side_effect = _fake_table
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s", model_name="m",
            customer_table="c.s.gold_features_X",
            filter_expression="REVENUE_MARKET_SEGMENT in ['Emerging', 'Small']",
            filter_via_table="c.s.landing_account",
        )
        with pytest.raises(_BoomError):
            _run_databricks(cfg)

        # The customer table AND the landing table must both be read.
        assert "c.s.gold_features_X" in table_calls
        assert "c.s.landing_account" in table_calls
        # The filter is applied to landing, NOT to gold directly.
        assert filter_calls == [
            ("c.s.landing_account", "REVENUE_MARKET_SEGMENT in ('Emerging', 'Small')")
        ], filter_calls
        # The customer table is inner-joined to the filtered entity_id set.
        assert len(join_calls) == 1
        left_label, right_label, on, how = join_calls[0]
        assert left_label == "c.s.gold_features_X"
        assert "landing_account" in right_label
        assert on == "entity_id"
        assert how == "inner"

    def test_databricks_path_reads_config_customer_table_and_dedups(self, monkeypatch):
        """Regression: the default ``gold_customers`` table doesn't exist in
        projects that use composite-name-qualified gold tables (``gold_features_{CN}``).
        The caller (c04) passes ``customer_table=GOLD_FEATURES_FQN``; the framework
        must read from that table and collapse to one row per entity via
        ``.distinct()`` so fe.score_batch only sees unique entity_ids.
        ``.distinct()`` shuffle is bounded by the entity count — one bounded
        Spark job, no wide aggregations."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        table_calls: list[str] = []
        distinct_calls = {"count": 0}

        class _FakeSparkDF:
            def filter(self, _expr):
                return self

            def select(self, *_a, **_kw):
                return self

            def distinct(self):
                distinct_calls["count"] += 1
                return self

            def count(self):
                return 100

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

        class _BoomError(RuntimeError):
            pass

        monkeypatch.setattr(
            batch_inference,
            "_score_with_feature_store",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomError()),
        )
        fake_spark = MagicMock()

        def _table(name):
            table_calls.append(name)
            return _FakeSparkDF()

        fake_spark.table.side_effect = _table
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s", model_name="m",
            customer_table="c.s.gold_features_abc",
        )
        with pytest.raises(_BoomError):
            _run_databricks(cfg)
        assert table_calls == ["c.s.gold_features_abc"]
        assert distinct_calls["count"] == 1

    def test_databricks_path_uses_configured_timestamp_column(self, monkeypatch):
        """Regression for `Unable to join feature table ... because timestamp
        lookup key 'event_timestamp' not found in DataFrame`. The entity_df's
        timestamp column MUST be named with ``config.timestamp_column`` so
        the FE registered lookup key resolves."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        with_column_names: list[str] = []
        score_call_kwargs: list[dict] = []

        class _FakeSparkDF:
            def filter(self, _expr):
                return self

            def select(self, *_a, **_kw):
                return self

            def distinct(self):
                return self

            def count(self): return 100

            def withColumn(self, name, _expr):  # noqa: N802 — mirrors Spark API
                with_column_names.append(name)
                return self

        class _BoomError(RuntimeError):
            pass

        def stub_score(spark, entity_df, feature_table, model_uri, timestamp_column):  # noqa: ARG001
            score_call_kwargs.append({"timestamp_column": timestamp_column})
            raise _BoomError()

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s",
            model_uri="models:/c.s.model_abc@production",
        )
        with pytest.raises(_BoomError):
            _run_databricks(cfg)

        assert with_column_names == ["event_timestamp"]
        assert score_call_kwargs == [{"timestamp_column": "event_timestamp"}]

    def test_score_with_feature_store_uses_timestamp_column_in_lookup(self, monkeypatch):
        """``FeatureLookup.timestamp_lookup_key`` must come from the config
        knob — otherwise the fallback path's training-set load fails with
        the same mismatch as ``fe.score_batch``."""
        pytest.importorskip("pyspark", reason="PySpark required")
        import sys
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference

        captured_kwargs: list[dict] = []

        class _FakeFeatureLookup:
            def __init__(self, **kwargs):
                captured_kwargs.append(kwargs)

        class _BoomError(RuntimeError):
            pass

        class _FakeClient:
            def score_batch(self, **_kw):
                raise _BoomError("stop after FeatureLookup constructed")

        fake_fe = type(sys)("databricks.feature_engineering")
        fake_fe.FeatureEngineeringClient = lambda *a, **kw: _FakeClient()
        fake_fe.FeatureLookup = _FakeFeatureLookup
        fake_databricks = type(sys)("databricks")
        fake_databricks.feature_engineering = fake_fe
        monkeypatch.setitem(sys.modules, "databricks", fake_databricks)
        monkeypatch.setitem(sys.modules, "databricks.feature_engineering", fake_fe)

        with pytest.raises(_BoomError):
            batch_inference._score_with_feature_store(
                spark=MagicMock(),
                entity_df=MagicMock(),
                feature_table="c.s.ft",
                model_uri="models:/c.s.m@production",
                timestamp_column="event_timestamp",
            )
        assert captured_kwargs == [{
            "table_name": "c.s.ft",
            "lookup_key": ["entity_id"],
            "timestamp_lookup_key": "event_timestamp",
        }]

    def test_databricks_path_falls_back_to_gold_customers(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        table_calls: list[str] = []

        class _FakeSparkDF:
            def filter(self, _expr):
                return self

            def select(self, *_a, **_kw):
                return self

            def distinct(self):
                return self

            def count(self): return 100

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

        class _BoomError(RuntimeError):
            pass

        monkeypatch.setattr(
            batch_inference,
            "_score_with_feature_store",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomError()),
        )
        fake_spark = MagicMock()
        fake_spark.table.side_effect = lambda n: table_calls.append(n) or _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        with pytest.raises(_BoomError):
            _run_databricks(BatchInferenceConfig(catalog="c", schema="s", model_name="m"))
        assert table_calls == ["c.s.gold_customers"]

    def test_databricks_path_skips_filter_when_expression_missing(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        filter_calls: list[str] = []

        class _FakeSparkDF:
            def filter(self, expr):
                filter_calls.append(expr)
                return self

            def select(self, *_a, **_kw):
                return self

            def distinct(self):
                return self

            def count(self): return 100

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        monkeypatch.setattr(
            batch_inference,
            "_score_with_feature_store",
            lambda *a, **kw: (_ for _ in ()).throw(_Boom()),
        )
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        with pytest.raises(_Boom):
            _run_databricks(BatchInferenceConfig(catalog="c", schema="s", model_name="m"))
        assert filter_calls == []

    def test_local_path_applies_safe_query_when_expression_set(self, monkeypatch, tmp_path):
        import pandas as _pd

        from customer_retention.core import compat
        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_local,
        )

        captured: list[str] = []

        def fake_safe_query(df, expr):
            captured.append(expr)
            return df.iloc[:1]

        monkeypatch.setattr(compat, "safe_query", fake_safe_query)

        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        monkeypatch.setattr(
            batch_inference,
            "_load_local_customers",
            lambda *_a, **_kw: _pd.DataFrame(
                {"customer_id": [1, 2, 3], "plan_type": ["free", "enterprise", "enterprise"]}
            ),
        )

        # Stub feature-store-manager so `get_inference_features` raises,
        # which sits *after* the filter application — letting us prove the
        # filter was applied.
        import sys
        fake_fs = type(sys)("customer_retention.integrations.feature_store")

        class _StubRegistry:
            @staticmethod
            def load(_p):
                class _R:
                    @staticmethod
                    def list_features():
                        return ["f1"]
                return _R()

        class _BoomManager:
            def get_inference_features(self, *_a, **_kw):
                raise _Boom("stop after filter")

        fake_fs.FeatureRegistry = _StubRegistry
        fake_fs.get_feature_store_manager = lambda *a, **kw: _BoomManager()
        monkeypatch.setitem(sys.modules, "customer_retention.integrations.feature_store", fake_fs)

        # Stub the delta factory too — same short-circuit style.
        fake_factory = type(sys)("customer_retention.integrations.adapters.factory")
        fake_factory.get_delta = lambda: object()
        monkeypatch.setitem(
            sys.modules,
            "customer_retention.integrations.adapters.factory",
            fake_factory,
        )

        cfg = BatchInferenceConfig(
            model_path=tmp_path / "stub_model.joblib",
            filter_expression="plan_type == 'enterprise'",
        )
        cfg.model_path.write_bytes(b"")
        import joblib

        class _StubModel:
            feature_names_in_ = ["f1"]

        monkeypatch.setattr(joblib, "load", lambda _p: _StubModel())

        with pytest.raises(_Boom):
            _run_local(cfg)
        assert captured == ["plan_type == 'enterprise'"]


class TestBatchInferenceResult:
    def test_summary_one_liner(self):
        result = BatchInferenceResult(
            inference_id="batch_20260408_120000",
            inference_timestamp=datetime(2026, 4, 8, 12, 0, 0),
            total_scored=10000,
            predicted_churners=1234,
            avg_probability=0.234,
            risk_distribution={"High": 500, "Medium": 2000, "Low": 7500},
            model_uri="models:/cat.sch.mdl@production",
        )
        text = result.summary()
        assert "10,000" in text
        assert "1,234" in text
        assert "0.234" in text

    def test_long_summary_includes_all_risk_tiers(self):
        result = BatchInferenceResult(
            inference_id="batch_20260408_120000",
            inference_timestamp=datetime(2026, 4, 8, 12, 0, 0),
            total_scored=100,
            predicted_churners=20,
            avg_probability=0.2,
            risk_distribution={"High": 5, "Medium": 15, "Low": 80},
            model_uri="models:/cat.sch.mdl@production",
        )
        text = result.long_summary()
        assert "High" in text and "5" in text
        assert "Medium" in text and "15" in text
        assert "Low" in text and "80" in text
        assert "models:/cat.sch.mdl@production" in text


class TestApplyRiskTiersPandas:
    def test_three_tier_buckets(self):
        df = pd.DataFrame({"p": [0.05, 0.25, 0.30, 0.45, 0.59, 0.60, 0.95]})
        out = apply_risk_tiers_pandas(df, "p", high=0.6, medium=0.3)
        tiers = out["risk_tier"].tolist()
        # bins=[0, medium=0.3, high=0.6, 1.0] with right-inclusive cuts
        assert tiers[0] == "Low"  # 0.05
        assert tiers[1] == "Low"  # 0.25
        assert tiers[2] == "Low"  # 0.30 (right edge of Low bucket)
        assert tiers[3] == "Medium"  # 0.45
        assert tiers[4] == "Medium"  # 0.59
        assert tiers[5] == "Medium"  # 0.60 (right edge of Medium bucket)
        assert tiers[6] == "High"  # 0.95

    def test_returns_new_dataframe_not_mutating(self):
        df = pd.DataFrame({"p": [0.5]})
        out = apply_risk_tiers_pandas(df, "p", high=0.6, medium=0.3)
        assert "risk_tier" not in df.columns
        assert "risk_tier" in out.columns

    def test_custom_thresholds(self):
        df = pd.DataFrame({"p": [0.1, 0.5, 0.9]})
        out = apply_risk_tiers_pandas(df, "p", high=0.8, medium=0.4)
        tiers = out["risk_tier"].tolist()
        assert tiers[0] == "Low"
        assert tiers[1] == "Medium"
        assert tiers[2] == "High"


class TestApplyRiskTiersSpark:
    """The Spark variant of risk-tier bucketing — tested via a mock DataFrame
    that records the column expression that gets applied.
    """

    def test_invokes_when_chain_with_thresholds(self):
        pytest.importorskip("pyspark", reason="PySpark required for Spark risk-tier tests")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring.batch_inference import apply_risk_tiers_spark

        # Mock DataFrame whose withColumn returns itself
        mock_df = MagicMock()
        mock_df.withColumn.return_value = mock_df

        result = apply_risk_tiers_spark(mock_df, "churn_probability", high=0.7, medium=0.4)

        # withColumn was called once with "risk_tier" + a Column expression
        assert mock_df.withColumn.call_count == 1
        call_args = mock_df.withColumn.call_args
        assert call_args[0][0] == "risk_tier"
        assert result is mock_df


class TestGetSpark:
    def test_raises_when_no_session(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "get_spark_session", lambda: None)
        with pytest.raises(RuntimeError, match="No active Spark session"):
            batch_inference._get_spark()

    def test_returns_session_when_present(self, monkeypatch):
        from unittest.mock import MagicMock

        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        fake_spark = MagicMock(name="SparkSession")
        monkeypatch.setattr(detection, "get_spark_session", lambda: fake_spark)
        assert batch_inference._get_spark() is fake_spark


class TestRunDatabricksRequiresIdentity:
    def test_missing_catalog_or_schema_raises(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required for Databricks inference tests")
        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        monkeypatch.setattr(batch_inference, "_get_spark", lambda: object())
        with pytest.raises(ValueError, match="catalog and schema are required"):
            _run_databricks(BatchInferenceConfig())

    def test_missing_model_identity_raises(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        monkeypatch.setattr(batch_inference, "_get_spark", lambda: object())
        with pytest.raises(ValueError, match="model_uri or model_name"):
            _run_databricks(BatchInferenceConfig(catalog="c", schema="s"))


class TestModelUriPassthrough:
    def test_model_uri_when_set_bypasses_assembly(self, monkeypatch):
        """Regression: ``ScoringConfig.registered_model_name`` is the 3-part
        FQN written by training (``{catalog}.{schema}.model_{CN}``). Assembling
        ``f"models:/{catalog}.{schema}.{model_name}@production"`` double-prefixes
        to ``models:/cat.sch.cat.sch.model_...`` — INVALID. When c04 provides
        ``model_uri=MODEL_URI`` directly, the framework must use it as-is."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        captured_uris: list[str] = []

        class _FakeSparkDF:
            def filter(self, _expr): return self
            def select(self, *_a, **_kw): return self
            def distinct(self): return self
            def count(self): return 100
            def withColumn(self, *_a, **_kw): return self  # noqa: N802

        class _BoomError(RuntimeError):
            pass

        def stub_score(spark, entity_df, feature_table, model_uri, **_kw):  # noqa: ARG001
            captured_uris.append(model_uri)
            raise _BoomError()

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s",
            model_uri="models:/c.s.model_abc@production",
        )
        with pytest.raises(_BoomError):
            _run_databricks(cfg)
        assert captured_uris == ["models:/c.s.model_abc@production"]

    def test_model_uri_falls_back_to_legacy_assembly(self, monkeypatch):
        """Legacy callers that still pass ``model_name`` (unqualified) get the
        assembled ``models:/{catalog}.{schema}.{model_name}@production`` URI."""
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        captured_uris: list[str] = []

        class _FakeSparkDF:
            def filter(self, _expr): return self
            def select(self, *_a, **_kw): return self
            def distinct(self): return self
            def count(self): return 100
            def withColumn(self, *_a, **_kw): return self  # noqa: N802

        class _BoomError(RuntimeError):
            pass

        def stub_score(spark, entity_df, feature_table, model_uri, **_kw):  # noqa: ARG001
            captured_uris.append(model_uri)
            raise _BoomError()

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(catalog="c", schema="s", model_name="m")
        with pytest.raises(_BoomError):
            _run_databricks(cfg)
        assert captured_uris == ["models:/c.s.m@production"]


class TestRunBatchInferenceDispatch:
    def test_dispatcher_picks_local_when_not_databricks(self, monkeypatch):
        # Force is_databricks() False and replace _run_local with a stub
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: False)

        sentinel = BatchInferenceResult(
            inference_id="local_stub",
            inference_timestamp=datetime(2026, 4, 8),
            total_scored=0,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="stub",
        )
        called = {}

        def stub_local(config):
            called["local"] = True
            return sentinel

        monkeypatch.setattr(batch_inference, "_run_local", stub_local)
        result = run_batch_inference(BatchInferenceConfig())
        assert called == {"local": True}
        assert result.inference_id == "local_stub"

    def test_dispatcher_picks_databricks_when_is_databricks_true(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: True)

        sentinel = BatchInferenceResult(
            inference_id="dbx_stub",
            inference_timestamp=datetime(2026, 4, 8),
            total_scored=0,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="stub",
        )
        called = {}

        def stub_dbx(config):
            called["dbx"] = True
            return sentinel

        monkeypatch.setattr(batch_inference, "_run_databricks", stub_dbx)
        result = run_batch_inference(BatchInferenceConfig(catalog="c", schema="s", model_name="m"))
        assert called == {"dbx": True}
        assert result.inference_id == "dbx_stub"

    def test_dispatcher_defaults_inference_timestamp(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: False)
        captured = {}

        def stub_local(config):
            captured["ts"] = config.inference_timestamp
            return BatchInferenceResult(
                inference_id="x",
                inference_timestamp=config.inference_timestamp,
                total_scored=0,
                predicted_churners=0,
                avg_probability=0.0,
                risk_distribution={},
                model_uri="x",
            )

        monkeypatch.setattr(batch_inference, "_run_local", stub_local)
        config = BatchInferenceConfig()  # inference_timestamp=None
        run_batch_inference(config)
        assert isinstance(captured["ts"], datetime), (
            "dispatcher must default inference_timestamp to datetime.now()"
        )


class TestS10CellsCallFramework:
    """Confirm the refactored s10 generator emits cells that import the
    framework module rather than inlining business logic.
    """

    @staticmethod
    def _make_stage():
        from unittest.mock import MagicMock

        from customer_retention.generators.notebook_generator.stages.s10_batch_inference import (
            BatchInferenceStage,
        )

        config = MagicMock()
        config.threshold = 0.5
        config.feature_store.catalog = "test_catalog"
        config.feature_store.schema = "test_schema"
        config.mlflow.model_name = "test_model"
        stage = BatchInferenceStage(config=config, findings=None)
        stage.header_cells = lambda: []
        stage.get_dataset_name = lambda: "test_dataset"
        stage.get_identifier_columns = lambda: ["customer_id"]
        return stage

    def _local_code(self):
        return "\n".join(c.source for c in self._make_stage().generate_local_cells())

    def _databricks_code(self):
        return "\n".join(c.source for c in self._make_stage().generate_databricks_cells())

    def _local_code_only(self):
        # Only code cells (excludes markdown explanations that may legitimately
        # mention the lifted helper names in prose).
        return "\n".join(
            c.source for c in self._make_stage().generate_local_cells() if c.cell_type == "code"
        )

    def _databricks_code_only(self):
        return "\n".join(
            c.source for c in self._make_stage().generate_databricks_cells() if c.cell_type == "code"
        )

    def test_local_cells_import_framework_callable(self):
        code = self._local_code()
        assert "from customer_retention.stages.scoring.batch_inference import" in code
        assert "BatchInferenceConfig" in code
        assert "run_batch_inference" in code

    def test_databricks_cells_import_framework_callable(self):
        code = self._databricks_code()
        assert "from customer_retention.stages.scoring.batch_inference import" in code
        assert "BatchInferenceConfig" in code
        assert "run_batch_inference" in code

    def test_local_cells_pass_dataset_name_through(self):
        code = self._local_code()
        # The dataset_name from the stage flows into BatchInferenceConfig
        assert "test_dataset" in code

    def test_databricks_cells_pass_catalog_schema_model(self):
        code = self._databricks_code()
        assert "test_catalog" in code
        assert "test_schema" in code
        assert "test_model" in code

    def test_no_inline_pd_cut_business_logic(self):
        # Risk-tier bucketing must NOT live in cell *code* — that was the
        # exact business logic that needed to move to the framework.
        # (Markdown cells may explain the bucketing in prose; that's fine.)
        code_local = self._local_code_only()
        code_dbx = self._databricks_code_only()
        assert "pd.cut" not in code_local
        assert "when(col" not in code_dbx
        assert "bins=[0, 0.3, 0.6, 1.0]" not in code_local

    def test_no_inline_fe_score_batch_call(self):
        # The fe.score_batch call now lives inside _run_databricks. Check
        # only code cells; markdown explanations may name the lifted call.
        code = self._databricks_code_only()
        assert "fe.score_batch(" not in code
        assert "FeatureEngineeringClient(" not in code

    def test_no_inline_save_table_calls(self):
        # Both write paths moved to framework
        code_local = self._local_code_only()
        code_dbx = self._databricks_code_only()
        assert ".saveAsTable(" not in code_dbx
        assert "storage.write" not in code_local


class TestS10ScoringReplaySplice:
    """Phase 7 splice point — BatchInferenceStage emits a replay cell iff
    any harvested function has replay_at_scoring=True. Empty / missing
    harvest preserves byte-parity with the pre-Phase-7 notebook."""

    @staticmethod
    def _make_stage(harvest_result=None):
        from unittest.mock import MagicMock

        from customer_retention.generators.notebook_generator.stages.s10_batch_inference import (
            BatchInferenceStage,
        )

        config = MagicMock()
        config.threshold = 0.5
        config.feature_store.catalog = "c"
        config.feature_store.schema = "s"
        config.mlflow.model_name = "m"
        stage = BatchInferenceStage(config=config, findings=None)
        stage.header_cells = lambda: []
        stage.get_dataset_name = lambda: "d"
        stage.get_identifier_columns = lambda: ["id"]
        stage.harvest_result = harvest_result
        return stage

    @staticmethod
    def _harvest_with_replay(names):
        from customer_retention.runtime.harvest import HarvestResult
        from customer_retention.runtime.registry import RegisteredFunction

        hr = HarvestResult.empty()
        for n in names:
            rf = RegisteredFunction(
                name=n,
                source=f"def {n}(df): return df",
                scope="dataset",
                dataset="request",
                replay_at_scoring=True,
                inferred_stage="landing_post",
            )
            hr.functions_by_target.setdefault(("landing_post", "request"), []).append(rf)
        return hr

    def test_no_harvest_emits_no_replay_cell_local(self):
        code = "\n".join(c.source for c in self._make_stage().generate_local_cells())
        assert "from user_extensions import" not in code
        assert "User-Extension Replay" not in code

    def test_no_harvest_emits_no_replay_cell_databricks(self):
        code = "\n".join(c.source for c in self._make_stage().generate_databricks_cells())
        assert "from user_extensions import" not in code

    def test_replay_function_adds_import_cell_local(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        code = "\n".join(c.source for c in stage.generate_local_cells())
        assert "from user_extensions import enrich_req" in code
        assert "TODO wire replay call-site for enrich_req" in code

    def test_replay_function_adds_import_cell_databricks(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        code = "\n".join(c.source for c in stage.generate_databricks_cells())
        assert "from user_extensions import enrich_req" in code

    def test_replay_cell_inserted_before_run_batch_inference(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        sources = [c.source for c in stage.generate_local_cells()]
        replay_idx = next(i for i, s in enumerate(sources) if "enrich_req" in s)
        run_idx = next(i for i, s in enumerate(sources) if "run_batch_inference(config)" in s)
        assert replay_idx < run_idx
