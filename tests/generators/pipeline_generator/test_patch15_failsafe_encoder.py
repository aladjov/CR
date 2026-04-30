"""Tests for the operator-side patches 15 + 16 (fail-safe gold encoder +
Spark-SQL-only label encoder) regexes and for the gold-encoding-targets
probe regex.

All three live in NB10 as user_code cells (see docs/sps_nb10_runtime_patches.md
sections 15, 16, and probe-15). The cells use only stdlib + pathlib, so the
critical correctness contract is the regex behaviour. These tests exercise
that regex against the actual rendered gold and bronze code paths.
"""
from __future__ import annotations

import re
from textwrap import dedent

# ---------------------------------------------------------------------------
# The regex strings are duplicated here from the NB10 patch cell body so
# they are testable on CI. Any change to the cell regex must be mirrored
# here, and vice versa.
# ---------------------------------------------------------------------------

PATCH15_PATTERN = re.compile(
    r'(    if col not in df\.columns:\n)'
    r'        raise RuntimeError\(\n'
    r'.*?'
    r'        \)\n',
    re.S,
)
PATCH15_REPLACEMENT = (
    r'\1'
    '        # [patch 15] fail-safe encoder rewrite\n'
    '        print(f"[patch15] gold encoder skipping {col!r} - not in df.columns; will be absent from gold")\n'
    '        return df\n'
)
PATCH15_MARKER = "[patch 15] fail-safe encoder rewrite"


PATCH16_PATTERN = re.compile(
    r'    from pyspark\.ml\.feature import StringIndexer\n'
    r'    tmp = f"__\{col\}_idx"\n'
    r'    indexer = StringIndexer\(inputCol=col, outputCol=tmp, handleInvalid="keep", stringOrderType="alphabetAsc"\)\n'
    r'    df = indexer\.fit\(df\)\.transform\(df\)\.drop\(col\)\.withColumnRenamed\(tmp, col\)\n'
    r'    return df\n'
)
PATCH16_REPLACEMENT = (
    '    # [patch 16] Spark-SQL label encoding (no pyspark.ml StringIndexer)\n'
    '    _vals = sorted({\n'
    '        row[0] for row in\n'
    '        df.select(col).where(F.col(col).isNotNull()).distinct().collect()\n'
    '    })\n'
    '    if not _vals:\n'
    '        print(f"[patch16] gold encoder: {col!r} has no non-null values; dropping")\n'
    '        return df.drop(col)\n'
    '    _n = len(_vals)\n'
    '    _col_dtype = df.schema[col].dataType.simpleString()\n'
    '    _lookup = df.sparkSession.createDataFrame(\n'
    '        [(_v, _i) for _i, _v in enumerate(_vals)],\n'
    '        schema=f"{col} {_col_dtype}, __idx int",\n'
    '    )\n'
    '    _tmp = f"__{col}_idx"\n'
    '    df = (\n'
    '        df.join(F.broadcast(_lookup), on=col, how="left")\n'
    '          .withColumn(_tmp, F.coalesce(F.col("__idx"), F.lit(_n)).cast("int"))\n'
    '          .drop("__idx")\n'
    '          .drop(col)\n'
    '          .withColumnRenamed(_tmp, col)\n'
    '    )\n'
    '    return df\n'
)
PATCH16_MARKER = "[patch 16] Spark-SQL label encoding"


# Probe regexes: extract gold encoding targets and bronze/silver column
# creators so the operator can diff them before the 6h cycle.
PROBE_ONE_HOT_PATTERN = re.compile(r'_encode_one_hot\(df,\s*"([^"]+)"')
PROBE_LABEL_ENCODE_PATTERN = re.compile(r'_label_encode\(df,\s*"([^"]+)"')
PROBE_WITH_COLUMN_PATTERN = re.compile(r'\.withColumn\(\s*"([^"]+)"')
PROBE_LIFECYCLE_FUNC_DEFINED = re.compile(r'^def add_lifecycle_quadrant\(df\):', re.M)
PROBE_RECENCY_FUNC_DEFINED = re.compile(r'^def add_recency_tenure\(', re.M)
PROBE_TEMPORAL_FEATURES_DEFINED = re.compile(r'^def compute_temporal_features\(', re.M)


# ---------------------------------------------------------------------------
# Patch 15 regex tests
# ---------------------------------------------------------------------------

class TestPatch15RegexRewrite:
    def test_rewrites_one_hot_raise_block(self):
        src = dedent('''\
            def _encode_one_hot(df, col, max_categories=100):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] one-hot encoding target '{col}' missing from DataFrame. "
                        "An upstream step (feature_selection, leakage exclusion, or silver merge) "
                        "dropped it before apply_encodings ran. Regenerate the pipeline — the "
                        "post-encoding '{col}_*' columns will otherwise be silently absent from gold."
                    )
                categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
            ''')
        new_src, n = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        assert n == 1, f"expected exactly one rewrite, got {n}"
        assert "raise RuntimeError" not in new_src
        assert PATCH15_MARKER in new_src
        assert "print(f\"[patch15] gold encoder skipping" in new_src
        assert "return df" in new_src
        assert "categories = [row[col] for row in df.select(col).distinct().collect()" in new_src

    def test_rewrites_label_encode_raise_block(self):
        src = dedent('''\
            def _label_encode(df, col):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] label encoding target '{col}' missing from DataFrame. "
                        "An upstream step dropped it before apply_encodings ran."
                    )
                from pyspark.ml.feature import StringIndexer
            ''')
        new_src, n = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        assert n == 1
        assert "raise RuntimeError" not in new_src
        assert PATCH15_MARKER in new_src
        assert "from pyspark.ml.feature import StringIndexer" in new_src

    def test_rewrites_both_in_same_file(self):
        """The full rendered gold script has both _encode_one_hot and
        _label_encode definitions; one regex pass must rewrite both."""
        src = dedent('''\
            def _encode_one_hot(df, col, max_categories=100):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] one-hot encoding target '{col}' missing."
                    )
                categories = []

            def _label_encode(df, col):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] label encoding target '{col}' missing."
                    )
                from pyspark.ml.feature import StringIndexer
            ''')
        new_src, n = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        assert n == 2
        assert new_src.count("raise RuntimeError") == 0
        assert new_src.count(PATCH15_MARKER) == 2

    def test_idempotent_via_marker(self):
        """Running the patch twice on the same source must produce the
        same file the second time — the marker is the operator's gate."""
        src = dedent('''\
            def _encode_one_hot(df, col, max_categories=100):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] one-hot encoding target '{col}' missing."
                    )
                categories = []
            ''')
        once, n1 = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        twice, n2 = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, once)
        assert n1 == 1
        assert n2 == 0
        assert once == twice

    def test_does_not_match_unrelated_raise(self):
        """The patch must not affect raise statements outside the encoder
        column-existence guard."""
        src = dedent('''\
            def some_other_func(df, col):
                if not df.count():
                    raise RuntimeError("empty df")
                return df
            ''')
        new_src, n = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        assert n == 0
        assert new_src == src

    def test_rewrite_compiles_as_python(self):
        """After the patch the rendered file must still parse."""
        src = dedent('''\
            def _encode_one_hot(df, col, max_categories=100):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] one-hot encoding target '{col}' missing from DataFrame. "
                        "An upstream step dropped it before apply_encodings ran."
                    )
                categories = []
                return df

            def _label_encode(df, col):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] label encoding target '{col}' missing."
                    )
                return df
            ''')
        new_src, _ = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        compile(new_src, "<patched>", "exec")  # raises SyntaxError on failure


# ---------------------------------------------------------------------------
# Patch 15 against actual rendered gold output
# ---------------------------------------------------------------------------

class TestPatch15AgainstRenderedGold:
    def test_rewrites_real_rendered_gold(self):
        """Render gold via DatabricksCodeRenderer and confirm the patch's
        regex matches both encoder definitions in the actual output."""
        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )

        encoding = TransformationStep(
            type=PipelineTransformationType.ENCODE,
            column="lifecycle_quadrant",
            parameters={"method": "one_hot"},
            rationale="encode lifecycle quadrant",
        )
        cfg = PipelineConfig(
            name="test",
            composite_name="test_xxxxxxx",
            target_column="target",
            sources=[SourceConfig(name="src", path="src.csv", format="csv", entity_key="id")],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(encodings=[encoding]),
            output_dir="/tmp/test",
        )
        renderer = DatabricksCodeRenderer(catalog="cat", schema="sch")
        rendered = renderer.render_gold(cfg)

        # Both encoder definitions must be present in the rendered file
        assert "def _encode_one_hot(df, col, max_categories=100):" in rendered
        assert "def _label_encode(df, col):" in rendered
        # Both must contain the raise-block the patch targets
        assert rendered.count("raise RuntimeError") == 2

        # Apply the patch
        patched, n = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, rendered)
        assert n == 2, f"expected 2 rewrites against real rendered gold, got {n}"
        assert "raise RuntimeError" not in patched
        assert patched.count(PATCH15_MARKER) == 2

        # Patched output still compiles
        compile(patched, "<rendered_patched>", "exec")


# ---------------------------------------------------------------------------
# Probe regex tests
# ---------------------------------------------------------------------------

class TestProbeRegexes:
    def test_extracts_one_hot_targets(self):
        src = dedent('''\
            def apply_encodings(df):
                # encode lifecycle quadrant
                df = _encode_one_hot(df, "lifecycle_quadrant")
                # encode region
                df = _encode_one_hot(df, "region_code")
                return df
            ''')
        targets = set(PROBE_ONE_HOT_PATTERN.findall(src))
        assert targets == {"lifecycle_quadrant", "region_code"}

    def test_extracts_label_encode_targets(self):
        src = dedent('''\
            def apply_encodings(df):
                df = _label_encode(df, "high_card_col")
                return df
            ''')
        targets = set(PROBE_LABEL_ENCODE_PATTERN.findall(src))
        assert targets == {"high_card_col"}

    def test_extracts_with_column_creators(self):
        src = dedent('''\
            def add_lifecycle_quadrant(df):
                df = df.withColumn("lifecycle_quadrant", F.when(...).otherwise(...))
                df = df.withColumn(
                    "another_col", F.lit(1)
                )
                return df
            ''')
        cols = set(PROBE_WITH_COLUMN_PATTERN.findall(src))
        assert "lifecycle_quadrant" in cols
        assert "another_col" in cols

    def test_lifecycle_prereq_detection(self):
        """Probe must detect when add_lifecycle_quadrant is defined but
        prereqs (add_recency_tenure, compute_temporal_features) are not."""
        src_entity_only_source = dedent('''\
            def add_lifecycle_quadrant(df):
                if "days_since_first" not in df.columns:
                    return df
                return df.withColumn("lifecycle_quadrant", F.lit("x"))
            ''')
        assert PROBE_LIFECYCLE_FUNC_DEFINED.search(src_entity_only_source) is not None
        assert PROBE_RECENCY_FUNC_DEFINED.search(src_entity_only_source) is None
        assert PROBE_TEMPORAL_FEATURES_DEFINED.search(src_entity_only_source) is None

        src_event_source_full = dedent('''\
            def compute_temporal_features(df, raw_df):
                return df

            def add_recency_tenure(df, raw_df):
                return df

            def add_lifecycle_quadrant(df):
                return df.withColumn("lifecycle_quadrant", F.lit("x"))
            ''')
        assert PROBE_LIFECYCLE_FUNC_DEFINED.search(src_event_source_full) is not None
        assert PROBE_RECENCY_FUNC_DEFINED.search(src_event_source_full) is not None
        assert PROBE_TEMPORAL_FEATURES_DEFINED.search(src_event_source_full) is not None

    def test_probe_diffs_targets_against_creators(self):
        """End-to-end: gold has 'X' and 'Y' encoding targets, bronze creates
        'X' but not 'Y' — probe must flag 'Y' as missing."""
        gold_src = dedent('''\
            df = _encode_one_hot(df, "X")
            df = _encode_one_hot(df, "Y")
            ''')
        bronze_src = dedent('''\
            df = df.withColumn("X", F.lit(1))
            df = df.withColumn("Z", F.lit(2))
            ''')

        targets = set(PROBE_ONE_HOT_PATTERN.findall(gold_src)) | set(
            PROBE_LABEL_ENCODE_PATTERN.findall(gold_src)
        )
        creators = set(PROBE_WITH_COLUMN_PATTERN.findall(bronze_src))
        missing = targets - creators

        assert missing == {"Y"}


class TestPatch16RegexRewrite:
    """Patch 16 replaces the StringIndexer body of _label_encode with a
    Spark-SQL broadcast-join body. Independent of patch 15 (different
    region of the same function)."""

    def _label_encode_block(self) -> str:
        return dedent('''\
            def _label_encode(df, col):
                if col not in df.columns:
                    raise RuntimeError(
                        f"[GOLD] label encoding target '{col}' missing."
                    )
                from pyspark.ml.feature import StringIndexer
                tmp = f"__{col}_idx"
                indexer = StringIndexer(inputCol=col, outputCol=tmp, handleInvalid="keep", stringOrderType="alphabetAsc")
                df = indexer.fit(df).transform(df).drop(col).withColumnRenamed(tmp, col)
                return df
            ''')

    def test_rewrites_label_encode_body(self):
        src = self._label_encode_block()
        new_src, n = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        assert n == 1, f"expected exactly one rewrite, got {n}"
        assert "from pyspark.ml.feature import StringIndexer" not in new_src
        assert "indexer.fit" not in new_src
        assert "indexer = StringIndexer(" not in new_src
        assert PATCH16_MARKER in new_src
        assert "F.broadcast(_lookup)" in new_src

    def test_idempotent_via_marker(self):
        src = self._label_encode_block()
        once, n1 = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        twice, n2 = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, once)
        assert n1 == 1
        assert n2 == 0
        assert once == twice

    def test_does_not_match_one_hot_body(self):
        """_encode_one_hot's body has 'categories = [...].collect()'; patch 16
        must not rewrite it."""
        src = dedent('''\
            def _encode_one_hot(df, col, max_categories=100):
                if col not in df.columns:
                    raise RuntimeError("missing")
                categories = [row[col] for row in df.select(col).distinct().collect() if row[col] is not None]
                for cat in sorted(categories):
                    df = df.withColumn(f"{col}_{cat}", F.when(F.col(col) == cat, 1).otherwise(0))
                return df
            ''')
        _, n = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        assert n == 0

    def test_does_not_match_unrelated_stringindexer(self):
        """Patch 16's regex anchors on the exact 4-line shape inside
        _label_encode. Unrelated StringIndexer use in another function must
        not trigger a rewrite."""
        src = dedent('''\
            def some_other_func(df):
                from pyspark.ml.feature import StringIndexer
                idx = StringIndexer(inputCol="x", outputCol="y")
                return idx.fit(df).transform(df)
            ''')
        _, n = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        assert n == 0

    def test_rewrite_compiles_as_python(self):
        src = "from pyspark.sql import functions as F\n\n" + self._label_encode_block()
        new_src, n = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        assert n == 1
        compile(new_src, "<patch16>", "exec")

    def test_independent_of_patch_15(self):
        """Patches 15 and 16 mutate disjoint regions. Applying in either
        order must yield the same final source."""
        src = self._label_encode_block()

        order_15_then_16, _ = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, src)
        order_15_then_16, _ = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, order_15_then_16)

        order_16_then_15, _ = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, src)
        order_16_then_15, _ = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, order_16_then_15)

        assert order_15_then_16 == order_16_then_15
        assert PATCH15_MARKER in order_15_then_16
        assert PATCH16_MARKER in order_15_then_16
        assert "raise RuntimeError" not in order_15_then_16
        assert "from pyspark.ml.feature import StringIndexer" not in order_15_then_16


class TestPatch16AgainstRenderedGold:
    def test_rewrites_real_rendered_gold(self):
        """Render gold via DatabricksCodeRenderer and confirm patch 16's
        regex matches the _label_encode body in the actual output. The
        StringIndexer body appears exactly once per gold file."""
        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )

        encoding = TransformationStep(
            type=PipelineTransformationType.ENCODE,
            column="lifecycle_quadrant",
            parameters={"method": "one_hot"},
            rationale="encode lifecycle quadrant",
        )
        cfg = PipelineConfig(
            name="test",
            composite_name="test_xxxxxxx",
            target_column="target",
            sources=[SourceConfig(name="src", path="src.csv", format="csv", entity_key="id")],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(encodings=[encoding]),
            output_dir="/tmp/test",
        )
        renderer = DatabricksCodeRenderer(catalog="cat", schema="sch")
        rendered = renderer.render_gold(cfg)

        assert "def _label_encode(df, col):" in rendered
        assert "from pyspark.ml.feature import StringIndexer" in rendered
        assert "indexer.fit(df).transform(df)" in rendered

        patched, n = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, rendered)
        assert n == 1, f"expected 1 rewrite against real rendered gold, got {n}"
        assert "from pyspark.ml.feature import StringIndexer" not in patched
        assert "indexer.fit" not in patched
        assert PATCH16_MARKER in patched

        compile(patched, "<rendered_patched_16>", "exec")

    def test_15_and_16_applied_together_against_rendered_gold(self):
        """Operator workflow applies patch 15 (encoder guards) and patch 16
        (label-encode body) in sequence. Both must complete and the result
        must compile."""
        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )

        encoding = TransformationStep(
            type=PipelineTransformationType.ENCODE,
            column="lifecycle_quadrant",
            parameters={"method": "one_hot"},
            rationale="encode lifecycle quadrant",
        )
        cfg = PipelineConfig(
            name="test",
            composite_name="test_xxxxxxx",
            target_column="target",
            sources=[SourceConfig(name="src", path="src.csv", format="csv", entity_key="id")],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(encodings=[encoding]),
            output_dir="/tmp/test",
        )
        renderer = DatabricksCodeRenderer(catalog="cat", schema="sch")
        rendered = renderer.render_gold(cfg)

        patched, n15 = PATCH15_PATTERN.subn(PATCH15_REPLACEMENT, rendered)
        patched, n16 = PATCH16_PATTERN.subn(PATCH16_REPLACEMENT, patched)

        assert n15 == 2
        assert n16 == 1
        assert patched.count(PATCH15_MARKER) == 2
        assert patched.count(PATCH16_MARKER) == 1
        assert "raise RuntimeError" not in patched
        assert "from pyspark.ml.feature import StringIndexer" not in patched

        compile(patched, "<rendered_patched_15_and_16>", "exec")
