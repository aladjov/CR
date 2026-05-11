"""Closed enum of operation categories audited for exploration/production parity.

A function decorated with `@apply_op(kind=ApplyOpKind.X)` is recorded in the
manifest under that kind. Adding a new value here is a deliberate framework
act, paired with the corresponding decoration somewhere in the codebase.

Layout: dotted values are `<stage>.<operation>` so the audit can group
manifest entries by stage without a separate mapping table.
"""
from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional

_EXPECTED_STAGE_TO_KIND = {
    "target_derive": "TARGET_DERIVE",
    "landing_post": "LIFECYCLE_ENRICH",
    "landing": "LIFECYCLE_ENRICH",
    "silver": "SILVER_DERIVED_FEATURE",
    "silver_post": "SILVER_DERIVED_FEATURE",
    "bronze": "BRONZE_AGGREGATE",
    "gold": "GOLD_TRANSFORMATION",
}


class ApplyOpKind(str, Enum):
    LIFECYCLE_ENRICH = "landing.lifecycle_enrich"
    SAMPLE_FILTER = "landing.sample_filter"
    LANDING_FILTER = "landing.filter"
    KEY_RESOLUTION = "landing.key_resolution"
    FEATURE_TIMESTAMP_DERIVE = "landing.feature_timestamp_derive"
    LABEL_TIMESTAMP_DERIVE = "landing.label_timestamp_derive"
    LABEL_AVAILABLE_FLAG = "landing.label_available_flag"
    DATETIME_DERIVE = "landing.datetime_derive"
    TEMPORAL_LOOKBACK = "landing.temporal_lookback"
    TIMESTAMP_NORMALIZE = "landing.timestamp_normalize"

    TARGET_DERIVE = "target_derive.user_code"

    BRONZE_AGGREGATE = "bronze.aggregate"
    BRONZE_VALUE_COUNTS = "bronze.value_counts"

    SILVER_TEMPORAL_MERGE = "silver.temporal_merge"
    SILVER_DERIVED_FEATURE = "silver.derived_feature"
    SILVER_HOLDOUT_MASK = "silver.holdout_mask"
    SILVER_TARGET_LABEL_MAP = "silver.target_label_map"

    GOLD_TRANSFORMATION = "gold.transformation"
    GOLD_ENCODING = "gold.encoding"
    GOLD_FEATURE_SPEC_GATE = "gold.feature_spec_gate"

    TRAINING_SPLIT = "training.split"
    TRAINING_FIT = "training.fit"
    TRAINING_EVALUATE = "training.evaluate"

    @property
    def stage(self) -> str:
        return self.value.split(".", 1)[0]

    @classmethod
    def kinds_for_stage(cls, stage: str) -> FrozenSet["ApplyOpKind"]:
        return frozenset(k for k in cls if k.stage == stage)

    @classmethod
    def from_expected_stage(cls, expected_stage: str) -> Optional["ApplyOpKind"]:
        name = _EXPECTED_STAGE_TO_KIND.get(expected_stage)
        return cls[name] if name else None
