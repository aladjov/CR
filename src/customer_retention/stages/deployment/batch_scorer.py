from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import time

from customer_retention.core.compat import pd, DataFrame
from customer_retention.core.components.enums import RiskSegment


@dataclass
class ScoringConfig:
    model_name: str
    model_stage: str = "Production"
    feature_table: str = "customer_features"
    output_table: str = "churn_predictions"
    batch_size: int = 10000
    parallelism: int = 8


@dataclass
class ScoringResult:
    predictions: DataFrame
    total_scored: int
    scoring_duration_seconds: float
    model_version: Optional[str] = None
    feature_table_version: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class BatchScorer:
    def __init__(self, model: Any, scaler: Any = None, threshold: float = 0.5,
                 model_version: Optional[str] = None, batch_size: int = 10000,
                 handle_nulls: str = "raise"):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.model_version = model_version
        self.batch_size = batch_size
        self.handle_nulls = handle_nulls
        self._segment_thresholds = {
            RiskSegment.CRITICAL: 0.75,
            RiskSegment.HIGH: 0.50,
            RiskSegment.MEDIUM: 0.25,
            RiskSegment.LOW: 0.0
        }

    def score(self, data: DataFrame, feature_columns: List[str],
              id_column: str) -> ScoringResult:
        start_time = time.time()
        errors = []
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        features = data[feature_columns].copy()
        if features.isnull().any().any():
            if self.handle_nulls == "raise":
                raise ValueError("Null values found in features")
            elif self.handle_nulls == "fill_zero":
                features = features.fillna(0)
            elif self.handle_nulls == "fill_mean":
                features = features.fillna(features.mean())
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values
        all_predictions = []
        n_batches = (len(data) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(data))
            batch_features = features_scaled[start_idx:end_idx]
            batch_ids = data[id_column].iloc[start_idx:end_idx]
            try:
                probabilities = self.model.predict_proba(batch_features)[:, 1]
            except Exception as e:
                errors.append(f"Batch {batch_idx} error: {str(e)}")
                continue
            batch_df = pd.DataFrame({
                "customer_id": batch_ids.values,
                "churn_probability": probabilities,
                "risk_segment": [self._assign_risk_segment(p) for p in probabilities],
                "predicted_churn": (probabilities >= self.threshold).astype(int),
                "score_timestamp": datetime.now()
            })
            if self.model_version:
                batch_df["model_version"] = self.model_version
            all_predictions.append(batch_df)
        predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        duration = time.time() - start_time
        return ScoringResult(
            predictions=predictions_df,
            total_scored=len(predictions_df),
            scoring_duration_seconds=duration,
            model_version=self.model_version,
            errors=errors
        )

    def _assign_risk_segment(self, probability: float) -> str:
        if probability >= self._segment_thresholds[RiskSegment.CRITICAL]:
            return RiskSegment.CRITICAL.value
        elif probability >= self._segment_thresholds[RiskSegment.HIGH]:
            return RiskSegment.HIGH.value
        elif probability >= self._segment_thresholds[RiskSegment.MEDIUM]:
            return RiskSegment.MEDIUM.value
        else:
            return RiskSegment.LOW.value
