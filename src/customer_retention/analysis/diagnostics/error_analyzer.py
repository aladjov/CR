"""Prediction error analysis probes."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from customer_retention.core.compat import DataFrame, Series


@dataclass
class ErrorPattern:
    error_type: str
    feature: str
    pattern: str
    count: int


@dataclass
class ErrorAnalysisResult:
    total_errors: int
    error_rate: float
    fp_count: int
    fn_count: int
    false_positives: DataFrame
    false_negatives: DataFrame
    high_confidence_fp: DataFrame
    high_confidence_fn: DataFrame
    fp_confidence_dist: Dict[str, int] = field(default_factory=dict)
    fn_confidence_dist: Dict[str, int] = field(default_factory=dict)
    error_patterns: List[ErrorPattern] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)


class ErrorAnalyzer:
    HIGH_CONFIDENCE_FP_THRESHOLD = 0.8
    HIGH_CONFIDENCE_FN_THRESHOLD = 0.2

    def analyze_errors(self, model, X: DataFrame, y: Series, threshold: float = 0.5) -> ErrorAnalysisResult:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
        fp_mask = (y_pred == 1) & (y == 0)
        fn_mask = (y_pred == 0) & (y == 1)
        false_positives = X[fp_mask].copy()
        false_negatives = X[fn_mask].copy()
        false_positives["probability"] = y_proba[fp_mask]
        false_negatives["probability"] = y_proba[fn_mask]
        high_conf_fp = false_positives[false_positives["probability"] > self.HIGH_CONFIDENCE_FP_THRESHOLD]
        high_conf_fn = false_negatives[false_negatives["probability"] < self.HIGH_CONFIDENCE_FN_THRESHOLD]
        fp_confidence_dist = self._compute_confidence_dist(false_positives["probability"].values if len(false_positives) > 0 else np.array([]))
        fn_confidence_dist = self._compute_confidence_dist(false_negatives["probability"].values if len(false_negatives) > 0 else np.array([]))
        error_patterns = self._find_patterns(X, y, y_pred, fp_mask, fn_mask)
        hypotheses = self._generate_hypotheses(false_positives, false_negatives, high_conf_fp, high_conf_fn)
        total_errors = fp_mask.sum() + fn_mask.sum()
        error_rate = total_errors / len(y) if len(y) > 0 else 0.0
        return ErrorAnalysisResult(
            total_errors=total_errors,
            error_rate=error_rate,
            fp_count=fp_mask.sum(),
            fn_count=fn_mask.sum(),
            false_positives=false_positives,
            false_negatives=false_negatives,
            high_confidence_fp=high_conf_fp,
            high_confidence_fn=high_conf_fn,
            fp_confidence_dist=fp_confidence_dist,
            fn_confidence_dist=fn_confidence_dist,
            error_patterns=error_patterns,
            hypotheses=hypotheses,
        )

    def _compute_confidence_dist(self, proba: np.ndarray) -> Dict[str, int]:
        if len(proba) == 0:
            return {"low": 0, "medium": 0, "high": 0}
        return {
            "low": int((proba < 0.4).sum()),
            "medium": int(((proba >= 0.4) & (proba < 0.7)).sum()),
            "high": int((proba >= 0.7).sum()),
        }

    def _find_patterns(self, X: DataFrame, y: Series, y_pred, fp_mask, fn_mask) -> List[ErrorPattern]:
        patterns = []
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                fp_mean = X.loc[fp_mask, col].mean() if fp_mask.sum() > 0 else 0
                correct_mean = X.loc[~fp_mask & ~fn_mask, col].mean()
                if abs(fp_mean - correct_mean) > X[col].std() * 0.5:
                    patterns.append(ErrorPattern(
                        error_type="FP",
                        feature=col,
                        pattern=f"FPs have {'higher' if fp_mean > correct_mean else 'lower'} {col}",
                        count=fp_mask.sum(),
                    ))
        return patterns

    def _generate_hypotheses(self, fps, fns, high_fp, high_fn) -> List[str]:
        hypotheses = []
        if len(high_fp) > 0:
            hypotheses.append(f"Model is overconfident on {len(high_fp)} false positives. Review these cases.")
        if len(high_fn) > 0:
            hypotheses.append(f"Model is overconfident on {len(high_fn)} false negatives. These are high-risk misses.")
        if len(fps) > len(fns) * 2:
            hypotheses.append("Model is biased toward positive predictions. Consider raising threshold.")
        if len(fns) > len(fps) * 2:
            hypotheses.append("Model is biased toward negative predictions. Consider lowering threshold.")
        if not hypotheses:
            hypotheses.append("Error distribution appears balanced. Focus on feature engineering to reduce errors.")
        return hypotheses
