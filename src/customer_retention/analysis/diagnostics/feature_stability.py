from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

_EPSILON = 1e-8


@dataclass
class FeatureStabilityResult:
    stable_features: List[str]
    volatile_features: List[str]
    importance_stats: Dict[str, Dict[str, float]]
    overall_stability: float


class FeatureStabilityAnalyzer:

    def analyze(self, fold_importances: List[Dict[str, float]],
                top_k: int = 20, majority_threshold: float = 0.6) -> FeatureStabilityResult:
        if not fold_importances:
            return FeatureStabilityResult([], [], {}, 0.0)

        all_features = set()
        for fi in fold_importances:
            all_features.update(fi.keys())

        values: Dict[str, list] = defaultdict(list)
        for fi in fold_importances:
            for feat in all_features:
                values[feat].append(fi.get(feat, 0.0))

        stats = {}
        for feat, vals in values.items():
            arr = np.array(vals)
            mean, std = float(arr.mean()), float(arr.std())
            stats[feat] = {"mean": mean, "std": std, "stability_score": mean / (std + _EPSILON)}

        n_folds = len(fold_importances)
        top_k_counts: Dict[str, int] = defaultdict(int)
        ever_top_k: set = set()
        for fi in fold_importances:
            sorted_feats = sorted(fi, key=fi.get, reverse=True)[:top_k]
            for feat in sorted_feats:
                top_k_counts[feat] += 1
                ever_top_k.add(feat)

        stable = sorted(f for f in ever_top_k if top_k_counts[f] >= majority_threshold * n_folds)
        volatile = sorted(f for f in ever_top_k if f not in set(stable))

        top_stability_scores = sorted(
            (stats[f]["stability_score"] for f in ever_top_k), reverse=True,
        )[:top_k]
        overall = float(np.mean(top_stability_scores)) if top_stability_scores else 0.0

        return FeatureStabilityResult(stable, volatile, stats, overall)
