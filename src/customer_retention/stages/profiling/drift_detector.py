import json
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from customer_retention.core.compat import pd, Series, DataFrame
from customer_retention.core.config import ColumnType
from customer_retention.core.utils.statistics import compute_psi_numeric, compute_psi_categorical, compute_ks_statistic, compute_chi_square
from .profile_result import ProfileResult


@dataclass
class DriftResult:
    """Result of drift detection for a single column."""
    column_name: str
    has_drift: bool
    severity: str  # "low", "medium", "high", "critical"
    metrics: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class BaselineDriftChecker:
    """Detects distribution drift between baseline and current data."""

    def __init__(self):
        self.baseline: Optional[Dict[str, Dict]] = None

    def set_baseline(self, column_name: str, series: pd.Series, column_type: ColumnType):
        """Set baseline distribution for a column."""
        if self.baseline is None:
            self.baseline = {}

        baseline_data = {
            "column_type": column_type.value,
            "sample_size": len(series),
        }

        if column_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            baseline_data.update(self._capture_numeric_baseline(series))
        elif column_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL,
                            ColumnType.CATEGORICAL_CYCLICAL, ColumnType.BINARY]:
            baseline_data.update(self._capture_categorical_baseline(series))

        self.baseline[column_name] = baseline_data

    def _capture_numeric_baseline(self, series: pd.Series) -> Dict:
        """Capture baseline statistics for numeric column."""
        clean_series = series.dropna()
        return {
            "mean": float(clean_series.mean()),
            "std": float(clean_series.std()),
            "median": float(clean_series.median()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "q1": float(clean_series.quantile(0.25)),
            "q3": float(clean_series.quantile(0.75)),
            # Store histogram for PSI calculation
            "histogram_bins": 10,
            "histogram_edges": [float(x) for x in np.histogram(clean_series, bins=10)[1]],
            "histogram_counts": [int(x) for x in np.histogram(clean_series, bins=10)[0]],
        }

    def _capture_categorical_baseline(self, series: pd.Series) -> Dict:
        """Capture baseline distribution for categorical column."""
        clean_series = series.dropna()
        value_counts = clean_series.value_counts()
        return {
            "categories": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "proportions": (value_counts / len(clean_series)).to_dict(),
        }

    def detect_drift(self, column_name: str, series: pd.Series, column_type: ColumnType) -> DriftResult:
        """Detect drift for a single column."""
        if self.baseline is None or column_name not in self.baseline:
            raise ValueError(f"No baseline found for column '{column_name}'")

        baseline = self.baseline[column_name]

        if column_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            return self._detect_numeric_drift(column_name, series, baseline)
        elif column_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL,
                            ColumnType.CATEGORICAL_CYCLICAL, ColumnType.BINARY]:
            return self._detect_categorical_drift(column_name, series, baseline)
        else:
            # Default: no drift detection for other types
            return DriftResult(
                column_name=column_name,
                has_drift=False,
                severity="low",
                metrics={},
                recommendations=[]
            )

    def _detect_numeric_drift(self, column_name: str, series: pd.Series, baseline: Dict) -> DriftResult:
        """Detect drift in numeric column."""
        clean_series = series.dropna()
        metrics = {}
        recommendations = []
        baseline_sample = self._reconstruct_numeric_baseline_sample(baseline)
        ks_statistic, ks_pvalue = compute_ks_statistic(pd.Series(baseline_sample), clean_series)
        metrics["ks_statistic"] = ks_statistic
        metrics["ks_pvalue"] = ks_pvalue
        psi = compute_psi_numeric(clean_series, baseline["histogram_edges"], baseline["histogram_counts"])
        metrics["psi"] = psi

        # Mean shift (normalized by baseline std)
        current_mean = clean_series.mean()
        mean_shift = (current_mean - baseline["mean"]) / baseline["std"] if baseline["std"] > 0 else 0
        metrics["mean_shift"] = float(mean_shift)

        # Variance ratio
        current_std = clean_series.std()
        variance_ratio = current_std / baseline["std"] if baseline["std"] > 0 else 1.0
        metrics["variance_ratio"] = float(variance_ratio)

        # Determine drift severity based on PSI thresholds
        if psi >= 0.5:
            severity = "critical"
            has_drift = True
            recommendations.append("Critical drift detected. Model performance likely degraded significantly.")
            recommendations.append("Consider retraining model with recent data.")
        elif psi >= 0.2:
            severity = "high"
            has_drift = True
            recommendations.append("Significant drift detected. Investigate data source changes.")
            recommendations.append("Monitor model performance closely.")
        elif psi >= 0.1:
            severity = "medium"
            has_drift = True
            recommendations.append("Moderate drift detected. Continue monitoring.")
        else:
            severity = "low"
            has_drift = False

        # Additional checks for mean shift and variance
        if abs(mean_shift) > 2:
            has_drift = True
            if severity == "low":
                severity = "medium"
            recommendations.append(f"Mean shifted by {mean_shift:.2f} standard deviations.")

        if variance_ratio > 2 or variance_ratio < 0.5:
            has_drift = True
            if severity == "low":
                severity = "medium"
            recommendations.append(f"Variance changed significantly (ratio: {variance_ratio:.2f}).")

        return DriftResult(
            column_name=column_name,
            has_drift=has_drift,
            severity=severity,
            metrics=metrics,
            recommendations=recommendations
        )

    def _detect_categorical_drift(self, column_name: str, series: pd.Series, baseline: Dict) -> DriftResult:
        """Detect drift in categorical column."""
        clean_series = series.dropna()
        metrics = {}
        recommendations = []

        # Get current distribution
        current_counts = clean_series.value_counts()
        current_categories = set(current_counts.index.tolist())
        baseline_categories = set(baseline["categories"])

        # New and missing categories
        new_categories = current_categories - baseline_categories
        missing_categories = baseline_categories - current_categories

        metrics["new_categories"] = list(new_categories)
        metrics["missing_categories"] = list(missing_categories)

        psi = compute_psi_categorical(pd.Series(baseline["categories"]).repeat([baseline["counts"][i] for i in range(len(baseline["categories"]))]), clean_series)
        metrics["psi"] = psi
        chi_square_stat, chi_pvalue = compute_chi_square(clean_series, baseline["proportions"])
        metrics["chi_square_statistic"] = chi_square_stat
        metrics["chi_square_pvalue"] = chi_pvalue

        # Determine severity
        if psi >= 0.5:
            severity = "critical"
            has_drift = True
            recommendations.append("Critical distribution shift detected.")
        elif psi >= 0.2:
            severity = "high"
            has_drift = True
            recommendations.append("Significant distribution change detected.")
        elif psi >= 0.1:
            severity = "medium"
            has_drift = True
            recommendations.append("Moderate distribution change detected.")
        else:
            severity = "low"
            has_drift = bool(chi_pvalue < 0.05)  # Convert numpy bool to Python bool

        # Check for new/missing categories
        if new_categories:
            has_drift = True
            if severity == "low":
                severity = "medium"
            recommendations.append(f"New categories detected: {', '.join(new_categories)}")

        if missing_categories:
            has_drift = True
            if severity == "low":
                severity = "medium"
            recommendations.append(f"Missing categories: {', '.join(missing_categories)}")

        return DriftResult(
            column_name=column_name,
            has_drift=has_drift,
            severity=severity,
            metrics=metrics,
            recommendations=recommendations
        )

    def _reconstruct_numeric_baseline_sample(self, baseline: Dict) -> np.ndarray:
        """Reconstruct a sample from baseline histogram for KS test."""
        edges = baseline["histogram_edges"]
        counts = baseline["histogram_counts"]

        # Generate samples from each bin
        samples = []
        for i, count in enumerate(counts):
            if count > 0:
                # Sample uniformly within each bin
                bin_samples = np.random.uniform(edges[i], edges[i + 1], count)
                samples.extend(bin_samples)

        return np.array(samples)

    def detect_drift_all(self, df: pd.DataFrame) -> List[DriftResult]:
        """Detect drift for all columns with baseline."""
        if self.baseline is None:
            raise ValueError("No baseline set. Call set_baseline first.")

        results = []
        for column_name in self.baseline.keys():
            if column_name in df.columns:
                column_type = ColumnType(self.baseline[column_name]["column_type"])
                result = self.detect_drift(column_name, df[column_name], column_type)
                results.append(result)

        return results

    def set_baseline_from_profile(self, profile: ProfileResult):
        """Set baseline from a ProfileResult."""
        self.baseline = {}

        for column_name, column_profile in profile.column_profiles.items():
            # Create a mock series for baseline (we'll use the metrics instead)
            baseline_data = {
                "column_type": column_profile.configured_type.value,
                "sample_size": profile.total_rows,
            }

            if column_profile.numeric_metrics:
                metrics = column_profile.numeric_metrics
                baseline_data.update({
                    "mean": metrics.mean,
                    "std": metrics.std,
                    "median": metrics.median,
                    "min": metrics.min_value,
                    "max": metrics.max_value,
                    "q1": metrics.q1,
                    "q3": metrics.q3,
                    "histogram_bins": 10,
                    "histogram_edges": metrics.histogram_edges if hasattr(metrics, 'histogram_edges') else [],
                    "histogram_counts": metrics.histogram_counts if hasattr(metrics, 'histogram_counts') else [],
                })

            elif column_profile.categorical_metrics:
                metrics = column_profile.categorical_metrics
                categories = list(metrics.value_counts.keys()) if metrics.value_counts else []
                counts = list(metrics.value_counts.values()) if metrics.value_counts else []
                total = sum(counts) if counts else 1

                baseline_data.update({
                    "categories": categories,
                    "counts": counts,
                    "proportions": {cat: count / total for cat, count in zip(categories, counts)},
                })

            self.baseline[column_name] = baseline_data

    def save_baseline(self, filepath: str):
        """Save baseline to JSON file."""
        if self.baseline is None:
            raise ValueError("No baseline to save")

        with open(filepath, 'w') as f:
            json.dump(self.baseline, f, indent=2)

    def load_baseline(self, filepath: str):
        """Load baseline from JSON file."""
        with open(filepath, 'r') as f:
            self.baseline = json.load(f)
