from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from customer_retention.core.compat import DataFrame, Series, to_pandas, ensure_pandas_series
from .number_formatter import NumberFormatter

if TYPE_CHECKING:
    from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalysis
    from customer_retention.stages.profiling.segment_analyzer import SegmentationResult
    from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalysis


class ChartBuilder:
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#ffbb00",
            "danger": "#d62728",
            "info": "#17becf"
        }

    def bar_chart(self, x: List[Any], y: List[Any], title: Optional[str] = None,
                  x_label: Optional[str] = None, y_label: Optional[str] = None,
                  horizontal: bool = False, color: Optional[str] = None) -> go.Figure:
        """Create a simple bar chart."""
        marker_color = color or self.colors["primary"]
        if horizontal:
            fig = go.Figure(go.Bar(y=x, x=y, orientation="h", marker_color=marker_color))
        else:
            fig = go.Figure(go.Bar(x=x, y=y, marker_color=marker_color))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme
        )
        return fig

    def column_type_distribution(self, type_counts: Dict[str, int]) -> go.Figure:
        if not type_counts:
            return go.Figure()
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Type Distribution",
            hole=0.4
        )
        fig.update_layout(template=self.theme)
        return fig

    def data_quality_scorecard(self, quality_scores: Dict[str, float]) -> go.Figure:
        columns = list(quality_scores.keys())
        scores = list(quality_scores.values())
        colors = [self.colors["success"] if s > 80 else self.colors["warning"] if s > 60 else self.colors["danger"] for s in scores]
        fig = go.Figure(go.Bar(y=columns, x=scores, orientation="h", marker_color=colors))
        fig.update_layout(
            title="Data Quality Scores by Column",
            xaxis_title="Quality Score (0-100)",
            template=self.theme,
            height=max(400, len(columns) * 25)
        )
        return fig

    def missing_value_bars(self, null_percentages: Dict[str, float]) -> go.Figure:
        columns = list(null_percentages.keys())
        pcts = list(null_percentages.values())
        colors = [self.colors["danger"] if p > 20 else self.colors["warning"] if p > 5 else self.colors["success"] for p in pcts]
        fig = go.Figure(go.Bar(x=columns, y=pcts, marker_color=colors))
        fig.update_layout(title="Missing Values by Column", yaxis_title="Missing %", template=self.theme)
        return fig

    def histogram_with_stats(self, series: Series, title: Optional[str] = None) -> go.Figure:
        series = ensure_pandas_series(series)
        clean = series.dropna()
        mean_val = clean.mean()
        median_val = clean.median()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=clean, nbinsx=30, name="Distribution"))
        fig.add_vline(x=mean_val, line_dash="dash", line_color=self.colors["primary"], annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color=self.colors["secondary"], annotation_text=f"Median: {median_val:.2f}")
        fig.update_layout(
            title=title or f"Distribution of {series.name}",
            xaxis_title=series.name,
            yaxis_title="Count",
            template=self.theme
        )
        return fig

    def box_plot(self, series: Series, title: Optional[str] = None) -> go.Figure:
        series = ensure_pandas_series(series)
        fig = px.box(y=series.dropna(), title=title or f"Box Plot: {series.name}")
        fig.update_layout(template=self.theme)
        return fig

    def outlier_visualization(self, series: Series, method: str = "iqr") -> go.Figure:
        series = ensure_pandas_series(series)
        clean = series.dropna().reset_index(drop=True)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = (clean < lower) | (clean > upper)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=clean[~is_outlier].index, y=clean[~is_outlier], mode="markers", name="Normal", marker_color=self.colors["primary"]))
        fig.add_trace(go.Scatter(x=clean[is_outlier].index, y=clean[is_outlier], mode="markers", name="Outliers", marker_color=self.colors["danger"]))
        fig.add_hline(y=upper, line_dash="dash", line_color="gray", annotation_text="Upper Bound")
        fig.add_hline(y=lower, line_dash="dash", line_color="gray", annotation_text="Lower Bound")
        fig.update_layout(title=f"Outlier Detection: {series.name}", template=self.theme)
        return fig

    def category_bar_chart(self, series: Series, top_n: int = 20) -> go.Figure:
        series = ensure_pandas_series(series)
        value_counts = series.value_counts().head(top_n)
        fig = go.Figure(go.Bar(x=value_counts.index.astype(str), y=value_counts.values, marker_color=self.colors["primary"]))
        fig.update_layout(
            title=f"Top {top_n} Categories: {series.name}",
            xaxis_title="Category",
            yaxis_title="Count",
            template=self.theme
        )
        return fig

    def correlation_heatmap(self, df: DataFrame, method: str = "pearson") -> go.Figure:
        df = to_pandas(df)
        corr = df.corr(method=method)
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
        fig.update_layout(
            title=f"Correlation Matrix ({method})",
            template=self.theme,
            height=max(400, len(corr.columns) * 25)
        )
        return fig

    def target_correlation_bars(self, correlations: Dict[str, float], target_name: str) -> go.Figure:
        cols = list(correlations.keys())
        vals = list(correlations.values())
        colors = [self.colors["success"] if v > 0 else self.colors["danger"] for v in vals]
        fig = go.Figure(go.Bar(y=cols, x=vals, orientation="h", marker_color=colors))
        fig.update_layout(
            title=f"Correlation with Target: {target_name}",
            xaxis_title="Correlation",
            template=self.theme,
            height=max(400, len(cols) * 25)
        )
        return fig

    def roc_curve(self, fpr, tpr, auc_score: float) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_score:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Random"))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=self.theme
        )
        return fig

    def precision_recall_curve(
        self,
        precision,
        recall,
        pr_auc: float,
        baseline: Optional[float] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a Precision-Recall curve.

        Args:
            precision: Array of precision values
            recall: Array of recall values
            pr_auc: Area under the PR curve
            baseline: Optional baseline (proportion of positives). If provided, shown as dashed line.
            title: Optional chart title
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"PR (AUC={pr_auc:.3f})",
            line={"color": self.colors["primary"], "width": 2}
        ))

        if baseline is not None:
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Baseline: {baseline:.2f}",
                annotation_position="right"
            )

        fig.update_layout(
            title=title or "Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1.05],
            template=self.theme
        )
        return fig

    def model_comparison_grid(self, model_results: Dict[str, Dict[str, Any]], y_test: Any,
                              class_labels: Optional[List[str]] = None, title: Optional[str] = None) -> go.Figure:
        from plotly.subplots import make_subplots
        model_names, n_models = list(model_results.keys()), len(model_results)
        class_labels = class_labels or ["0", "1"]
        subplot_titles = [f"{name[:15]}<br>{row}" for row in ["Confusion Matrix", "ROC Curve", "Precision-Recall"] for name in model_names]
        fig = make_subplots(rows=3, cols=n_models, subplot_titles=subplot_titles, vertical_spacing=0.12, horizontal_spacing=0.08,
                            specs=[[{"type": "heatmap"} for _ in range(n_models)], [{"type": "xy"} for _ in range(n_models)], [{"type": "xy"} for _ in range(n_models)]])
        model_colors = [self.colors["primary"], self.colors["secondary"], self.colors["success"], self.colors["info"], self.colors["warning"]]
        baseline = np.mean(y_test)
        for i, model_name in enumerate(model_names):
            col, color = i + 1, model_colors[i % len(model_colors)]
            y_pred, y_pred_proba = model_results[model_name]["y_pred"], model_results[model_name]["y_pred_proba"]
            self._add_confusion_matrix_to_grid(fig, y_test, y_pred, class_labels, col)
            self._add_roc_curve_to_grid(fig, y_test, y_pred_proba, color, col, n_models)
            self._add_pr_curve_to_grid(fig, y_test, y_pred_proba, color, col, n_models, baseline)
        self._update_comparison_grid_axes(fig, n_models)
        fig.update_layout(title=title or "Model Comparison", height=300 * 3 + 100, width=350 * n_models + 50, template=self.theme, showlegend=False)
        return fig

    def _add_confusion_matrix_to_grid(self, fig: go.Figure, y_test: Any, y_pred: Any, class_labels: List[str], col: int) -> None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_text = [[f"{cm[i][j]}<br>({cm_normalized[i][j]:.0%})" for j in range(len(class_labels))] for i in range(len(class_labels))]
        fig.add_trace(go.Heatmap(z=cm, x=class_labels, y=class_labels, colorscale="Blues", text=cm_text, texttemplate="%{text}",
                                  textfont={"size": 11}, showscale=False, hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"), row=1, col=col)

    def _add_roc_curve_to_grid(self, fig: go.Figure, y_test: Any, y_pred_proba: Any, color: str, col: int, n_models: int) -> None:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": color, "width": 2}, name=f"AUC={auc:.3f}", showlegend=False,
                                  hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"), row=2, col=col)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": "gray", "width": 1, "dash": "dash"}, showlegend=False, hoverinfo="skip"), row=2, col=col)
        xref = f"x{col + n_models}" if col > 1 else "x" + str(n_models + 1) if n_models > 1 else "x2"
        yref = f"y{col + n_models}" if col > 1 else "y" + str(n_models + 1) if n_models > 1 else "y2"
        fig.add_annotation(x=0.95, y=0.05, xref=xref, yref=yref, text=f"AUC={auc:.3f}", showarrow=False, font={"size": 11, "color": color}, bgcolor="rgba(255,255,255,0.8)", xanchor="right")

    def _add_pr_curve_to_grid(self, fig: go.Figure, y_test: Any, y_pred_proba: Any, color: str, col: int, n_models: int, baseline: float) -> None:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line={"color": color, "width": 2}, name=f"PR-AUC={pr_auc:.3f}", showlegend=False,
                                  hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>"), row=3, col=col)
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", line={"color": "gray", "width": 1, "dash": "dash"}, showlegend=False, hoverinfo="skip"), row=3, col=col)
        pr_row_offset = 2 * n_models
        xref = f"x{col + pr_row_offset}" if col + pr_row_offset > 1 else "x"
        yref = f"y{col + pr_row_offset}" if col + pr_row_offset > 1 else "y"
        fig.add_annotation(x=0.05, y=0.05, xref=xref, yref=yref, text=f"PR-AUC={pr_auc:.3f}", showarrow=False, font={"size": 11, "color": color}, bgcolor="rgba(255,255,255,0.8)", xanchor="left")

    def _update_comparison_grid_axes(self, fig: go.Figure, n_models: int) -> None:
        for i in range(n_models):
            col = i + 1
            fig.update_xaxes(title_text="Predicted", row=1, col=col)
            fig.update_yaxes(title_text="Actual", row=1, col=col)
            fig.update_xaxes(title_text="FPR", row=2, col=col, range=[0, 1])
            fig.update_yaxes(title_text="TPR", row=2, col=col, range=[0, 1.02])
            fig.update_xaxes(title_text="Recall", row=3, col=col, range=[0, 1])
            fig.update_yaxes(title_text="Precision", row=3, col=col, range=[0, 1.05])

    def confusion_matrix_heatmap(self, cm, labels: Optional[List[str]] = None) -> go.Figure:
        cm_array = np.array(cm)
        if labels is None:
            labels = [str(i) for i in range(len(cm_array))]
        fig = go.Figure(go.Heatmap(
            z=cm_array,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}"
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=self.theme
        )
        return fig

    def feature_importance_plot(self, importance_df: DataFrame) -> go.Figure:
        importance_df = to_pandas(importance_df)
        fig = go.Figure(go.Bar(
            y=importance_df["feature"],
            x=importance_df["importance"],
            orientation="h",
            marker_color=self.colors["primary"]
        ))
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            template=self.theme,
            height=max(400, len(importance_df) * 25)
        )
        return fig

    def lift_curve(self, percentiles, lift_values) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=percentiles, y=lift_values, mode="lines+markers", name="Model Lift"))
        fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Baseline")
        fig.update_layout(
            title="Lift Curve",
            xaxis_title="Percentile",
            yaxis_title="Lift",
            template=self.theme
        )
        return fig

    def time_series_plot(self, df: DataFrame, date_col: str, value_col: str) -> go.Figure:
        df = to_pandas(df)
        fig = px.line(df, x=date_col, y=value_col)
        fig.update_layout(title=f"{value_col} over Time", template=self.theme)
        return fig

    def cohort_retention_heatmap(self, retention_matrix: DataFrame) -> go.Figure:
        retention_matrix = to_pandas(retention_matrix)
        fig = go.Figure(go.Heatmap(
            z=retention_matrix.values,
            x=retention_matrix.columns,
            y=retention_matrix.index,
            colorscale="Greens",
            text=np.round(retention_matrix.values, 2),
            texttemplate="%{text:.0%}"
        ))
        fig.update_layout(
            title="Cohort Retention",
            xaxis_title="Months Since Start",
            yaxis_title="Cohort",
            template=self.theme
        )
        return fig

    def histogram(self, series: Series, title: Optional[str] = None, nbins: int = 30) -> go.Figure:
        """Create a simple histogram."""
        series = ensure_pandas_series(series)
        fig = go.Figure(go.Histogram(x=series.dropna(), nbinsx=nbins, marker_color=self.colors["primary"]))
        fig.update_layout(
            title=title or f"Distribution of {series.name}",
            xaxis_title=series.name,
            yaxis_title="Count",
            template=self.theme
        )
        return fig

    def heatmap(self, z: Any, x_labels: List[str], y_labels: List[str],
                title: Optional[str] = None, colorscale: str = "RdBu") -> go.Figure:
        """Create a generic heatmap."""
        z_array = np.array(z) if not isinstance(z, np.ndarray) else z
        fig = go.Figure(go.Heatmap(
            z=z_array, x=x_labels, y=y_labels,
            colorscale=colorscale, zmid=0 if colorscale == "RdBu" else None
        ))
        fig.update_layout(
            title=title,
            template=self.theme,
            height=max(400, len(y_labels) * 25)
        )
        return fig

    def scatter_matrix(self, df: DataFrame, title: Optional[str] = None,
                        height: Optional[int] = None, width: Optional[int] = None) -> go.Figure:
        """Create a scatter plot matrix for numeric columns.

        Args:
            df: DataFrame with numeric columns to plot
            title: Optional chart title
            height: Chart height in pixels (default: auto-sized based on columns)
            width: Chart width in pixels (default: None for full-width responsive)
        """
        df = to_pandas(df)
        n_cols = len(df.columns)

        # Auto-size height based on number of columns (min 500, ~150px per column)
        auto_height = max(500, n_cols * 150)

        fig = px.scatter_matrix(df, title=title)
        fig.update_layout(
            template=self.theme,
            height=height or auto_height,
            autosize=True,  # Enable responsive width
        )
        if width:
            fig.update_layout(width=width)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        return fig

    def multi_line_chart(self, data: List[Dict[str, Any]], x_key: str, y_key: str,
                         name_key: str, title: Optional[str] = None,
                         x_title: Optional[str] = None, y_title: Optional[str] = None) -> go.Figure:
        """Create a multi-line chart from a list of data series."""
        fig = go.Figure()
        for series in data:
            fig.add_trace(go.Scatter(
                x=series[x_key], y=series[y_key],
                mode="lines", name=series[name_key]
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash",
                                  line_color="gray", name="Random"))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template=self.theme)
        return fig

    def temporal_distribution(
        self,
        analysis: "TemporalAnalysis",
        title: Optional[str] = None,
        chart_type: str = "bar",
    ) -> go.Figure:
        """Create a temporal distribution chart from TemporalAnalysis."""
        period_counts = analysis.period_counts
        if period_counts.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        x_values = period_counts["period"].astype(str)
        y_values = period_counts["count"]

        fig = go.Figure()
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values,
                mode="lines+markers",
                line={"color": self.colors["primary"], "width": 2},
                marker={"size": 6},
                name="Record Count"
            ))
        else:
            fig.add_trace(go.Bar(
                x=x_values, y=y_values,
                marker_color=self.colors["primary"],
                name="Record Count"
            ))

        # Add mean line
        mean_count = y_values.mean()
        fig.add_hline(
            y=mean_count,
            line_dash="dash",
            line_color=self.colors["secondary"],
            annotation_text=f"Avg: {mean_count:.0f}",
            annotation_position="top right"
        )

        granularity_label = analysis.granularity.value.capitalize()
        default_title = f"Records by {granularity_label}"
        fig.update_layout(
            title=title or default_title,
            xaxis_title=granularity_label,
            yaxis_title="Count",
            template=self.theme,
            xaxis_tickangle=-45 if len(x_values) > 12 else 0
        )
        return fig

    def temporal_trend(
        self,
        analysis: "TemporalAnalysis",
        title: Optional[str] = None,
        show_trend: bool = True,
    ) -> go.Figure:
        """Create a temporal trend line chart with optional trend line."""
        period_counts = analysis.period_counts
        if period_counts.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        x_values = list(range(len(period_counts)))
        x_labels = period_counts["period"].astype(str)
        y_values = period_counts["count"].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_values,
            mode="lines+markers",
            line={"color": self.colors["primary"], "width": 2},
            marker={"size": 8},
            name="Actual"
        ))

        if show_trend and len(x_values) >= 2:
            z = np.polyfit(x_values, y_values, 1)
            trend_line = np.poly1d(z)(x_values)
            slope_pct = ((trend_line[-1] - trend_line[0]) / trend_line[0] * 100) if trend_line[0] != 0 else 0
            trend_direction = "increasing" if z[0] > 0 else "decreasing"
            trend_color = self.colors["success"] if z[0] > 0 else self.colors["danger"]

            fig.add_trace(go.Scatter(
                x=x_labels, y=trend_line,
                mode="lines",
                line={"color": trend_color, "width": 2, "dash": "dash"},
                name=f"Trend ({trend_direction}, {abs(slope_pct):.1f}%)"
            ))

        granularity_label = analysis.granularity.value.capitalize()
        default_title = f"Temporal Trend by {granularity_label}"
        fig.update_layout(
            title=title or default_title,
            xaxis_title=granularity_label,
            yaxis_title="Count",
            template=self.theme,
            xaxis_tickangle=-45 if len(x_labels) > 12 else 0,
            showlegend=True
        )
        return fig

    def temporal_heatmap(
        self,
        dates: Series,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a day-of-week by hour heatmap for datetime data."""
        import pandas as pd
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce").dropna()

        if len(parsed) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No valid dates", x=0.5, y=0.5, showarrow=False)
            return fig

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        counts = parsed.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)

        fig = go.Figure(go.Bar(
            x=dow_names,
            y=counts.values,
            marker_color=[self.colors["info"] if i < 5 else self.colors["warning"] for i in range(7)]
        ))

        fig.update_layout(
            title=title or "Records by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Count",
            template=self.theme
        )
        return fig

    def year_month_heatmap(
        self,
        pivot_df: "DataFrame",
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a year x month heatmap showing record counts."""
        pivot_df = to_pandas(pivot_df)
        if pivot_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure(go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.astype(str).tolist(),
            colorscale="Blues",
            text=pivot_df.values,
            texttemplate="%{text:,}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Count: %{z:,}<extra></extra>"
        ))

        fig.update_layout(
            title=title or "Records by Year and Month",
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.theme,
            height=max(300, len(pivot_df) * 40 + 100)
        )
        return fig

    def cumulative_growth_chart(
        self,
        cumulative_series: Series,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a cumulative growth chart."""
        cumulative_series = ensure_pandas_series(cumulative_series)
        if len(cumulative_series) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[str(p) for p in cumulative_series.index],
            y=cumulative_series.values,
            mode="lines+markers",
            fill="tozeroy",
            line={"color": self.colors["primary"], "width": 2},
            marker={"size": 6},
            name="Cumulative Count"
        ))

        fig.update_layout(
            title=title or "Cumulative Records Over Time",
            xaxis_title="Period",
            yaxis_title="Cumulative Count",
            template=self.theme,
            xaxis_tickangle=-45
        )
        return fig

    def year_over_year_lines(
        self,
        pivot_df: "DataFrame",
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create year-over-year comparison line chart."""
        pivot_df = to_pandas(pivot_df)
        if pivot_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        colors = px.colors.qualitative.Set1
        fig = go.Figure()

        for i, year in enumerate(pivot_df.index):
            fig.add_trace(go.Scatter(
                x=pivot_df.columns.tolist(),
                y=pivot_df.loc[year].values,
                mode="lines+markers",
                name=str(year),
                line={"color": colors[i % len(colors)], "width": 2},
                marker={"size": 8}
            ))

        fig.update_layout(
            title=title or "Year-over-Year Comparison",
            xaxis_title="Month",
            yaxis_title="Count",
            template=self.theme,
            showlegend=True,
            legend={"title": "Year"}
        )
        return fig

    def growth_summary_indicators(
        self,
        growth_data: Dict[str, Any],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create growth summary with key indicators using compact number formatting."""
        if not growth_data.get("has_data"):
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data", x=0.5, y=0.5, showarrow=False)
            return fig

        formatter = NumberFormatter()
        fig = go.Figure()

        # Define indicator positions (x_center, label)
        indicators = [
            (0.15, "Overall Growth", growth_data["overall_growth_pct"], "%"),
            (0.5, "Avg Monthly", growth_data["avg_monthly_growth"], "%/mo"),
            (0.85, f"Trend: {growth_data['trend_direction'].upper()}", growth_data["trend_slope"], "/mo"),
        ]

        for x_pos, label, value, suffix in indicators:
            color = self.colors["success"] if value >= 0 else self.colors["danger"]
            formatted_value = formatter.compact(abs(value))
            sign = "+" if value >= 0 else "-"
            display_text = f"{sign}{formatted_value}{suffix}"

            # Value annotation
            fig.add_annotation(
                x=x_pos, y=0.55,
                text=display_text,
                font={"size": 36, "color": color, "family": "Arial Black"},
                showarrow=False,
                xref="paper", yref="paper"
            )
            # Label annotation
            fig.add_annotation(
                x=x_pos, y=0.15,
                text=label,
                font={"size": 14, "color": "#666666"},
                showarrow=False,
                xref="paper", yref="paper"
            )

        fig.update_layout(
            title={"text": title or "Growth Summary", "font": {"size": 16}},
            template=self.theme,
            height=180,
            margin={"t": 60, "b": 20, "l": 20, "r": 20},
            xaxis={"visible": False},
            yaxis={"visible": False}
        )
        return fig

    def segment_overview(
        self,
        result: "SegmentationResult",
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create overview of segments showing size and target rate."""
        from plotly.subplots import make_subplots

        profiles = result.profiles
        if not profiles:
            fig = go.Figure()
            fig.add_annotation(text="No segments found", x=0.5, y=0.5, showarrow=False)
            return fig

        segment_names = [f"Segment {p.segment_id}" for p in profiles]
        sizes = [p.size_pct for p in profiles]
        target_rates = [p.target_rate for p in profiles]
        has_target = any(tr is not None for tr in target_rates)

        fig = make_subplots(
            rows=1, cols=2 if has_target else 1,
            specs=[[{"type": "pie"}, {"type": "bar"}]] if has_target else [[{"type": "pie"}]],
            subplot_titles=["Segment Sizes", "Target Rate by Segment"] if has_target else ["Segment Sizes"],
        )

        colors = px.colors.qualitative.Set2[:len(profiles)]
        fig.add_trace(
            go.Pie(
                labels=segment_names,
                values=sizes,
                marker_colors=colors,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Size: %{value:.1f}%<extra></extra>",
            ),
            row=1, col=1
        )

        if has_target:
            target_rates_clean = [tr if tr is not None else 0 for tr in target_rates]
            fig.add_trace(
                go.Bar(
                    x=segment_names,
                    y=[tr * 100 for tr in target_rates_clean],
                    marker_color=colors,
                    text=[f"{tr*100:.1f}%" for tr in target_rates_clean],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Target Rate: %{y:.1f}%<extra></extra>",
                ),
                row=1, col=2
            )
            fig.update_yaxes(title_text="Target Rate (%)", row=1, col=2, range=[0, 100])

        fig.update_layout(
            title=title or f"Segment Overview ({result.n_segments} segments)",
            template=self.theme,
            height=400,
            showlegend=False,
        )
        return fig

    def segment_feature_comparison(
        self,
        result: "SegmentationResult",
        features: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Compare feature distributions across segments using grouped bars."""
        profiles = result.profiles
        if not profiles:
            fig = go.Figure()
            fig.add_annotation(text="No segments found", x=0.5, y=0.5, showarrow=False)
            return fig

        all_features = set()
        for p in profiles:
            all_features.update(p.defining_features.keys())

        if features:
            all_features = [f for f in features if f in all_features]
        else:
            all_features = sorted(all_features)[:8]

        if not all_features:
            fig = go.Figure()
            fig.add_annotation(text="No features to compare", x=0.5, y=0.5, showarrow=False)
            return fig

        colors = px.colors.qualitative.Set2[:len(profiles)]
        fig = go.Figure()

        for i, profile in enumerate(profiles):
            means = []
            for feat in all_features:
                feat_data = profile.defining_features.get(feat, {})
                means.append(feat_data.get("mean", 0))

            fig.add_trace(go.Bar(
                name=f"Segment {profile.segment_id}",
                x=list(all_features),
                y=means,
                marker_color=colors[i],
            ))

        fig.update_layout(
            title=title or "Feature Comparison Across Segments",
            xaxis_title="Feature",
            yaxis_title="Mean Value",
            barmode="group",
            template=self.theme,
            height=400,
            legend={"title": "Segment"},
        )
        return fig

    def segment_recommendation_card(
        self,
        result: "SegmentationResult",
        title: Optional[str] = None,
    ) -> go.Figure:
        """Display segmentation recommendation with rationale."""
        recommendation_colors = {
            "single_model": self.colors["success"],
            "consider_segmentation": self.colors["warning"],
            "strong_segmentation": self.colors["danger"],
        }
        recommendation_labels = {
            "single_model": "Single Model Recommended",
            "consider_segmentation": "Consider Segmentation",
            "strong_segmentation": "Segmentation Strongly Recommended",
        }

        rec_color = recommendation_colors.get(result.recommendation, self.colors["info"])
        rec_label = recommendation_labels.get(result.recommendation, result.recommendation)

        fig = go.Figure()

        # Recommendation header
        fig.add_annotation(
            x=0.5, y=0.85,
            text=rec_label,
            font={"size": 24, "color": rec_color, "family": "Arial Black"},
            showarrow=False,
            xref="paper", yref="paper"
        )

        # Confidence indicator
        fig.add_annotation(
            x=0.5, y=0.65,
            text=f"Confidence: {result.confidence*100:.0f}%",
            font={"size": 16, "color": "#666666"},
            showarrow=False,
            xref="paper", yref="paper"
        )

        # Key metrics
        metrics_text = (
            f"Segments: {result.n_segments} | "
            f"Quality: {result.quality_score:.2f} | "
            f"Target Variance: {result.target_variance_ratio:.2f}"
            if result.target_variance_ratio is not None
            else f"Segments: {result.n_segments} | Quality: {result.quality_score:.2f}"
        )
        fig.add_annotation(
            x=0.5, y=0.48,
            text=metrics_text,
            font={"size": 14, "color": "#888888"},
            showarrow=False,
            xref="paper", yref="paper"
        )

        # Rationale
        rationale_text = "<br>".join(f"• {r}" for r in result.rationale[:4])
        fig.add_annotation(
            x=0.5, y=0.2,
            text=rationale_text,
            font={"size": 12, "color": "#666666"},
            showarrow=False,
            xref="paper", yref="paper",
            align="center"
        )

        fig.update_layout(
            title=title or "Segmentation Recommendation",
            template=self.theme,
            height=280,
            margin={"t": 50, "b": 20, "l": 20, "r": 20},
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]},
        )
        return fig

    # =========================================================================
    # Advanced Time Series Visualizations
    # =========================================================================

    def sparkline(
        self,
        values: List[float],
        title: Optional[str] = None,
        show_endpoints: bool = True,
        show_min_max: bool = True,
        height: int = 60,
        width: int = 200,
    ) -> go.Figure:
        """Create a compact sparkline for inline time series display.

        Sparklines are small, word-sized graphics that show trends at a glance.
        Ideal for dashboards and tables where space is limited.
        """
        x = list(range(len(values)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=values,
            mode="lines",
            line={"color": self.colors["primary"], "width": 1.5},
            hoverinfo="y"
        ))

        if show_endpoints and len(values) >= 2:
            fig.add_trace(go.Scatter(
                x=[0, len(values) - 1],
                y=[values[0], values[-1]],
                mode="markers",
                marker={"color": self.colors["primary"], "size": 6},
                hoverinfo="y"
            ))

        if show_min_max and len(values) >= 2:
            min_idx, max_idx = int(np.argmin(values)), int(np.argmax(values))
            fig.add_trace(go.Scatter(
                x=[min_idx], y=[values[min_idx]],
                mode="markers",
                marker={"color": self.colors["danger"], "size": 5},
                hovertemplate=f"Min: {values[min_idx]:.2f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[max_idx], y=[values[max_idx]],
                mode="markers",
                marker={"color": self.colors["success"], "size": 5},
                hovertemplate=f"Max: {values[max_idx]:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title={"text": title, "font": {"size": 10}} if title else None,
            height=height,
            width=width,
            margin={"t": 20 if title else 5, "b": 5, "l": 5, "r": 5},
            xaxis={"visible": False},
            yaxis={"visible": False},
            showlegend=False,
            template=self.theme,
        )
        return fig

    def sparkline_grid(
        self,
        data: Dict[str, List[float]],
        columns: int = 4,
        sparkline_height: int = 60,
        sparkline_width: int = 180,
    ) -> go.Figure:
        """Create a grid of sparklines for multiple time series comparison."""
        from plotly.subplots import make_subplots

        names = list(data.keys())
        n_rows = (len(names) + columns - 1) // columns

        fig = make_subplots(
            rows=n_rows, cols=columns,
            subplot_titles=names,
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        for i, (name, values) in enumerate(data.items()):
            row, col = (i // columns) + 1, (i % columns) + 1
            x = list(range(len(values)))

            fig.add_trace(
                go.Scatter(x=x, y=values, mode="lines",
                          line={"color": self.colors["primary"], "width": 1.5},
                          showlegend=False),
                row=row, col=col
            )

            if len(values) >= 2:
                trend = values[-1] - values[0]
                color = self.colors["success"] if trend >= 0 else self.colors["danger"]
                fig.add_trace(
                    go.Scatter(x=[len(values) - 1], y=[values[-1]], mode="markers",
                              marker={"color": color, "size": 6}, showlegend=False),
                    row=row, col=col
                )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            height=n_rows * sparkline_height + 50,
            template=self.theme,
            margin={"t": 40, "b": 20},
        )
        return fig

    def calendar_heatmap(
        self,
        dates: Series,
        values: Optional[Series] = None,
        title: Optional[str] = None,
        colorscale: str = "Blues",
    ) -> go.Figure:
        """Create a calendar heatmap showing patterns by day-of-week and week-of-year.

        Similar to GitHub contribution graphs. Shows temporal patterns at a glance.
        If values not provided, shows count of occurrences per day.
        """
        import pandas as pd
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce")

        if values is not None:
            values = ensure_pandas_series(values)
            df_cal = pd.DataFrame({"date": parsed, "value": values}).dropna()
            daily = df_cal.groupby(df_cal["date"].dt.date)["value"].sum()
        else:
            daily = parsed.dropna().dt.date.value_counts().sort_index()

        if len(daily) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No valid dates", x=0.5, y=0.5, showarrow=False)
            return fig

        df_daily = pd.DataFrame({"date": pd.to_datetime(daily.index), "value": daily.values})
        df_daily["week"] = df_daily["date"].dt.isocalendar().week
        df_daily["year"] = df_daily["date"].dt.year
        df_daily["dow"] = df_daily["date"].dt.dayofweek
        df_daily["year_week"] = df_daily["year"].astype(str) + "-W" + df_daily["week"].astype(str).str.zfill(2)

        pivot = df_daily.pivot_table(index="dow", columns="year_week", values="value", aggfunc="sum")
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[dow_labels[i] for i in pivot.index],
            colorscale=colorscale,
            hovertemplate="Week: %{x}<br>Day: %{y}<br>Value: %{z:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            title=title or "Calendar Heatmap",
            xaxis_title="Week",
            yaxis_title="Day of Week",
            template=self.theme,
            height=250,
            xaxis={"tickangle": -45, "dtick": 4},
        )
        return fig

    def monthly_calendar_heatmap(
        self,
        dates: Series,
        values: Optional[Series] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a month x day-of-week heatmap for pattern discovery."""
        import pandas as pd
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce").dropna()

        if values is not None:
            values = ensure_pandas_series(values)
            df_cal = pd.DataFrame({"date": parsed, "value": values}).dropna()
            df_cal["month"] = df_cal["date"].dt.month
            df_cal["dow"] = df_cal["date"].dt.dayofweek
            pivot = df_cal.pivot_table(index="dow", columns="month", values="value", aggfunc="mean")
        else:
            df_cal = pd.DataFrame({"date": parsed})
            df_cal["month"] = df_cal["date"].dt.month
            df_cal["dow"] = df_cal["date"].dt.dayofweek
            pivot = df_cal.groupby(["dow", "month"]).size().unstack(fill_value=0)

        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[month_labels[i-1] for i in pivot.columns],
            y=[dow_labels[i] for i in pivot.index],
            colorscale="YlOrRd",
            hovertemplate="Month: %{x}<br>Day: %{y}<br>Value: %{z:,.1f}<extra></extra>",
        ))

        fig.update_layout(
            title=title or "Activity by Month and Day of Week",
            template=self.theme,
            height=280,
        )
        return fig

    def time_series_with_anomalies(
        self,
        dates: Series,
        values: Series,
        window: int = 7,
        n_std: float = 2.0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create time series plot with anomaly detection bands.

        Uses rolling mean ± n_std * rolling_std to define normal bounds.
        Points outside bounds are highlighted as anomalies.
        """
        import pandas as pd
        dates = ensure_pandas_series(dates)
        values = ensure_pandas_series(values)

        df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna()
        df = df.sort_values("date")

        df["rolling_mean"] = df["value"].rolling(window=window, center=True, min_periods=1).mean()
        df["rolling_std"] = df["value"].rolling(window=window, center=True, min_periods=1).std()
        df["upper"] = df["rolling_mean"] + n_std * df["rolling_std"]
        df["lower"] = df["rolling_mean"] - n_std * df["rolling_std"]
        df["is_anomaly"] = (df["value"] > df["upper"]) | (df["value"] < df["lower"])

        anomaly_count = df["is_anomaly"].sum()
        anomaly_pct = anomaly_count / len(df) * 100

        fig = go.Figure()

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["upper"], df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name=f"Normal Range (±{n_std}σ)",
            hoverinfo="skip",
        ))

        # Rolling mean
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rolling_mean"],
            mode="lines",
            line={"color": self.colors["info"], "width": 1, "dash": "dash"},
            name="Rolling Mean",
        ))

        # Normal points
        normal = df[~df["is_anomaly"]]
        fig.add_trace(go.Scatter(
            x=normal["date"], y=normal["value"],
            mode="lines+markers",
            line={"color": self.colors["primary"], "width": 1.5},
            marker={"size": 4},
            name="Normal",
        ))

        # Anomaly points
        anomalies = df[df["is_anomaly"]]
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies["date"], y=anomalies["value"],
                mode="markers",
                marker={"color": self.colors["danger"], "size": 10, "symbol": "x"},
                name=f"Anomalies ({anomaly_count})",
            ))

        fig.update_layout(
            title=title or f"Time Series with Anomalies ({anomaly_pct:.1f}% anomalous)",
            xaxis_title="Date",
            yaxis_title="Value",
            template=self.theme,
            height=400,
            legend={"orientation": "h", "y": -0.15},
        )
        return fig

    def waterfall_chart(
        self,
        categories: List[str],
        values: List[float],
        title: Optional[str] = None,
        initial_label: str = "Start",
        final_label: str = "End",
    ) -> go.Figure:
        """Create a waterfall chart showing cumulative impact.

        Shows how sequential changes contribute to a final result.
        Useful for explaining score breakdowns or cumulative effects.
        """
        measures = ["absolute"] + ["relative"] * len(values) + ["total"]
        x_labels = [initial_label] + categories + [final_label]

        initial_value = 0
        cumulative = initial_value
        y_values = [initial_value]
        text_values = [f"{initial_value:,.0f}"]

        for v in values:
            y_values.append(v)
            cumulative += v
            sign = "+" if v >= 0 else ""
            text_values.append(f"{sign}{v:,.0f}")

        y_values.append(cumulative)
        text_values.append(f"{cumulative:,.0f}")

        colors = [self.colors["info"]]  # Initial
        for v in values:
            colors.append(self.colors["success"] if v >= 0 else self.colors["danger"])
        colors.append(self.colors["primary"])  # Total

        fig = go.Figure(go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            text=text_values,
            textposition="outside",
            connector={"line": {"color": "gray", "width": 1, "dash": "dot"}},
            increasing={"marker": {"color": self.colors["success"]}},
            decreasing={"marker": {"color": self.colors["danger"]}},
            totals={"marker": {"color": self.colors["primary"]}},
        ))

        fig.update_layout(
            title=title or "Waterfall Chart",
            template=self.theme,
            height=400,
            showlegend=False,
        )
        return fig

    def quality_waterfall(
        self,
        check_results: List[Dict[str, Any]],
        max_score: int = 100,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a waterfall chart specifically for quality score breakdown.

        Shows how each check contributes to or detracts from the total score.

        Args:
            check_results: List of dicts with 'name', 'passed', 'weight' keys
            max_score: Maximum possible score (default 100)
            title: Chart title
        """
        categories = []
        values = []

        for check in check_results:
            categories.append(check["name"])
            if check["passed"]:
                values.append(0)  # No penalty
            else:
                penalty = -check["weight"] * (max_score / sum(c["weight"] for c in check_results))
                values.append(penalty)

        return self.waterfall_chart(
            categories=categories,
            values=values,
            title=title or "Quality Score Breakdown",
            initial_label="Max Score",
            final_label="Final Score",
        )

    def velocity_acceleration_chart(
        self,
        data: Dict[str, Dict[str, List[float]]],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create side-by-side Value/Velocity/Acceleration chart for cohort comparison.

        Args:
            data: Dict with structure {column: {"retained": [...], "churned": [...], "velocity_retained": [...], ...}}
            title: Chart title
        """
        from plotly.subplots import make_subplots

        columns = list(data.keys())
        n_cols = len(columns)

        fig = make_subplots(
            rows=n_cols, cols=3,
            subplot_titles=[f"{col[:12]} - Value" for col in columns] +
                          [f"{col[:12]} - Velocity" for col in columns] +
                          [f"{col[:12]} - Accel." for col in columns],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        for i, col in enumerate(columns):
            row = i + 1
            col_data = data[col]

            # Value
            if "retained" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["retained"], mode="lines",
                    line={"color": self.colors["success"], "width": 1.5},
                    name="Retained", showlegend=(i == 0), legendgroup="retained"
                ), row=row, col=1)
            if "churned" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["churned"], mode="lines",
                    line={"color": self.colors["danger"], "width": 1.5},
                    name="Churned", showlegend=(i == 0), legendgroup="churned"
                ), row=row, col=1)

            # Velocity
            if "velocity_retained" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["velocity_retained"], mode="lines",
                    line={"color": self.colors["success"], "width": 1.5},
                    showlegend=False, legendgroup="retained"
                ), row=row, col=2)
            if "velocity_churned" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["velocity_churned"], mode="lines",
                    line={"color": self.colors["danger"], "width": 1.5},
                    showlegend=False, legendgroup="churned"
                ), row=row, col=2)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=2)

            # Acceleration
            if "accel_retained" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["accel_retained"], mode="lines",
                    line={"color": self.colors["success"], "width": 1.5},
                    showlegend=False, legendgroup="retained"
                ), row=row, col=3)
            if "accel_churned" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["accel_churned"], mode="lines",
                    line={"color": self.colors["danger"], "width": 1.5},
                    showlegend=False, legendgroup="churned"
                ), row=row, col=3)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=3)

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            height=150 * n_cols + 80,
            title=title or "Value → Velocity → Acceleration",
            template=self.theme,
            legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
            margin={"t": 100},
        )
        return fig

    def lag_correlation_heatmap(
        self,
        data: Dict[str, List[float]],
        max_lag: int = 14,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create heatmap of lag correlations for multiple variables.

        Args:
            data: Dict with {column_name: [corr_lag1, corr_lag2, ...]}
            max_lag: Maximum lag shown
            title: Chart title
        """
        columns = list(data.keys())
        z_values = [data[col][:max_lag] for col in columns]
        lag_labels = [f"Lag {i}" for i in range(1, max_lag + 1)]

        fig = go.Figure(go.Heatmap(
            z=z_values,
            x=lag_labels,
            y=[col[:15] for col in columns],
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar={"title": "Correlation"},
        ))

        fig.update_layout(
            title=title or "Autocorrelation by Lag",
            xaxis_title="Lag (periods)",
            yaxis_title="Variable",
            template=self.theme,
            height=50 * len(columns) + 150,
        )
        return fig

    def predictive_power_chart(
        self,
        iv_values: Dict[str, float],
        ks_values: Dict[str, float],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create side-by-side IV and KS statistic bar charts.

        Args:
            iv_values: Dict with {column: iv_value}
            ks_values: Dict with {column: ks_value}
            title: Chart title
        """
        from plotly.subplots import make_subplots

        # Sort by IV
        sorted_cols = sorted(iv_values.keys(), key=lambda x: iv_values[x], reverse=True)
        ivs = [iv_values[c] for c in sorted_cols]
        kss = [ks_values.get(c, 0) for c in sorted_cols]
        col_labels = [c[:15] for c in sorted_cols]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Information Value (IV)", "KS Statistic"),
        )

        # IV thresholds
        iv_colors = [
            self.colors["danger"] if iv > 0.5 else
            self.colors["success"] if iv > 0.3 else
            self.colors["warning"] if iv > 0.1 else
            self.colors["primary"]
            for iv in ivs
        ]

        fig.add_trace(go.Bar(
            x=col_labels, y=ivs, marker_color=iv_colors, name="IV"
        ), row=1, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", row=1, col=1)

        # KS thresholds
        ks_colors = [
            self.colors["success"] if ks > 0.4 else
            self.colors["warning"] if ks > 0.2 else
            self.colors["primary"]
            for ks in kss
        ]

        fig.add_trace(go.Bar(
            x=col_labels, y=kss, marker_color=ks_colors, name="KS"
        ), row=1, col=2)
        fig.add_hline(y=0.4, line_dash="dash", line_color="green", row=1, col=2)
        fig.add_hline(y=0.2, line_dash="dash", line_color="orange", row=1, col=2)

        fig.update_layout(
            title=title or "Variable Predictive Power",
            template=self.theme,
            height=400,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=45)
        return fig

    def momentum_comparison_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create grouped bar chart comparing momentum between cohorts.

        Args:
            data: Dict with {column: {"retained_7_30": x, "churned_7_30": y, "retained_30_90": z, ...}}
            title: Chart title
        """
        from plotly.subplots import make_subplots

        columns = list(data.keys())
        col_labels = [c[:12] for c in columns]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("7d/30d Momentum", "30d/90d Momentum"),
        )

        # 7d/30d
        fig.add_trace(go.Bar(
            name="Retained", x=col_labels,
            y=[data[c].get("retained_7_30", 1) for c in columns],
            marker_color=self.colors["success"],
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            name="Churned", x=col_labels,
            y=[data[c].get("churned_7_30", 1) for c in columns],
            marker_color=self.colors["danger"],
        ), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

        # 30d/90d
        fig.add_trace(go.Bar(
            name="Retained", x=col_labels,
            y=[data[c].get("retained_30_90", 1) for c in columns],
            marker_color=self.colors["success"], showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            name="Churned", x=col_labels,
            y=[data[c].get("churned_30_90", 1) for c in columns],
            marker_color=self.colors["danger"], showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_layout(
            title=title or "Momentum by Retention Status",
            template=self.theme,
            height=400,
            barmode="group",
            legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        )
        return fig

    def cohort_sparklines(
        self,
        data: Dict[str, Dict[str, List[float]]],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create sparkline grid comparing retained vs churned trends.

        Args:
            data: Dict with {column: {"retained": [...], "churned": [...]}}
            title: Chart title
        """
        from plotly.subplots import make_subplots

        columns = list(data.keys())
        n_cols = len(columns)

        fig = make_subplots(
            rows=2, cols=n_cols,
            row_titles=["Retained", "Churned"],
            subplot_titles=[c[:15] for c in columns],
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        for i, col in enumerate(columns):
            col_num = i + 1
            col_data = data[col]

            # Retained (top row)
            if "retained" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["retained"], mode="lines",
                    line={"color": self.colors["success"], "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(44, 160, 44, 0.2)",
                    showlegend=False,
                ), row=1, col=col_num)

            # Churned (bottom row)
            if "churned" in col_data:
                fig.add_trace(go.Scatter(
                    y=col_data["churned"], mode="lines",
                    line={"color": self.colors["danger"], "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(214, 39, 40, 0.2)",
                    showlegend=False,
                ), row=2, col=col_num)

        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(
            title=title or "Retained vs Churned Trends",
            height=300,
            template=self.theme,
            margin={"t": 80, "b": 30, "l": 60, "r": 20},
        )
        return fig

    def descriptive_stats_tiles(
        self,
        df: DataFrame,
        findings: Any,
        max_columns: int = 12,
        columns_per_row: int = 4,
    ) -> go.Figure:
        """Create a grid of mini chart tiles showing descriptive statistics for each column.

        Each tile shows a type-appropriate visualization:
        - Numeric: histogram with mean/median markers and key stats
        - Categorical: top categories bar chart with cardinality
        - Binary: pie chart with class balance
        - Datetime: date range indicator
        - Identifier: uniqueness gauge

        Args:
            df: DataFrame to visualize
            findings: ExplorationFindings object with column metadata
            max_columns: Maximum number of columns to display
            columns_per_row: Number of tiles per row
        """
        from plotly.subplots import make_subplots

        df = to_pandas(df)
        formatter = NumberFormatter()

        # Exclude temporal metadata columns from visualization
        temporal_metadata_cols = {"feature_timestamp", "label_timestamp", "label_available_flag"}
        available_cols = {k: v for k, v in findings.columns.items() if k not in temporal_metadata_cols}

        # Select columns to display (prioritize by type)
        type_priority = ['target', 'binary', 'numeric_continuous', 'numeric_discrete',
                         'categorical_nominal', 'categorical_ordinal', 'datetime', 'identifier']
        sorted_cols = []
        for col_type in type_priority:
            for name, col in available_cols.items():
                if col.inferred_type.value == col_type and name not in sorted_cols:
                    sorted_cols.append(name)
        for name in available_cols.keys():
            if name not in sorted_cols:
                sorted_cols.append(name)
        display_cols = sorted_cols[:max_columns]

        n_cols = min(columns_per_row, len(display_cols))
        n_rows = (len(display_cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"<b>{c[:20]}</b>" for c in display_cols],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[[{"type": "xy"} for _ in range(n_cols)] for _ in range(n_rows)]
        )

        for i, col_name in enumerate(display_cols):
            row, col = (i // n_cols) + 1, (i % n_cols) + 1
            col_finding = findings.columns.get(col_name)
            col_type = col_finding.inferred_type.value if col_finding else "unknown"
            series = df[col_name] if col_name in df.columns else None

            if series is None:
                continue

            self._add_column_tile(fig, series, col_finding, col_type, row, col, formatter, n_cols)

        fig.update_layout(
            height=250 * n_rows,
            template=self.theme,
            showlegend=False,
            margin={"t": 40, "b": 20, "l": 40, "r": 20},
        )

        return fig

    def dataset_at_a_glance(
        self,
        df: DataFrame,
        findings: Any,
        source_path: str = "",
        granularity: str = "entity",
        max_columns: int = 12,
        columns_per_row: int = 4,
    ) -> go.Figure:
        """Create a unified dataset overview with key metrics and column distribution tiles.

        Combines dataset-level stats (rows, columns, format, granularity) with
        small multiples of column distributions for a complete first look.

        Args:
            df: DataFrame to visualize
            findings: ExplorationFindings object with column metadata
            source_path: Path to data source (for format detection)
            granularity: Dataset granularity ("entity" or "event")
            max_columns: Maximum number of column tiles to display
            columns_per_row: Number of tiles per row
        """
        from plotly.subplots import make_subplots
        from pathlib import Path

        df = to_pandas(df)
        formatter = NumberFormatter()

        # Calculate metrics
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        completeness = 100 - sum(
            col.universal_metrics.get("null_percentage", 0)
            for col in findings.columns.values()
        ) / max(len(findings.columns), 1)

        # Detect format from path
        path = Path(source_path) if source_path else Path("data.csv")
        fmt = path.suffix.lstrip('.').upper() or "CSV"
        if fmt == "":
            fmt = "CSV"

        # Exclude temporal metadata columns from visualization
        temporal_metadata_cols = {"feature_timestamp", "label_timestamp", "label_available_flag"}
        available_cols = {k: v for k, v in findings.columns.items() if k not in temporal_metadata_cols}

        # Select columns to display (prioritize by type)
        type_priority = ['target', 'binary', 'numeric_continuous', 'numeric_discrete',
                         'categorical_nominal', 'categorical_ordinal', 'datetime', 'identifier']
        sorted_cols = []
        for col_type in type_priority:
            for name, col in available_cols.items():
                if col.inferred_type.value == col_type and name not in sorted_cols:
                    sorted_cols.append(name)
        for name in available_cols.keys():
            if name not in sorted_cols:
                sorted_cols.append(name)
        display_cols = sorted_cols[:max_columns]

        n_cols = min(columns_per_row, len(display_cols))
        n_tile_rows = (len(display_cols) + n_cols - 1) // n_cols

        # Build specs: 1 header row + tile rows
        header_specs = [{"type": "indicator"} for _ in range(n_cols)]
        tile_specs = [[{"type": "xy"} for _ in range(n_cols)] for _ in range(n_tile_rows)]

        # Subplot titles: empty for header, column names for tiles
        titles = [""] * n_cols + [f"<b>{c[:18]}</b>" for c in display_cols]

        fig = make_subplots(
            rows=1 + n_tile_rows,
            cols=n_cols,
            row_heights=[0.15] + [0.85 / n_tile_rows] * n_tile_rows,
            specs=[header_specs] + tile_specs,
            subplot_titles=titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        # Header row: Order is Rows, Columns, Structure, Format, Memory
        # Use annotations for all to ensure consistent appearance
        structure_label = "Event" if granularity.lower() == "event" else "Entity"
        memory_str = f"{memory_mb:.1f} MB"

        # Calculate header column positions for paper coordinates
        h_spacing = 0.06
        col_width = (1.0 - h_spacing * (n_cols - 1)) / n_cols

        def get_header_x(col_idx: int) -> float:
            """Get x center position for header column (1-indexed)."""
            return (col_idx - 1) * (col_width + h_spacing) + col_width / 2

        # Header data: (label, value)
        header_items = [
            ("Rows", f"{findings.row_count:,}"),
            ("Columns", str(findings.column_count)),
            ("Structure", structure_label),
            ("Format", fmt),
            ("Memory", memory_str),
        ]

        # Add placeholder indicators (needed for subplot structure)
        for i in range(min(n_cols, len(header_items))):
            fig.add_trace(go.Indicator(
                mode="number", value=0,
                number={"font": {"size": 1, "color": "rgba(0,0,0,0)"}}
            ), row=1, col=i+1)

        # Add labels (small, gray, top) and values (large, blue, below) as annotations
        label_y = 0.96
        value_y = 0.92

        for i, (label, value) in enumerate(header_items[:n_cols]):
            x_pos = get_header_x(i + 1)

            # Label
            fig.add_annotation(
                x=x_pos, y=label_y,
                xref="paper", yref="paper",
                text=label, showarrow=False,
                font={"size": 12, "color": "#666"},
                xanchor="center", yanchor="middle"
            )

            # Value
            fig.add_annotation(
                x=x_pos, y=value_y,
                xref="paper", yref="paper",
                text=value, showarrow=False,
                font={"size": 28, "color": self.colors["primary"]},
                xanchor="center", yanchor="middle"
            )

        # Column tiles (starting from row 2)
        for i, col_name in enumerate(display_cols):
            tile_row = (i // n_cols) + 2  # +2 because row 1 is header
            tile_col = (i % n_cols) + 1
            col_finding = findings.columns.get(col_name)
            col_type = col_finding.inferred_type.value if col_finding else "unknown"
            series = df[col_name] if col_name in df.columns else None

            if series is None:
                continue

            self._add_column_tile(fig, series, col_finding, col_type, tile_row, tile_col, formatter, n_cols)

        fig.update_layout(
            height=120 + 220 * n_tile_rows,
            template=self.theme,
            showlegend=False,
            margin={"t": 30, "b": 20, "l": 40, "r": 20},
        )

        return fig

    def _add_column_tile(
        self,
        fig: go.Figure,
        series: Series,
        col_finding: Any,
        col_type: str,
        row: int,
        col: int,
        formatter: "NumberFormatter",
        n_cols: int = 4,
    ) -> None:
        """Add a single column tile to the subplot grid."""
        series = ensure_pandas_series(series)
        metrics = col_finding.universal_metrics if col_finding else {}
        type_metrics = col_finding.type_metrics if col_finding else {}

        if col_type in ('numeric_continuous', 'numeric_discrete'):
            self._add_numeric_tile(fig, series, metrics, type_metrics, row, col, n_cols, formatter)
        elif col_type in ('categorical_nominal', 'categorical_ordinal', 'categorical_cyclical'):
            self._add_categorical_tile(fig, series, metrics, row, col, n_cols, formatter)
        elif col_type == 'binary':
            self._add_binary_tile(fig, series, metrics, row, col, n_cols, formatter)
        elif col_type in ('datetime', 'date'):
            self._add_datetime_tile(fig, series, metrics, row, col, n_cols)
        elif col_type == 'identifier':
            self._add_identifier_tile(fig, series, metrics, row, col, n_cols, formatter)
        elif col_type == 'target':
            self._add_target_tile(fig, series, metrics, row, col, n_cols, formatter)
        else:
            self._add_generic_tile(fig, series, metrics, row, col, n_cols, formatter)

    def _get_axis_ref(self, row: int, col: int, n_cols: int, axis: str = "x") -> str:
        """Get the correct axis reference for subplot annotations."""
        # Calculate linear index (0-based)
        idx = (row - 1) * n_cols + col
        # First subplot uses 'x'/'y', others use 'x2', 'x3', etc.
        if idx == 1:
            return axis
        return f"{axis}{idx}"

    def _add_numeric_tile(
        self, fig: go.Figure, series: Series, metrics: Dict, type_metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add numeric column tile with histogram and stats."""
        clean = series.dropna()
        if len(clean) == 0:
            return

        mean_val = type_metrics.get('mean', clean.mean())
        median_val = type_metrics.get('median', clean.median())
        std_val = type_metrics.get('std', clean.std())
        null_pct = metrics.get('null_percentage', 0)

        fig.add_trace(go.Histogram(
            x=clean, nbinsx=20,
            marker_color=self.colors["primary"],
            opacity=0.7,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
        ), row=row, col=col)

        xaxis_ref = self._get_axis_ref(row, col, n_cols, 'x')
        yaxis_ref = self._get_axis_ref(row, col, n_cols, 'y')
        fig.add_shape(type="line", x0=mean_val, x1=mean_val, y0=0, y1=1,
                     xref=xaxis_ref, yref=f"{yaxis_ref} domain",
                     line={"color": self.colors["secondary"], "width": 2, "dash": "dash"})
        fig.add_shape(type="line", x0=median_val, x1=median_val, y0=0, y1=1,
                     xref=xaxis_ref, yref=f"{yaxis_ref} domain",
                     line={"color": self.colors["success"], "width": 2, "dash": "dot"})

        stats_text = (f"μ={formatter.compact(mean_val)} | "
                     f"σ={formatter.compact(std_val)}" +
                     (f"<br>null={null_pct:.0f}%" if null_pct > 0 else ""))
        fig.add_annotation(
            x=0.98, y=0.98, xref=f"{xaxis_ref} domain", yref=f"{yaxis_ref} domain",
            text=stats_text, showarrow=False,
            font={"size": 9, "color": "#666"},
            bgcolor="rgba(255,255,255,0.8)",
            xanchor="right", yanchor="top"
        )

    def _add_categorical_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add categorical column tile with top categories bar."""
        value_counts = series.value_counts().head(5)
        cardinality = metrics.get('distinct_count', series.nunique())
        null_pct = metrics.get('null_percentage', 0)

        # Gradient colors to show rank
        colors = [self.colors["info"]] + [self.colors["primary"]] * (len(value_counts) - 1)

        fig.add_trace(go.Bar(
            x=value_counts.values,
            y=[str(v)[:10] for v in value_counts.index],
            orientation='h',
            marker_color=colors[:len(value_counts)],
            hovertemplate="%{y}: %{x:,}<extra></extra>"
        ), row=row, col=col)

    def _add_binary_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add binary column tile with horizontal bars showing labels clearly."""
        value_counts = series.value_counts()
        if len(value_counts) == 0:
            return

        labels = [str(v) for v in value_counts.index]
        values = value_counts.values.tolist()
        total = sum(values)
        percentages = [v/total*100 for v in values]

        balance_ratio = max(values) / min(values) if min(values) > 0 else float('inf')
        balance_color = (self.colors["success"] if balance_ratio < 3
                        else self.colors["warning"] if balance_ratio < 10
                        else self.colors["danger"])

        # Horizontal bars with labels on y-axis
        colors = [self.colors["primary"], self.colors["secondary"]]
        fig.add_trace(go.Bar(
            y=labels[:2],
            x=percentages[:2],
            orientation='h',
            marker_color=colors[:len(labels)],
            text=[f"{p:.0f}%" for p in percentages[:2]],
            textposition="inside",
            textfont={"size": 11, "color": "white"},
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            showlegend=False
        ), row=row, col=col)

        ratio_text = f"{balance_ratio:.1f}:1"
        xref = f"{self._get_axis_ref(row, col, n_cols, 'x')} domain"
        yref = f"{self._get_axis_ref(row, col, n_cols, 'y')} domain"
        fig.add_annotation(
            x=0.98, y=0.98, xref=xref, yref=yref,
            text=ratio_text, showarrow=False,
            font={"size": 10, "color": balance_color, "family": "Arial Black"},
            xanchor="right", yanchor="top"
        )

    def _add_datetime_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int
    ) -> None:
        """Add datetime column tile with date range visualization."""
        import pandas as pd
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates = pd.to_datetime(series, errors='coerce').dropna()
        if len(dates) == 0:
            return

        # Monthly distribution as area chart for cleaner look
        counts = dates.dt.to_period('M').value_counts().sort_index()
        x_labels = [str(p) for p in counts.index]
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=counts.values,
            mode='lines',
            fill='tozeroy',
            line={"color": self.colors["info"]},
            fillcolor=f"rgba(23, 190, 207, 0.3)",
            hovertemplate="%{x}: %{y:,}<extra></extra>"
        ), row=row, col=col)

        # Force categorical x-axis to prevent Plotly from interpreting as dates
        xaxis_name = f"xaxis{(row - 1) * n_cols + col}" if (row - 1) * n_cols + col > 1 else "xaxis"
        fig.update_layout(**{xaxis_name: {"type": "category", "tickangle": -45}})

    def _add_identifier_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add identifier column tile with uniqueness gauge."""
        total = len(series)
        unique = metrics.get('distinct_count', series.nunique())
        unique_pct = (unique / total * 100) if total > 0 else 0

        gauge_color = (self.colors["success"] if unique_pct >= 99
                      else self.colors["warning"] if unique_pct >= 95
                      else self.colors["danger"])

        # Progress bar style for uniqueness
        fig.add_trace(go.Bar(
            x=[unique_pct], y=[""],
            orientation='h',
            marker_color=gauge_color,
            text=f"{unique_pct:.1f}% unique",
            textposition="inside",
            textfont={"color": "white", "size": 11},
            hovertemplate=f"Unique: {unique:,} / {total:,}<extra></extra>",
            showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Bar(
            x=[100 - unique_pct], y=[""],
            orientation='h',
            marker_color="#ecf0f1",
            hoverinfo="skip",
            showlegend=False
        ), row=row, col=col)

    def _add_target_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add target column tile with horizontal bars showing class distribution."""
        value_counts = series.value_counts()
        total = len(series)

        colors_list = [self.colors["success"], self.colors["danger"]] + \
                     [self.colors["warning"], self.colors["info"]]

        labels = [str(v) for v in value_counts.head(4).index]
        percentages = [(c / total * 100) for c in value_counts.head(4).values]

        # Horizontal bars with labels on y-axis
        fig.add_trace(go.Bar(
            y=labels,
            x=percentages,
            orientation='h',
            marker_color=colors_list[:len(labels)],
            text=[f"{p:.0f}%" for p in percentages],
            textposition="inside",
            textfont={"size": 11, "color": "white"},
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            showlegend=False
        ), row=row, col=col)

        xref = f"{self._get_axis_ref(row, col, n_cols, 'x')} domain"
        yref = f"{self._get_axis_ref(row, col, n_cols, 'y')} domain"
        if len(value_counts) == 2:
            ratio = value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf')
            ratio_color = (self.colors["success"] if ratio < 3
                          else self.colors["warning"] if ratio < 10
                          else self.colors["danger"])
            fig.add_annotation(
                x=0.98, y=0.98, xref=xref, yref=yref,
                text=f"{ratio:.1f}:1",
                showarrow=False, font={"size": 10, "color": ratio_color, "family": "Arial Black"},
                xanchor="right", yanchor="top"
            )

    def _add_generic_tile(
        self, fig: go.Figure, series: Series, metrics: Dict,
        row: int, col: int, n_cols: int, formatter: "NumberFormatter"
    ) -> None:
        """Add generic tile for unknown column types."""
        distinct = metrics.get('distinct_count', series.nunique())
        null_pct = metrics.get('null_percentage', 0)

        value_counts = series.value_counts().head(5)

        fig.add_trace(go.Bar(
            x=value_counts.values,
            y=[str(v)[:10] for v in value_counts.index],
            orientation='h',
            marker_color=self.colors["primary"],
            hovertemplate="%{y}: %{x:,}<extra></extra>"
        ), row=row, col=col)

    def cutoff_selection_chart(
        self, cutoff_analysis: "CutoffAnalysis", suggested_cutoff: Optional[datetime] = None,
        current_cutoff: Optional[datetime] = None, title: str = "Point-in-Time Cutoff Selection"
    ) -> go.Figure:
        df = cutoff_analysis.to_dataframe()
        if len(df) == 0:
            return go.Figure().add_annotation(text="No temporal data available", showarrow=False)

        # Get data date range to check if cutoffs are within bounds
        min_date = df["date"].min()
        max_date = df["date"].max()

        fig = go.Figure()

        # Add 100% baseline first (invisible, for fill reference)
        fig.add_trace(go.Scatter(
            x=df["date"], y=[100] * len(df), name="_baseline",
            mode="lines", line={"color": "rgba(0,0,0,0)", "width": 0},
            showlegend=False, hoverinfo="skip"
        ))

        # Score area fills from 100% down to train_pct line
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["train_pct"], name="Score Set %",
            mode="lines", line={"color": self.colors["warning"], "width": 2},
            fill="tonexty", fillcolor="rgba(255, 193, 7, 0.3)",
            hovertemplate="Cutoff: %{x|%Y-%m-%d}<br>Score: %{customdata:.1f}%<extra></extra>",
            customdata=df["score_pct"], showlegend=True
        ))

        # Train area fills from train_pct down to 0
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["train_pct"], name="Training Set %",
            mode="lines", line={"color": self.colors["success"], "width": 2},
            fill="tozeroy", fillcolor="rgba(40, 167, 69, 0.3)",
            hovertemplate="Cutoff: %{x|%Y-%m-%d}<br>Train: %{y:.1f}%<extra></extra>",
            showlegend=True
        ))

        milestones = cutoff_analysis.get_percentage_milestones(step=5)
        if milestones:
            milestone_dates = [m["date"] for m in milestones]
            milestone_pcts = [m["train_pct"] for m in milestones]
            fig.add_trace(go.Scatter(
                x=milestone_dates, y=milestone_pcts, name="Train % Reference",
                mode="markers+text", marker={"size": 8, "color": self.colors["success"], "symbol": "circle"},
                text=[f"{int(p)}%" for p in milestone_pcts], textposition="top center",
                textfont={"size": 8, "color": self.colors["success"]},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Train: %{y:.0f}%<extra></extra>",
                showlegend=False
            ))

        # Add cutoff lines - only if within data range
        if suggested_cutoff:
            split = cutoff_analysis.get_split_at_date(suggested_cutoff)
            # Check if suggested cutoff is within data range
            if min_date <= suggested_cutoff <= max_date:
                fig.add_vline(
                    x=suggested_cutoff, line={"color": self.colors["info"], "dash": "dash", "width": 2}
                )
                # Add text annotation label on chart for selected cutoff
                fig.add_annotation(
                    x=suggested_cutoff, y=1.02, xref="x", yref="paper",
                    text=f"Selected: {suggested_cutoff.strftime('%Y-%m-%d')}",
                    showarrow=False, font={"size": 9, "color": self.colors["info"]},
                    xanchor="center", yanchor="bottom"
                )
            # Add legend entry with visible line sample
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line={"color": self.colors["info"], "dash": "dash", "width": 2},
                name=f"Selected: {suggested_cutoff.strftime('%Y-%m-%d')} ({split['train_pct']:.0f}% train)",
                showlegend=True
            ))

        if current_cutoff:
            split = cutoff_analysis.get_split_at_date(current_cutoff)
            # Check if registry cutoff is within data range
            cutoff_in_range = min_date <= current_cutoff <= max_date
            # Determine if registry and selected cutoffs are at the same position
            same_as_selected = suggested_cutoff and current_cutoff == suggested_cutoff
            if cutoff_in_range:
                fig.add_vline(
                    x=current_cutoff, line={"color": self.colors["danger"], "dash": "dot", "width": 2}
                )
                # Add text annotation label on chart for registry cutoff
                # Offset vertically if same as selected to avoid overlap
                annotation_y = 1.08 if same_as_selected else 1.02
                fig.add_annotation(
                    x=current_cutoff, y=annotation_y, xref="x", yref="paper",
                    text=f"Registry: {current_cutoff.strftime('%Y-%m-%d')}",
                    showarrow=False, font={"size": 9, "color": self.colors["danger"]},
                    xanchor="center", yanchor="bottom"
                )
                legend_label = f"Registry: {current_cutoff.strftime('%Y-%m-%d')} ({split['train_pct']:.0f}% train)"
            else:
                # Registry cutoff is outside data range
                legend_label = f"Registry: {current_cutoff.strftime('%Y-%m-%d')} (outside data range)"
            # Add legend entry
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line={"color": self.colors["danger"], "dash": "dot", "width": 2},
                name=legend_label,
                showlegend=True
            ))

        fig.update_layout(
            title={"text": "Train/Score Split by Cutoff Date", "x": 0.5, "xanchor": "center"},
            width=800, height=300, autosize=False, template=self.theme, showlegend=True,
            legend={
                "orientation": "h", "yanchor": "top", "y": -0.15,
                "xanchor": "center", "x": 0.5, "bgcolor": "rgba(255,255,255,0.8)",
                "font": {"size": 9}
            },
            margin={"t": 40, "b": 60, "l": 55, "r": 55},
            yaxis={"title": "Percentage", "range": [0, 100]},
            xaxis={"title": ""},
        )

        return fig
