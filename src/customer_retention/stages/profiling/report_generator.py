import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from customer_retention.core.compat import pd, DataFrame
from customer_retention.core.config import ColumnType
from .profile_result import ProfileResult


class ReportGenerator:
    """Generate profiling reports in multiple formats (JSON, HTML, Markdown)."""

    def __init__(self, profile: Optional[ProfileResult] = None):
        self.profile = profile

    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report from profile result."""
        if self.profile is None:
            raise ValueError("No profile set. Provide profile in constructor or set profile attribute.")

        # Convert profile to dict using Pydantic's model_dump
        report_dict = self.profile.model_dump()

        return json.dumps(report_dict, indent=indent, default=str)

    def save_json(self, filepath: str):
        """Save JSON report to file."""
        json_report = self.to_json()

        with open(filepath, 'w') as f:
            f.write(json_report)

    def to_html(self) -> str:
        """Generate HTML report from profile result."""
        if self.profile is None:
            raise ValueError("No profile set.")

        summary = self.generate_executive_summary()

        html = self._generate_html_template(summary)

        return html

    def save_html(self, filepath: str):
        """Save HTML report to file."""
        html_report = self.to_html()

        with open(filepath, 'w') as f:
            f.write(html_report)

    def to_markdown(self) -> str:
        """Generate Markdown report from profile result."""
        if self.profile is None:
            raise ValueError("No profile set.")

        summary = self.generate_executive_summary()

        md = self._generate_markdown_template(summary)

        return md

    def save_markdown(self, filepath: str):
        """Save Markdown report to file."""
        md_report = self.to_markdown()

        with open(filepath, 'w') as f:
            f.write(md_report)

    def save_all_formats(self, directory: str, base_filename: str):
        """Save reports in all formats to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.save_json(str(dir_path / f"{base_filename}.json"))
        self.save_html(str(dir_path / f"{base_filename}.html"))
        self.save_markdown(str(dir_path / f"{base_filename}.md"))

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of profiling results."""
        if self.profile is None:
            raise ValueError("No profile set.")

        # Basic dataset info
        summary = {
            "dataset_name": self.profile.dataset_name,
            "total_rows": self.profile.total_rows,
            "total_columns": self.profile.total_columns,
            "profiling_timestamp": self.profile.profiling_timestamp,
            "profiling_duration_seconds": self.profile.profiling_duration_seconds,
        }

        # Column type breakdown
        type_counts = {}
        for col_profile in self.profile.column_profiles.values():
            col_type = col_profile.configured_type.value
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        summary["column_types"] = type_counts

        # Missing data summary
        total_missing = 0
        columns_with_missing = 0

        for col_profile in self.profile.column_profiles.values():
            if col_profile.universal_metrics.null_count > 0:
                columns_with_missing += 1
                total_missing += col_profile.universal_metrics.null_count

        total_cells = self.profile.total_rows * self.profile.total_columns
        missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0

        summary["total_missing_cells"] = total_missing
        summary["columns_with_missing"] = columns_with_missing
        summary["missing_percentage"] = round(missing_percentage, 2)

        # Quality score calculation (0-100)
        quality_score = self._calculate_quality_score()
        summary["quality_score"] = quality_score

        # Memory usage estimate
        total_memory = sum(
            col_profile.universal_metrics.memory_size_bytes
            for col_profile in self.profile.column_profiles.values()
            if hasattr(col_profile.universal_metrics, 'memory_size_bytes') and
               col_profile.universal_metrics.memory_size_bytes is not None
        )
        summary["estimated_memory_mb"] = round(total_memory / (1024 * 1024), 2) if total_memory > 0 else 0.0

        return summary

    def _calculate_quality_score(self) -> int:
        """Calculate overall data quality score (0-100)."""
        if not self.profile or not self.profile.column_profiles:
            return 0

        penalties = 0
        max_penalties = 100

        for col_profile in self.profile.column_profiles.values():
            metrics = col_profile.universal_metrics

            # Penalize missing values
            if metrics.null_percentage > 50:
                penalties += 20
            elif metrics.null_percentage > 20:
                penalties += 10
            elif metrics.null_percentage > 5:
                penalties += 5

            # Penalize constant columns
            if metrics.distinct_count == 1:
                penalties += 15

            # Penalize very high cardinality (possible identifiers)
            if metrics.distinct_percentage > 95 and col_profile.configured_type not in [
                ColumnType.IDENTIFIER, ColumnType.TEXT
            ]:
                penalties += 5

        # Cap penalties at max
        penalties = min(penalties, max_penalties)

        return max(0, 100 - penalties)

    def calculate_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate correlation matrix for numeric columns."""
        if self.profile is None:
            return None

        # Get numeric columns from profile
        numeric_columns = [
            col_name for col_name, col_profile in self.profile.column_profiles.items()
            if col_profile.configured_type in [
                ColumnType.NUMERIC_CONTINUOUS,
                ColumnType.NUMERIC_DISCRETE
            ]
        ]

        if len(numeric_columns) < 2:
            return None

        # Filter dataframe to numeric columns that exist
        numeric_cols_in_df = [col for col in numeric_columns if col in df.columns]

        if len(numeric_cols_in_df) < 2:
            return None

        # Calculate correlations
        corr_matrix = df[numeric_cols_in_df].corr()

        # Convert to dictionary
        correlations = {
            "matrix": corr_matrix.to_dict(),
            "high_correlations": []
        }

        # Find high correlations (>0.8 or <-0.8)
        for i, col1 in enumerate(numeric_cols_in_df):
            for col2 in numeric_cols_in_df[i + 1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) > 0.8:
                    correlations["high_correlations"].append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(corr_value, 3)
                    })

        return correlations

    def _generate_html_template(self, summary: Dict[str, Any]) -> str:
        """Generate HTML report template."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profiling Report - {self.profile.dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #333;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .quality-score {{
            font-size: 3em;
            font-weight: bold;
            color: {self._get_quality_color(summary['quality_score'])};
            text-align: center;
        }}
        .column-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .column-header {{
            font-size: 1.3em;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .column-type {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 10px;
            border-radius: 5px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: #667eea;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Profiling Report</h1>
        <p>{self.profile.dataset_name}</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Generated on {summary['profiling_timestamp']}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div class="metric">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{summary['total_rows']:,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{summary['total_columns']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Missing Data</div>
                <div class="metric-value">{summary['missing_percentage']}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Quality Score</div>
                <div class="quality-score">{summary['quality_score']}</div>
            </div>
        </div>

        <h3>Column Types</h3>
        <table>
            <tr>
                <th>Type</th>
                <th>Count</th>
            </tr>
"""

        for col_type, count in summary['column_types'].items():
            html += f"""            <tr>
                <td>{col_type}</td>
                <td>{count}</td>
            </tr>
"""

        html += """        </table>
    </div>

    <h2>Column Details</h2>
"""

        # Add column sections
        for col_name, col_profile in self.profile.column_profiles.items():
            html += self._generate_column_section_html(col_name, col_profile)

        html += """
</body>
</html>"""

        return html

    def _generate_column_section_html(self, col_name: str, col_profile) -> str:
        """Generate HTML section for a single column."""
        metrics = col_profile.universal_metrics

        html = f"""
    <div class="column-section">
        <div class="column-header">
            {col_name}
            <span class="column-type">{col_profile.configured_type.value}</span>
        </div>

        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Count</td>
                <td>{metrics.total_count:,}</td>
            </tr>
            <tr>
                <td>Missing Values</td>
                <td>{metrics.null_count:,} ({metrics.null_percentage:.1f}%)</td>
            </tr>
            <tr>
                <td>Unique Values</td>
                <td>{metrics.distinct_count:,} ({metrics.distinct_percentage:.1f}%)</td>
            </tr>
"""

        # Add type-specific metrics
        if col_profile.numeric_metrics:
            nm = col_profile.numeric_metrics
            html += f"""            <tr>
                <td>Mean</td>
                <td>{nm.mean:.2f}</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>{nm.std:.2f}</td>
            </tr>
            <tr>
                <td>Min / Max</td>
                <td>{nm.min_value:.2f} / {nm.max_value:.2f}</td>
            </tr>
"""

        elif col_profile.categorical_metrics:
            cm = col_profile.categorical_metrics
            top_cats = ', '.join(f"{cat}({count})" for cat, count in cm.top_categories[:5])
            html += f"""            <tr>
                <td>Cardinality</td>
                <td>{cm.cardinality}</td>
            </tr>
            <tr>
                <td>Top Categories</td>
                <td>{top_cats}</td>
            </tr>
"""

        html += """        </table>
    </div>
"""

        return html

    def _generate_markdown_template(self, summary: Dict[str, Any]) -> str:
        """Generate Markdown report template."""
        md = f"""# Data Profiling Report

## Dataset: {self.profile.dataset_name}

**Generated:** {summary['profiling_timestamp']}
**Duration:** {summary['profiling_duration_seconds']:.2f} seconds

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Rows | {summary['total_rows']:,} |
| Total Columns | {summary['total_columns']} |
| Missing Data | {summary['missing_percentage']}% |
| Quality Score | **{summary['quality_score']}/100** |

### Column Types

| Type | Count |
|------|-------|
"""

        for col_type, count in summary['column_types'].items():
            md += f"| {col_type} | {count} |\n"

        md += "\n---\n\n## Column Details\n\n"

        # Add column sections
        for col_name, col_profile in self.profile.column_profiles.items():
            md += self._generate_column_section_markdown(col_name, col_profile)

        return md

    def _generate_column_section_markdown(self, col_name: str, col_profile) -> str:
        """Generate Markdown section for a single column."""
        metrics = col_profile.universal_metrics

        md = f"""### {col_name} `({col_profile.configured_type.value})`

| Metric | Value |
|--------|-------|
| Total Count | {metrics.total_count:,} |
| Missing Values | {metrics.null_count:,} ({metrics.null_percentage:.1f}%) |
| Unique Values | {metrics.distinct_count:,} ({metrics.distinct_percentage:.1f}%) |
"""

        # Add type-specific metrics
        if col_profile.numeric_metrics:
            nm = col_profile.numeric_metrics
            md += f"""| Mean | {nm.mean:.2f} |
| Std Dev | {nm.std:.2f} |
| Min / Max | {nm.min_value:.2f} / {nm.max_value:.2f} |
| Median | {nm.median:.2f} |
"""

        elif col_profile.categorical_metrics:
            cm = col_profile.categorical_metrics
            top_cats = ', '.join(f"{cat}({count})" for cat, count in cm.top_categories[:5])
            md += f"""| Cardinality | {cm.cardinality} |
| Top Categories | {top_cats} |
"""

        md += "\n"

        return md

    def _get_quality_color(self, score: int) -> str:
        """Get color based on quality score."""
        if score >= 90:
            return "#28a745"  # Green
        elif score >= 70:
            return "#ffc107"  # Yellow
        elif score >= 50:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
