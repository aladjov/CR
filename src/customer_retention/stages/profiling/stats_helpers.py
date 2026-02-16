from customer_retention.core.compat import DataFrame


def calculate_group_retention_stats(
    df: DataFrame,
    group_col: str,
    target_col: str,
    overall_rate: float,
    min_samples: int = 0,
    sort_by: str = "group",
    ascending: bool = True,
) -> DataFrame:
    stats = df.groupby(group_col)[target_col].agg(["sum", "count", "mean"]).reset_index()
    stats.columns = ["group", "retained_count", "count", "retention_rate"]
    stats["lift"] = stats["retention_rate"] / overall_rate if overall_rate > 0 else 0
    if min_samples > 0:
        stats = stats[stats["count"] >= min_samples]
    return stats.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
