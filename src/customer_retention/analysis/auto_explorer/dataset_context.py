from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from customer_retention.core.compat import pd
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.profiling.relationship_detector import RelationshipDetector
from customer_retention.stages.profiling.type_detector import TypeDetector


@dataclass
class DatasetEntry:
    path: str
    has_target: bool
    entity_column: Optional[str] = None
    target_column: Optional[str] = None
    join_key: Optional[str] = None
    join_to: Optional[str] = None
    relationship: Optional[str] = None


@dataclass
class DatasetContext:
    target_dataset: Optional[str] = None
    target_column: Optional[str] = None
    entity_column: Optional[str] = None
    cutoff_date: Optional[_dt.datetime] = None
    join_dataset_name: Optional[str] = None
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        data = {
            "target_dataset": self.target_dataset,
            "target_column": self.target_column,
            "entity_column": self.entity_column,
            "datasets": {
                name: {k: v for k, v in entry.__dict__.items() if v is not None}
                for name, entry in self.datasets.items()
            },
        }
        if self.cutoff_date is not None:
            naive = self.cutoff_date.replace(tzinfo=None) if self.cutoff_date.tzinfo else self.cutoff_date
            data["cutoff_date"] = naive.isoformat()
        if self.join_dataset_name is not None:
            data["join_dataset_name"] = self.join_dataset_name
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> DatasetContext:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset context file not found: {p}")
        data = yaml.safe_load(p.read_text())
        datasets = {}
        for name, entry_data in (data.get("datasets") or {}).items():
            datasets[name] = DatasetEntry(
                path=entry_data.get("path", ""),
                has_target=entry_data.get("has_target", False),
                entity_column=entry_data.get("entity_column"),
                target_column=entry_data.get("target_column"),
                join_key=entry_data.get("join_key"),
                join_to=entry_data.get("join_to"),
                relationship=entry_data.get("relationship"),
            )
        raw_cutoff = data.get("cutoff_date")
        cutoff_date = None
        if raw_cutoff is not None:
            if isinstance(raw_cutoff, str):
                cutoff_date = _dt.datetime.fromisoformat(raw_cutoff)
            elif isinstance(raw_cutoff, (_dt.datetime, _dt.date)):
                cutoff_date = _dt.datetime(raw_cutoff.year, raw_cutoff.month, raw_cutoff.day)
            if cutoff_date and cutoff_date.tzinfo:
                cutoff_date = cutoff_date.replace(tzinfo=None)
        return cls(
            target_dataset=data.get("target_dataset"),
            target_column=data.get("target_column"),
            entity_column=data.get("entity_column"),
            cutoff_date=cutoff_date,
            join_dataset_name=data.get("join_dataset_name"),
            datasets=datasets,
        )

    @classmethod
    def for_single_dataset(
        cls,
        path: str | Path,
        dataset_name: str,
        target_column: Optional[str] = None,
        entity_column: Optional[str] = None,
        cutoff_date: Optional[_dt.datetime] = None,
        join_dataset_name: Optional[str] = None,
    ) -> DatasetContext:
        entry = DatasetEntry(
            path=str(path),
            has_target=target_column is not None,
            target_column=target_column,
            entity_column=entity_column,
        )
        return cls(
            target_dataset=dataset_name if target_column else None,
            target_column=target_column,
            entity_column=entity_column,
            cutoff_date=cutoff_date,
            join_dataset_name=join_dataset_name,
            datasets={dataset_name: entry},
        )


class DatasetContextScanner:
    def __init__(self, nrows: int = 1000):
        self.nrows = nrows
        self.type_detector = TypeDetector()
        self.relationship_detector = RelationshipDetector()

    def scan(self, dataset_paths: list[str | Path]) -> DatasetContext:
        frames: dict[str, pd.DataFrame] = {}
        for p in dataset_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found: {p}")
            frames[p.stem] = pd.read_csv(str(p), nrows=self.nrows)

        target_name, target_col, entity_col = self._find_target_and_entity(frames)

        datasets: dict[str, DatasetEntry] = {}
        for name, df in frames.items():
            if name == target_name:
                datasets[name] = DatasetEntry(
                    path=str(self._find_path(dataset_paths, name)),
                    has_target=True, target_column=target_col, entity_column=entity_col,
                )
            else:
                join_key, join_to, relationship = self._detect_join(
                    df, name, frames, target_name,
                )
                datasets[name] = DatasetEntry(
                    path=str(self._find_path(dataset_paths, name)),
                    has_target=False, join_key=join_key, join_to=join_to,
                    relationship=relationship,
                )

        return DatasetContext(
            target_dataset=target_name, target_column=target_col,
            entity_column=entity_col, datasets=datasets,
        )

    def _find_target_and_entity(
        self, frames: dict[str, pd.DataFrame],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        for name, df in frames.items():
            candidates = []
            entity_col = None
            for col in df.columns:
                inference = self.type_detector.detect_type(df[col], col)
                if inference.inferred_type == ColumnType.TARGET:
                    candidates.append((col, self._target_score(df[col])))
                if inference.inferred_type == ColumnType.IDENTIFIER and entity_col is None:
                    entity_col = col
            if candidates:
                best = max(candidates, key=lambda c: c[1])[0]
                return name, best, entity_col
        return None, None, None

    @staticmethod
    def _target_score(series: pd.Series) -> float:
        from customer_retention.core.compat import is_numeric_dtype
        nunique = series.dropna().nunique()
        score = 0.0
        if is_numeric_dtype(series):
            score += 10
        if nunique == 2:
            score += 5
        score -= nunique
        return score

    def _detect_join(
        self, df: pd.DataFrame, name: str,
        frames: dict[str, pd.DataFrame], target_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if target_name is None or target_name not in frames:
            return None, None, None
        target_df = frames[target_name]
        rel = self.relationship_detector.detect(df, target_df, df1_name=name, df2_name=target_name)
        if rel.suggested_join is None:
            return None, None, None
        return rel.suggested_join.left_column, target_name, rel.relationship_type.value

    @staticmethod
    def _find_path(dataset_paths: list[str | Path], stem: str) -> Path:
        for p in dataset_paths:
            if Path(p).stem == stem:
                return Path(p)
        return Path(stem)


_LABEL_COLS = ("label_timestamp", "label_available_flag")


def mount_target_column(
    df: pd.DataFrame, context: DatasetContext, dataset_name: str,
    target_label_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Optional[dict]]:
    if not context.target_dataset or not context.target_column:
        return df, None

    entry = context.datasets.get(dataset_name)
    if entry is None or entry.has_target or not entry.join_key:
        return df, None

    target_entry = context.datasets.get(context.target_dataset)
    if target_entry is None:
        return df, None

    left_key = _resolve_left_key(entry.join_key, df)
    if left_key is None:
        return df, None

    mount_df, label_mounted = _build_mount_df(
        target_label_df, target_entry, context,
    )

    drop_before_merge = [c for c in _LABEL_COLS if label_mounted and c in df.columns]
    merge_input = df.drop(columns=drop_before_merge) if drop_before_merge else df
    result = merge_input.merge(mount_df, left_on=left_key, right_on=context.entity_column, how="left")

    if entry.join_key != context.entity_column:
        extra = f"{context.entity_column}_y"
        if extra in result.columns:
            result = result.drop(columns=[extra])

    info = {
        "target_mounted_from": context.target_dataset,
        "target_mount_key": entry.join_key,
    }
    if label_mounted:
        info["label_timestamp_mounted"] = True
    return result, info


def _resolve_left_key(join_key: str, df: pd.DataFrame) -> Optional[str]:
    if join_key in df.columns:
        return join_key
    if "entity_id" in df.columns:
        return "entity_id"
    return None


def _build_mount_df(
    target_label_df: Optional[pd.DataFrame],
    target_entry: DatasetEntry, context: DatasetContext,
) -> tuple[pd.DataFrame, bool]:
    if target_label_df is not None:
        return _mount_df_from_label_source(target_label_df, context)
    usecols = [context.entity_column, context.target_column]
    target_df = pd.read_csv(target_entry.path, usecols=usecols)
    return target_df.drop_duplicates(subset=[context.entity_column]), False


def _mount_df_from_label_source(
    target_label_df: pd.DataFrame, context: DatasetContext,
) -> tuple[pd.DataFrame, bool]:
    entity_col = "entity_id" if "entity_id" in target_label_df.columns else context.entity_column
    target_col = "target" if "target" in target_label_df.columns else context.target_column
    keep = [entity_col, target_col]
    available_label_cols = [c for c in _LABEL_COLS if c in target_label_df.columns]
    keep.extend(available_label_cols)
    mount_df = target_label_df[keep].drop_duplicates(subset=[entity_col])
    if entity_col != context.entity_column:
        mount_df = mount_df.rename(columns={entity_col: context.entity_column})
    if target_col != context.target_column:
        mount_df = mount_df.rename(columns={target_col: context.target_column})
    return mount_df, len(available_label_cols) > 0
