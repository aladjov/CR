from abc import ABC, abstractmethod
from typing import List, Optional
import nbformat

from ..config import NotebookConfig, Platform
from ..base import NotebookStage
from ..cell_builder import CellBuilder


class StageGenerator(ABC):
    def __init__(self, config: NotebookConfig, findings: Optional["ExplorationFindings"]):
        self.config = config
        self.findings = findings
        self.cb = CellBuilder

    @property
    @abstractmethod
    def stage(self) -> NotebookStage:
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @property
    def description(self) -> str:
        return ""

    def generate(self, platform: Platform) -> List[nbformat.NotebookNode]:
        if platform == Platform.LOCAL:
            return self.generate_local_cells()
        return self.generate_databricks_cells()

    @abstractmethod
    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        pass

    @abstractmethod
    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        pass

    def header_cells(self) -> List[nbformat.NotebookNode]:
        cells = [self.cb.header(self.title)]
        if self.description:
            cells.append(self.cb.markdown(self.description))
        return cells

    def get_target_column(self) -> str:
        if self.findings and hasattr(self.findings, "target_column") and self.findings.target_column:
            return self.findings.target_column
        return "target"

    def get_identifier_columns(self) -> List[str]:
        if self.findings and hasattr(self.findings, "identifier_columns") and self.findings.identifier_columns:
            return self.findings.identifier_columns
        return ["customer_id"]

    def get_feature_columns(self) -> List[str]:
        if not self.findings or not hasattr(self.findings, "columns"):
            return []
        from customer_retention.core.config import ColumnType
        feature_types = {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE,
                        ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.BINARY}
        return [name for name, col in self.findings.columns.items()
                if hasattr(col, "inferred_type") and col.inferred_type in feature_types]

    def get_numeric_columns(self) -> List[str]:
        if not self.findings or not hasattr(self.findings, "columns"):
            return []
        from customer_retention.core.config import ColumnType
        return [name for name, col in self.findings.columns.items()
                if hasattr(col, "inferred_type") and col.inferred_type in
                {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}]

    def get_categorical_columns(self) -> List[str]:
        if not self.findings or not hasattr(self.findings, "columns"):
            return []
        from customer_retention.core.config import ColumnType
        return [name for name, col in self.findings.columns.items()
                if hasattr(col, "inferred_type") and col.inferred_type in
                {ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL}]
