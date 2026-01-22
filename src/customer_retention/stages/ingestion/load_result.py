from pydantic import BaseModel


class LoadResult(BaseModel):
    success: bool
    row_count: int
    column_count: int
    duration_seconds: float
    source_name: str
    warnings: list[str] = []
    errors: list[str] = []
    schema_info: dict[str, str] = {}

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def get_summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"{status}: {self.source_name} - "
            f"{self.row_count} rows, {self.column_count} columns "
            f"({self.duration_seconds:.2f}s)"
        )
