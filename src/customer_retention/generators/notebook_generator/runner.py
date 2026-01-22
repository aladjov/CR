from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import time
import nbformat


@dataclass
class NotebookValidationResult:
    notebook_name: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None
    cell_errors: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    results: List[NotebookValidationResult]
    platform: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def all_passed(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def total_notebooks(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def total_duration_seconds(self) -> float:
        return sum(r.duration_seconds for r in self.results)

    def to_markdown(self) -> str:
        lines = [
            f"# Notebook Validation Report - {self.platform.upper()}",
            f"**Timestamp:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Duration:** {self.total_duration_seconds:.2f}s",
            "",
            "## Summary",
            f"- **Total Notebooks:** {self.total_notebooks}",
            f"- **Passed:** {self.passed_count}",
            f"- **Failed:** {self.failed_count}",
            f"- **Status:** {'PASSED' if self.all_passed else 'FAILED'}",
            "",
            "## Results",
            "| Notebook | Status | Duration | Error |",
            "|----------|--------|----------|-------|",
        ]
        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            error = r.error[:50] + "..." if r.error and len(r.error) > 50 else (r.error or "-")
            lines.append(f"| {r.notebook_name} | {status} | {r.duration_seconds:.2f}s | {error} |")
        return "\n".join(lines)


class NotebookRunner:
    def __init__(self, dry_run: bool = False, stop_on_failure: bool = False):
        self.dry_run = dry_run
        self.stop_on_failure = stop_on_failure

    def validate_syntax(self, code: str) -> bool:
        try:
            compile(code, "<notebook>", "exec")
            return True
        except SyntaxError:
            return False

    def extract_code(self, notebook_path: str) -> str:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    def validate_notebook(self, notebook_path: str) -> NotebookValidationResult:
        notebook_name = Path(notebook_path).stem
        start_time = time.time()
        try:
            code = self.extract_code(notebook_path)
            if self.validate_syntax(code):
                return NotebookValidationResult(notebook_name, True, time.time() - start_time)
            return NotebookValidationResult(notebook_name, False, time.time() - start_time, error="Syntax validation failed")
        except Exception as e:
            return NotebookValidationResult(notebook_name, False, time.time() - start_time, error=str(e))

    def validate_sequence(self, notebooks_dir: str, platform: str) -> ValidationReport:
        notebook_files = sorted(Path(notebooks_dir).glob("*.ipynb"))
        results = []
        for nb_path in notebook_files:
            result = self.validate_notebook(str(nb_path))
            results.append(result)
            if self.stop_on_failure and not result.success:
                break
        return ValidationReport(results=results, platform=platform)


def validate_generated_notebooks(output_dir: str, platforms: Optional[List[str]] = None) -> dict:
    if platforms is None:
        platforms = ["local", "databricks"]
    runner = NotebookRunner(dry_run=True)
    reports = {}
    for platform in platforms:
        platform_dir = Path(output_dir) / platform
        if platform_dir.exists():
            reports[platform] = runner.validate_sequence(str(platform_dir), platform)
    return reports


class ScriptRunner:
    def __init__(self, dry_run: bool = False, stop_on_failure: bool = False):
        self.dry_run = dry_run
        self.stop_on_failure = stop_on_failure

    def validate_syntax(self, code: str) -> bool:
        try:
            compile(code, "<script>", "exec")
            return True
        except SyntaxError:
            return False

    def validate_script(self, script_path: str) -> NotebookValidationResult:
        script_name = Path(script_path).stem
        start_time = time.time()
        try:
            code = Path(script_path).read_text(encoding="utf-8")
            if self.validate_syntax(code):
                return NotebookValidationResult(script_name, True, time.time() - start_time)
            return NotebookValidationResult(script_name, False, time.time() - start_time, error="Syntax validation failed")
        except Exception as e:
            return NotebookValidationResult(script_name, False, time.time() - start_time, error=str(e))

    def validate_sequence(self, scripts_dir: str, platform: str) -> ValidationReport:
        script_files = sorted(Path(scripts_dir).glob("*.py"))
        results = []
        for script_path in script_files:
            result = self.validate_script(str(script_path))
            results.append(result)
            if self.stop_on_failure and not result.success:
                break
        return ValidationReport(results=results, platform=platform)
