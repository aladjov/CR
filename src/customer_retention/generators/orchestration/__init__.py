# Import context first to avoid circular imports
from .context import ContextManager, PipelineContext, setup_notebook_context
from .data_materializer import DataMaterializer
from .databricks_exporter import DatabricksExporter
from .doc_generator import PipelineDocGenerator

# Deferred import - code_generator has heavy dependencies
def __getattr__(name):
    if name == "PipelineCodeGenerator":
        from .code_generator import PipelineCodeGenerator
        return PipelineCodeGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PipelineContext",
    "ContextManager",
    "setup_notebook_context",
    "PipelineCodeGenerator",
    "PipelineDocGenerator",
    "DataMaterializer",
    "DatabricksExporter",
]
