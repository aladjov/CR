from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AdapterResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
