"""Console output utilities using Markdown for rich formatting."""

from typing import Any, Dict, List, Optional

try:
    from IPython.display import Markdown, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

_buffer: List[str] = []
_auto_flush = True


def _add(line: str) -> None:
    if _auto_flush:
        if HAS_IPYTHON:
            display(Markdown(line))
        else:
            print(line)
    else:
        _buffer.append(line)


def _bar(value: float, width: int = 20) -> str:
    value = max(0, min(100, value))
    filled = int(value / 100 * width)
    if value >= 100:
        return "█" * width
    return "█" * filled + "░" * (width - filled)


def start_section() -> None:
    global _auto_flush
    _auto_flush = False
    _buffer.clear()


def end_section() -> None:
    global _auto_flush
    if _buffer:
        text = "  \n".join(_buffer)
        if HAS_IPYTHON:
            display(Markdown(text))
        else:
            print("\n".join(_buffer))
    _buffer.clear()
    _auto_flush = True


def header(text: str) -> None:
    _add(f"#### {text.upper()}")


def subheader(text: str) -> None:
    _add(f"**{text}**")


def success(text: str) -> None:
    _add(f"[OK] {text}")


def warning(text: str) -> None:
    _add(f"[!] {text}")


def error(text: str) -> None:
    _add(f"[X] {text}")


def info(text: str) -> None:
    _add(f"*(i) {text}*")


def metric(label: str, value: Any) -> None:
    _add(f"{label}: **{value}**")


def kv(data: Dict[str, Any], inline: bool = False) -> None:
    if inline:
        parts = [f"{k}: **{v}**" for k, v in data.items()]
        _add(" | ".join(parts))
    else:
        for k, v in data.items():
            _add(f"{k}: **{v}**")


def bullets(items: List[str]) -> None:
    for item in items:
        _add(f"- {item}")


def numbers(items: List[str]) -> None:
    for i, item in enumerate(items, 1):
        _add(f"{i}. {item}")


def score(value: float, label: str = "Score") -> None:
    rating = "Excellent" if value >= 90 else "Good" if value >= 70 else "Fair" if value >= 50 else "Poor"
    _add(f"{label}: `{_bar(value, 25)}` **{value:.0f}/100** ({rating})")


def progress(label: str, value: float) -> None:
    _add(f"{label}: `{_bar(value, 15)}` **{value:.1f}%**")


def check(name: str, passed: bool, detail: Optional[str] = None) -> None:
    icon = "[OK]" if passed else "[X]"
    line = f"{icon} {name}"
    if detail:
        line += f" — {detail}"
    _add(line)


def overview(rows: int, cols: int, memory_mb: float, completeness: float,
             target: Optional[str] = None) -> None:
    _add(f"Rows: **{rows:,}**")
    _add(f"Columns: **{cols}**")
    _add(f"Memory: **{memory_mb:.1f} MB**")
    _add(f"Completeness: `{_bar(completeness, 15)}` **{completeness:.1f}%**")
    if target:
        _add(f"Target: **{target}**")
