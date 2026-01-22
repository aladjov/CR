from typing import Tuple


class NumberFormatter:
    SUFFIXES: Tuple[Tuple[float, str], ...] = (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )

    def compact(self, value: float, precision: int = 2) -> str:
        if value == 0:
            return f"{value:.{precision}f}"

        sign = "-" if value < 0 else ""
        abs_value = abs(value)

        for threshold, suffix in self.SUFFIXES:
            if abs_value >= threshold:
                scaled = abs_value / threshold
                return f"{sign}{scaled:.{precision}f}{suffix}"

        return f"{sign}{abs_value:.{precision}f}"

    def percentage(self, value: float, precision: int = 2, show_sign: bool = True) -> str:
        sign = "+" if value >= 0 and show_sign else ""
        return f"{sign}{value:.{precision}f}%"

    def rate(self, value: float, suffix: str, precision: int = 2) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.{precision}f}{suffix}"

    def plotly_format(self, precision: int = 2, show_sign: bool = True) -> str:
        sign_char = "+" if show_sign else ""
        return f"{sign_char}.{precision}f"
