from __future__ import annotations

from typing import Iterable, List


class ScoringSpecMismatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        missing: Iterable[str] = (),
        extra: Iterable[str] = (),
        model_uri: str = "",
    ) -> None:
        self.missing: List[str] = sorted(missing)
        self.extra: List[str] = sorted(extra)
        self.model_uri = model_uri
        super().__init__(message)

    @classmethod
    def from_diff(
        cls,
        spec_features: Iterable[str],
        actual_features: Iterable[str],
        *,
        model_uri: str = "",
        context: str = "model",
    ) -> "ScoringSpecMismatchError":
        spec_set = set(spec_features)
        actual_set = set(actual_features)
        missing = sorted(spec_set - actual_set)
        extra = sorted(actual_set - spec_set)
        message = (
            f"FeatureSpec parity violation in {context}{f' {model_uri}' if model_uri else ''}:\n"
            f"  missing: {missing[:10]}{'…' if len(missing) > 10 else ''} ({len(missing)} total)\n"
            f"  extra:   {extra[:10]}{'…' if len(extra) > 10 else ''} ({len(extra)} total)\n"
            "Production training did not consume the spec — re-run training (NB10 → run_all)."
        )
        return cls(message, missing=missing, extra=extra, model_uri=model_uri)
