"""Map a raw feature value to an ordinal band via quantile lookup.

Consumed by two places (plan §2):

- **Eligibility rule prose**: ``predicate_compiler`` renders a SQL predicate;
  the prose sibling uses this function to say "NPS is very low (≤ 4)"
  instead of "nps_score <= 4".
- **Driver salience**: the ``mean_value`` carried on ``top_positive_drivers``
  is lifted through the same band.

Bands are fixed (plan §2): ``< q05`` → ``very low``, ``q05..q25`` → ``low``,
``q25..q75`` → ``typical``, ``q75..q95`` → ``elevated``, ``> q95`` → ``very high``.

Polarity inversion is cosmetic: when ``polarity == "high_is_bad"`` the band
maps invert so "very high NPS" vs "very high churn_days" read naturally —
the *rank* is unchanged, only the word the narrator uses. Callers that
want the raw ordinal should pass ``polarity="neutral"`` or leave it unset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PopulationStats:
    """Minimum subset of ``PopulationStatsRow`` that the phraser needs.

    Decoupled from the Delta-row dataclass so renderers and tests don't
    need to build a full population-stats object for every call.
    """
    q05: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    q95: Optional[float] = None


_NEUTRAL_BANDS = ("very low", "low", "typical", "elevated", "very high")
_INVERTED_BANDS = ("very high", "elevated", "typical", "low", "very low")


def quantile_phrase(
    value: Optional[float],
    stats: PopulationStats,
    polarity: str = "neutral",
) -> str:
    """Return the ordinal band for ``value`` given ``stats``.

    ``None`` value or stats missing any of q05/q25/q75/q95 → ``"unknown"`` —
    the renderer degrades gracefully. Polarity ``"high_is_bad"`` inverts
    the band direction for narrative use; every other polarity is neutral.
    """
    if value is None or not _has_required_quantiles(stats):
        return "unknown"
    bands = _INVERTED_BANDS if polarity == "high_is_bad" else _NEUTRAL_BANDS
    if value < stats.q05:  # type: ignore[operator]
        return bands[0]
    if value < stats.q25:  # type: ignore[operator]
        return bands[1]
    if value <= stats.q75:  # type: ignore[operator]
        return bands[2]
    if value <= stats.q95:  # type: ignore[operator]
        return bands[3]
    return bands[4]


def _has_required_quantiles(stats: PopulationStats) -> bool:
    return all(q is not None for q in (stats.q05, stats.q25, stats.q75, stats.q95))
