from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from pydantic import BaseModel, Field


class SectionRecord(BaseModel):
    section_id: str
    signals: dict[str, int]
    why: dict[str, list[str]]
    positives: list[str] = Field(default_factory=list)
    negatives: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ObjectiveSynthesis(BaseModel):
    combined_strength: int
    drivers: list[tuple[str, int]] = Field(default_factory=list)
    frictions: list[tuple[str, int]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    confidence: int = 0


_DEFAULT_OBJECTIVES = ("ImmediateRisk", "Renewal", "Disengagement")
_MAX_WHY = 5
_BAR_WIDTH = 3
_FULL = "█"
_EMPTY = "░"


def signal_bar(strength: int) -> str:
    clamped = max(0, min(_BAR_WIDTH, strength))
    return _FULL * clamped + _EMPTY * (_BAR_WIDTH - clamped)


@dataclass
class SignalRule:
    active: bool
    decrement: int = 0
    cap: int | None = field(default=None)
    why: str = ""


def apply_signal_rules(base: int, rules: list[SignalRule], default_why: str = "") -> tuple[int, list[str]]:
    signal = base
    whys: list[str] = []
    for rule in rules:
        if rule.active:
            if rule.decrement:
                signal = max(0, signal - rule.decrement)
            if rule.cap is not None:
                signal = min(signal, rule.cap)
            if rule.why:
                whys.append(rule.why)
    if not whys and default_why:
        whys.append(default_why)
    return max(0, min(3, signal)), whys


def collect_indicators(indicators: list[tuple[bool, str, str]]) -> tuple[list[str], list[str]]:
    positives = [pos for active, pos, _ in indicators if not active and pos]
    negatives = [neg for active, _, neg in indicators if active and neg]
    return positives, negatives


def _display_output(text: str) -> None:
    try:
        from IPython.display import HTML, display
        display(HTML(f"<pre>{text}</pre>"))
    except ImportError:
        print(text)


class ObjectiveSupportCommunicator:
    def __init__(self, objectives: Optional[tuple[str, ...]] = None):
        self._objectives = objectives or _DEFAULT_OBJECTIVES
        self._sections: dict[str, SectionRecord] = {}
        self._section_order: list[str] = []

    def record_section(
        self, section_id: str, signals: dict[str, int], why: dict[str, list[str]],
        positives: Optional[list[str]] = None, negatives: Optional[list[str]] = None,
        gaps: Optional[list[str]] = None,
    ) -> None:
        for obj_name in signals:
            if obj_name not in self._objectives:
                raise ValueError(f"Unknown objective: {obj_name}. Valid: {self._objectives}")
        for obj_name, strength in signals.items():
            if strength < 0 or strength > 3:
                raise ValueError(f"Signal strength must be 0-3, got {strength} for {obj_name}")
        for obj_name in why:
            if obj_name not in self._objectives:
                raise ValueError(f"Unknown objective in why: {obj_name}. Valid: {self._objectives}")
        record = SectionRecord(
            section_id=section_id, signals=signals,
            why={obj: lines[:_MAX_WHY] for obj, lines in why.items()},
            positives=positives or [], negatives=negatives or [], gaps=gaps or [],
        )
        if section_id not in self._sections:
            self._section_order.append(section_id)
        self._sections[section_id] = record

    signal = record_section

    def render_section(self, section_id: Optional[str] = None) -> str:
        if section_id is None:
            if not self._section_order:
                raise KeyError("No sections recorded")
            section_id = self._section_order[-1]
        if section_id not in self._sections:
            raise KeyError(f"Section not found: {section_id}")
        rec = self._sections[section_id]
        lines = ["OBJECTIVES IMPACT", ""]
        for obj in self._objectives:
            strength = rec.signals.get(obj, 0)
            lines.append(f"{self._format_objective_label(obj)}: ({signal_bar(strength)})")
            prefix = "+" if strength >= 2 else "-"
            for item in rec.why.get(obj, []):
                lines.append(f"  {prefix} {item}")
        for item in rec.positives:
            lines.append(f"  + {item}")
        for item in rec.negatives:
            lines.append(f"  - {item}")
        if rec.gaps:
            lines.append("")
            lines.append("Gaps:")
            for item in rec.gaps:
                lines.append(f"  ? {item}")
        text = "\n".join(lines)
        _display_output(text)
        return text

    def render_summary(self) -> str:
        if not self._sections:
            raise ValueError("No sections recorded")
        synthesis = self.aggregate()
        lines = ["OBJECTIVES SYNTHESIS", ""]
        for obj in self._objectives:
            label = self._format_objective_label(obj)
            synth = synthesis[obj]
            parts = [f"{label}: ({signal_bar(synth.combined_strength)}]"]
            for phrase, count in synth.drivers[:3]:
                suffix = f" ({count}x)" if count > 1 else ""
                parts.append(f"+ {phrase}{suffix}")
            for phrase, count in synth.frictions[:3]:
                suffix = f" ({count}x)" if count > 1 else ""
                parts.append(f"- {phrase}{suffix}")
            lines.append(" ".join(parts))
        all_gaps: list[str] = []
        for synth in synthesis.values():
            for g in synth.gaps:
                if g not in all_gaps:
                    all_gaps.append(g)
        if all_gaps:
            lines.append("")
            lines.append("Gaps:")
            for g in all_gaps:
                lines.append(f"  ? {g}")
        lines.append("")
        lines.append("Confidence:")
        for obj in self._objectives:
            synth = synthesis[obj]
            lines.append(f"  {self._format_objective_label(obj)}: ({signal_bar(synth.confidence)})")
        text = "\n".join(lines)
        _display_output(text)
        return text

    def aggregate(self) -> dict[str, ObjectiveSynthesis]:
        result: dict[str, ObjectiveSynthesis] = {}
        for obj in self._objectives:
            strengths = [rec.signals.get(obj, 0) for rec in self._sections.values()]
            if strengths and any(s > 0 for s in strengths):
                combined = min(3, max(0, int(sum(strengths) / len(strengths) + 0.5)))
            else:
                combined = 0
            pos_phrases: list[str] = []
            neg_phrases: list[str] = []
            gap_set: list[str] = []
            for rec in self._sections.values():
                if rec.signals.get(obj, 0) >= 1:
                    pos_phrases.extend(rec.positives)
                    neg_phrases.extend(rec.negatives)
                for g in rec.gaps:
                    if g not in gap_set:
                        gap_set.append(g)
            result[obj] = ObjectiveSynthesis(
                combined_strength=combined,
                drivers=self._rank_phrases(pos_phrases),
                frictions=self._rank_phrases(neg_phrases),
                gaps=gap_set,
                confidence=self._compute_confidence(strengths),
            )
        return result

    def _format_objective_label(self, name: str) -> str:
        spaced = re.sub(r"([A-Z])", r" \1", name).strip()
        return spaced[0] + spaced[1:].lower()

    def _rank_phrases(self, phrases: list[str]) -> list[tuple[str, int]]:
        if not phrases:
            return []
        counter: Counter[str] = Counter()
        for p in phrases:
            counter[p.lower()] += 1
        return counter.most_common()

    def _compute_confidence(self, strengths: list[int]) -> int:
        if not strengths:
            return 0
        mean = sum(strengths) / len(strengths)
        variance = sum((s - mean) ** 2 for s in strengths) / len(strengths)
        std = sqrt(variance)
        if std == 0:
            return 3
        if std <= 0.5:
            return 2
        if std <= 1:
            return 1
        return 0
