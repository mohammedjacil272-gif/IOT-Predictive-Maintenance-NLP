"""Chronological Azure PdM split and two-sided purge policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class SplitPolicy:
    coverage_start: datetime
    coverage_end: datetime
    embargo_hours: int = 24
    train_fraction: float = 0.60
    validation_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.coverage_end <= self.coverage_start:
            raise ValueError("coverage_end must be after coverage_start")
        if self.embargo_hours < 0:
            raise ValueError("embargo_hours must be non-negative")
        if self.train_fraction <= 0 or self.validation_fraction <= 0:
            raise ValueError("split fractions must be positive")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave test coverage")

    @property
    def first_cutpoint(self) -> datetime:
        return self.coverage_start + (self.coverage_end - self.coverage_start) * self.train_fraction

    @property
    def second_cutpoint(self) -> datetime:
        return self.coverage_start + (self.coverage_end - self.coverage_start) * (self.train_fraction + self.validation_fraction)

    @property
    def embargo(self) -> timedelta:
        return timedelta(hours=self.embargo_hours)

    def describe(self) -> dict[str, Any]:
        return {
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "first_cutpoint": self.first_cutpoint.isoformat(),
            "second_cutpoint": self.second_cutpoint.isoformat(),
            "embargo_hours": self.embargo_hours,
            "boundary_rule": "exclude anchors in [B-embargo, B+embargo) around each boundary",
            "train_rule": f"anchor < { (self.first_cutpoint - self.embargo).isoformat() }",
            "validation_rule": f"{ (self.first_cutpoint + self.embargo).isoformat() } <= anchor < { (self.second_cutpoint - self.embargo).isoformat() }",
            "test_rule": f"anchor >= { (self.second_cutpoint + self.embargo).isoformat() }",
            "lookback_context_rule": "past observations before the anchor may be read causally; no observation at or after the anchor is allowed",
        }

    def assign(self, anchor: datetime) -> tuple[str | None, str | None]:
        first, second, gap = self.first_cutpoint, self.second_cutpoint, self.embargo
        if anchor < first - gap:
            return "train", None
        if first + gap <= anchor < second - gap:
            return "validation", None
        if anchor >= second + gap:
            return "test", None
        if first - gap <= anchor < first + gap:
            return None, "first_boundary_embargo_or_purge"
        if second - gap <= anchor < second + gap:
            return None, "second_boundary_embargo_or_purge"
        return None, "outside_defined_coverage"
