"""Explicit elapsed-time lookback windows for the Azure PdM benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class LookbackWindow:
    """A half-open lookback interval ending immediately before an anchor."""

    hours: int

    def __post_init__(self) -> None:
        if self.hours not in (3, 24):
            raise ValueError("Supported Azure PdM lookbacks are 3 or 24 hours.")

    def bounds(self, anchor: datetime) -> tuple[datetime, datetime]:
        """Return [anchor - hours, anchor), never including the anchor itself."""
        return anchor - timedelta(hours=self.hours), anchor

    def expected_hourly_timestamps(self, anchor: datetime) -> tuple[datetime, ...]:
        start, end = self.bounds(anchor)
        return tuple(start + timedelta(hours=i) for i in range(self.hours))
