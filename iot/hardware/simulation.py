"""Deterministic demonstration telemetry profile shared conceptually with firmware."""

from __future__ import annotations

from dataclasses import dataclass

STAGES = (("normal", 6), ("warming", 4), ("abnormal", 4), ("failure", 3), ("recovery", 2))


@dataclass(frozen=True)
class SimulatedReading:
    temperature: float
    vibration: float
    rpm: int
    pressure: float
    operating_status: str


class SyntheticSensorSimulator:
    """A repeatable 19-sample operating cycle with gradual ramps and recovery."""

    def at(self, sample_number: int) -> SimulatedReading:
        if isinstance(sample_number, bool) or not isinstance(sample_number, int) or sample_number < 0:
            raise ValueError("sample_number must be a non-negative integer")
        period = sum(count for _, count in STAGES)
        position = sample_number % period
        start = 0
        for stage, count in STAGES:
            if position < start + count:
                phase = (position - start) / max(count - 1, 1)
                # Triangular envelope rises through abnormal/failure, then cools during recovery.
                if stage == "normal":
                    severity = 0.0
                elif stage == "warming":
                    severity = 0.15 + 0.35 * phase
                elif stage == "abnormal":
                    severity = 0.55 + 0.35 * phase
                elif stage == "failure":
                    severity = 1.0
                else:
                    severity = 0.65 * (1.0 - phase)
                return SimulatedReading(
                    temperature=round(45.0 + 50.0 * severity, 2),
                    vibration=round(1.2 + 7.3 * severity, 2),
                    rpm=round(1500 + 600 * severity),
                    pressure=round(3.2 + 2.6 * severity, 2),
                    operating_status=stage,
                )
            start += count
        raise AssertionError("unreachable stage selection")
