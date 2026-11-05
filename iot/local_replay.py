"""Local deterministic replay of FD001 training sensor rows.

This module reads raw C-MAPSS FD001 training rows and converts each row into
the flat ``SensorMessage`` schema. It intentionally does not compute RUL,
failure labels, timestamps, or any future-looking fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

from .sensor_message import FD001_SENSOR_FEATURES, SensorMessage, validate_sensor_message


DEFAULT_TRAIN_PATH = Path("data/train_FD001.txt")

FD001_CANONICAL_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
)

_COLUMN_INDEX = {name: index for index, name in enumerate(FD001_CANONICAL_COLUMNS)}


class SensorReplayError(ValueError):
    """Raised when local FD001 replay input cannot be converted safely."""


def iter_fd001_training_messages(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    limit: int | None = None,
) -> Iterator[SensorMessage]:
    """Yield validated FD001 training rows as ``SensorMessage`` objects.

    Args:
        train_path: Path to the whitespace-delimited ``train_FD001.txt`` file.
        limit: Optional maximum number of messages to emit. ``None`` replays
            the whole file; ``0`` performs input checks and emits no messages.

    Raises:
        FileNotFoundError: If ``train_path`` does not exist.
        SensorReplayError: If ``limit`` is invalid or a row has malformed
            width, identifiers, or numeric values.
        SensorMessageValidationError: If a converted row violates the current
            sensor message schema.
    """

    normalized_limit = _validate_limit(limit)
    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(f"FD001 training file not found: {path}")
    if not path.is_file():
        raise SensorReplayError(f"FD001 training path is not a file: {path}")
    if normalized_limit == 0:
        return

    emitted = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                raise SensorReplayError(f"Malformed FD001 row {line_number}: blank rows are not allowed.")

            values = stripped.split()
            if len(values) != len(FD001_CANONICAL_COLUMNS):
                raise SensorReplayError(
                    f"Malformed FD001 row {line_number}: expected {len(FD001_CANONICAL_COLUMNS)} columns, "
                    f"found {len(values)}."
                )

            yield _row_to_sensor_message(values, line_number)
            emitted += 1
            if normalized_limit is not None and emitted >= normalized_limit:
                break


def replay_fd001_training_messages(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    limit: int | None = None,
) -> list[SensorMessage]:
    """Return validated FD001 training replay messages as a list."""

    return list(iter_fd001_training_messages(train_path, limit=limit))


def _row_to_sensor_message(values: Sequence[str], line_number: int) -> SensorMessage:
    try:
        unit_id = _parse_identifier(values[_COLUMN_INDEX["unit_id"]], "unit_id")
        cycle = _parse_identifier(values[_COLUMN_INDEX["cycle"]], "cycle")
        message: dict[str, int | float] = {"unit_id": unit_id, "cycle": cycle}
        for feature in FD001_SENSOR_FEATURES:
            message[feature] = float(values[_COLUMN_INDEX[feature]])
    except ValueError as exc:
        raise SensorReplayError(f"Malformed FD001 row {line_number}: {exc}") from exc

    return validate_sensor_message(message)


def _parse_identifier(raw_value: str, field_name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer, found {raw_value!r}") from exc

    if str(value) != raw_value:
        raise ValueError(f"{field_name} must be an integer token, found {raw_value!r}")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return value


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise SensorReplayError("Replay limit must be a non-negative integer or None.")
    if limit < 0:
        raise SensorReplayError("Replay limit must be non-negative.")
    return limit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay local FD001 training rows as validated sensor messages.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="Path to train_FD001.txt.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum messages to emit as JSON lines.")
    args = parser.parse_args(argv)

    for message in iter_fd001_training_messages(args.train_path, limit=args.limit):
        print(json.dumps(message.to_dict(), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
