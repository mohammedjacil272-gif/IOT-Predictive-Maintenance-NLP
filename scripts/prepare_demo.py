"""Create an isolated, provenance-labelled MaintainIQ demonstration dataset.

This script never writes the repository database or model artifacts. It reads
30 rows from the existing FD001 benchmark test file, records their benchmark
identity, and stores them under an explicitly synthetic demo machine ID.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.history import LocalHistoryStore
from assistant.storage import AssistantStorage
from iot.lstm_inference import predict_records
from preprocessing.cmapss_fd001 import CANONICAL_COLUMNS, SELECTED_FEATURES, WINDOW_SIZE


MACHINE_ID = "MAINTAINIQ-DEMO-001"
MODEL_ID = "fd001_lstm_baseline"
SENSOR_PROVENANCE = "fd001_benchmark_demo_fixture"
TECHNICIAN_PROVENANCE = "synthetic_technician_demo_fixture"
TECHNICIAN_TEXT = (
    "Synthetic demonstration observation: elevated vibration near the bearing; "
    "inspect lubrication and bearing condition before continued operation."
)


def _read_first_unit_window(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != len(CANONICAL_COLUMNS):
                raise ValueError(f"Unexpected FD001 row width {len(parts)} in {source}.")
            row = dict(zip(CANONICAL_COLUMNS, parts))
            if int(row["unit_id"]) != 1:
                if rows:
                    break
                continue
            rows.append(
                {
                    "unit_id": int(row["unit_id"]),
                    "cycle": int(row["cycle"]),
                    **{feature: float(row[feature]) for feature in SELECTED_FEATURES},
                }
            )
            if len(rows) == WINDOW_SIZE:
                break
    if len(rows) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} rows for FD001 unit 1; found {len(rows)}.")
    return rows


def prepare_demo(data_dir: Path, source: Path, *, model_path: Path, scaler_path: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FD001 source file not found: {source}")
    if (data_dir / ".assistant.sqlite3").exists() or (data_dir / ".local_history.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing application data in {data_dir}; choose a new data directory."
        )
    records = _read_first_unit_window(source)
    prediction = predict_records(records, model_path=model_path.resolve(), scaler_path=scaler_path.resolve())
    data_dir.mkdir(parents=True, exist_ok=True)

    storage = AssistantStorage(data_dir / ".assistant.sqlite3")
    storage.save_machine(
        MACHINE_ID,
        {
            "machine_id": MACHINE_ID,
            "display_name": "MaintainIQ FD001 Benchmark Demo (Synthetic)",
            "machine_type": "benchmark_demo_fixture",
            "location": "isolated_demo_data",
            "is_synthetic": True,
        },
    )
    history = LocalHistoryStore(data_dir / ".local_history.json")
    timestamp = datetime.now(timezone.utc).isoformat()
    history.append(
        mode="sensor_only",
        model_id=MODEL_ID,
        source_provenance=SENSOR_PROVENANCE,
        is_synthetic=True,
        request={
            "machine_id": MACHINE_ID,
            "records": records,
            "source_file": source.as_posix(),
            "benchmark_unit_id": 1,
            "benchmark_window": "test_FD001 unit 1 cycles 1-30",
            "timestamp": timestamp,
        },
        response={
            "mode": "sensor_only",
            "model_id": MODEL_ID,
            "model_provenance": "REGENERATED_CHECKPOINT_NOT_ORIGINAL",
            "prediction": prediction.prediction,
            "window_start_cycle": prediction.window_start_cycle,
            "window_end_cycle": prediction.window_end_cycle,
            "prediction_semantics": "Independent FD001 benchmark LSTM result; not live industrial telemetry.",
        },
    )
    history.append(
        mode="nlp_only",
        model_id="rule_based_nlp_v1",
        source_provenance=TECHNICIAN_PROVENANCE,
        is_synthetic=True,
        request={
            "machine_id": MACHINE_ID,
            "log_id": "MAINTAINIQ-DEMO-001-TECH-001",
            "text": TECHNICIAN_TEXT,
            "timestamp": timestamp,
        },
        response={
            "mode": "nlp_only",
            "model_id": "rule_based_nlp_v1",
            "prediction": None,
            "prediction_semantics": "NLP-only findings; no numerical RUL is produced by this path.",
        },
    )
    return {
        "data_dir": str(data_dir),
        "machine_id": MACHINE_ID,
        "records": len(records),
        "benchmark_source": str(source),
        "benchmark_unit_id": 1,
        "window_start_cycle": prediction.window_start_cycle,
        "window_end_cycle": prediction.window_end_cycle,
        "prediction": prediction.prediction,
        "sensor_provenance": SENSOR_PROVENANCE,
        "technician_provenance": TECHNICIAN_PROVENANCE,
        "fusion_status": "NOT VALIDATED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="New isolated application-data directory.")
    parser.add_argument("--source", type=Path, default=Path("data/test_FD001.txt"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/fd001_lstm_baseline/best_model.pt"))
    parser.add_argument("--scaler", type=Path, default=Path("artifacts/data/fd001_lstm_baseline/scaler.joblib"))
    args = parser.parse_args()
    result = prepare_demo(args.data_dir, args.source, model_path=args.model, scaler_path=args.scaler)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
