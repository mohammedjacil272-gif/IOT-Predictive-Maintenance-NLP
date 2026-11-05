"""Windows-friendly offline Azure PdM demonstration CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .streaming import run_offline_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Azure PdM telemetry through the saved Phase 7 model")
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--machines", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-predictions", type=int, default=10)
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    result = run_offline_replay(telemetry_csv=args.telemetry, machines_csv=args.machines, model_path=args.model, output_jsonl=args.output, max_predictions=args.max_predictions)
    print(json.dumps({"output_path": result["output_path"], "model_version": result["model_version"], "benchmark_status": result["benchmark_status"], "summary": result["summary"], "example_predictions": result["predictions"][:3]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
