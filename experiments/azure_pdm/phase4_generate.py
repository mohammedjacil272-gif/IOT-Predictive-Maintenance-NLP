"""Generate the Phase 4 Azure PdM temporal foundation outside Git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_builder import build_temporal_dataset
from .dataset_validation import validate_temporal_dataset
from .splits import SplitPolicy
from .anchors import read_and_audit_telemetry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_temporal_dataset(dataset_root=args.dataset_root, output_root=args.output_root)
    telemetry = read_and_audit_telemetry(args.dataset_root / "PdM_telemetry.csv")
    policy = SplitPolicy(telemetry.start, telemetry.end)
    validation = validate_temporal_dataset(result["dataset_path"], split_policy=policy)
    validation_path = args.output_root / "phase4_dataset_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": str(result["dataset_path"]),
        "audit": str(result["audit_path"]),
        "manifest": str(result["manifest_path"]),
        "validation": str(validation_path),
        "status": validation["status"],
        "rows": validation["row_count"],
    }, sort_keys=True))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
