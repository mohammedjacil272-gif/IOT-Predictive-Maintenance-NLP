"""Dry-run validation for an owner-supplied genuine technician-log CSV.

This command never downloads models, generates embeddings, or writes artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.technician_logs import GenuineLogValidationConfig, load_genuine_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--approved-status", default="approved")
    parser.add_argument("--reject-empty-text", action="store_true")
    args = parser.parse_args()
    try:
        _, report = load_genuine_csv(
            args.csv_path,
            GenuineLogValidationConfig(
                expected_dataset=args.dataset,
                expected_dataset_version=args.dataset_version,
                approved_status=args.approved_status,
                allow_empty_text=not args.reject_empty_text,
            ),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("No embeddings or artifacts were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
