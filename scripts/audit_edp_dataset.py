"""Audit locally acquired EDP workbooks without changing raw files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from the repository root or any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.edp_wind.schema import EDPValidationError, audit_directory, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_directory(args.raw_directory)
        write_report(report, args.report)
    except (EDPValidationError, OSError) as exc:
        parser.error(str(exc))
    print(f"Audited {len(report['workbooks'])} EDP workbook(s); report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
