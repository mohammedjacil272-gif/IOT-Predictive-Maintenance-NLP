
"""Validate an NLP artifact package without modifying or training anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.nlp_artifact_validation import (
    ArtifactValidationLimits,
    DEFAULT_LIMITS,
    validate_nlp_artifact_package,
)


def _validate_report_destination(package_dir: Path, report_path: Path) -> None:
    if report_path.exists() and report_path.is_symlink():
        raise ValueError("--report must not be a symlink.")
    package_root = package_dir.resolve()
    destination = report_path.resolve()
    try:
        destination.relative_to(package_root)
    except ValueError:
        return
    raise ValueError("--report must point outside the input package directory.")


def main(
    package_dir: Path,
    report_path: Path | None = None,
    limits: ArtifactValidationLimits | None = None,
) -> int:
    if report_path is not None:
        _validate_report_destination(package_dir, report_path)
    report = validate_nlp_artifact_package(package_dir, limits=limits)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if report_path is not None:
        report_path.write_text(output + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate supplied NLP artifacts; no artifacts are generated."
    )
    parser.add_argument(
        "--package-dir",
        default="artifacts",
        help=("Flat package directory, a root containing data/ and embeddings/, "
              "or the repository artifacts root containing the namespaced FD001 NLP fixture."),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the structured validation report.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_LIMITS.max_file_bytes,
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=DEFAULT_LIMITS.max_csv_rows,
    )
    parser.add_argument(
        "--max-jsonl-line-bytes",
        type=int,
        default=DEFAULT_LIMITS.max_jsonl_line_bytes,
    )
    parser.add_argument(
        "--max-jsonl-records",
        type=int,
        default=DEFAULT_LIMITS.max_jsonl_records,
    )
    parser.add_argument(
        "--max-tensor-elements",
        type=int,
        default=DEFAULT_LIMITS.max_tensor_elements,
    )
    args = parser.parse_args()
    package_dir = Path(args.package_dir)
    if not package_dir.is_absolute():
        package_dir = ROOT / package_dir
    report_path = args.report
    if report_path is not None and not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        limits = ArtifactValidationLimits(
            max_file_bytes=args.max_file_bytes,
            max_csv_rows=args.max_csv_rows,
            max_jsonl_line_bytes=args.max_jsonl_line_bytes,
            max_jsonl_records=args.max_jsonl_records,
            max_tensor_elements=args.max_tensor_elements,
        )
        raise SystemExit(main(package_dir, report_path, limits))
    except ValueError as exc:
        parser.error(str(exc))
