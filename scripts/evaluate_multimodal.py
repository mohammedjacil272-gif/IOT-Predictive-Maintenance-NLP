
"""Create a validation comparison report for FD001 RUL models."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = (
    ROOT / "artifacts/metrics/fd001_lstm_baseline/validation_metrics.json"
)
OUTPUT_DIR = ROOT / "artifacts/metrics/fd001_multimodal"
OUTPUT_JSON = OUTPUT_DIR / "comparison_report.json"
OUTPUT_CSV = OUTPUT_DIR / "comparison_report.csv"


def main() -> None:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Frozen sensor baseline metrics not found: {BASELINE_FILE}"
        )

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))

    report = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation_split": "validation",
        "official_test_evaluated": False,
        "models": [
            {
                "model": "sensor_only_lstm",
                "status": "measured_baseline",
                "MAE": baseline["mae"],
                "RMSE": baseline["rmse"],
                "MSE": baseline["mse"],
                "source": str(BASELINE_FILE.relative_to(ROOT)),
            },
            {
                "model": "nlp_only",
                "status": "pending_real_nlp_artifacts",
                "MAE": None,
                "RMSE": None,
                "MSE": None,
                "source": None,
            },
            {
                "model": "multimodal_fusion",
                "status": "pending_real_nlp_artifacts_and_training",
                "MAE": None,
                "RMSE": None,
                "MSE": None,
                "source": None,
            },
        ],
        "notes": [
            "All metrics must use the same frozen validation engine split.",
            "Lower MAE, RMSE, and MSE indicate better predictions.",
            "Pending metrics are not zero and must not be treated as results.",
            "Synthetic smoke-test metrics are excluded from this comparison.",
            "Official test labels were not used for model selection.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    import csv

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "status", "MAE", "RMSE", "MSE", "source"],
        )
        writer.writeheader()
        writer.writerows(report["models"])

    print("Comparison report created.")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")
    print("\nValidation comparison:")
    for item in report["models"]:
        print(
            f"{item['model']}: status={item['status']}, "
            f"MAE={item['MAE']}, RMSE={item['RMSE']}, MSE={item['MSE']}"
        )
    print("\nOfficial test evaluation: NOT PERFORMED")


if __name__ == "__main__":
    main()
