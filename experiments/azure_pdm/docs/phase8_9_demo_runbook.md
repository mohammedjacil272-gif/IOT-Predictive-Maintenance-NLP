# Azure PdM college-project demo runbook

## Demonstration flow

1. Show the pinned source and acquisition manifest, explaining that the dataset is a simulated benchmark.
2. Show the saved Phase 7 model, preprocessing, feature schema, thresholds, and hash manifest.
3. Run the PowerShell quickstart with ten predictions.
4. Point out that the first 23 records for machine 1 are reported as incomplete because a complete 24-hour history is required; no prediction is emitted for them.
5. Show one JSONL prediction: machine, observation timestamp, input-window bounds, probabilities, predicted class, model version, provenance, latency, and benchmark warning.
6. Explain that only telemetry and approved machine metadata enter the classifier. Failure rows are target-only; errors and maintenance are excluded because availability is unknown.
7. Run the focused tests and show duplicate, gap, malformed-value, timezone, and deterministic-inference coverage.

## Presentation wording

“Telemetry observations are replayed as timezone-naive benchmark timestamps. The adapter validates finite sensor values, rejects duplicates, resets on gaps, waits for a contiguous 24-hour window, applies the saved training preprocessing and feature order, and produces a six-class benchmark prediction. This is not a verified real-world failure alarm.”

## Do not claim

- Do not call the output a confirmed equipment failure.
- Do not describe structured error records as technician narratives.
- Do not claim live cloud ingestion; the guaranteed path is local replay.
- Do not claim the benchmark establishes factory deployment performance.
