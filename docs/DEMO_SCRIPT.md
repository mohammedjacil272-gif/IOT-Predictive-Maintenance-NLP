# MaintainIQ evaluator demo script

## 1. Select the asset

Open MaintainIQ, choose **Machines**, and select:

`MAINTAINIQ-DEMO-001 — MaintainIQ FD001 Benchmark Demo (Synthetic)`

Explain that the identity is explicit and isolated. No FD001 engine ID is silently treated as a plant machine.

## 2. Run analysis

Open **Assistant Workspace** and click **ANALYZE MACHINE**.

The result shows:

- A numerical RUL only when the exact machine-linked sensor sequence is valid.
- The raw LSTM prediction and its window/cycle details.
- Rule-based technician entities and severity.
- Source, synthetic, benchmark, and model provenance.
- A local Qwen explanation when Ollama reports `qwen2.5:3b` available.

The LSTM owns the number. Qwen is given structured evidence and is instructed not to calculate, alter, or invent RUL. The text and sensor paths are orchestrated together for explanation; this is not trained multimodal fusion.

## 3. Explain the evidence honestly

The sensor window is drawn from the existing FD001 benchmark file and is labelled synthetic demo evidence, not live telemetry. The technician note is synthetic demonstration text processed by `rule_based_nlp_v1`; DistilBERT is not claimed unless the backend explicitly reports it available.

Fusion must remain **NOT VALIDATED** because there is no approved paired industrial dataset, identity crosswalk, temporal-alignment policy, or trained fusion checkpoint.

## 4. Show the fallback

If Ollama is stopped, rerun the analysis. The UI should retain the independent LSTM and NLP results while clearly reporting that the local assistant is unavailable. If the sensor evidence is missing, the UI must show no numerical RUL.

## 5. Closing statement

This demonstration proves evidence-grounded LSTM inference, rule-based maintenance-log extraction, and optional local explanation over an explicitly synthetic benchmark fixture. It does not prove live industrial deployment, genuine paired data, validated fusion, or production readiness.
