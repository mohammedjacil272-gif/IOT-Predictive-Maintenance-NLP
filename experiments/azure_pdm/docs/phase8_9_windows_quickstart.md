# Azure PdM Phase 8–9 Windows quickstart

Run from PowerShell in the repository root. The demonstration uses the real Phase 7 tuned Extra Trees artifact and the acquired telemetry CSV; it does not train, connect to a broker, or connect to InfluxDB.

```powershell
Set-Location $env:IOT_FUSION_REPO_ROOT
$azureRoot = $env:AZURE_PDM_ROOT
if (-not $azureRoot) { throw 'Set AZURE_PDM_ROOT to the external Azure PdM data directory.' }
$demoRoot = Join-Path $azureRoot 'phase8_9_outputs'
New-Item -ItemType Directory -Path $demoRoot -Force | Out-Null

python -m experiments.azure_pdm.demo `
  --telemetry (Join-Path $azureRoot 'raw_acquisition_02\PdM_telemetry.csv') `
  --machines (Join-Path $azureRoot 'raw_acquisition_02\PdM_machines.csv') `
  --model (Join-Path $azureRoot 'phase7_outputs\selected_extra_trees\model_tuned.pkl') `
  --output (Join-Path $demoRoot 'offline_replay_predictions.jsonl') `
  --max-predictions 10 `
  --log-level INFO
```

The command prints three example predictions and a replay summary. Each prediction contains machine ID, observation time, the 24-hour input-window bounds, six probabilities, threshold-adjusted class, model version, provenance, latency, and an explicit simulated-benchmark warning.

Run the integration tests:

```powershell
pytest -q experiments/azure_pdm/tests
```

The live MQTT/InfluxDB path is intentionally not started by this quickstart.

## Day 2 bounded multi-machine smoke test

```powershell
$multi = Join-Path $demoRoot 'multi_machine_replay_input.csv'
$multiOut = Join-Path $demoRoot 'multi_machine_predictions_09.jsonl'

python -m experiments.azure_pdm.demo `
  --telemetry $multi `
  --machines (Join-Path $azureRoot 'raw_acquisition_02\PdM_machines.csv') `
  --model (Join-Path $azureRoot 'phase7_outputs\selected_extra_trees\model_tuned.pkl') `
  --output $multiOut `
  --max-predictions 9

pytest -q experiments/azure_pdm/tests `
  tests/test_azure_pdm_validator.py `
  tests/test_baseline_verifier_readonly.py `
  tests/test_fusion_alignment.py `
  tests/test_fusion_provenance.py `
  tests/test_fusion_shapes.py `
  tests/test_missing_text.py `
  tests/test_ner_measurements.py `
  tests/test_nlp_embeddings_diagnostics.py `
  tests/test_nlp_fusion_integration.py `
  tests/test_nlp_fusion_pipeline.py `
  tests/test_training_contract.py `
  tests/test_sensor_message.py `
  tests/test_local_replay.py `
  tests/test_influx_ingestion.py `
  tests/test_mqtt_publisher.py
```

The checked-in multi-machine input is a bounded derived replay fixture under the external output directory, not a replacement for the raw dataset.
