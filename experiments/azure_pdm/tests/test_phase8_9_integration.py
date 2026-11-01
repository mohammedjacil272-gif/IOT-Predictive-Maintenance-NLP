from datetime import datetime, timedelta

import pytest

from experiments.azure_pdm.streaming import AzureStreamError, AzureTelemetryWindowAdapter


class FakePredictor:
    artifact = {"lookback_hours": 3, "provenance": {"model_version": "test"}}

    def predict_one(self, payload):
        assert len(payload["telemetry"]) == 3
        return {"predicted_class": "none", "probabilities": {"none": 1.0}}


def record(timestamp, machine=1, value=1.0):
    return {"datetime": timestamp.isoformat(), "machineID": str(machine), "volt": value, "rotate": value, "pressure": value, "vibration": value}


def test_adapter_emits_only_complete_contiguous_windows_and_counts_duplicates():
    start = datetime(2020, 1, 1)
    adapter = AzureTelemetryWindowAdapter(FakePredictor(), {1: {"machine_model": "A", "machine_age": 1.0}}, lookback_hours=3)
    output = list(adapter.replay([record(start), record(start + timedelta(hours=1)), record(start + timedelta(hours=1)), record(start + timedelta(hours=2))]))
    assert len(output) == 1
    assert adapter.summary()["duplicates_rejected"] == 1
    assert adapter.summary()["incomplete_windows"] == 2


def test_adapter_resets_after_gap_and_rejects_unknown_machine():
    start = datetime(2020, 1, 1)
    adapter = AzureTelemetryWindowAdapter(FakePredictor(), {1: {"machine_model": "A", "machine_age": 1.0}}, lookback_hours=3)
    list(adapter.replay([record(start), record(start + timedelta(hours=2)), record(start + timedelta(hours=3))]))
    assert adapter.summary()["gap_resets"] == 1
    with pytest.raises(AzureStreamError):
        adapter.ingest(record(start + timedelta(hours=4), machine=2))


def test_adapter_rejects_timezone_aware_and_nonfinite_inputs():
    adapter = AzureTelemetryWindowAdapter(FakePredictor(), {1: {"machine_model": "A", "machine_age": 1.0}}, lookback_hours=3)
    with pytest.raises(AzureStreamError):
        adapter.ingest(record(datetime(2020, 1, 1).astimezone()))
    bad = record(datetime(2020, 1, 1)); bad["volt"] = "nan"
    with pytest.raises(AzureStreamError):
        adapter.ingest(bad)
