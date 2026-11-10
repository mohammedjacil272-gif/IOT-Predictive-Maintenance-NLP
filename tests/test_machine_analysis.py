from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.machine_analysis import analyze_machine


NOW = datetime.now(timezone.utc).replace(microsecond=0)


class FakeGateway:
    def __init__(self):
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return {"text": "Evidence-only report", "provider": "ollama", "model": "qwen2.5:3b", "response_type": "model"}


def history_item(mode, machine_id, *, timestamp=None, synthetic=False, text=None, records=None, provenance="user_submitted"):
    request = {"machine_id": machine_id}
    if text is not None:
        request.update({"log_id": "log-1", "text": text})
    if records is not None:
        request["records"] = records
    return {"timestamp": (timestamp or NOW).isoformat(), "mode": mode, "request": request, "source_provenance": provenance, "is_synthetic": synthetic, "response": {}}


def test_benchmark_history_is_not_implicitly_associated_with_machine():
    result = analyze_machine(
        "MAINTAINIQ-LOCAL-TEST-001",
        machine={"machine_id": "MAINTAINIQ-LOCAL-TEST-001", "is_synthetic": True},
        history_records=[history_item("sensor_only", None, records=[{"unit_id": 1, "cycle": c} for c in range(1, 31)], provenance="fd001_benchmark")],
        gateway=FakeGateway(), model_path="missing.pt", scaler_path="missing.joblib", now=NOW,
    )
    assert result["status"] == "no_verified_evidence"
    assert result["sensor_evidence"]["status"] == "unavailable"


def test_explicit_machine_log_supports_nlp_only_and_qwen_context():
    gateway = FakeGateway()
    result = analyze_machine(
        "machine-1",
        machine={"machine_id": "machine-1", "is_synthetic": True},
        history_records=[history_item("nlp_only", "machine-1", text="Bearing vibration increased; inspection recommended.", synthetic=True)],
        gateway=gateway, model_path="missing.pt", scaler_path="missing.joblib", now=NOW,
    )
    assert result["status"] == "nlp_only"
    assert result["sensor_evidence"]["status"] == "unavailable"
    assert result["technician_evidence"]["status"] == "verified"
    assert result["assistant"]["status"] == "ready"
    assert "machine-1" in gateway.messages[1]["content"]
    assert "synthetic" in " ".join(result["limitations"])


def test_valid_sensor_result_is_independent_of_nlp_severity(monkeypatch):
    class FakeResult:
        unit_id = 1
        window_start_cycle = 1
        window_end_cycle = 30
        prediction = 42.5

    monkeypatch.setattr("app.machine_analysis.predict_records", lambda records, scaler_path, model_path: FakeResult())
    result = analyze_machine(
        "machine-2",
        machine={"machine_id": "machine-2"},
        history_records=[
            history_item("sensor_only", "machine-2", records=[{"unit_id": 1, "cycle": c} for c in range(1, 31)]),
            history_item("nlp_only", "machine-2", text="Critical vibration", provenance="technician_log"),
        ],
        gateway=FakeGateway(), model_path="model.pt", scaler_path="scaler.joblib", now=NOW,
    )
    assert result["status"] == "both_sources_verified"
    assert result["sensor_evidence"]["prediction"] == 42.5
    assert result["fusion"]["status"] == "not_validated"


def test_stale_and_mismatched_records_are_not_used():
    result = analyze_machine(
        "machine-3",
        machine={"machine_id": "machine-3"},
        history_records=[
            history_item("nlp_only", "machine-3", timestamp=NOW - timedelta(days=60), text="old observation"),
            history_item("nlp_only", "other-machine", text="wrong machine"),
        ],
        gateway=FakeGateway(), model_path="missing.pt", scaler_path="missing.joblib", now=NOW, max_age_hours=24,
    )
    assert result["status"] == "no_verified_evidence"
    assert result["technician_evidence"]["status"] == "stale"
