from __future__ import annotations
from typing import Any

def machine_health(machine: dict[str, Any] | None, *, sensor_prediction: dict[str, Any] | None = None, observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not machine: return {"status": "unavailable", "machine": None, "evidence": [], "unknown": ["Machine registry record not found."]}
    evidence=[]; unknown=[]
    if sensor_prediction: evidence.append({"kind": "model_estimated", "source": sensor_prediction.get("model_id"), "prediction": sensor_prediction.get("prediction"), "semantics": "sensor-only FD001 RUL; not a health percentage"})
    else: unknown.append("No sensor prediction available.")
    for item in observations or []: evidence.append({"kind": "technician_reported" if item.get("status") == "confirmed" else "pending_observation", "source": item.get("observation_id"), "content": item.get("interpretation", item)})
    if not observations: unknown.append("No confirmed technician observations available.")
    return {"status": "available", "machine": machine, "evidence": evidence, "unknown": unknown}
