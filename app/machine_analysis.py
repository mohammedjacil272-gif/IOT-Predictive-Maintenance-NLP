"""Evidence-aware machine analysis orchestration.

Only records carrying an explicit matching machine_id are eligible. Unit IDs,
timestamps, benchmark provenance, and record order never create an association.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any, Mapping, Sequence
from uuid import uuid4

from assistant.gateway import ModelGatewayError, bounded_context
from iot.lstm_inference import LSTMInferenceError, predict_records
from nlp.embeddings import embedding_diagnostics
from nlp.ner import extract_entities
from nlp.preprocessing import preprocess_log
from nlp.severity import score_severity

DEFAULT_MAX_AGE_HOURS = 24 * 30


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _fresh(value: object, *, now: datetime, max_age_hours: float) -> bool:
    parsed = _timestamp(value)
    return parsed is not None and parsed >= now - timedelta(hours=max_age_hours) and parsed <= now + timedelta(minutes=5)


def _associated(item: Mapping[str, Any], machine_id: str) -> bool:
    request = item.get("request")
    return isinstance(request, Mapping) and request.get("machine_id") == machine_id


def _latest(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return max(items, key=lambda item: _timestamp(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), default=None)


def _sensor_evidence(machine_id: str, candidates: Sequence[Mapping[str, Any]], *, model_path: Any, scaler_path: Any, now: datetime, max_age_hours: float) -> dict[str, Any]:
    item = _latest(candidates)
    if item is None:
        return {"status": "unavailable", "reason": "No sensor inference record explicitly associated with this machine_id."}
    request = item.get("request")
    records = request.get("records") if isinstance(request, Mapping) else None
    base = {"history_timestamp": item.get("timestamp"), "source_provenance": item.get("source_provenance", "unknown"), "is_synthetic": bool(item.get("is_synthetic", False)), "machine_id": request.get("machine_id") if isinstance(request, Mapping) else None, "fresh": _fresh(item.get("timestamp"), now=now, max_age_hours=max_age_hours)}
    if not base["fresh"]:
        return {"status": "stale", **base, "reason": "The latest explicitly associated sensor record is outside the freshness window."}
    if not isinstance(records, list):
        return {"status": "invalid", **base, "reason": "The associated sensor record does not contain a valid records array."}
    try:
        result = predict_records(records, scaler_path=scaler_path, model_path=model_path)
    except (LSTMInferenceError, OSError, ValueError, TypeError) as exc:
        return {"status": "invalid", **base, "record_count": len(records), "reason": str(exc)}
    response = item.get("response") if isinstance(item.get("response"), Mapping) else {}
    return {"status": "verified", **base, "record_count": len(records), "unit_id": result.unit_id, "window_start_cycle": result.window_start_cycle, "window_end_cycle": result.window_end_cycle, "prediction": result.prediction, "model_id": "fd001_lstm_baseline", "model_provenance": response.get("model_provenance", "REGENERATED_CHECKPOINT_NOT_ORIGINAL"), "prediction_semantics": "Independent FD001 LSTM RUL result; not a validated multimodal fusion prediction."}


def _technician_evidence(machine_id: str, candidates: Sequence[Mapping[str, Any]], *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    item = _latest(candidates)
    if item is None:
        return {"status": "unavailable", "reason": "No technician log explicitly associated with this machine_id."}
    request = item.get("request")
    text = request.get("text") if isinstance(request, Mapping) else None
    base = {"history_timestamp": item.get("timestamp"), "source_provenance": item.get("source_provenance", "unknown"), "is_synthetic": bool(item.get("is_synthetic", False)), "machine_id": request.get("machine_id") if isinstance(request, Mapping) else None, "fresh": _fresh(item.get("timestamp"), now=now, max_age_hours=max_age_hours)}
    if not base["fresh"]:
        return {"status": "stale", **base, "reason": "The latest explicitly associated technician log is outside the freshness window."}
    if not isinstance(text, str) or not text.strip():
        return {"status": "invalid", **base, "reason": "The associated technician log has no usable text."}
    processed = preprocess_log({"log_id": request.get("log_id", "unknown"), "text": text})
    return {"status": "verified", **base, "log_id": request.get("log_id"), "original_text": processed["original_text"], "normalized_text": processed["normalized_text"], "entities": extract_entities(text, log_id=request.get("log_id")), "severity": score_severity(text), "model_id": "rule_based_nlp_v1", "embedding": {"status": "not_generated", "diagnostics": embedding_diagnostics()}}


def _assistant_prompt(machine_id: str, sensor: Mapping[str, Any], technician: Mapping[str, Any], limitations: list[str]) -> list[dict[str, str]]:
    system = ("You are MaintainIQ's evidence-aware machine health analyst. Use only the supplied structured evidence. Never invent readings, logs, timestamps, severity, or a numerical RUL. Never recalculate or alter the LSTM prediction. State clearly when evidence is synthetic, benchmark-derived, stale, missing, or unpaired. This is evidence interpretation, not a validated multimodal fusion prediction. Return concise sections: RUL interpretation, technician findings, consistency, maintenance recommendations, and limitations.")
    context = {"machine_id": machine_id, "sensor_evidence": dict(sensor), "technician_evidence": dict(technician), "limitations": limitations, "fusion_status": "not_validated"}
    return bounded_context([{"role": "system", "text": system}, {"role": "user", "text": json.dumps(context, ensure_ascii=False, sort_keys=True)}], max_messages=2, max_chars=12000)


def analyze_machine(machine_id: str, *, machine: Mapping[str, Any], history_records: Sequence[Mapping[str, Any]], gateway: Any, model_path: Any, scaler_path: Any, now: datetime | None = None, max_age_hours: float = DEFAULT_MAX_AGE_HOURS, include_model: bool = True) -> dict[str, Any]:
    """Analyze one machine using only explicitly machine-associated evidence."""
    current = now or datetime.now(timezone.utc)
    sensor_candidates = [item for item in history_records if item.get("mode") == "sensor_only" and _associated(item, machine_id)]
    technician_candidates = [item for item in history_records if item.get("mode") == "nlp_only" and _associated(item, machine_id)]
    sensor = _sensor_evidence(machine_id, sensor_candidates, model_path=model_path, scaler_path=scaler_path, now=current, max_age_hours=max_age_hours)
    technician = _technician_evidence(machine_id, technician_candidates, now=current, max_age_hours=max_age_hours)
    limitations: list[str] = ["Fusion remains NOT VALIDATED; this result does not alter the independent LSTM RUL."]
    if sensor["status"] != "verified":
        limitations.append(sensor.get("reason", "No verified sensor evidence."))
    if technician["status"] != "verified":
        limitations.append(technician.get("reason", "No verified technician evidence."))
    if sensor.get("is_synthetic") or technician.get("is_synthetic"):
        limitations.append("One or more evidence sources are synthetic/demo data and are not live industrial telemetry.")
    if "benchmark" in str(sensor.get("source_provenance", "")).lower():
        limitations.append("Benchmark-derived sensor evidence preserves its engine identity and is not machine telemetry.")
    has_sensor = sensor["status"] == "verified"
    has_technician = technician["status"] == "verified"
    mode = "both_sources_verified" if has_sensor and has_technician else "sensor_only" if has_sensor else "nlp_only" if has_technician else "no_verified_evidence"
    assistant: dict[str, Any] = {"status": "not_run", "reason": "No verified evidence is available for local model analysis."}
    if include_model and mode != "no_verified_evidence":
        try:
            generated = gateway.generate(_assistant_prompt(machine_id, sensor, technician, limitations))
            assistant = {"status": "ready", "response": generated["text"], "provider": generated["provider"], "model": generated["model"], "response_type": generated["response_type"]}
        except ModelGatewayError as exc:
            assistant = {"status": "unavailable", "reason": str(exc)}
    return {"analysis_id": str(uuid4()), "machine": dict(machine), "machine_id": machine_id, "status": mode, "sensor_evidence": sensor, "technician_evidence": technician, "assistant": assistant, "fusion": {"status": "not_validated", "reason": "No approved trained fusion checkpoint or paired sensor/log dataset."}, "limitations": limitations}
