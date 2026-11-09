"""Small local FastAPI application over the project's existing components.

The API deliberately reports unavailable models instead of inventing outputs.
It can run in database-disabled development mode and serves the static UI.
"""
from __future__ import annotations

import json
import base64
import math
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from nlp.embeddings import embedding_diagnostics
from nlp.ner import extract_entities
from nlp.preprocessing import preprocess_log
from nlp.severity import score_severity
from app.history import LocalHistoryStore
from assistant.confirmation import confirmation_payload, missing_information
from assistant.contracts import AssistantState
from assistant.health import machine_health
from assistant.intents import classify_intent
from assistant.storage import AssistantStorage
from assistant.voice import TranscriptionRequest, UnavailableTranscriptionProvider, UnavailableTTSProvider
from assistant.attachments import attachment_contract, store_attachment
from assistant.gateway import ModelGatewayConfig, ModelGatewayError, OllamaGateway, bounded_context
from assistant.grounding import evidence_summary, grounded_system_prompt
from assistant.tools import MaintenanceToolError, ReadOnlyMaintenanceTools, tool_evidence_text
from assistant.retrieval import LocalKnowledgeIndex, RetrievalError

DATA_ROOT = Path(os.environ.get("IOT_INTEGRATED_ROOT", Path(__file__).resolve().parents[1])).resolve()
# Bundled resources are read-only. In a PyInstaller one-file build, _MEIPASS
# is the temporary extraction directory; in development this resolves to the
# repository root. User state never falls back to this directory.
RESOURCE_ROOT = Path(os.environ.get("IOT_RESOURCE_ROOT", getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))).resolve()
# Backward-compatible read-only resource alias for existing integrations/tests.
ROOT = RESOURCE_ROOT
STATIC = Path(__file__).resolve().parent / "static"
MODEL_PATH = RESOURCE_ROOT / "artifacts/models/fd001_lstm_baseline/best_model.pt"
SCALER_PATH = RESOURCE_ROOT / "artifacts/data/fd001_lstm_baseline/scaler.joblib"
SENSOR_EMBEDDING_PATH = RESOURCE_ROOT / "artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy"
FUSION_ARTIFACT_ROOT = RESOURCE_ROOT / "artifacts/models/fusion"
VALIDATION_METRICS_PATH = RESOURCE_ROOT / "artifacts/metrics/fd001_lstm_baseline/validation_metrics.json"
OFFICIAL_TEST_METRICS_PATH = RESOURCE_ROOT / "artifacts/metrics/fd001_lstm_baseline/test_metrics.json"
HISTORY_PATH = DATA_ROOT / ".local_history.json"
HISTORY = LocalHistoryStore(HISTORY_PATH)
ASSISTANT_STORAGE = AssistantStorage(DATA_ROOT / ".assistant.sqlite3")
MODEL_GATEWAY = OllamaGateway(ModelGatewayConfig.from_env())
KNOWLEDGE_INDEX = LocalKnowledgeIndex(DATA_ROOT / ".knowledge", allowed_roots=(RESOURCE_ROOT / "docs",))


def fusion_checkpoint_present() -> bool:
    """Return whether a candidate fusion checkpoint exists under the resource root."""
    return FUSION_ARTIFACT_ROOT.is_dir() and any(FUSION_ARTIFACT_ROOT.glob("*.pt"))


def official_test_evaluation() -> dict[str, Any]:
    """Report official-test state from an explicit artifact, never inference."""
    if not OFFICIAL_TEST_METRICS_PATH.is_file():
        return {
            "status": "not_run",
            "split": "official_test",
            "metrics_path": str(OFFICIAL_TEST_METRICS_PATH),
            "reason": "Official FD001 test evaluation has not been run; validation metrics are not official-test metrics.",
        }
    try:
        payload = json.loads(OFFICIAL_TEST_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "split": "official_test", "reason": f"Official-test metrics are unreadable: {exc}"}
    required = {"mae", "rmse", "mse", "official_test_labels_used"}
    if not required.issubset(payload):
        return {"status": "failed", "split": "official_test", "reason": "Official-test metrics artifact is incomplete."}
    return {
        "status": "completed",
        "split": "official_test",
        "metrics_path": str(OFFICIAL_TEST_METRICS_PATH),
        "metrics": {key: payload[key] for key in ("mae", "rmse", "mse")},
        "evaluated_at": payload.get("evaluated_at"),
        "checkpoint_id": payload.get("checkpoint_id"),
    }


def iot_configuration_status() -> dict[str, Any]:
    """Expose configuration readiness without probing or revealing secrets."""
    mqtt_ready = all(os.environ.get(key, "").strip() for key in ("IOT_MQTT_HOST", "IOT_MQTT_TOPIC"))
    influx_ready = all(os.environ.get(key, "").strip() for key in ("IOT_INFLUXDB_URL", "IOT_INFLUXDB_ORG", "IOT_INFLUXDB_BUCKET", "IOT_INFLUXDB_TOKEN"))
    return {
        "status": "configured" if mqtt_ready and influx_ready else "local_replay_only",
        "local_replay": "available",
        # Keep the existing public status values stable; configuration details
        # are additive and do not imply a live connection.
        "mqtt": "not_checked",
        "influxdb": "not_checked",
        "configuration": {"mqtt_configured": mqtt_ready, "influxdb_configured": influx_ready},
    }

app = FastAPI(title="IoT Predictive Maintenance", version="0.1.0")
API_TOKEN = os.environ.get("IOT_API_TOKEN")


@app.middleware("http")
async def local_api_token(request: Request, call_next):
    if API_TOKEN and request.url.path.startswith("/api") and request.url.path != "/api/health":
        if request.headers.get("authorization") != f"Bearer {API_TOKEN}":
            return JSONResponse(status_code=401, content={"detail": "local API authorization required"})
    return await call_next(request)


class SensorRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


class TechnicianLogRequest(BaseModel):
    log_id: str = Field(min_length=1)
    text: str
    unit_id: str | int | None = None
    event_time: str | None = None
    available_at: str | None = None
    source_provenance: str = "user_submitted"
    is_synthetic: bool = False


class MachineRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=256)
    machine_type: str | None = None
    location: str | None = None
    is_synthetic: bool = False


class MachinePatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    machine_type: str | None = None
    location: str | None = None
    is_synthetic: bool | None = None


class SessionRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=128)


class MessageRequest(BaseModel):
    role: str = "user"
    text: str = Field(min_length=1, max_length=10000)


class ConfirmationDecision(BaseModel):
    action: str = Field(pattern="^(confirm|reject|cancel)$")
    interpretation: dict[str, Any] | None = None


class AttachmentRequest(BaseModel):
    session_id: str = Field(min_length=1)
    machine_id: str | None = None
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=128)
    data_base64: str = Field(min_length=1)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    include_retrieval: bool = True


class KnowledgePathRequest(BaseModel):
    source_path: str = Field(min_length=1)


class FusionInferRequest(BaseModel):
    """Optional precomputed fusion inputs for contract-only validation."""

    sensor_embedding: list[float] | None = Field(default=None, min_length=64, max_length=64)
    text_embedding: list[float] | None = Field(default=None, min_length=768, max_length=768)
    unit_id: str | int | None = None
    log_id: str | None = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/static/{filename}", include_in_schema=False)
def static_file(filename: str) -> FileResponse:
    candidate = (STATIC / filename).resolve()
    if STATIC.resolve() not in candidate.parents:
        raise HTTPException(status_code=404, detail="Asset not found")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(candidate)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "database_mode": "disabled_local", "version": app.version}


@app.get("/api/status")
def status() -> dict[str, Any]:
    test_evaluation = official_test_evaluation()
    return {
        "sensor": {
            "mode": "fd001_lstm",
            "status": "ready" if SCALER_PATH.exists() and MODEL_PATH.exists() else "blocked_missing_checkpoint",
            "checkpoint_present": MODEL_PATH.exists(),
            "scaler_present": SCALER_PATH.exists(),
            "official_test_evaluation": test_evaluation["status"],
            "validation_metrics": {
                "status": "available" if VALIDATION_METRICS_PATH.is_file() else "unavailable",
                "split": "validation",
                "metrics_path": str(VALIDATION_METRICS_PATH),
            },
            "official_test_evaluation_detail": test_evaluation,
        },
        "nlp": {"status": "rule_based_ready", "embedding": embedding_diagnostics()},
        "fusion": {
            "status": "not_validated",
            "prediction_available": False,
            "reason": "No approved trained fusion checkpoint is present; synthetic forward passes are not production predictions.",
        },
        "iot": iot_configuration_status(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {
        "modes": ["sensor_only", "nlp_only", "fusion"],
        "sensor_only": {"model": "LSTMBaseline", "input_shape": [30, 17], "status": "available" if MODEL_PATH.exists() else "unavailable"},
        "nlp_only": {"model": "rule_based_contract", "status": "available"},
        "fusion": {"model": "MultimodalFusionModel", "status": "untrained_or_unvalidated"},
    }


@app.get("/api/attachments/contract")
def attachments_contract() -> dict[str, object]:
    return attachment_contract()


@app.get("/api/fusion/readiness")
def fusion_readiness() -> dict[str, Any]:
    """Expose evidence-backed readiness conditions without implying a prediction exists."""
    embedding = embedding_diagnostics()
    conditions = {
        "sensor_checkpoint": {
            "status": "READY" if MODEL_PATH.is_file() and SCALER_PATH.is_file() else "BLOCKED",
            "detail": "FD001 baseline checkpoint and scaler are present." if MODEL_PATH.is_file() and SCALER_PATH.is_file() else "FD001 checkpoint or scaler is missing.",
        },
        "sensor_embeddings": {
            "status": "READY" if SENSOR_EMBEDDING_PATH.is_file() else "BLOCKED",
            "detail": "Existing 64-D sensor embedding artifact is present." if SENSOR_EMBEDDING_PATH.is_file() else "Sensor embedding artifact is missing.",
        },
        "text_embedding_dependency": {
            "status": "READY" if embedding["transformers_available"] else "BLOCKED",
            "detail": "Transformers dependency is installed." if embedding["transformers_available"] else "transformers is not installed; no download was attempted.",
        },
        "local_transformer_weights": {
            "status": "READY" if embedding["ready"] else "BLOCKED",
            "detail": "Local DistilBERT weights are loadable." if embedding["ready"] else "Approved local DistilBERT weights are unavailable.",
        },
        "paired_dataset": {"status": "BLOCKED", "detail": "No approved paired sensor/log dataset is present."},
        "machine_identity_mapping": {"status": "BLOCKED", "detail": "No approved crosswalk links FD001 engines to technician entities."},
        "temporal_alignment_policy": {"status": "NOT_CONFIGURED", "detail": "Shared event/cycle semantics are not approved."},
        "prediction_cutoff_policy": {"status": "NOT_CONFIGURED", "detail": "Point-in-time cutoff policy is not approved for a genuine run."},
        "provenance_validation": {"status": "BLOCKED", "detail": "Available NLP package is synthetic and lacks genuine provenance."},
        "leakage_safe_splits": {"status": "NOT_VALIDATED", "detail": "No approved paired-entity split exists."},
        "fusion_checkpoint": {"status": "BLOCKED", "detail": "No verified trained fusion checkpoint is present."},
        "genuine_evaluation": {"status": "BLOCKED", "detail": "No genuine paired evaluation has been authorized or run."},
    }
    return {
        "status": "not_validated",
        "ready_for_genuine_fusion": False,
        "checkpoint_present": fusion_checkpoint_present(),
        "paired_data": "not_ready",
        "conditions": conditions,
        "reason": "Overall readiness is blocked until every mandatory condition is READY; no genuine fusion prediction is available.",
    }


@app.post("/api/sensor/infer")
def sensor_infer(request: SensorRequest) -> dict[str, Any]:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise HTTPException(status_code=503, detail="Sensor model artifacts are unavailable in this worktree")
    try:
        from iot.lstm_inference import predict_records

        result = predict_records(request.records, scaler_path=SCALER_PATH, model_path=MODEL_PATH)
        response = {"mode": "sensor_only", "model_id": "fd001_lstm_baseline", "model_status": "ready",
                "unit_id": result.unit_id, "window_start_cycle": result.window_start_cycle,
                "window_end_cycle": result.window_end_cycle, "prediction": result.prediction,
                "prediction_semantics": "FD001 uncapped RUL cycles from the validation-trained baseline; official-test evaluation status is reported separately and this prediction is not an official test result"}
        HISTORY.append(mode="sensor_only", model_id="fd001_lstm_baseline", request={"records": request.records},
                       response=response, source_provenance="user_submitted", is_synthetic=False)
        return response
    except Exception as exc:  # adapter normalizes user-facing validation errors
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/technician-log/analyze")
def analyze_log(request: TechnicianLogRequest) -> dict[str, Any]:
    raw = request.model_dump()
    try:
        processed = preprocess_log(raw)
        text = processed["original_text"] if isinstance(processed.get("original_text"), str) else ""
        response = {
            "mode": "nlp_only",
            "model_id": "rule_based_nlp_v1",
            "model_status": "ready",
            "record": processed,
            "entities": extract_entities(text, log_id=request.log_id),
            "severity": score_severity(text),
            "embedding": {"status": "not_generated", "diagnostics": embedding_diagnostics(),
                          "reason": "Transformer embeddings require an approved local model and optional dependency."},
        }
        HISTORY.append(mode="nlp_only", model_id="rule_based_nlp_v1", request=raw, response=response,
                       source_provenance=request.source_provenance, is_synthetic=request.is_synthetic)
        return response
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/fusion/infer")
def fusion_infer(request: FusionInferRequest) -> dict[str, Any]:
    supplied = request.sensor_embedding is not None or request.text_embedding is not None
    if supplied and (request.sensor_embedding is None or request.text_embedding is None):
        raise HTTPException(status_code=422, detail="sensor_embedding and text_embedding must be supplied together")
    if supplied:
        vectors = [*request.sensor_embedding, *request.text_embedding]  # type: ignore[arg-type]
        if not all(math.isfinite(value) for value in vectors):
            raise HTTPException(status_code=422, detail="Fusion embeddings must contain only finite values")
    return {
        "mode": "fusion",
        "status": "not_validated",
        "prediction": None,
        "validated_inputs": {
            "sensor_embedding_dim": len(request.sensor_embedding) if request.sensor_embedding is not None else None,
            "text_embedding_dim": len(request.text_embedding) if request.text_embedding is not None else None,
            "unit_id": request.unit_id,
            "log_id": request.log_id,
        },
        "reason": "No approved trained fusion checkpoint and no approved real paired data are available.",
    }


@app.post("/api/infer")
def unified_infer(payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("mode")
    if mode == "sensor_only":
        return sensor_infer(SensorRequest.model_validate(payload))
    if mode == "nlp_only":
        return analyze_log(TechnicianLogRequest.model_validate(payload))
    if mode == "multimodal_fusion":
        return fusion_infer(FusionInferRequest.model_validate(payload))
    raise HTTPException(status_code=422, detail="mode must be sensor_only, nlp_only, or multimodal_fusion")


@app.get("/api/history")
def history() -> dict[str, Any]:
    try:
        return {"records": HISTORY.list()}
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/history")
def clear_history() -> dict[str, Any]:
    try:
        HISTORY.clear()
        return {"cleared": True, "records": []}
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/diagnostics")
def diagnostics() -> dict[str, Any]:
    return {"root": str(ROOT), "python": __import__("sys").version, "model_path": str(MODEL_PATH),
            "scaler_path": str(SCALER_PATH), "environment": "local_development_only"}


@app.get("/api/assistant/model-status")
def assistant_model_status() -> dict[str, Any]:
    """Report local model availability without loading or downloading weights."""
    return MODEL_GATEWAY.status()


@app.post("/api/assistant/sessions/{session_id}/chat")
def chat_session(session_id: str, request: ChatRequest) -> dict[str, Any]:
    session = ASSISTANT_STORAGE.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["state"] == "CANCELLED":
        raise HTTPException(status_code=409, detail="Session is cancelled")
    try:
        ASSISTANT_STORAGE.set_state(session_id, "PROCESSING")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ASSISTANT_STORAGE.add_message(session_id, "user", request.text, {"provenance": "user_submitted"})
    messages = ASSISTANT_STORAGE.list_messages(session_id)
    references = KNOWLEDGE_INDEX.search(request.text) if request.include_retrieval else []
    tool_activity: list[dict[str, Any]] = []
    evidence = list(references)
    if session.get("machine_id"):
        try:
            health = ReadOnlyMaintenanceTools(ASSISTANT_STORAGE).machine_health(session["machine_id"])
            evidence.append({"source_name": "machine_health_tool", "line_start": 1, "line_end": 1, "text": tool_evidence_text(health)})
            tool_activity.append({"name": "machine_health", "status": "read_only_success"})
        except MaintenanceToolError as exc:
            tool_activity.append({"name": "machine_health", "status": "unavailable", "reason": str(exc)})
    system = grounded_system_prompt(machine_id=session.get("machine_id"), evidence=evidence)
    prompt = bounded_context([{"role": "system", "text": system}, *messages])
    try:
        result = MODEL_GATEWAY.generate(prompt)
    except ModelGatewayError as exc:
        try:
            ASSISTANT_STORAGE.set_state(session_id, "ERROR")
        except ValueError:
            pass
        return {"session_id": session_id, "state": AssistantState.ERROR, "response": None, "response_type": "model_unavailable", "references": references, "reason": str(exc)}
    ASSISTANT_STORAGE.add_message(session_id, "assistant", result["text"], {"response_type": result["response_type"], "provider": result["provider"], "model": result["model"], "references": references})
    ASSISTANT_STORAGE.set_state(session_id, "RESPONDING")
    return {"session_id": session_id, "state": AssistantState.RESPONDING, "response": result["text"], "response_type": result["response_type"], "references": references, "grounding": evidence_summary(machine_id=session.get("machine_id"), references=evidence), "tool_activity": tool_activity, "provider": result["provider"], "model": result["model"]}


@app.post("/api/knowledge/ingest")
def ingest_knowledge(request: KnowledgePathRequest) -> dict[str, Any]:
    try:
        return KNOWLEDGE_INDEX.ingest(request.source_path)
    except (OSError, RetrievalError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/knowledge/search")
def search_knowledge(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")
    return {"query": query, "results": KNOWLEDGE_INDEX.search(query, limit=payload.get("limit", 5))}


@app.delete("/api/knowledge/source")
def remove_knowledge(request: KnowledgePathRequest) -> dict[str, Any]:
    try:
        return KNOWLEDGE_INDEX.remove(request.source_path)
    except OSError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/assistant/transcribe")
def assistant_transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    request = TranscriptionRequest(typed_text=payload.get("typed_text"), content_type=payload.get("content_type"), duration_seconds=payload.get("duration_seconds"), provenance=payload.get("provenance", "user_submitted"))
    return {"state": AssistantState.PROCESSING, "transcription": UnavailableTranscriptionProvider().transcribe(request)}


@app.get("/api/assistant/voice-status")
def assistant_voice_status() -> dict[str, Any]:
    return {"input": {"status": "typed_fallback", "audio_provider": "unavailable"}, "output": {"status": "unavailable", "provider": "none"}}


@app.post("/api/assistant/speak")
def assistant_speak(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    return UnavailableTTSProvider().synthesize(text)


@app.post("/api/assistant/analyze")
def assistant_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    session_id = payload.get("session_id") or "local-session"
    machine_id = payload.get("machine_id")
    processed = preprocess_log({"log_id": payload.get("log_id", session_id), "text": text})
    entities = extract_entities(text, log_id=processed["log_id"])
    severity = score_severity(text)
    component = next((e["entity_text"] for e in entities if e["label"] == "COMPONENT"), None)
    interpretation = {"machine_id": machine_id, "component": component, "severity": severity["severity_label"], "entities": entities}
    result = confirmation_payload(text, interpretation)
    return {"session_id": session_id, "state": AssistantState.RESPONDING, "intent": classify_intent(text), "analysis": {"record": processed, "entities": entities, "severity": severity}, "confirmation": result, "provenance": payload.get("provenance", "user_submitted"), "is_synthetic": bool(payload.get("is_synthetic", False))}


@app.get("/api/machines/{machine_id}/health")
def assistant_machine_health(machine_id: str) -> dict[str, Any]:
    return machine_health(ASSISTANT_STORAGE.get_machine(machine_id), observations=ASSISTANT_STORAGE.list_observations(machine_id))


@app.get("/api/machines")
def list_machines() -> dict[str, Any]:
    return {"machines": ASSISTANT_STORAGE.list_machines()}


@app.post("/api/machines", status_code=201)
def register_machine(request: MachineRequest) -> dict[str, Any]:
    payload = request.model_dump()
    machine_id = payload.pop("machine_id")
    try:
        ASSISTANT_STORAGE.save_machine(machine_id, {"machine_id": machine_id, **payload})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ASSISTANT_STORAGE.get_machine(machine_id)


@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str) -> dict[str, Any]:
    result = ASSISTANT_STORAGE.get_machine(machine_id)
    if result is None: raise HTTPException(status_code=404, detail="Machine not registered")
    return result


@app.patch("/api/machines/{machine_id}")
def update_machine(machine_id: str, request: MachinePatchRequest) -> dict[str, Any]:
    current = ASSISTANT_STORAGE.get_machine(machine_id)
    if current is None: raise HTTPException(status_code=404, detail="Machine not registered")
    current.update(request.model_dump(exclude_unset=True))
    try: ASSISTANT_STORAGE.update_machine(machine_id, current)
    except (KeyError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    return current


@app.post("/api/assistant/sessions", status_code=201)
def create_session(request: SessionRequest) -> dict[str, Any]:
    if ASSISTANT_STORAGE.get_machine(request.machine_id) is None: raise HTTPException(status_code=404, detail="Machine not registered")
    session_id = str(uuid4())
    ASSISTANT_STORAGE.save_session(session_id, request.machine_id, "IDLE", {"provenance": "user_submitted"})
    return ASSISTANT_STORAGE.get_session(session_id)


@app.get("/api/assistant/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    result = ASSISTANT_STORAGE.get_session(session_id)
    if result is None: raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.get("/api/assistant/sessions/{session_id}/messages")
def get_session_messages(session_id: str) -> dict[str, Any]:
    if ASSISTANT_STORAGE.get_session(session_id) is None: raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": ASSISTANT_STORAGE.list_messages(session_id)}


@app.post("/api/assistant/sessions/{session_id}/messages")
def append_session_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
    if request.role not in {"user", "assistant", "system"}: raise HTTPException(status_code=422, detail="Invalid message role")
    try: message_id = ASSISTANT_STORAGE.add_message(session_id, request.role, request.text)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"id": message_id, "session_id": session_id, "role": request.role, "text": request.text}


@app.post("/api/assistant/sessions/{session_id}/cancel")
def cancel_session(session_id: str) -> dict[str, Any]:
    try: ASSISTANT_STORAGE.set_state(session_id, "CANCELLED")
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ASSISTANT_STORAGE.get_session(session_id)


@app.post("/api/assistant/sessions/{session_id}/analyze")
def analyze_session(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = ASSISTANT_STORAGE.get_session(session_id)
    if session is None: raise HTTPException(status_code=404, detail="Session not found")
    if session["state"] in {"IDLE", "RESPONDING"}:
        ASSISTANT_STORAGE.set_state(session_id, "PROCESSING")
    payload["session_id"] = session_id; payload.setdefault("machine_id", session["machine_id"])
    result = assistant_analyze(payload)
    observation_id, confirmation_id = str(uuid4()), str(uuid4())
    ASSISTANT_STORAGE.add_message(session_id, "user", str(payload["text"]), {"provenance": result["provenance"], "is_synthetic": result["is_synthetic"]})
    ASSISTANT_STORAGE.add_message(session_id, "assistant", "Interpretation is ready for technician confirmation.", {"confirmation_id": confirmation_id})
    ASSISTANT_STORAGE.save_observation(observation_id, session_id, session["machine_id"], "pending", result["confirmation"])
    ASSISTANT_STORAGE.save_confirmation(confirmation_id, observation_id, session_id, {"status": "pending", "original": result["confirmation"]})
    ASSISTANT_STORAGE.set_state(session_id, "RESPONDING")
    return {**result, "observation_id": observation_id, "confirmation_id": confirmation_id}


@app.get("/api/assistant/confirmations/{confirmation_id}")
def get_confirmation(confirmation_id: str) -> dict[str, Any]:
    result = ASSISTANT_STORAGE.get_confirmation(confirmation_id)
    if result is None: raise HTTPException(status_code=404, detail="Confirmation not found")
    return result


@app.post("/api/assistant/confirmations/{confirmation_id}/{action}")
def decide_confirmation(confirmation_id: str, action: str, decision: ConfirmationDecision | None = None) -> dict[str, Any]:
    if action not in {"confirm", "reject", "cancel"}: raise HTTPException(status_code=422, detail="Unsupported confirmation action")
    confirmation = ASSISTANT_STORAGE.get_confirmation(confirmation_id)
    if confirmation is None: raise HTTPException(status_code=404, detail="Confirmation not found")
    status = "confirmed" if action == "confirm" else "rejected" if action == "reject" else "cancelled"
    payload = confirmation.get("original", {})
    if decision and decision.interpretation is not None: payload = {**payload, "interpretation": decision.interpretation}
    try:
        ASSISTANT_STORAGE.save_observation(confirmation["observation_id"], confirmation["session_id"], confirmation.get("machine_id"), status, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"confirmation_id": confirmation_id, "observation_id": confirmation["observation_id"], "status": status, "original_transcript": payload.get("original_transcript")}


@app.post("/api/attachments", status_code=201)
def upload_attachment(request: AttachmentRequest) -> dict[str, Any]:
    if ASSISTANT_STORAGE.get_session(request.session_id) is None: raise HTTPException(status_code=404, detail="Session not found")
    if request.machine_id and ASSISTANT_STORAGE.get_machine(request.machine_id) is None: raise HTTPException(status_code=404, detail="Machine not registered")
    try: data = base64.b64decode(request.data_base64, validate=True)
    except (ValueError, TypeError) as exc: raise HTTPException(status_code=422, detail="data_base64 is invalid") from exc
    try:
        metadata = store_attachment(data, request.filename, request.mime_type, DATA_ROOT / ".attachments", session_id=request.session_id, machine_id=request.machine_id)
        try:
            ASSISTANT_STORAGE.save_attachment(metadata["file_id"], request.session_id, request.machine_id, metadata)
        except Exception:
            (DATA_ROOT / ".attachments" / metadata["file_id"]).unlink(missing_ok=True)
            raise
    except (ValueError, KeyError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {key: value for key, value in metadata.items() if key not in {"stored_filename"}}


@app.get("/api/attachments/{file_id}")
def get_attachment(file_id: str):
    metadata = ASSISTANT_STORAGE.get_attachment(file_id)
    if metadata is None: raise HTTPException(status_code=404, detail="Attachment not found")
    path = DATA_ROOT / ".attachments" / file_id
    if not path.is_file(): raise HTTPException(status_code=410, detail="Attachment content is unavailable")
    return FileResponse(path, media_type=metadata.get("mime_type", "application/octet-stream"), filename=metadata.get("original_filename"))
