from __future__ import annotations
from typing import Any

def missing_information(interpretation: dict[str, Any]) -> list[str]:
    missing=[]
    if not interpretation.get("machine_id"): missing.append("Which machine is this about?")
    if interpretation.get("component") and not interpretation.get("observation_time"): missing.append("When was this observed?")
    if interpretation.get("component") in {"bearing", "bearings"} and not interpretation.get("supporting_symptoms"): missing.append("Were there vibration, temperature, noise, or inspection findings?")
    if interpretation.get("temperature_issue") and interpretation.get("measurement") is None: missing.append("Was a temperature measurement recorded?")
    if interpretation.get("maintenance_action") and not interpretation.get("action_result"): missing.append("What was the result of the maintenance action?")
    return missing

def confirmation_payload(transcript: str, interpretation: dict[str, Any]) -> dict[str, Any]:
    return {"original_transcript": transcript, "interpretation": interpretation, "missing_information": missing_information(interpretation), "confirmation_question": "Is this what you meant?"}
