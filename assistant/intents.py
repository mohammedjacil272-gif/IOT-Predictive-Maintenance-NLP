from __future__ import annotations
import re

INTENTS = ("machine_rundown", "health_status", "faults", "maintenance_summary", "link_observation", "missing_information", "session_summary", "observation")

def classify_intent(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): raise ValueError("text must be non-empty")
    t = text.lower()
    if "missing" in t or "what information" in t: return "missing_information"
    if "rundown" in t or "overview" in t: return "machine_rundown"
    if "health" in t: return "health_status"
    if "fault" in t or "failure" in t: return "faults"
    if "maintenance" in t or "repair" in t: return "maintenance_summary"
    if "summarize" in t or "reported so far" in t: return "session_summary"
    if "link" in t or "associate" in t: return "link_observation"
    return "observation"
