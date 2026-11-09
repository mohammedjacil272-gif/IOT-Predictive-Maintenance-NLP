from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class TranscriptionRequest:
    audio_bytes: bytes | None = None
    content_type: str | None = None
    duration_seconds: float | None = None
    typed_text: str | None = None
    provenance: str = "user_submitted"

class TranscriptionProvider(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]: ...

class UnavailableTranscriptionProvider:
    def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        if request.typed_text and request.typed_text.strip():
            return {"status": "typed_input", "text": request.typed_text, "provenance": request.provenance}
        return {"status": "unavailable", "text": None, "reason": "No local speech provider is configured; typed input is available."}

class UnavailableTTSProvider:
    def synthesize(self, text: str) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "No local text-to-speech provider is configured.", "text": text}
