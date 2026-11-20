from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class AssistantState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class AssistantSession:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    machine_id: str | None = None
    state: AssistantState = AssistantState.IDLE
    pending_confirmation: dict[str, Any] | None = None
    current_transcription: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    provenance: str = "user_submitted"


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    response_type: str
    session_id: str
    state: AssistantState
    machine_id: str | None
    references: tuple[dict[str, Any], ...] = ()
    follow_up_questions: tuple[str, ...] = ()
    confirmation_required: bool = False
    uncertainty: str | None = None
