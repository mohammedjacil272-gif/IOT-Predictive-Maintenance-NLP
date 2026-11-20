"""Allowlisted, read-only maintenance evidence tools for assistant grounding."""

from __future__ import annotations

import json
from typing import Any

from assistant.health import machine_health


class MaintenanceToolError(ValueError):
    """Raised for an unknown or invalid assistant tool request."""


class ReadOnlyMaintenanceTools:
    """Expose current application evidence without mutation or shell access."""

    def __init__(self, storage: Any):
        self.storage = storage

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "machine_health", "read_only": True, "requires_confirmation": False}]

    def machine_health(self, machine_id: str) -> dict[str, Any]:
        if not isinstance(machine_id, str) or not machine_id.strip():
            raise MaintenanceToolError("machine_id must be non-empty")
        return machine_health(
            self.storage.get_machine(machine_id),
            observations=self.storage.list_observations(machine_id),
        )

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != "machine_health":
            raise MaintenanceToolError(f"Unsupported assistant tool: {name}")
        return self.machine_health(arguments.get("machine_id"))


def tool_evidence_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
