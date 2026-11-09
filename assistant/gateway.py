"""Provider-independent local conversational model gateway.

The default adapter speaks to an already-running Ollama-compatible HTTP API.
It never downloads models and does not log prompts, responses, or credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable
from urllib import error, request


class ModelGatewayError(RuntimeError):
    """A safe, user-facing model gateway failure."""


@dataclass(frozen=True)
class ModelGatewayConfig:
    provider: str = "ollama"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 20.0
    max_response_tokens: int | None = None
    temperature: float = 0.2
    retries: int = 1

    @classmethod
    def from_env(cls) -> "ModelGatewayConfig":
        raw_retries = os.environ.get("IOT_ASSISTANT_MODEL_RETRIES", "1")
        try:
            retries = max(0, min(3, int(raw_retries)))
        except ValueError:
            retries = 1
        raw_timeout = os.environ.get("IOT_ASSISTANT_MODEL_TIMEOUT", "20")
        try:
            timeout = max(0.1, min(120.0, float(raw_timeout)))
        except ValueError:
            timeout = 20.0
        raw_temperature = os.environ.get("IOT_ASSISTANT_TEMPERATURE", "0.2")
        try:
            temperature = max(0.0, min(2.0, float(raw_temperature)))
        except ValueError:
            temperature = 0.2
        return cls(
            provider=os.environ.get("IOT_ASSISTANT_MODEL_PROVIDER", "ollama").strip().lower(),
            model=os.environ.get("IOT_ASSISTANT_MODEL", "").strip(),
            base_url=os.environ.get("IOT_ASSISTANT_MODEL_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            timeout_seconds=timeout,
            max_response_tokens=(int(os.environ["IOT_ASSISTANT_MAX_RESPONSE_TOKENS"]) if os.environ.get("IOT_ASSISTANT_MAX_RESPONSE_TOKENS", "").isdigit() else None),
            temperature=temperature,
            retries=retries,
        )


def bounded_context(messages: list[dict[str, Any]], *, max_messages: int = 12, max_chars: int = 12000) -> list[dict[str, str]]:
    """Select recent messages without sending an entire local history."""
    selected: list[dict[str, str]] = []
    remaining = max(1, max_chars)
    for message in reversed(messages[-max(1, max_messages):]):
        role = str(message.get("role", "user"))
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        text = str(message.get("text", ""))
        if not text:
            continue
        text = text[:remaining]
        selected.append({"role": role, "content": text})
        remaining -= len(text)
        if remaining <= 0:
            break
    return list(reversed(selected))


class OllamaGateway:
    def __init__(self, config: ModelGatewayConfig, *, opener: Callable[..., Any] | None = None):
        self.config = config
        self._opener = opener or request.urlopen

    def status(self) -> dict[str, Any]:
        if self.config.provider != "ollama":
            return {"status": "unavailable", "provider": self.config.provider, "reason": "Only the local Ollama adapter is implemented."}
        if not self.config.model:
            return {"status": "unconfigured", "provider": "ollama", "reason": "Set IOT_ASSISTANT_MODEL to an installed local model name."}
        try:
            payload = self._request("/api/tags", {})
            names = [str(item.get("name", "")) for item in payload.get("models", []) if isinstance(item, dict)]
            available = self.config.model in names or any(name.split(":", 1)[0] == self.config.model for name in names)
            return {"status": "available" if available else "model_unavailable", "provider": "ollama", "model": self.config.model, "installed_models": names}
        except ModelGatewayError as exc:
            return {"status": "unavailable", "provider": "ollama", "model": self.config.model, "reason": str(exc)}

    def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config.provider != "ollama":
            raise ModelGatewayError("Configured model provider is not implemented locally.")
        if not self.config.model:
            raise ModelGatewayError("No local model is configured; set IOT_ASSISTANT_MODEL.")
        options: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.max_response_tokens is not None:
            options["num_predict"] = self.config.max_response_tokens
        payload = self._request("/api/chat", {"model": self.config.model, "messages": messages, "stream": False, "options": options})
        content = payload.get("message", {}).get("content") if isinstance(payload.get("message"), dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("The model returned an empty response.")
        return {"text": content, "provider": "ollama", "model": self.config.model, "response_type": "model"}

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                req = request.Request(self.config.base_url + path, data=body, headers={"content-type": "application/json"}, method="POST")
                if path == "/api/tags":
                    req = request.Request(self.config.base_url + path, headers={"accept": "application/json"}, method="GET")
                with self._opener(req, timeout=self.config.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ModelGatewayError("Model service returned an invalid JSON object.")
                return decoded
            except (TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError, OSError, ModelGatewayError) as exc:
                last = exc
                if attempt < self.config.retries:
                    time.sleep(min(0.25 * (attempt + 1), 1.0))
        if isinstance(last, ModelGatewayError):
            raise last
        raise ModelGatewayError("Local model service is unavailable or timed out.") from last
