"""Small local JSON history store for successful non-fusion analyses."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class LocalHistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Local history is unreadable; clear it before continuing.") from exc
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("Local history must contain a JSON list of records.")
        return value

    def append(self, *, mode: str, model_id: str, request: dict[str, Any], response: dict[str, Any],
               source_provenance: str, is_synthetic: bool) -> dict[str, Any]:
        if mode not in {"sensor_only", "nlp_only"}:
            raise ValueError("Only successful sensor_only and nlp_only results may be stored.")
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "model_id": model_id,
            "source_provenance": source_provenance,
            "is_synthetic": bool(is_synthetic),
            "request": request,
            "response": response,
        }
        records = self._read()
        records.append(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return item

    def list(self) -> list[dict[str, Any]]:
        return self._read()

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError("Unable to clear local history.") from exc
