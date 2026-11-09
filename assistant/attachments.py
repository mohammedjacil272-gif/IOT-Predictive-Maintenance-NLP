from __future__ import annotations
import hashlib
import csv
import io
import json
from pathlib import Path
import re
from uuid import uuid4

ALLOWED = {
    "image/jpeg", "image/png", "application/pdf",
    "text/plain", "text/csv", "application/csv", "application/json", "text/json",
}
MAX_BYTES = 10 * 1024 * 1024

MAGIC = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG\r\n\x1a\n",), "application/pdf": (b"%PDF-",)}
EXTENSION_MIME = {
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain"},
    ".csv": {"text/csv", "application/csv"},
    ".json": {"application/json", "text/json"},
}


def attachment_contract() -> dict[str, object]:
    """Return the shared upload contract without exposing filesystem details."""
    return {
        "max_bytes": MAX_BYTES,
        "formats": {
            extension: {"mime_types": sorted(mimes), "validation": "signature" if extension in {".jpg", ".jpeg", ".png", ".pdf"} else "utf8_and_format"}
            for extension, mimes in sorted(EXTENSION_MIME.items())
        },
        "storage": "local_application_data_outside_static_assets",
        "analysis": "not_performed",
    }

def _signature_matches(data: bytes, mime_type: str) -> bool:
    return any(data.startswith(prefix) for prefix in MAGIC.get(mime_type, ()))


def _content_matches(data: bytes, filename: str, mime_type: str) -> bool:
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSION_MIME or mime_type not in EXTENSION_MIME[extension]:
        return False
    if mime_type in MAGIC:
        return _signature_matches(data, mime_type)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text.strip():
        return False
    if mime_type in {"application/json", "text/json"}:
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return False
    elif mime_type in {"text/csv", "application/csv"}:
        try:
            rows = list(csv.reader(io.StringIO(text)))
        except csv.Error:
            return False
        if not rows or not any(cell.strip() for row in rows for cell in row):
            return False
    return True

def store_attachment(data: bytes, filename: str, mime_type: str, root: str | Path, *, session_id: str, machine_id: str | None = None) -> dict:
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES: raise ValueError("Attachment is missing, empty, or exceeds the 10 MiB limit.")
    if mime_type not in ALLOWED: raise ValueError("Unsupported attachment MIME type.")
    if not isinstance(filename, str) or not filename.strip(): raise ValueError("Attachment filename is required.")
    if not _content_matches(data, filename, mime_type): raise ValueError("Attachment filename, MIME type, or content is invalid.")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    if not safe or safe in {".", ".."}: raise ValueError("Invalid attachment filename.")
    file_id = str(uuid4()); target = Path(root) / file_id
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    return {"file_id": file_id, "original_filename": safe, "stored_filename": file_id, "mime_type": mime_type, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "session_id": session_id, "machine_id": machine_id, "validation_status": "stored_unanalyzed"}
