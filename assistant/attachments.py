from __future__ import annotations
import hashlib
from pathlib import Path
import re
from uuid import uuid4

ALLOWED = {"image/jpeg", "image/png", "application/pdf"}
MAX_BYTES = 10 * 1024 * 1024

MAGIC = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG\r\n\x1a\n",), "application/pdf": (b"%PDF-",)}

def _signature_matches(data: bytes, mime_type: str) -> bool:
    return any(data.startswith(prefix) for prefix in MAGIC.get(mime_type, ()))

def store_attachment(data: bytes, filename: str, mime_type: str, root: str | Path, *, session_id: str, machine_id: str | None = None) -> dict:
    if not isinstance(data, bytes) or len(data) > MAX_BYTES: raise ValueError("Attachment is missing or exceeds the 10 MiB limit.")
    if mime_type not in ALLOWED: raise ValueError("Unsupported attachment MIME type.")
    if not _signature_matches(data, mime_type): raise ValueError("Attachment content does not match its declared MIME type.")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    if not safe or safe in {".", ".."}: raise ValueError("Invalid attachment filename.")
    file_id = str(uuid4()); target = Path(root) / file_id
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    return {"file_id": file_id, "original_filename": safe, "stored_filename": file_id, "mime_type": mime_type, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "session_id": session_id, "machine_id": machine_id, "validation_status": "stored_unanalyzed"}
