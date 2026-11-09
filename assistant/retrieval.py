"""Small local, source-grounded document index.

This implementation intentionally uses deterministic lexical retrieval. A real
embedding provider can be added behind the same interface later; fixture scores
must not be described as neural embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}
EXCLUDED_NAMES = {".env", ".assistant.sqlite3", ".local_history.json"}
MAX_BYTES = 2 * 1024 * 1024


class RetrievalError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_name: str
    source_path: str
    line_start: int
    line_end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "source_name": self.source_name, "source_path": self.source_path, "line_start": self.line_start, "line_end": self.line_end, "text": self.text}


class LocalKnowledgeIndex:
    def __init__(self, root: str | Path, *, allowed_roots: tuple[str | Path, ...] = ()):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.allowed_roots = tuple(Path(item).resolve() for item in allowed_roots)
        self._chunks: list[Chunk] = []
        self._load()

    def ingest(self, path: str | Path) -> dict[str, Any]:
        source = Path(path).resolve()
        self._check_path(source)
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            raise RetrievalError("Only Markdown and text documents are supported by the local reader.")
        if source.stat().st_size > MAX_BYTES:
            raise RetrievalError("Document exceeds the 2 MiB local ingestion limit.")
        text = source.read_text(encoding="utf-8")
        if not text.strip():
            raise RetrievalError("Document is empty.")
        self._chunks = [chunk for chunk in self._chunks if chunk.source_path != str(source)]
        lines = text.splitlines()
        size = 40
        for start in range(0, len(lines), size):
            body = "\n".join(lines[start:start + size]).strip()
            if not body:
                continue
            digest = hashlib.sha256(f"{source}\0{start}\0{body}".encode()).hexdigest()[:20]
            self._chunks.append(Chunk(digest, source.name, str(source), start + 1, min(len(lines), start + size), body))
        self._save()
        return {"status": "indexed", "source_name": source.name, "source_path": str(source), "chunks": sum(c.source_path == str(source) for c in self._chunks)}

    def remove(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path).resolve()
        before = len(self._chunks)
        self._chunks = [chunk for chunk in self._chunks if chunk.source_path != str(source)]
        self._save()
        return {"status": "removed" if len(self._chunks) != before else "not_found", "source_path": str(source)}

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
        if not tokens:
            return []
        scored: list[tuple[int, Chunk]] = []
        for chunk in self._chunks:
            words = set(re.findall(r"[a-z0-9_]+", chunk.text.lower()))
            score = len(tokens & words)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].source_path, item[1].line_start))
        return [{**chunk.to_dict(), "score": score, "retrieval": "lexical_testable_provider"} for score, chunk in scored[:max(1, min(limit, 20))]]

    def _check_path(self, source: Path) -> None:
        if source.name in EXCLUDED_NAMES or source.name.startswith(".env") or source.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            raise RetrievalError("Private or database files cannot be indexed.")
        if not self.allowed_roots:
            raise RetrievalError("No approved knowledge source root is configured.")
        if not any(root == source.parent or root in source.parents for root in self.allowed_roots):
            raise RetrievalError("Document is outside the approved knowledge source roots.")

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            rows = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**row) for row in rows if isinstance(row, dict)]
        except (OSError, ValueError, TypeError):
            self._chunks = []

    def _save(self) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps([chunk.to_dict() for chunk in self._chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.index_path)
