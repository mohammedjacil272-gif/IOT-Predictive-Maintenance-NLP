from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any

SCHEMA_VERSION = 3
STATES = {"IDLE", "LISTENING", "PROCESSING", "RESPONDING", "ERROR", "CANCELLED"}
ROLES = {"user", "assistant", "system"}
OBSERVATION_STATUSES = {"pending", "confirmed", "rejected", "cancelled"}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class AssistantStorage:
    """Transactional local store with explicit ownership and provenance checks."""
    def __init__(self, path: str | Path = ".assistant.sqlite3"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level="DEFERRED")
        conn.row_factory = sqlite3.Row; conn.execute("PRAGMA foreign_keys=ON"); conn.execute("PRAGMA busy_timeout=10000"); conn.execute("PRAGMA journal_mode=WAL"); return conn

    def _init(self) -> None:
        with self._connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS machines (machine_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, machine_id TEXT REFERENCES machines(machine_id), state TEXT NOT NULL CHECK(state IN ('IDLE','LISTENING','PROCESSING','RESPONDING','ERROR','CANCELLED')), payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user','assistant','system')), text TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (observation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE, machine_id TEXT REFERENCES machines(machine_id), status TEXT NOT NULL CHECK(status IN ('pending','confirmed','rejected','cancelled')), payload TEXT NOT NULL, created_at TEXT NOT NULL, confirmed_at TEXT);
            CREATE TABLE IF NOT EXISTS confirmations (confirmation_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL UNIQUE REFERENCES observations(observation_id) ON DELETE CASCADE, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attachments (file_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE, machine_id TEXT REFERENCES machines(machine_id), payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sensor_references (reference_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE, machine_id TEXT NOT NULL REFERENCES machines(machine_id), payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS components (component_id TEXT PRIMARY KEY, machine_id TEXT NOT NULL REFERENCES machines(machine_id), payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(machine_id, component_id));
            CREATE TABLE IF NOT EXISTS maintenance_events (event_id TEXT PRIMARY KEY, machine_id TEXT NOT NULL REFERENCES machines(machine_id), component_id TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS alerts (alert_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, machine_id TEXT NOT NULL REFERENCES machines(machine_id), payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_components_machine ON components(machine_id);
            CREATE INDEX IF NOT EXISTS idx_maintenance_machine_time ON maintenance_events(machine_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_alerts_machine_status ON alerts(machine_id, json_extract(payload, '$.status'));
            CREATE TABLE IF NOT EXISTS audit_events (audit_id TEXT PRIMARY KEY, actor TEXT NOT NULL, action TEXT NOT NULL, resource_type TEXT NOT NULL, resource_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_audit_resource_time ON audit_events(resource_type, resource_id, created_at);
            """)
            c.execute("INSERT OR IGNORE INTO schema_meta VALUES ('schema_version', '3')")
            current = int(c.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])
            if current < SCHEMA_VERSION:
                c.execute("UPDATE schema_meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
            if current > SCHEMA_VERSION: raise RuntimeError(f"Unsupported database schema version {current}.")

    def backup(self, destination: str | Path) -> Path:
        """Create a recoverable SQLite backup without mutating the store."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(target) as backup:
            source.backup(backup)
        return target

    @staticmethod
    def _json(payload: dict[str, Any]) -> str: return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def save_machine(self, machine_id: str, payload: dict[str, Any]) -> None:
        if not isinstance(machine_id, str) or not machine_id.strip(): raise ValueError("machine_id must be non-empty")
        now = utc_now()
        with self._connect() as c:
            if c.execute("SELECT 1 FROM machines WHERE machine_id=?", (machine_id,)).fetchone(): raise ValueError(f"Machine {machine_id!r} already exists.")
            c.execute("INSERT INTO machines VALUES (?,?,?,?)", (machine_id, self._json(payload), now, now))

    def update_machine(self, machine_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as c:
            cur = c.execute("UPDATE machines SET payload=?,updated_at=? WHERE machine_id=?", (self._json(payload), utc_now(), machine_id))
            if cur.rowcount != 1: raise KeyError(f"Unknown machine {machine_id!r}.")

    def get_machine(self, machine_id: str) -> dict[str, Any] | None:
        with self._connect() as c: row = c.execute("SELECT payload FROM machines WHERE machine_id=?", (machine_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_machines(self) -> list[dict[str, Any]]:
        with self._connect() as c: rows = c.execute("SELECT payload FROM machines ORDER BY machine_id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_component(self, component_id: str, machine_id: str, payload: dict[str, Any]) -> None:
        if not component_id.strip() or self.get_machine(machine_id) is None:
            raise ValueError("component_id must be non-empty and machine must exist")
        now = utc_now()
        with self._connect() as c:
            if c.execute("SELECT 1 FROM components WHERE component_id=?", (component_id,)).fetchone():
                raise ValueError(f"Component {component_id!r} already exists.")
            c.execute("INSERT INTO components VALUES (?,?,?,?,?)", (component_id, machine_id, self._json({"component_id": component_id, "machine_id": machine_id, **payload}), now, now))

    def list_components(self, machine_id: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT payload FROM components", []
        if machine_id: query += " WHERE machine_id=?"; args.append(machine_id)
        with self._connect() as c: rows = c.execute(query + " ORDER BY component_id", args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_maintenance(self, event_id: str, machine_id: str, component_id: str | None, payload: dict[str, Any]) -> None:
        if self.get_machine(machine_id) is None: raise ValueError("machine must exist")
        if component_id and not any(x["component_id"] == component_id and x["machine_id"] == machine_id for x in self.list_components(machine_id)):
            raise ValueError("component does not belong to machine")
        now = utc_now()
        with self._connect() as c:
            c.execute("INSERT INTO maintenance_events VALUES (?,?,?,?,?,?)", (event_id, machine_id, component_id, self._json({"event_id": event_id, "machine_id": machine_id, "component_id": component_id, **payload}), now, now))

    def list_maintenance(self, machine_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query, args = "SELECT payload FROM maintenance_events", []
        if machine_id: query += " WHERE machine_id=?"; args.append(machine_id)
        with self._connect() as c: rows = c.execute(query + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        values = [json.loads(r[0]) for r in rows]
        return [x for x in values if not status or x.get("status") == status]

    def save_alert(self, alert_id: str, fingerprint: str, machine_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if self.get_machine(machine_id) is None: raise ValueError("machine must exist")
        with self._connect() as c:
            prior = c.execute("SELECT payload FROM alerts WHERE fingerprint=? AND json_extract(payload, '$.status') IN ('open','acknowledged')", (fingerprint,)).fetchone()
            if prior: return json.loads(prior[0]), False
            now = utc_now(); value = {"alert_id": alert_id, "fingerprint": fingerprint, "machine_id": machine_id, **payload}
            c.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?)", (alert_id, fingerprint, machine_id, self._json(value), now, now))
            return value, True

    def list_alerts(self, machine_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query, args = "SELECT payload FROM alerts", []
        if machine_id: query += " WHERE machine_id=?"; args.append(machine_id)
        with self._connect() as c: rows = c.execute(query + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        values = [json.loads(r[0]) for r in rows]
        return [x for x in values if not status or x.get("status") == status]

    def update_alert_status(self, alert_id: str, status: str) -> dict[str, Any] | None:
        if status not in {"open", "acknowledged", "resolved"}: raise ValueError("invalid alert status")
        with self._connect() as c:
            row = c.execute("SELECT payload FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: return None
            value = json.loads(row[0]); value["status"] = status
            c.execute("UPDATE alerts SET payload=?,updated_at=? WHERE alert_id=?", (self._json(value), utc_now(), alert_id))
            return value

    def add_audit(self, audit_id: str, actor: str, action: str, resource_type: str, resource_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        value = {"audit_id": audit_id, "actor": actor, "action": action, "resource_type": resource_type, "resource_id": resource_id, "payload": payload, "created_at": utc_now()}
        with self._connect() as c: c.execute("INSERT INTO audit_events VALUES (?,?,?,?,?,?,?)", (audit_id, actor, action, resource_type, resource_id, self._json(payload), value["created_at"]))
        return value

    def list_audit(self, resource_type: str | None = None, resource_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM audit_events", []
        clauses = []
        if resource_type: clauses.append("resource_type=?"); args.append(resource_type)
        if resource_id: clauses.append("resource_id=?"); args.append(resource_id)
        if clauses: query += " WHERE " + " AND ".join(clauses)
        with self._connect() as c: rows = c.execute(query + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [{"audit_id": r["audit_id"], "actor": r["actor"], "action": r["action"], "resource_type": r["resource_type"], "resource_id": r["resource_id"], "payload": json.loads(r["payload"]), "created_at": r["created_at"]} for r in rows]

    def save_session(self, session_id: str, machine_id: str | None, state: str, payload: dict[str, Any]) -> None:
        if state not in STATES: raise ValueError(f"Invalid assistant state {state!r}.")
        if machine_id and self.get_machine(machine_id) is None: raise KeyError(f"Unknown machine {machine_id!r}.")
        now = utc_now()
        with self._connect() as c:
            exists = c.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if exists: c.execute("UPDATE sessions SET machine_id=?,state=?,payload=?,updated_at=? WHERE session_id=?", (machine_id, state, self._json(payload), now, session_id))
            else: c.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", (session_id, machine_id, state, self._json(payload), now, now))

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as c: row = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row: return None
        return {"session_id": row["session_id"], "machine_id": row["machine_id"], "state": row["state"], "payload": json.loads(row["payload"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def set_state(self, session_id: str, state: str) -> None:
        if state not in STATES: raise ValueError(f"Invalid assistant state {state!r}.")
        allowed = {"IDLE": {"LISTENING","PROCESSING","CANCELLED","ERROR"}, "LISTENING": {"PROCESSING","CANCELLED","ERROR"}, "PROCESSING": {"RESPONDING","ERROR","CANCELLED"}, "RESPONDING": {"IDLE","PROCESSING","CANCELLED"}, "ERROR": {"IDLE","PROCESSING"}, "CANCELLED": {"IDLE"}}
        with self._connect() as c:
            row = c.execute("SELECT state FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row: raise KeyError(f"Unknown session {session_id!r}.")
            if state != row["state"] and state not in allowed[row["state"]]: raise ValueError(f"Illegal state transition {row['state']} -> {state}.")
            c.execute("UPDATE sessions SET state=?,updated_at=? WHERE session_id=?", (state, utc_now(), session_id))

    def add_message(self, session_id: str, role: str, text: str, payload: dict[str, Any] | None = None) -> int:
        if role not in ROLES or not isinstance(text, str) or not text.strip(): raise ValueError("Invalid message role or text.")
        with self._connect() as c:
            if not c.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone(): raise KeyError(f"Unknown session {session_id!r}.")
            return int(c.execute("INSERT INTO messages(session_id,role,text,payload,created_at) VALUES (?,?,?,?,?)", (session_id, role, text, self._json(payload or {}), utc_now())).lastrowid)

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as c: rows = c.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{"id": r["id"], "role": r["role"], "text": r["text"], "payload": json.loads(r["payload"]), "created_at": r["created_at"]} for r in rows]

    def save_observation(self, observation_id: str, session_id: str, machine_id: str | None, status: str, payload: dict[str, Any]) -> None:
        if status not in OBSERVATION_STATUSES: raise ValueError(f"Invalid observation status {status!r}.")
        session = self.get_session(session_id)
        if not session: raise KeyError(f"Unknown session {session_id!r}.")
        if machine_id and session["machine_id"] != machine_id: raise ValueError("Observation machine does not match session machine.")
        if machine_id and self.get_machine(machine_id) is None: raise KeyError(f"Unknown machine {machine_id!r}.")
        with self._connect() as c:
            prior = c.execute("SELECT status FROM observations WHERE observation_id=?", (observation_id,)).fetchone()
            if prior and prior["status"] in {"confirmed", "rejected", "cancelled"} and prior["status"] != status:
                raise ValueError(f"Observation is already terminal: {prior['status']}.")
            c.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?) ON CONFLICT(observation_id) DO UPDATE SET status=excluded.status,payload=excluded.payload,confirmed_at=excluded.confirmed_at", (observation_id, session_id, machine_id, status, self._json(payload), utc_now(), utc_now() if status == "confirmed" else None))

    def list_observations(self, machine_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM observations WHERE 1=1", []
        if machine_id: query += " AND machine_id=?"; args.append(machine_id)
        if session_id: query += " AND session_id=?"; args.append(session_id)
        with self._connect() as c: rows = c.execute(query + " ORDER BY created_at", args).fetchall()
        return [{**json.loads(r["payload"]), "observation_id": r["observation_id"], "status": r["status"], "session_id": r["session_id"], "machine_id": r["machine_id"]} for r in rows]

    def save_confirmation(self, confirmation_id: str, observation_id: str, session_id: str, payload: dict[str, Any]) -> None:
        if not self.get_session(session_id): raise KeyError(f"Unknown session {session_id!r}.")
        if not any(row["observation_id"] == observation_id for row in self.list_observations(session_id=session_id)):
            raise KeyError(f"Unknown observation {observation_id!r}.")
        with self._connect() as c:
            c.execute("INSERT INTO confirmations VALUES (?,?,?,?,?) ON CONFLICT(confirmation_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at", (confirmation_id, observation_id, session_id, self._json(payload), utc_now()))

    def get_confirmation(self, confirmation_id: str) -> dict[str, Any] | None:
        with self._connect() as c: row = c.execute("SELECT * FROM confirmations WHERE confirmation_id=?", (confirmation_id,)).fetchone()
        if not row: return None
        return {"confirmation_id": row["confirmation_id"], "observation_id": row["observation_id"], "session_id": row["session_id"], **json.loads(row["payload"])}

    def save_attachment(self, file_id: str, session_id: str, machine_id: str | None, payload: dict[str, Any]) -> None:
        session = self.get_session(session_id)
        if not session: raise KeyError(f"Unknown session {session_id!r}.")
        if machine_id and session["machine_id"] != machine_id: raise ValueError("Attachment machine does not match session machine.")
        with self._connect() as c: c.execute("INSERT INTO attachments VALUES (?,?,?,?,?)", (file_id, session_id, machine_id, self._json(payload), utc_now()))

    def get_attachment(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as c: row = c.execute("SELECT * FROM attachments WHERE file_id=?", (file_id,)).fetchone()
        if not row: return None
        return {"file_id": row["file_id"], "session_id": row["session_id"], "machine_id": row["machine_id"], **json.loads(row["payload"])}

    def migrate_json_history(self, json_path: str | Path, *, backup_path: str | Path | None = None) -> dict[str, Any]:
        source = Path(json_path)
        if not source.is_file(): return {"status": "not_found", "imported": 0}
        backup = Path(backup_path) if backup_path else source.with_suffix(source.suffix + ".backup")
        if not backup.exists(): shutil.copy2(source, backup)
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list): raise ValueError("History JSON must contain a list.")
        sid = "migration-local-history"
        if self.get_session(sid) is None: self.save_session(sid, None, "IDLE", {"source": str(source), "migration": True})
        existing = self.list_messages(sid); imported = 0
        for item in data:
            digest = hashlib.sha256(self._json(item).encode()).hexdigest()
            if not any(m["payload"].get("migration_digest") == digest for m in existing):
                self.add_message(sid, "system", "Imported local history record", {"migration_digest": digest, "record": item}); imported += 1
        return {"status": "completed", "imported": imported, "backup": str(backup), "session_id": sid}
