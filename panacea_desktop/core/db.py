from __future__ import annotations

import json
import os
import sqlite3
import threading
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .defaults import DEFAULTS, ENV_MAP


def _app_data_dir() -> Path:
    """Return per-user app data directory.

    Windows: %APPDATA%/MiriamDesktop
    macOS:   ~/Library/Application Support/MiriamDesktop
    Linux:   $XDG_DATA_HOME/miriam_desktop or ~/.local/share/miriam_desktop
    """
    # Windows
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "MiriamDesktop"

    # macOS
    if platform.system().lower() == "darwin":
        return Path.home() / "Library" / "Application Support" / "MiriamDesktop"

    # Linux / other
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "miriam_desktop"
    return Path.home() / ".local" / "share" / "miriam_desktop"


def default_db_path() -> Path:
    d = _app_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "miriam.sqlite3"


class Database:
    """Thread-safe SQLite access.

    UI uses worker threads (LLM streaming / voice). sqlite3 connections are not
    thread-safe by default; we therefore open a fresh connection per operation
    and guard DB writes with a re-entrant lock.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = str(Path(db_path) if db_path else default_db_path())
        self._lock = threading.RLock()
        self.init_db()
        self._ensure_defaults()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def close(self) -> None:
        # kept for API compatibility
        return

    def init_db(self) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dialogs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    attachments TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS histo_state (
                    dialog_id INTEGER PRIMARY KEY,
                    stain TEXT DEFAULT '',
                    magnification TEXT DEFAULT '',
                    quality TEXT DEFAULT '',
                    note TEXT DEFAULT '',
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS imaging_studies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_id INTEGER,
                    modality TEXT NOT NULL DEFAULT '',
                    body_region TEXT NOT NULL DEFAULT '',
                    contrast TEXT NOT NULL DEFAULT '',
                    clinical_question TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT '',           -- 'dicom_zip' | 'dicom_dir' | 'nifti' | 'unknown'
                    source_storage_path TEXT NOT NULL DEFAULT '',   -- stable path in app data
                    meta_json TEXT NOT NULL DEFAULT '{}',           -- extracted DICOM/NIfTI metadata
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS imaging_previews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT '',                  -- e.g. 'axial', 'coronal', 'sagittal', 'mip'
                    label TEXT NOT NULL DEFAULT '',
                    storage_path TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(study_id) REFERENCES imaging_studies(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS imaging_state (
                    dialog_id INTEGER PRIMARY KEY,
                    active_study_id INTEGER,
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE,
                    FOREIGN KEY(active_study_id) REFERENCES imaging_studies(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS global_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    tags TEXT NOT NULL DEFAULT '',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS prompt_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pack TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(pack, kind)
                );

                CREATE TABLE IF NOT EXISTS attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_id INTEGER,
                    message_id INTEGER,
                    original_name TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'attachments',
                    sha256 TEXT NOT NULL,
                    mime TEXT NOT NULL DEFAULT '',
                    size INTEGER NOT NULL DEFAULT 0,
                    storage_path TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE SET NULL,
                    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS rag_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    scope TEXT NOT NULL DEFAULT 'global',          -- 'global' or 'dialog'
                    dialog_id INTEGER,
                    sha256 TEXT DEFAULT '',
                    mime TEXT DEFAULT '',
                    size INTEGER NOT NULL DEFAULT 0,
                    storage_path TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(dialog_id) REFERENCES dialogs(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(doc_id) REFERENCES rag_docs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    chunk_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY(chunk_id, model),
                    FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()

        # add missing columns for users with older DBs (best-effort)
        self._ensure_columns()

        # Ensure at least one dialog exists & active dialog set
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) AS n FROM dialogs")
            n = int(cur.fetchone()["n"])
            if n == 0:
                cur.execute("INSERT INTO dialogs(title) VALUES(?)", ("New dialog",))
                did = int(cur.lastrowid)
                cur.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('active_dialog',?)", (str(did),))
                conn.commit()
            else:
                cur.execute("SELECT value FROM app_state WHERE key='active_dialog'")
                row = cur.fetchone()
                if not row:
                    cur.execute("SELECT id FROM dialogs ORDER BY id ASC LIMIT 1")
                    did = int(cur.fetchone()["id"])
                    cur.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('active_dialog',?)", (str(did),))
                    conn.commit()

    def _ensure_columns(self) -> None:
        """Best-effort migrations for older DB files."""
        # NOTE: sqlite "ALTER TABLE ADD COLUMN" is cheap; ignore failures.
        alters = [
            ("global_memory", "source", "TEXT DEFAULT 'manual'"),
            ("rag_docs", "scope", "TEXT NOT NULL DEFAULT 'global'"),
            ("rag_docs", "dialog_id", "INTEGER"),
            ("rag_docs", "sha256", "TEXT DEFAULT ''"),
            ("rag_docs", "mime", "TEXT DEFAULT ''"),
            ("rag_docs", "size", "INTEGER NOT NULL DEFAULT 0"),
            ("rag_docs", "storage_path", "TEXT DEFAULT ''"),
            ("attachments", "dialog_id", "INTEGER"),
            ("attachments", "message_id", "INTEGER"),
            ("imaging_studies", "dialog_id", "INTEGER"),
        ]
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            for table, col, decl in alters:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                except Exception:
                    pass
            conn.commit()

    def _ensure_defaults(self) -> None:
        # seed defaults into settings if absent
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            for k, v in DEFAULTS.items():
                cur.execute("SELECT value FROM settings WHERE key=?", (k,))
                if cur.fetchone() is None:
                    cur.execute("INSERT INTO settings(key, value) VALUES(?, ?)", (k, str(v)))
            conn.commit()

    # ---------------- settings ----------------
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key=?", (key,))
            row = cur.fetchone()
            if row is None:
                return default
            return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
            conn.commit()

    # ---------------- prompt overrides ----------------
    def get_prompt_override(self, pack: str, kind: str | None = None) -> Any:
        """Return prompt override(s).

        Backward/forward compatible API:
        - get_prompt_override(pack, kind) -> str | None
        - get_prompt_override(pack) -> {"system_prompt": str|None, "arbiter_prompt": str|None}
        """
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            if kind is not None:
                cur.execute("SELECT content FROM prompt_overrides WHERE pack=? AND kind=?", (pack, kind))
                row = cur.fetchone()
                return str(row["content"]) if row else None
            cur.execute(
                "SELECT kind, content FROM prompt_overrides WHERE pack=? AND kind IN (?,?)",
                (pack, "system_prompt", "arbiter_prompt"),
            )
            out = {"system_prompt": None, "arbiter_prompt": None}
            for r in cur.fetchall():
                if r["kind"] in out:
                    out[r["kind"]] = str(r["content"])
            return out

    def _set_prompt_kind(self, pack: str, kind: str, content: str) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO prompt_overrides(pack,kind,content) VALUES(?,?,?) "
                "ON CONFLICT(pack,kind) DO UPDATE SET content=excluded.content, updated_at=datetime('now')",
                (pack, kind, content),
            )
            conn.commit()

    def set_prompt_override(self, pack: str, a: str, b: str | None = None) -> None:
        """Set prompt override(s).

        Compatible call styles:
        - set_prompt_override(pack, kind, content)
        - set_prompt_override(pack, system_prompt, arbiter_prompt)
        """
        if b is None:
            # interpret as (kind, content) is impossible; keep strict
            raise TypeError("set_prompt_override(pack, kind, content) or set_prompt_override(pack, system, arbiter)")

        # If first arg looks like a known kind, treat as (kind, content)
        if a in {"system_prompt", "arbiter_prompt"}:
            self._set_prompt_kind(pack, a, b)
            return
        # Otherwise treat as (system, arbiter)
        self._set_prompt_kind(pack, "system_prompt", a)
        self._set_prompt_kind(pack, "arbiter_prompt", b)

    def delete_prompt_override(self, pack: str, kind: str | None = None) -> None:
        """Delete prompt override(s).

        - delete_prompt_override(pack, kind)
        - delete_prompt_override(pack) -> delete both system+arbiter overrides for pack
        """
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            if kind is None:
                cur.execute(
                    "DELETE FROM prompt_overrides WHERE pack=? AND kind IN (?,?)",
                    (pack, "system_prompt", "arbiter_prompt"),
                )
            else:
                cur.execute("DELETE FROM prompt_overrides WHERE pack=? AND kind=?", (pack, kind))
            conn.commit()

    # ---------------- dialogs ----------------
    def list_dialogs(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, title FROM dialogs ORDER BY updated_at DESC, id DESC")
            return [dict(r) for r in cur.fetchall()]

    def create_dialog(self, title: str = "New dialog") -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO dialogs(title) VALUES(?)", (title,))
            did = int(cur.lastrowid)
            cur.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('active_dialog',?)", (str(did),))
            conn.commit()
            return did

    def set_active_dialog(self, dialog_id: int) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('active_dialog',?)", (str(int(dialog_id)),))
            conn.commit()

    def get_active_dialog(self) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM app_state WHERE key='active_dialog'")
            row = cur.fetchone()
            if row and str(row["value"]).isdigit():
                return int(row["value"])
            cur.execute("SELECT id FROM dialogs ORDER BY id ASC LIMIT 1")
            r = cur.fetchone()
            did = int(r["id"]) if r else self.create_dialog()
            self.set_active_dialog(did)
            return did

    # ---------------- messages ----------------
    def add_message(self, dialog_id: int, role: str, content: str, attachments: dict[str, Any] | None = None) -> int:
        att = json.dumps(attachments or {}, ensure_ascii=False)
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages(dialog_id, role, content, attachments) VALUES(?, ?, ?, ?)",
                (int(dialog_id), role, content, att),
            )
            # bump updated_at
            cur.execute("UPDATE dialogs SET updated_at=datetime('now') WHERE id=?", (int(dialog_id),))
            conn.commit()
            return int(cur.lastrowid)

    def get_messages(self, dialog_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            q = "SELECT id, role, content, attachments, created_at FROM messages WHERE dialog_id=? ORDER BY id ASC"
            params: list[Any] = [int(dialog_id)]
            if limit is not None and int(limit) > 0:
                q += " LIMIT ?"
                params.append(int(limit))
            cur.execute(q, tuple(params))
            out: list[dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(r)
                try:
                    d["attachments"] = json.loads(d.get("attachments") or "{}")
                except Exception:
                    d["attachments"] = {}
                out.append(d)
            return out

    def set_message_attachments(self, message_id: int, attachments: dict[str, Any]) -> None:
        att = json.dumps(attachments or {}, ensure_ascii=False)
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE messages SET attachments=? WHERE id=?", (att, int(message_id)))
            conn.commit()

    # ---------------- histo per-dialog ----------------
    def histo_get(self, dialog_id: int) -> dict[str, str]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT stain, magnification, quality, note FROM histo_state WHERE dialog_id=?", (int(dialog_id),))
            row = cur.fetchone()
            if not row:
                return {"stain": "", "magnification": "", "quality": "", "note": ""}
            return {k: str(row[k] or "") for k in ("stain", "magnification", "quality", "note")}

    def histo_set(self, dialog_id: int, stain: str, magnification: str, quality: str, note: str) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO histo_state(dialog_id, stain, magnification, quality, note) VALUES(?,?,?,?,?) "
                "ON CONFLICT(dialog_id) DO UPDATE SET stain=excluded.stain, magnification=excluded.magnification, "
                "quality=excluded.quality, note=excluded.note",
                (int(dialog_id), stain or "", magnification or "", quality or "", note or ""),
            )
            conn.commit()

    # ---------------- imaging (radiology) ----------------
    def imaging_create_study(
        self,
        *,
        dialog_id: int | None,
        modality: str = "",
        body_region: str = "",
        contrast: str = "",
        clinical_question: str = "",
        source_type: str = "",
        source_storage_path: str = "",
        meta: dict[str, Any] | None = None,
    ) -> int:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO imaging_studies(dialog_id, modality, body_region, contrast, clinical_question, source_type, source_storage_path, meta_json) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    int(dialog_id) if dialog_id is not None else None,
                    modality or "",
                    body_region or "",
                    contrast or "",
                    clinical_question or "",
                    source_type or "",
                    source_storage_path or "",
                    meta_json,
                ),
            )
            sid = int(cur.lastrowid)
            if dialog_id is not None:
                cur.execute(
                    "INSERT INTO imaging_state(dialog_id, active_study_id) VALUES(?,?) "
                    "ON CONFLICT(dialog_id) DO UPDATE SET active_study_id=excluded.active_study_id",
                    (int(dialog_id), sid),
                )
            conn.commit()
            return sid

    def imaging_add_preview(self, *, study_id: int, kind: str, label: str, storage_path: str) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO imaging_previews(study_id, kind, label, storage_path) VALUES(?,?,?,?)",
                (int(study_id), kind or "", label or "", storage_path),
            )
            conn.commit()
            return int(cur.lastrowid)

    def imaging_set_active(self, dialog_id: int, study_id: int | None) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO imaging_state(dialog_id, active_study_id) VALUES(?,?) "
                "ON CONFLICT(dialog_id) DO UPDATE SET active_study_id=excluded.active_study_id",
                (int(dialog_id), int(study_id) if study_id is not None else None),
            )
            conn.commit()

    def imaging_get_active(self, dialog_id: int) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT active_study_id FROM imaging_state WHERE dialog_id=?", (int(dialog_id),))
            row = cur.fetchone()
            if not row or row["active_study_id"] is None:
                return None
            sid = int(row["active_study_id"])
            cur.execute(
                "SELECT id, dialog_id, modality, body_region, contrast, clinical_question, source_type, source_storage_path, meta_json, created_at "
                "FROM imaging_studies WHERE id=?",
                (sid,),
            )
            st = cur.fetchone()
            if not st:
                return None
            d = dict(st)
            try:
                d["meta"] = json.loads(d.get("meta_json") or "{}")
            except Exception:
                d["meta"] = {}
            d.pop("meta_json", None)
            # previews
            cur.execute(
                "SELECT id, kind, label, storage_path FROM imaging_previews WHERE study_id=? ORDER BY id ASC",
                (sid,),
            )
            d["previews"] = [dict(r) for r in cur.fetchall()]
            return d

    def imaging_list_studies(self, dialog_id: int) -> list[dict[str, Any]]:
        """List imaging studies for a dialog (newest first)."""
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, dialog_id, modality, body_region, contrast, clinical_question, source_type, source_storage_path, meta_json, created_at "
                "FROM imaging_studies WHERE dialog_id=? ORDER BY id DESC",
                (int(dialog_id),),
            )
            out: list[dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(r)
                try:
                    d["meta"] = json.loads(d.get("meta_json") or "{}")
                except Exception:
                    d["meta"] = {}
                d.pop("meta_json", None)
                out.append(d)
            return out

    def imaging_get_study(self, study_id: int) -> dict[str, Any] | None:
        """Get a study by id (includes previews)."""
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, dialog_id, modality, body_region, contrast, clinical_question, source_type, source_storage_path, meta_json, created_at "
                "FROM imaging_studies WHERE id=?",
                (int(study_id),),
            )
            st = cur.fetchone()
            if not st:
                return None
            d = dict(st)
            try:
                d["meta"] = json.loads(d.get("meta_json") or "{}")
            except Exception:
                d["meta"] = {}
            d.pop("meta_json", None)
            cur.execute(
                "SELECT id, kind, label, storage_path FROM imaging_previews WHERE study_id=? ORDER BY id ASC",
                (int(study_id),),
            )
            d["previews"] = [dict(r) for r in cur.fetchall()]
            return d

    def imaging_delete_study(self, study_id: int) -> None:
        """Delete a study and its previews (files remain in store)."""
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            # If study is active in imaging_state, unset it.
            cur.execute(
                "UPDATE imaging_state SET active_study_id=NULL WHERE active_study_id=?",
                (int(study_id),),
            )
            cur.execute("DELETE FROM imaging_studies WHERE id=?", (int(study_id),))
            conn.commit()

    # ---------------- global memory ----------------
    def list_global_memories(
        self,
        *,
        include_disabled: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            sql = (
                "SELECT id, text, source, tags, pinned, enabled, created_at, updated_at "
                "FROM global_memory "
            )
            params: list[Any] = []
            if not include_disabled:
                sql += "WHERE enabled=1 "
            sql += "ORDER BY pinned DESC, id DESC "
            if limit is not None:
                sql += "LIMIT ?"
                params.append(int(limit))
            cur.execute(sql, params)
            out: list[dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(r)
                # Backward/forward-compatible keys
                d["content"] = d.get("text", "")
                out.append(d)
            return out

    def add_global_memory(
        self,
        text: str | None = None,
        *,
        content: str | None = None,
        tags: str = "",
        pinned: bool = False,
        enabled: bool = True,
        source: str = "manual",
    ) -> int:
        # accept both `text` and `content`
        if content is None:
            content = text or ""
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO global_memory(text,source,tags,pinned,enabled) VALUES(?,?,?,?,?)",
                (content, source or "manual", tags or "", int(bool(pinned)), int(bool(enabled))),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_global_memory(
        self,
        memory_id: int,
        text: str | None = None,
        *,
        content: str | None = None,
        tags: str = "",
        pinned: bool = False,
        enabled: bool = True,
        source: str | None = None,
    ) -> None:
        if content is None:
            content = text or ""
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE global_memory SET text=?, source=COALESCE(?, source), tags=?, pinned=?, enabled=?, updated_at=datetime('now') WHERE id=?",
                (content, source, tags or "", int(bool(pinned)), int(bool(enabled)), int(memory_id)),
            )
            conn.commit()

    def delete_global_memory(self, memory_id: int) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM global_memory WHERE id=?", (int(memory_id),))
            conn.commit()

    # ---------------- attachments registry ----------------
    def add_attachment(
        self,
        *,
        dialog_id: int | None,
        message_id: int | None,
        original_name: str,
        category: str,
        sha256: str,
        mime: str,
        size: int,
        storage_path: str,
    ) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO attachments(dialog_id, message_id, original_name, category, sha256, mime, size, storage_path) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    int(dialog_id) if dialog_id is not None else None,
                    int(message_id) if message_id is not None else None,
                    original_name,
                    category,
                    sha256,
                    mime or "",
                    int(size or 0),
                    storage_path,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    # ---------------- RAG documents & chunks ----------------
    def rag_add_doc(
        self,
        *,
        title: str,
        path: str,
        enabled: bool,
        scope: str = "global",
        dialog_id: int | None = None,
        sha256: str = "",
        mime: str = "",
        size: int = 0,
        storage_path: str = "",
    ) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO rag_docs(title, path, enabled, scope, dialog_id, sha256, mime, size, storage_path) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    title,
                    path,
                    int(bool(enabled)),
                    scope or "global",
                    int(dialog_id) if dialog_id is not None else None,
                    sha256 or "",
                    mime or "",
                    int(size or 0),
                    storage_path or "",
                ),
            )
            conn.commit()
            # mark FAISS dirty
            self.set_setting("rag_index_dirty", "1")
            return int(cur.lastrowid)

    def rag_add_chunks(self, doc_id: int, chunks: Iterable[str]) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            for i, ch in enumerate(chunks):
                t = (ch or "").strip()
                if not t:
                    continue
                cur.execute("INSERT INTO rag_chunks(doc_id, idx, text) VALUES(?,?,?)", (int(doc_id), int(i), t))
            conn.commit()
        self.set_setting("rag_index_dirty", "1")

    def rag_list_docs(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, title, enabled, scope, dialog_id, created_at FROM rag_docs ORDER BY id DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def rag_set_doc_enabled(self, doc_id: int, enabled: bool) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE rag_docs SET enabled=? WHERE id=?", (int(bool(enabled)), int(doc_id)))
            conn.commit()
        self.set_setting("rag_index_dirty", "1")

    def rag_delete_doc(self, doc_id: int) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM rag_docs WHERE id=?", (int(doc_id),))
            conn.commit()
        self.set_setting("rag_index_dirty", "1")

    def rag_get_doc_chunks(self, doc_id: int, limit: int = 6) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT c.id AS chunk_id, c.doc_id AS doc_id, d.title AS title, c.text AS text "
                "FROM rag_chunks c JOIN rag_docs d ON d.id=c.doc_id WHERE c.doc_id=? "
                "ORDER BY c.idx ASC LIMIT ?",
                (int(doc_id), int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]

    def rag_latest_enabled_doc(self, dialog_id: int | None = None) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            if dialog_id is None:
                cur.execute(
                    "SELECT id, title, scope, dialog_id FROM rag_docs WHERE enabled=1 ORDER BY id DESC LIMIT 1"
                )
            else:
                # prefer dialog-scoped docs, fallback to global
                cur.execute(
                    "SELECT id, title, scope, dialog_id FROM rag_docs "
                    "WHERE enabled=1 AND ((scope='dialog' AND dialog_id=?) OR scope='global') "
                    "ORDER BY (scope='dialog') DESC, id DESC LIMIT 1",
                    (int(dialog_id),),
                )
            row = cur.fetchone()
            return dict(row) if row else None

    def rag_get_enabled_chunks(self, dialog_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            if dialog_id is None:
                cur.execute(
                    "SELECT c.id AS chunk_id, c.doc_id AS doc_id, d.title AS title, c.text AS text "
                    "FROM rag_chunks c JOIN rag_docs d ON d.id=c.doc_id "
                    "WHERE d.enabled=1 ORDER BY c.id ASC"
                )
            else:
                cur.execute(
                    "SELECT c.id AS chunk_id, c.doc_id AS doc_id, d.title AS title, c.text AS text "
                    "FROM rag_chunks c JOIN rag_docs d ON d.id=c.doc_id "
                    "WHERE d.enabled=1 AND ((d.scope='dialog' AND d.dialog_id=?) OR d.scope='global') "
                    "ORDER BY c.id ASC",
                    (int(dialog_id),),
                )
            return [dict(r) for r in cur.fetchall()]

    def rag_get_embedding(self, chunk_id: int, model: str) -> bytes | None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT vector FROM rag_embeddings WHERE chunk_id=? AND model=?", (int(chunk_id), model))
            row = cur.fetchone()
            return bytes(row["vector"]) if row else None

    def rag_upsert_embedding(self, chunk_id: int, model: str, vector: bytes) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO rag_embeddings(chunk_id, model, vector) VALUES(?,?,?) "
                "ON CONFLICT(chunk_id, model) DO UPDATE SET vector=excluded.vector, updated_at=datetime('now')",
                (int(chunk_id), model, sqlite3.Binary(vector)),
            )
            conn.commit()
