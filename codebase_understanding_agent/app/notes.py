from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing
from threading import RLock
from typing import Any

from .contracts import utc_now
from .embeddings import EmbeddingClient, cosine_similarity


KINDS = {"fact", "architecture", "open_question"}
STATUSES = {"confirmed", "hypothesis"}


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return compact[:64] or "topic"


class NoteStore:
    """Repository-scoped, evidence-bearing knowledge notes."""

    def __init__(self, database_path: Path, embedder: EmbeddingClient | None = None):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.embedder = embedder or EmbeddingClient()
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL DEFAULT '[]',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    related_note_ids_json TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (repo_id, subject_key)
                );
                CREATE TABLE IF NOT EXISTS evidence_refs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    revision TEXT NOT NULL,
                    observation_event_id TEXT,
                    FOREIGN KEY(note_id) REFERENCES notes(note_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS note_embeddings (
                    note_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(note_id) REFERENCES notes(note_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_notes_repo ON notes(repo_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_note ON evidence_refs(note_id);
                """
            )
            connection.commit()

    @staticmethod
    def _decode_json(raw: str) -> list[Any]:
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _embedding_text(
        *,
        subject_key: str,
        title: str,
        kind: str,
        content: str,
        tags_json: str,
        open_questions_json: str,
    ) -> str:
        return "\n".join(
            [
                f"subject: {subject_key}",
                f"title: {title}",
                f"kind: {kind}",
                f"content: {content}",
                f"tags: {tags_json}",
                f"open_questions: {open_questions_json}",
            ]
        )

    @classmethod
    def _row_embedding_text(cls, row: sqlite3.Row) -> str:
        return cls._embedding_text(
            subject_key=row["subject_key"],
            title=row["title"],
            kind=row["kind"],
            content=row["content"],
            tags_json=row["tags_json"],
            open_questions_json=row["open_questions_json"],
        )

    def search(self, repo_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        limit = max(0, min(int(limit), 20))
        if limit == 0:
            return []
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT n.*, e.signature AS embedding_signature, "
                "e.content_hash AS embedding_content_hash, "
                "e.vector_json AS embedding_vector_json "
                "FROM notes n LEFT JOIN note_embeddings e ON e.note_id = n.note_id "
                "WHERE n.repo_id = ? ORDER BY n.updated_at DESC LIMIT 200",
                (repo_id,),
            ).fetchall()
            if not rows:
                return []

            vectors: dict[str, list[float]] = {}
            missing_rows: list[sqlite3.Row] = []
            missing_texts: list[str] = []
            for row in rows:
                text = self._row_embedding_text(row)
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if (
                    row["embedding_signature"] != self.embedder.signature
                    or row["embedding_content_hash"] != content_hash
                    or not row["embedding_vector_json"]
                ):
                    missing_rows.append(row)
                    missing_texts.append(text)
                    continue
                try:
                    vector = json.loads(row["embedding_vector_json"])
                except (TypeError, json.JSONDecodeError):
                    missing_rows.append(row)
                    missing_texts.append(text)
                    continue
                if isinstance(vector, list):
                    vectors[row["note_id"]] = [float(value) for value in vector]

            if missing_rows:
                embedded = self.embedder.embed(missing_texts)
                now = utc_now()
                for row, text, vector in zip(missing_rows, missing_texts, embedded):
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    vectors[row["note_id"]] = vector
                    connection.execute(
                        "INSERT INTO note_embeddings "
                        "(note_id, signature, content_hash, vector_json, updated_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(note_id) DO UPDATE SET "
                        "signature=excluded.signature, content_hash=excluded.content_hash, "
                        "vector_json=excluded.vector_json, updated_at=excluded.updated_at",
                        (
                            row["note_id"],
                            self.embedder.signature,
                            content_hash,
                            json.dumps(vector),
                            now,
                        ),
                    )
                connection.commit()

        query_vector = self.embedder.embed([query])[0]
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            vector = vectors.get(row["note_id"])
            if vector is None:
                continue
            scored.append((cosine_similarity(query_vector, vector), row))
        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        return [
            {
                "note_id": row["note_id"],
                "subject_key": row["subject_key"],
                "title": row["title"],
                "kind": row["kind"],
                "status": row["status"],
                "summary": row["content"][:600],
                "tags": self._decode_json(row["tags_json"]),
                "version": row["version"],
                "updated_at": row["updated_at"],
                "similarity": score,
            }
            for score, row in scored[:limit]
        ]

    def get(self, repo_id: str, note_id: str, workspace: Any | None = None) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM notes WHERE repo_id = ? AND note_id = ?",
                (repo_id, note_id),
            ).fetchone()
            if row is None:
                return None
            evidence_rows = connection.execute(
                "SELECT path, start_line, end_line, revision, observation_event_id "
                "FROM evidence_refs WHERE note_id = ? ORDER BY id",
                (note_id,),
            ).fetchall()

        evidence: list[dict[str, Any]] = []
        stale = False
        for item in evidence_rows:
            entry = dict(item)
            if workspace is not None:
                try:
                    current = workspace.file_hash(entry["path"])
                    entry["stale"] = current != entry["revision"]
                    stale = stale or entry["stale"]
                except Exception:
                    entry["stale"] = True
                    stale = True
            evidence.append(entry)

        return {
            "note_id": row["note_id"],
            "repo_id": row["repo_id"],
            "subject_key": row["subject_key"],
            "title": row["title"],
            "kind": row["kind"],
            "status": "stale" if stale else row["status"],
            "content": row["content"],
            "evidence": evidence,
            "open_questions": self._decode_json(row["open_questions_json"]),
            "tags": self._decode_json(row["tags_json"]),
            "related_note_ids": self._decode_json(row["related_note_ids_json"]),
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert(
        self,
        *,
        repo_id: str,
        workspace: Any,
        subject_key: str,
        title: str,
        kind: str,
        content: str,
        evidence: list[dict[str, Any]],
        status: str = "confirmed",
        tags: list[str] | None = None,
        open_questions: list[str] | None = None,
        related_note_ids: list[str] | None = None,
        note_id: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind 必须是 {sorted(KINDS)} 之一。")
        if status not in STATUSES:
            raise ValueError(f"status 必须是 {sorted(STATUSES)} 之一。")
        if not subject_key.strip() or not title.strip() or not content.strip():
            raise ValueError("subject_key、title 和 content 不能为空。")
        if status == "confirmed" and not evidence:
            raise ValueError("confirmed 笔记必须至少包含一条源码证据。")

        normalized_evidence: list[dict[str, Any]] = []
        for raw in evidence:
            path = str(raw.get("path", ""))
            start_line = int(raw.get("start_line", 0))
            end_line = int(raw.get("end_line", start_line))
            if start_line < 1 or end_line < start_line:
                raise ValueError(f"证据行号无效：{path}:{start_line}-{end_line}")
            workspace.resolve_relative(path, require_file=True)
            normalized_evidence.append(
                {
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "revision": workspace.file_hash(path),
                    "observation_event_id": raw.get("observation_event_id"),
                }
            )

        now = utc_now()
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        open_questions_json = json.dumps(open_questions or [], ensure_ascii=False)
        related_note_ids_json = json.dumps(related_note_ids or [], ensure_ascii=False)
        embedding_text = self._embedding_text(
            subject_key=subject_key,
            title=title,
            kind=kind,
            content=content,
            tags_json=tags_json,
            open_questions_json=open_questions_json,
        )
        embedding = self.embedder.embed([embedding_text])[0]
        embedding_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
        stable_id = note_id or (
            f"{_slug(repo_id)}:{_slug(subject_key)}:"
            f"{hashlib.sha1(subject_key.encode('utf-8')).hexdigest()[:8]}"
        )
        with self._lock, closing(self._connect()) as connection:
            current = connection.execute(
                "SELECT note_id, version, created_at FROM notes "
                "WHERE repo_id = ? AND subject_key = ?",
                (repo_id, subject_key),
            ).fetchone()
            if current and expected_version is not None and current["version"] != expected_version:
                raise ValueError(
                    f"VERSION_CONFLICT：当前版本为 {current['version']}，"
                    f"不是 {expected_version}。"
                )
            if current:
                stable_id = current["note_id"]
                version = current["version"] + 1
                created_at = current["created_at"]
                action = "updated"
                connection.execute("DELETE FROM evidence_refs WHERE note_id = ?", (stable_id,))
                connection.execute(
                    """
                    UPDATE notes SET title=?, kind=?, status=?, content=?,
                        open_questions_json=?, tags_json=?, related_note_ids_json=?,
                        version=?, updated_at=?
                    WHERE note_id=? AND repo_id=?
                    """,
                    (
                        title,
                        kind,
                        status,
                        content,
                        open_questions_json,
                        tags_json,
                        related_note_ids_json,
                        version,
                        now,
                        stable_id,
                        repo_id,
                    ),
                )
            else:
                version = 1
                created_at = now
                action = "created"
                connection.execute(
                    """
                    INSERT INTO notes (
                        note_id, repo_id, subject_key, title, kind, status, content,
                        open_questions_json, tags_json, related_note_ids_json,
                        version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id,
                        repo_id,
                        subject_key,
                        title,
                        kind,
                        status,
                        content,
                        open_questions_json,
                        tags_json,
                        related_note_ids_json,
                        version,
                        created_at,
                        now,
                    ),
                )
            for entry in normalized_evidence:
                connection.execute(
                    """
                    INSERT INTO evidence_refs (
                        note_id, path, start_line, end_line, revision, observation_event_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id,
                        entry["path"],
                        entry["start_line"],
                        entry["end_line"],
                        entry["revision"],
                        entry["observation_event_id"],
                    ),
                )
            connection.execute(
                "INSERT INTO note_embeddings "
                "(note_id, signature, content_hash, vector_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(note_id) DO UPDATE SET "
                "signature=excluded.signature, content_hash=excluded.content_hash, "
                "vector_json=excluded.vector_json, updated_at=excluded.updated_at",
                (
                    stable_id,
                    self.embedder.signature,
                    embedding_hash,
                    json.dumps(embedding),
                    now,
                ),
            )
            connection.commit()

        return {
            "note_id": stable_id,
            "version": version,
            "action": action,
            "subject_key": subject_key,
            "updated_at": now,
        }
