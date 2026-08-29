

from __future__ import annotations
import sqlite3
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "data", "app.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            mode TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()


def create_session() -> str:
    session_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return session_id


def session_exists(session_id: str) -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row is not None


def add_message(session_id: str, role: str, content: str, mode: Optional[str] = None) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, mode, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, mode, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, mode, created_at FROM messages "
        "WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    sid = create_session()
    print("Created session:", sid)
    add_message(sid, "user", "How does caffeine affect sleep?")
    add_message(sid, "assistant", "Caffeine can delay sleep onset...", mode="rag")
    for msg in get_history(sid):
        print(msg)
