import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "chat_history.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            think INTEGER NOT NULL,
            run_index INTEGER NOT NULL,
            load_ms REAL,
            ttft_ms REAL,
            total_ms REAL,
            eval_count INTEGER,
            tokens_per_s REAL,
            response_chars INTEGER,
            thinking_chars INTEGER,
            mem_total_mb REAL,
            mem_vram_mb REAL,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(title: str, model: str) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO conversations (title, model, created_at) VALUES (?, ?, ?)",
        (title, model, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def list_conversations():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, model, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_messages(conversation_id: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_message(conversation_id: int, role: str, content: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def add_benchmark_run(row: dict):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO benchmark_runs (
            batch_id, created_at, model, prompt, think, run_index, load_ms,
            ttft_ms, total_ms, eval_count, tokens_per_s, response_chars,
            thinking_chars, mem_total_mb, mem_vram_mb, response
        ) VALUES (
            :batch_id, :created_at, :model, :prompt, :think, :run_index, :load_ms,
            :ttft_ms, :total_ms, :eval_count, :tokens_per_s, :response_chars,
            :thinking_chars, :mem_total_mb, :mem_vram_mb, :response
        )
        """,
        {**row, "created_at": datetime.now(timezone.utc).isoformat()},
    )
    conn.commit()
    conn.close()


def list_benchmark_runs(limit: int = 300):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_benchmark_batch(batch_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM benchmark_runs WHERE batch_id = ?", (batch_id,))
    conn.commit()
    conn.close()