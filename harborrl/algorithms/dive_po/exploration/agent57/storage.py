from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


_SQLITE_SCHEMA_LOCK = threading.Lock()
_SQLITE_SCHEMA_INITIALIZED: set[str] = set()


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    name: str,
    ddl: str,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if name in columns:
        return
    try:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def connect(
    path: str,
    *,
    busy_timeout_ms: int = 5000,
    wal: bool = False,
) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_ms = max(1, int(busy_timeout_ms))
    connection = sqlite3.connect(
        str(db_path),
        timeout=float(timeout_ms) / 1000.0,
        isolation_level=None,
    )
    connection.execute(f"PRAGMA busy_timeout={timeout_ms}")
    if wal:
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass

    path_key = str(db_path)
    if path_key not in _SQLITE_SCHEMA_INITIALIZED:
        with _SQLITE_SCHEMA_LOCK:
            if path_key not in _SQLITE_SCHEMA_INITIALIZED:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS lifelong_counts "
                    "(key TEXT PRIMARY KEY, count REAL NOT NULL)"
                )
                _ensure_column(
                    connection,
                    "lifelong_counts",
                    "last_seen",
                    "last_seen INTEGER NOT NULL DEFAULT 0",
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS meta "
                    "(name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS arm_events "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
                    "arm_id INTEGER NOT NULL, base_score REAL NOT NULL, "
                    "final_score REAL NOT NULL, success INTEGER NOT NULL, "
                    "parse_error INTEGER NOT NULL, truncated INTEGER NOT NULL, "
                    "bonus REAL NOT NULL)"
                )
                _ensure_column(
                    connection,
                    "arm_events",
                    "dataset",
                    "dataset TEXT NOT NULL DEFAULT ''",
                )
                _ensure_column(
                    connection,
                    "arm_events",
                    "normalized_base_score",
                    "normalized_base_score REAL NOT NULL DEFAULT 0.0",
                )
                _ensure_column(
                    connection,
                    "arm_events",
                    "infra_failure",
                    "infra_failure INTEGER NOT NULL DEFAULT 0",
                )
                _SQLITE_SCHEMA_INITIALIZED.add(path_key)
    return connection
