from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    schema = files("predic_v2").joinpath("schema.sql").read_text(encoding="utf-8")
    connection.executescript(schema)
    connection.commit()
