"""
database.py
──────────────────────────────────────────────────────
SQLite database layer — Vision-Based Smart Tailoring System.

Tables:
  users        — authentication and role management
  measurements — body/garment measurements, linked to users via user_id
"""

from __future__ import annotations
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "tailoring_system.db")


# ── Schema DDL ────────────────────────────────────────────────────────────────

_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    role          TEXT    CHECK(role IN ('user', 'tailor')) NOT NULL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_MEASUREMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS measurements (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER,
    name                 TEXT    NOT NULL,
    height_cm            REAL    NOT NULL,
    gender               TEXT    NOT NULL,
    fit_preference       TEXT    NOT NULL,
    sleeve_type          TEXT    NOT NULL,
    -- Body measurements (all in INCHES)
    shoulder_width       REAL,
    full_chest           REAL,
    waist                REAL,
    hip                  REAL,
    arm_length           REAL,
    front_length         REAL,
    collar               REAL,
    sleeve_open          REAL,
    -- Garment measurements (all in INCHES)
    garment_chest        REAL,
    garment_waist        REAL,
    garment_hip          REAL,
    garment_sleeve       REAL,
    garment_front_length REAL,
    garment_collar       REAL,
    garment_sleeve_open  REAL,
    recommended_size     TEXT,
    confidence_score     REAL,
    timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
"""

_INSERT_MEASUREMENT = """
INSERT INTO measurements (
    user_id, name, height_cm, gender, fit_preference, sleeve_type,
    shoulder_width, full_chest, waist,
    arm_length, front_length, collar, sleeve_open,
    garment_chest, garment_waist, garment_sleeve,
    garment_front_length, garment_collar, garment_sleeve_open,
    recommended_size, confidence_score, timestamp
) VALUES (
    :user_id, :name, :height_cm, :gender, :fit_preference, :sleeve_type,
    :shoulder_width, :full_chest, :waist,
    :arm_length, :front_length, :collar, :sleeve_open,
    :garment_chest, :garment_waist, :garment_sleeve,
    :garment_front_length, :garment_collar, :garment_sleeve_open,
    :recommended_size, :confidence_score, :timestamp
);
"""


class Database:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Initialization ────────────────────────────────────────────────────────

    def init_db(self) -> None:
        """Create tables and run safe migrations."""
        with self._connect() as conn:
            conn.execute(_USERS_TABLE)
            conn.execute(_MEASUREMENTS_TABLE)
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Safely add new columns to legacy measurements tables."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(measurements)")}
        # Legacy: add user_id if missing
        if "user_id" not in cols:
            conn.execute("ALTER TABLE measurements ADD COLUMN user_id INTEGER")
        # New inch-based columns
        new_cols = [
            ("full_chest",           "REAL"),
            ("waist",                "REAL"),
            ("hip",                  "REAL"),
            ("front_length",         "REAL"),
            ("collar",               "REAL"),
            ("sleeve_open",          "REAL"),
            ("garment_front_length", "REAL"),
            ("garment_collar",       "REAL"),
            ("garment_sleeve_open",  "REAL"),
        ]
        for col_name, col_type in new_cols:
            if col_name not in cols:
                conn.execute(
                    f"ALTER TABLE measurements ADD COLUMN {col_name} {col_type}"
                )

    # ── User operations ───────────────────────────────────────────────────────

    def get_connection(self) -> sqlite3.Connection:
        """Return a raw connection (used by auth_manager)."""
        return self._connect()

    def get_user_by_email(self, email: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return dict(row) if row else None

    # ── Measurements ──────────────────────────────────────────────────────────

    def save_measurement(
        self,
        name:                 str,
        height_cm:            float,
        gender:               str,
        fit_preference:       str,
        sleeve_type:          str,
        body_measurements:    dict,
        garment_measurements: dict,
        recommended_size:     str,
        confidence_score:     float,
        user_id:              int | None = None,
    ) -> int:
        row = {
            "user_id":              user_id,
            "name":                 name,
            "height_cm":            height_cm,
            "gender":               gender,
            "fit_preference":       fit_preference,
            "sleeve_type":          sleeve_type,
            # Body measurements (inches)
            "shoulder_width":       body_measurements.get("shoulder_width"),
            "full_chest":           body_measurements.get("full_chest"),
            "waist":                body_measurements.get("waist"),
            "arm_length":           body_measurements.get("arm_length"),
            "front_length":         body_measurements.get("front_length"),
            "collar":               body_measurements.get("collar"),
            "sleeve_open":          body_measurements.get("sleeve_open"),
            # Garment measurements (inches)
            "garment_chest":        garment_measurements.get("garment_chest"),
            "garment_waist":        garment_measurements.get("garment_waist"),
            "garment_sleeve":       garment_measurements.get("garment_sleeve"),
            "garment_front_length": garment_measurements.get("garment_front_length"),
            "garment_collar":       garment_measurements.get("garment_collar"),
            "garment_sleeve_open":  garment_measurements.get("garment_sleeve_open"),
            "recommended_size":     recommended_size,
            "confidence_score":     round(confidence_score, 4),
            "timestamp":            datetime.now().isoformat(sep=" ", timespec="seconds"),
        }
        with self._connect() as conn:
            cursor = conn.execute(_INSERT_MEASUREMENT, row)
            conn.commit()
            return cursor.lastrowid

    def get_measurements_by_user(self, user_id: int) -> list[dict]:
        """Return all measurements for a specific user, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM measurements WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_record_by_id(self, record_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM measurements WHERE id = ?", (record_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_record(self, record_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM measurements WHERE id = ?", (record_id,))
            conn.commit()

    def get_analytics(self, user_id: int) -> dict:
        """Return analytics scoped to a single user/tailor."""
        records = self.get_measurements_by_user(user_id)
        if not records:
            return {}

        import pandas as pd
        df = pd.DataFrame(records)

        analytics = {
            "total_records":       len(df),
            "size_distribution":   df["recommended_size"].value_counts().to_dict(),
            "fit_distribution":    df["fit_preference"].value_counts().to_dict(),
            "sleeve_distribution": df["sleeve_type"].value_counts().to_dict(),
            "gender_distribution": df["gender"].value_counts().to_dict(),
        }
        for col in ["garment_chest", "garment_waist",
                    "garment_sleeve", "confidence_score"]:
            if col in df.columns:
                analytics[f"avg_{col}"] = round(float(df[col].dropna().mean()), 1)
        return analytics


if __name__ == "__main__":
    db = Database()
    db.init_db()
    print(f"Database initialized at: {db.db_path}")
