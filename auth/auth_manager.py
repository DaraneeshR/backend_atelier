"""
auth/auth_manager.py
──────────────────────────────────────────────────────
PBKDF2-HMAC-SHA256 hashing. No external dependencies.

Password storage format:
    pbkdf2_sha256$<iterations>$<hex_salt>$<hex_derived_key>

The iteration count is embedded so verify_password ALWAYS uses the
correct count, even if ITERATIONS constant changes in the future.
Salt is 16 random bytes (32 hex chars), unique per user.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3


ALGORITHM  = "pbkdf2_sha256"
ITERATIONS = 100_000     # ~0.8 s on modern hardware; embedded in every hash
KEY_LENGTH = 32          # bytes → 64 hex chars


def hash_password(plaintext: str) -> str:
    """Return  pbkdf2_sha256$<iters>$<hex_salt>$<hex_hash>"""
    salt = os.urandom(16)
    dk   = hashlib.pbkdf2_hmac("sha256", plaintext.encode(), salt, ITERATIONS, KEY_LENGTH)
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Re-derive the key using the embedded iteration count + salt and compare.
    Constant-time comparison guards against timing attacks.
    """
    try:
        parts = stored_hash.split("$")
        if len(parts) == 4:
            # New format: algo$iters$salt$hash
            algo, iters_str, salt_hex, dk_hex = parts
            iters = int(iters_str)
        elif len(parts) == 3:
            # Legacy format (no iter count stored) — assume old default
            algo, salt_hex, dk_hex = parts
            iters = 260_000
        else:
            return False
        if algo != ALGORITHM:
            return False
        salt = bytes.fromhex(salt_hex)
        dk   = hashlib.pbkdf2_hmac("sha256", plaintext.encode(), salt, iters, KEY_LENGTH)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# ── User operations ───────────────────────────────────────────────────────────

def create_user(conn: sqlite3.Connection, name: str, email: str,
                plaintext_password: str, role: str) -> int:
    """Insert a new user. Returns the new user id.
    Raises sqlite3.IntegrityError if email already exists."""
    ph  = hash_password(plaintext_password)
    cur = conn.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (name.strip(), email.strip().lower(), ph, role),
    )
    conn.commit()
    return cur.lastrowid


def get_user_by_email(conn: sqlite3.Connection, email: str) -> dict | None:
    """Return user dict or None."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    return dict(row) if row else None


def authenticate(conn: sqlite3.Connection, email: str,
                 plaintext_password: str) -> dict | None:
    """Verify credentials. Returns user dict on success, None on failure."""
    user = get_user_by_email(conn, email)
    if user and verify_password(plaintext_password, user["password_hash"]):
        return user
    return None
