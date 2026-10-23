"""Isolated per-job bearer capabilities.

Tokens are random and returned only when issued. SQLite stores
SHA256 digests, never plaintext tokens. Possession of a token
grants access to its job; this is not account authentication.
"""

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID


def _job_id(value):
    if not isinstance(value, str):
        raise ValueError("Invalid job ID")
    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid job ID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Invalid job ID")
    return value


def _token_digest(token):
    if not isinstance(token, str):
        return None
    if len(token) != 64:
        return None
    try:
        raw = bytes.fromhex(token)
    except ValueError:
        return None
    if len(raw) != 32:
        return None
    return hashlib.sha256(raw).digest()


class JobAccessCapabilities:
    def __init__(self, database):
        self.database = Path(database)
        if not self.database.is_file():
            raise FileNotFoundError(
                "Capability database must be initialized explicitly"
            )

    @classmethod
    def initialize(cls, database):
        path = Path(database)
        if not path.parent.is_dir():
            raise ValueError("Capability database parent missing")

        with closing(sqlite3.connect(path, timeout=10)) as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS job_capabilities (
                        job_id TEXT PRIMARY KEY,
                        token_digest BLOB NOT NULL
                            CHECK(length(token_digest) = 32)
                    )
                """)
        return cls(path)

    def issue(self, job_id):
        job_id = _job_id(job_id)
        token = secrets.token_hex(32)
        digest = _token_digest(token)

        with closing(sqlite3.connect(self.database, timeout=10)) as conn:
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO job_capabilities
                            (job_id, token_digest)
                        VALUES (?, ?)
                        """,
                        (job_id, digest),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "Capability already issued for this job"
                ) from exc
        return token

    def verify(self, job_id, token):
        try:
            job_id = _job_id(job_id)
        except ValueError:
            return False

        supplied = _token_digest(token)
        if supplied is None:
            return False

        if not self.database.is_file():
            return False

        # Open read-only; verification must never create a DB.
        uri = self.database.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=10)) as conn:
            row = conn.execute(
                """
                SELECT token_digest
                FROM job_capabilities
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        return (
            row is not None
            and isinstance(row[0], bytes)
            and hmac.compare_digest(row[0], supplied)
        )

    def revoke(self, job_id):
        job_id = _job_id(job_id)
        with closing(sqlite3.connect(self.database, timeout=10)) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM job_capabilities WHERE job_id = ?",
                    (job_id,),
                )
                return cursor.rowcount == 1