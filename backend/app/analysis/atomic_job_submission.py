"""Atomic queue submission with per-job bearer capability.

Opt-in component. Not registered with the live HTTP application.
Requires a new SQLite database initialized explicitly.
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import UUID, uuid4

from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue


def valid_job_id(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value


def token_digest(token):
    if not isinstance(token, str) or len(token) != 64:
        return None
    try:
        raw = bytes.fromhex(token)
    except ValueError:
        return None
    if len(raw) != 32:
        return None
    return hashlib.sha256(raw).digest()


class AtomicJobSubmission:
    def __init__(self, database):
        self.database = Path(database)
        if not self.database.is_file():
            raise FileNotFoundError(
                "Submission database must be initialized explicitly"
            )

    @classmethod
    def initialize(cls, database):
        path = Path(database)
        if path.exists():
            raise FileExistsError(
                "Refusing to initialize existing database"
            )
        if not path.parent.is_dir():
            raise ValueError("Database parent directory missing")

        # Exclusive creation prevents overwriting an existing DB.
        with path.open("xb"):
            pass

        try:
            SingleAttemptLeasedQueue(path)

            with closing(sqlite3.connect(path, timeout=30)) as conn:
                with conn:
                    conn.execute("""
                        CREATE TABLE job_capabilities (
                            job_id TEXT PRIMARY KEY
                                REFERENCES leased_jobs(job_id),
                            token_digest BLOB NOT NULL
                                CHECK(length(token_digest) = 32)
                        )
                    """)

            return cls(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def submit(self, video_path):
        if not isinstance(video_path, str):
            raise ValueError("Invalid video path")
        if not video_path.strip():
            raise ValueError("Empty video path")
        if len(video_path.encode("utf-8")) > 4096:
            raise ValueError("Video path too long")
        if "\x00" in video_path:
            raise ValueError("Invalid video path")

        # Upload-root containment and file validation are the
        # responsibility of the future HTTP submission boundary.
        payload = json.dumps(
            {"video_path": video_path},
            allow_nan=False,
        )

        job_id = str(uuid4())
        token = secrets.token_hex(32)
        digest = token_digest(token)
        timestamp = time.time()

        with closing(sqlite3.connect(self.database, timeout=30)) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")

                conn.execute(
                    """
                    INSERT INTO leased_jobs (
                        job_id, state, payload,
                        created_at, updated_at
                    ) VALUES (?, 'queued', ?, ?, ?)
                    """,
                    (job_id, payload, timestamp, timestamp),
                )

                conn.execute(
                    """
                    INSERT INTO job_capabilities (
                        job_id, token_digest
                    ) VALUES (?, ?)
                    """,
                    (job_id, digest),
                )

                conn.commit()
            except BaseException:
                conn.rollback()
                raise

        return {"job_id": job_id, "access_token": token}

    def verify(self, job_id, token):
        if not valid_job_id(job_id):
            return False

        digest = token_digest(token)
        if digest is None or not self.database.is_file():
            return False

        uri = self.database.resolve().as_uri() + "?mode=ro"

        with closing(sqlite3.connect(uri, uri=True, timeout=30)) as conn:
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
            and hmac.compare_digest(row[0], digest)
        )

    def revoke(self, job_id):
        if not valid_job_id(job_id):
            raise ValueError("Invalid job ID")

        with closing(sqlite3.connect(self.database, timeout=30)) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    DELETE FROM job_capabilities
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )
                return cursor.rowcount == 1