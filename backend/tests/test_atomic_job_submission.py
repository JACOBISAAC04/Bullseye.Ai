import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from uuid import uuid4

import pytest

from app.analysis.atomic_job_submission import AtomicJobSubmission
from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue


@pytest.fixture
def store(tmp_path):
    return AtomicJobSubmission.initialize(
        tmp_path / "atomic.sqlite3"
    )


def counts(store):
    with closing(sqlite3.connect(store.database)) as conn:
        jobs = conn.execute(
            "SELECT COUNT(*) FROM leased_jobs"
        ).fetchone()[0]
        capabilities = conn.execute(
            "SELECT COUNT(*) FROM job_capabilities"
        ).fetchone()[0]
    return jobs, capabilities


def test_submission_commits_both_records(store):
    issued = store.submit("test-video.mp4")
    assert counts(store) == (1, 1)
    assert store.verify(issued["job_id"], issued["access_token"])


def test_queue_claim_receives_authoritative_identity(store):
    issued = store.submit("test-video.mp4")
    queue = SingleAttemptLeasedQueue(store.database)
    claimed = queue.claim_next(lease_seconds=300)
    assert claimed is not None
    assert claimed["job_id"] == issued["job_id"]
    assert claimed["payload"]["job_id"] == issued["job_id"]
    assert claimed["payload"]["video_path"] == "test-video.mp4"


def test_rollback_when_capability_insert_fails(store):
    with closing(sqlite3.connect(store.database)) as conn:
        with conn:
            conn.execute("""
                CREATE TRIGGER reject_capability
                BEFORE INSERT ON job_capabilities
                BEGIN
                    SELECT RAISE(ABORT, 'test failure');
                END
            """)

    with pytest.raises(sqlite3.DatabaseError):
        store.submit("test-video.mp4")

    assert counts(store) == (0, 0)
    assert SingleAttemptLeasedQueue(
        store.database
    ).claim_next(lease_seconds=300) is None


def test_rollback_when_queue_insert_fails(store):
    with closing(sqlite3.connect(store.database)) as conn:
        with conn:
            conn.execute("""
                CREATE TRIGGER reject_job
                BEFORE INSERT ON leased_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'test failure');
                END
            """)

    with pytest.raises(sqlite3.DatabaseError):
        store.submit("test-video.mp4")

    assert counts(store) == (0, 0)


def test_token_is_not_plaintext_in_database(store):
    issued = store.submit("test-video.mp4")
    token = issued["access_token"]

    with closing(sqlite3.connect(store.database)) as conn:
        digest = conn.execute(
            "SELECT token_digest FROM job_capabilities"
        ).fetchone()[0]

    assert digest == hashlib.sha256(
        bytes.fromhex(token)
    ).digest()
    assert token.encode() not in store.database.read_bytes()


def test_cross_job_access_denied(store):
    first = store.submit("first.mp4")
    second = store.submit("second.mp4")

    assert not store.verify(
        first["job_id"], second["access_token"]
    )
    assert not store.verify(
        second["job_id"], first["access_token"]
    )


def test_revocation(store):
    issued = store.submit("test-video.mp4")
    assert store.revoke(issued["job_id"])
    assert not store.verify(
        issued["job_id"], issued["access_token"]
    )


@pytest.mark.parametrize(
    "value",
    ["", "   ", None, 12, "a" * 4097, "bad\x00path"],
)
def test_invalid_video_path_rejected(store, value):
    with pytest.raises(ValueError):
        store.submit(value)
    assert counts(store) == (0, 0)


def test_invalid_token_rejected(store):
    issued = store.submit("test-video.mp4")
    assert not store.verify(issued["job_id"], "invalid")
    assert not store.verify(issued["job_id"], None)
    assert not store.verify(str(uuid4()), issued["access_token"])


def test_reopen_preserves_authorization(store):
    issued = store.submit("test-video.mp4")
    reopened = AtomicJobSubmission(store.database)
    assert reopened.verify(
        issued["job_id"], issued["access_token"]
    )


def test_existing_database_not_overwritten(store):
    before = store.database.read_bytes()

    with pytest.raises(FileExistsError):
        AtomicJobSubmission.initialize(store.database)

    assert store.database.read_bytes() == before


def test_missing_database_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"

    with pytest.raises(FileNotFoundError):
        AtomicJobSubmission(path)

    assert not path.exists()


def test_concurrent_submissions(store):
    def submit(i):
        return store.submit(f"video-{i}.mp4")

    with ThreadPoolExecutor(max_workers=4) as pool:
        issued = list(pool.map(submit, range(8)))

    assert len({item["job_id"] for item in issued}) == 8
    assert counts(store) == (8, 8)

    for item in issued:
        assert store.verify(
            item["job_id"], item["access_token"]
        )