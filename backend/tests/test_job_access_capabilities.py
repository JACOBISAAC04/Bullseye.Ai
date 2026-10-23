import hashlib
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.analysis.job_access_capabilities import JobAccessCapabilities


@pytest.fixture
def store(tmp_path):
    return JobAccessCapabilities.initialize(
        tmp_path / "capabilities.sqlite3"
    )


def test_issue_and_verify(store):
    job = str(uuid4())
    token = store.issue(job)
    assert len(token) == 64
    assert store.verify(job, token)


def test_wrong_job_is_denied(store):
    first = str(uuid4())
    second = str(uuid4())
    token = store.issue(first)
    assert not store.verify(second, token)


def test_wrong_token_is_denied(store):
    job = str(uuid4())
    store.issue(job)
    assert not store.verify(job, "ab" * 32)


@pytest.mark.parametrize(
    "token",
    [None, "", "invalid", "00" * 31, "gg" * 32, True],
)
def test_invalid_token_is_denied(store, token):
    job = str(uuid4())
    store.issue(job)
    assert not store.verify(job, token)


def test_duplicate_issue_rejected(store):
    job = str(uuid4())
    original = store.issue(job)
    with pytest.raises(ValueError):
        store.issue(job)
    assert store.verify(job, original)


def test_database_stores_only_digest(store):
    job = str(uuid4())
    token = store.issue(job)

    with closing(sqlite3.connect(store.database)) as conn:
        row = conn.execute(
            "SELECT token_digest FROM job_capabilities WHERE job_id = ?",
            (job,),
        ).fetchone()

    assert row is not None
    assert row[0] == hashlib.sha256(bytes.fromhex(token)).digest()
    assert token.encode() not in store.database.read_bytes()


def test_revoke(store):
    job = str(uuid4())
    token = store.issue(job)
    assert store.revoke(job)
    assert not store.verify(job, token)
    assert not store.revoke(job)


def test_unknown_job_denied(store):
    assert not store.verify(str(uuid4()), "ab" * 32)


def test_invalid_job_id_denied(store):
    assert not store.verify("../other", "ab" * 32)
    with pytest.raises(ValueError):
        store.issue("../other")


def test_reopen_preserves_access(store):
    job = str(uuid4())
    token = store.issue(job)
    reopened = JobAccessCapabilities(store.database)
    assert reopened.verify(job, token)


def test_constructor_does_not_create_database(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        JobAccessCapabilities(missing)
    assert not missing.exists()


def test_concurrent_duplicate_issue_has_one_winner(store):
    job = str(uuid4())

    def attempt(_):
        try:
            return store.issue(job)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(pool.map(attempt, range(2)))

    winners = [token for token in tokens if token is not None]
    assert len(winners) == 1
    assert store.verify(job, winners[0])


def test_multiple_jobs_isolated(store):
    jobs = [str(uuid4()) for _ in range(3)]
    tokens = [store.issue(job) for job in jobs]

    for i, job in enumerate(jobs):
        for j, token in enumerate(tokens):
            assert store.verify(job, token) == (i == j)