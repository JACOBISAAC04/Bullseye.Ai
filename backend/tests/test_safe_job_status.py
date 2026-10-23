from uuid import uuid4

import pytest

from app.analysis.safe_job_status import public_job_status
from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue


def record(state="queued", attempts=0, lease_until=None):
    return {
        "job_id": str(uuid4()),
        "state": state,
        "attempts": attempts,
        "lease_until": lease_until,
        "owner_token": "SECRET-OWNER",
        "payload": {"video_path": "PRIVATE-VIDEO-PATH"},
        "result": {"private": "PRIVATE-RESULT"},
        "error": "PRIVATE-ERROR",
    }


@pytest.mark.parametrize(
    "state,expected",
    [
        ("queued", "queued"),
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_terminal_and_queued_status(state, expected):
    result = public_job_status(record(state), now=100)
    assert result["status"] == expected
    assert result["automatic_retry"] is False


def test_active_running_status():
    result = public_job_status(
        record("running", attempts=1, lease_until=101),
        now=100,
    )
    assert result["status"] == "processing"


def test_expired_running_status_requires_review():
    result = public_job_status(
        record("running", attempts=1, lease_until=99),
        now=100,
    )
    assert result["status"] == "needs_review"


def test_lease_boundary_is_expired():
    result = public_job_status(
        record("running", attempts=1, lease_until=100),
        now=100,
    )
    assert result["status"] == "needs_review"


def test_does_not_expose_private_fields():
    result = public_job_status(
        record("running", attempts=1, lease_until=101),
        now=100,
    )
    assert set(result) == {
        "job_id", "status", "attempts", "automatic_retry"
    }
    serialized = str(result)
    for secret in (
        "SECRET-OWNER",
        "PRIVATE-VIDEO-PATH",
        "PRIVATE-RESULT",
        "PRIVATE-ERROR",
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": "../escape"},
        {"job_id": "not-a-uuid"},
        {"state": "mystery"},
        {"attempts": -1},
        {"attempts": True},
        {"lease_until": float("nan"), "state": "running"},
        {"lease_until": None, "state": "running"},
    ],
)
def test_rejects_invalid_records(changes):
    value = record()
    value.update(changes)
    with pytest.raises(ValueError):
        public_job_status(value, now=100)


def test_rejects_invalid_clock():
    with pytest.raises(ValueError):
        public_job_status(record(), now=float("nan"))


def test_reads_real_single_attempt_queue(tmp_path):
    now = [100.0]
    queue = SingleAttemptLeasedQueue(
        tmp_path / "jobs.sqlite3",
        clock=lambda: now[0],
    )
    job_id = queue.enqueue({"video_path": "example.mp4"})

    queued = public_job_status(queue.get(job_id), now=100)
    assert queued["status"] == "queued"

    queue.claim_next(lease_seconds=5)
    active = public_job_status(queue.get(job_id), now=102)
    assert active["status"] == "processing"

    now[0] = 110.0
    expired = public_job_status(queue.get(job_id), now=110)
    assert expired["status"] == "needs_review"
    assert queue.claim_next(lease_seconds=5) is None

    saved = queue.get(job_id)
    assert saved["state"] == "running"
    assert saved["attempts"] == 1