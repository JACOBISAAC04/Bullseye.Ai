from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.analysis.identity_bound_queue import (
    make_identity_bound_worker,
)
from app.analysis.leased_job_worker import process_one_leased
from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue


def make_queue(tmp_path, clock=None):
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return SingleAttemptLeasedQueue(
        tmp_path / "jobs.sqlite3", **kwargs
    )


def test_claim_binds_authoritative_identity(tmp_path):
    queue = make_queue(tmp_path)
    job_id = queue.enqueue({"video_path": "sample.mp4"})
    claimed = queue.claim_next(lease_seconds=30)
    assert claimed["job_id"] == job_id
    assert claimed["payload"] == {
        "video_path": "sample.mp4",
        "job_id": job_id,
    }
    assert claimed["attempts"] == 1


def test_expired_job_is_never_automatically_reclaimed(tmp_path):
    now = [100.0]
    queue = make_queue(tmp_path, clock=lambda: now[0])
    job_id = queue.enqueue({"video_path": "sample.mp4"})

    first = queue.claim_next(lease_seconds=5)
    assert first["job_id"] == job_id

    now[0] = 1000.0
    assert queue.claim_next(lease_seconds=5) is None

    saved = queue.get(job_id)
    assert saved["state"] == "running"
    assert saved["attempts"] == 1
    assert saved["owner_token"] == first["owner_token"]


def test_new_job_can_run_while_old_lease_expired(tmp_path):
    now = [100.0]
    queue = make_queue(tmp_path, clock=lambda: now[0])

    old_id = queue.enqueue({"video_path": "old.mp4"})
    queue.claim_next(lease_seconds=5)

    now[0] = 1000.0
    new_id = queue.enqueue({"video_path": "new.mp4"})
    claimed = queue.claim_next(lease_seconds=5)

    assert claimed["job_id"] == new_id
    assert queue.get(old_id)["attempts"] == 1
    assert queue.get(new_id)["attempts"] == 1


def test_fake_worker_completes_once(tmp_path):
    queue = make_queue(tmp_path)
    job_id = queue.enqueue({"video_path": "sample.mp4"})

    worker = make_identity_bound_worker(
        lambda payload: {
            "job_id": payload["job_id"],
            "fake": True,
        }
    )
    record = process_one_leased(
        queue, worker,
        lease_seconds=30,
        heartbeat_interval=1,
    )

    assert record["state"] == "completed"
    assert record["job_id"] == job_id
    assert record["attempts"] == 1
    assert process_one_leased(queue, worker) is None


def test_cannot_supply_payload_job_id(tmp_path):
    queue = make_queue(tmp_path)
    with pytest.raises(ValueError):
        queue.enqueue({
            "video_path": "sample.mp4",
            "job_id": str(uuid4()),
        })


def test_concurrent_claim_only_once(tmp_path):
    queue = make_queue(tmp_path)
    job_id = queue.enqueue({"video_path": "sample.mp4"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: queue.claim_next(lease_seconds=30),
            range(2),
        ))

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
    assert claimed[0]["job_id"] == job_id
    assert queue.get(job_id)["attempts"] == 1


def test_invalid_lease_duration(tmp_path):
    queue = make_queue(tmp_path)
    for value in (0, -1, True, 86401, "30"):
        with pytest.raises(ValueError):
            queue.claim_next(lease_seconds=value)


def test_empty_queue(tmp_path):
    queue = make_queue(tmp_path)
    assert queue.claim_next() is None