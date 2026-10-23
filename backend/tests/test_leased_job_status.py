import sqlite3
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue
from app.api.leased_job_status import router


TOKEN = "phase29-87R-test-token-not-for-production-123456789"


@pytest.fixture
def setup_api(tmp_path, monkeypatch):
    database = tmp_path / "test.sqlite3"
    queue = SingleAttemptLeasedQueue(database)
    job_id = queue.enqueue({"video_path": "PRIVATE-VIDEO-PATH"})

    monkeypatch.setenv("BULLSEYE_LEASED_STATUS_TOKEN", TOKEN)
    monkeypatch.setenv("BULLSEYE_LEASED_STATUS_DB", str(database))

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), queue, job_id, database


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_queued_status(setup_api):
    client, _, job_id, _ = setup_api
    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json() == {
        "job_id": job_id,
        "status": "queued",
        "attempts": 0,
        "automatic_retry": False,
    }


def test_running_status(setup_api):
    client, queue, job_id, _ = setup_api
    queue.claim_next(lease_seconds=300)
    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "processing"


def test_expired_lease_needs_review(setup_api):
    client, queue, job_id, database = setup_api
    queue.claim_next(lease_seconds=300)

    # Test-only modification to a temporary SQLite database.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE leased_jobs SET lease_until = 1 WHERE job_id = ?",
            (job_id,),
        )

    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert queue.get(job_id)["attempts"] == 1


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer wrong", "Basic abc", "Bearer "],
)
def test_unauthorized(setup_api, authorization):
    client, _, job_id, _ = setup_api
    kwargs = {}
    if authorization is not None:
        kwargs["headers"] = {"Authorization": authorization}
    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        **kwargs,
    )
    assert response.status_code == 401
    assert "PRIVATE-VIDEO-PATH" not in response.text


def test_unknown_job_is_404(setup_api):
    client, _, _, _ = setup_api
    response = client.get(
        f"/api/leased-jobs/{uuid4()}/status",
        headers=headers(),
    )
    assert response.status_code == 404


def test_invalid_job_id_is_404(setup_api):
    client, _, _, _ = setup_api
    response = client.get(
        "/api/leased-jobs/not-a-uuid/status",
        headers=headers(),
    )
    assert response.status_code == 404


def test_missing_database_returns_503(setup_api, monkeypatch, tmp_path):
    client, _, job_id, _ = setup_api
    missing = tmp_path / "missing.sqlite3"
    monkeypatch.setenv("BULLSEYE_LEASED_STATUS_DB", str(missing))

    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers=headers(),
    )
    assert response.status_code == 503
    assert not missing.exists()


def test_short_token_returns_503(setup_api, monkeypatch):
    client, _, job_id, _ = setup_api
    monkeypatch.setenv("BULLSEYE_LEASED_STATUS_TOKEN", "short")
    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers={"Authorization": "Bearer short"},
    )
    assert response.status_code == 503


def test_response_excludes_private_fields(setup_api):
    client, _, job_id, _ = setup_api
    response = client.get(
        f"/api/leased-jobs/{job_id}/status",
        headers=headers(),
    )
    assert response.status_code == 200
    assert set(response.json()) == {
        "job_id", "status", "attempts", "automatic_retry"
    }
    for secret in (
        "PRIVATE-VIDEO-PATH",
        "owner_token",
        "lease_until",
        "result",
    ):
        assert secret not in response.text