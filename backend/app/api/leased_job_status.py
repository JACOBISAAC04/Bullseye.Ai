"""Protected, read-only leased-job status router.

Not registered in the live application. The shared bearer token
is suitable only for controlled infrastructure tests.
"""

import hmac
import os
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException

from app.analysis.safe_job_status import public_job_status
from app.analysis.single_attempt_queue import SingleAttemptLeasedQueue


router = APIRouter()


@router.get("/api/leased-jobs/{job_id}/status")
def get_leased_job_status(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    token = os.environ.get("BULLSEYE_LEASED_STATUS_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(
            status_code=503,
            detail="Status service unavailable",
        )

    if (
        not authorization
        or not authorization.startswith("Bearer ")
        or not hmac.compare_digest(
            authorization[7:].encode("utf-8"),
            token.encode("utf-8"),
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        parsed = UUID(job_id)
        if parsed.version != 4 or str(parsed) != job_id:
            raise ValueError("Noncanonical UUIDv4")
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Job not found")

    db_setting = os.environ.get("BULLSEYE_LEASED_STATUS_DB", "")
    if not db_setting:
        raise HTTPException(
            status_code=503,
            detail="Status service unavailable",
        )

    database = Path(db_setting)
    if not database.is_file():
        raise HTTPException(
            status_code=503,
            detail="Status service unavailable",
        )

    try:
        record = SingleAttemptLeasedQueue(database).get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return public_job_status(record)
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Status service unavailable",
        )