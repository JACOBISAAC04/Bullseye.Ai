"""Read-only, privacy-minimal status projection for leased jobs.

An expired lease means ownership is uncertain. It does NOT prove
that the analysis subprocess stopped or that its output is invalid.
"""

import math
import time
from uuid import UUID


def public_job_status(record, *, now=None):
    if not isinstance(record, dict):
        raise ValueError("Job record must be a dictionary")

    job_id = record.get("job_id")
    try:
        if not isinstance(job_id, str) or str(UUID(job_id, version=4)) != job_id:
            raise ValueError("Noncanonical job ID")
    except (TypeError, AttributeError) as exc:
        raise ValueError("Noncanonical job ID") from exc

    state = record.get("state")
    if state not in {"queued", "running", "completed", "failed"}:
        raise ValueError("Invalid job state")

    attempts = record.get("attempts")
    if type(attempts) is not int or attempts < 0:
        raise ValueError("Invalid attempt count")

    if now is None:
        now = time.time()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("Invalid clock")

    if state == "running":
        lease_until = record.get("lease_until")
        if (
            type(lease_until) not in (int, float)
            or not math.isfinite(lease_until)
        ):
            raise ValueError("Running job missing valid lease")
        status = "needs_review" if lease_until <= now else "processing"
    else:
        status = {
            "queued": "queued",
            "completed": "completed",
            "failed": "failed",
        }[state]

    return {
        "job_id": job_id,
        "status": status,
        "attempts": attempts,
        "automatic_retry": False,
    }