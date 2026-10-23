"""Identity-bound leased queue with no automatic reclaims.

An expired running job remains running until explicitly inspected.
This is intentionally conservative for video jobs with filesystem
side effects. It does not terminate an abandoned subprocess.
"""

import uuid

from app.analysis.identity_bound_queue import (
    IdentityBoundLeasedJobQueue,
)


class SingleAttemptLeasedQueue(IdentityBoundLeasedJobQueue):
    def claim_next(self, lease_seconds=300):
        if type(lease_seconds) not in (int, float):
            raise ValueError("Invalid lease duration")
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Lease duration out of range")

        now = float(self.clock())
        owner_token = str(uuid.uuid4())

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT job_id FROM leased_jobs
                WHERE state = 'queued' AND attempts = 0
                ORDER BY created_at, rowid
                LIMIT 1
            """).fetchone()

            if row is None:
                return None

            job_id = row["job_id"]
            updated = connection.execute("""
                UPDATE leased_jobs
                SET state = 'running',
                    owner_token = ?,
                    lease_until = ?,
                    attempts = attempts + 1,
                    updated_at = ?
                WHERE job_id = ?
                  AND state = 'queued'
                  AND attempts = 0
            """, (
                owner_token,
                now + lease_seconds,
                now,
                job_id,
            ))
            if updated.rowcount != 1:
                raise RuntimeError("Atomic queue claim failed")

        # Parent class injects queue identity into validated payload.
        # Its claim_next() cannot be used here because it also
        # automatically reclaims expired jobs.
        job = self.get(job_id)
        payload = job["payload"]

        if (
            not isinstance(payload, dict)
            or set(payload) != {"video_path"}
            or not isinstance(payload["video_path"], str)
            or not payload["video_path"].strip()
        ):
            job["payload"] = {
                "_invalid_identity_payload": True,
                "job_id": job_id,
            }
        else:
            job["payload"] = {
                "video_path": payload["video_path"],
                "job_id": job_id,
            }
        return job