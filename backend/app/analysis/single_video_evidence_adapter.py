"""Conservative adapter from shot records and observed pose frames.

No outcome classification, hand-contact inference, arc estimation,
or foot-ground-contact inference occurs here.
"""

import math

from app.analysis.single_video_mechanics_schema import (
    new_shot_mechanics,
    set_metric,
    set_phase_frame,
)


def _frame(value):
    if type(value) is not int or value < 1:
        raise ValueError("Invalid frame number")
    return value


def _pose_index(frames):
    if not isinstance(frames, list):
        raise ValueError("Pose frames must be a list")

    indexed = {}
    for pose in frames:
        if not isinstance(pose, dict):
            raise ValueError("Invalid pose record")
        number = _frame(pose.get("frame"))
        if number in indexed:
            raise ValueError("Duplicate pose frame")
        indexed[number] = pose

    return indexed


def _angle(pose, name):
    if not isinstance(pose, dict):
        return None
    if pose.get("pose_detected") is not True:
        return None

    value = pose.get(name)
    if type(value) not in (int, float):
        return None
    if not math.isfinite(value) or not 0 <= value <= 180:
        return None

    return float(value)


def adapt_shot(job_id, shot, pose_frames):
    """Map one tracked shot into evidence-first mechanics output.

    Setup and release are tracker events, not biomechanically
    certified shot phases. Pose angles are descriptive 2D values.
    """
    if not isinstance(shot, dict):
        raise ValueError("Shot must be an object")

    number = shot.get("shot_number")
    arm = shot.get("arm")
    if arm not in ("left", "right"):
        raise ValueError("Shooting arm must be left or right")

    start = _frame(shot.get("start_frame"))
    release = _frame(shot.get("release_frame"))
    end = _frame(shot.get("end_frame"))
    if not start <= release <= end:
        raise ValueError("Invalid shot frame order")

    poses = _pose_index(pose_frames)
    report = new_shot_mechanics(job_id, number, arm)

    report = set_phase_frame(report, "setup", start)
    report = set_phase_frame(report, "release", release)

    for phase, frame in (("setup", start), ("release", release)):
        pose = poses.get(frame)

        for joint in ("elbow", "hip"):
            # Never compute missing hip angles from unrelated joints.
            # Never treat shot-level summaries as frame-level evidence.
            name = f"{arm}_{joint}_angle"
            value = _angle(pose, name)

            if value is None:
                continue

            report = set_metric(
                report,
                f"{joint}_angle_{phase}",
                state="observed",
                value=value,
                evidence_frames=[frame],
                reason=(
                    "Observed 2D pose angle at tracked "
                    f"{phase} frame; perspective-dependent"
                ),
            )

    return report


def adapt_shots(job_id, shots, pose_frames):
    if not isinstance(shots, list):
        raise ValueError("Shots must be a list")

    # Validate once and reject duplicate shot numbers.
    poses = _pose_index(pose_frames)
    indexed_frames = list(poses.values())
    result = []
    seen = set()

    for shot in shots:
        report = adapt_shot(job_id, shot, indexed_frames)
        number = report["shot_number"]
        if number in seen:
            raise ValueError("Duplicate shot number")
        seen.add(number)
        result.append(report)

    return result