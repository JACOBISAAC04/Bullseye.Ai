"""Evidence-first single-video basketball shooting mechanics schema.

This module defines output structure only. It does not infer technique.
Unknown observations must never be converted to a pass or failure.
"""

import math
from copy import deepcopy
from uuid import UUID

SCHEMA_VERSION = "1.0"

SHOT_PHASES = (
    "setup",
    "upward_motion",
    "release",
    "follow_through",
    "landing",
)

METRICS = {
    "ball_arc": ("degrees", "release"),
    "release_angle": ("degrees", "release"),
    "follow_through": ("assessment", "follow_through"),
    "support_thumb_thumbing": ("assessment", "release"),
    "shooting_finger_alignment": ("assessment", "follow_through"),
    "support_hand_position": ("assessment", "release"),
    "grip_assessment": ("assessment", "release"),
    "elbow_angle_setup": ("degrees", "setup"),
    "elbow_angle_release": ("degrees", "release"),
    "elbow_angle_follow_through": ("degrees", "follow_through"),
    "hip_angle_setup": ("degrees", "setup"),
    "hip_angle_release": ("degrees", "release"),
    "hip_angle_follow_through": ("degrees", "follow_through"),
    "vertical_jump": ("assessment", "landing"),
    "landing_forward_displacement": ("normalized_body_units", "landing"),
    "landing_lateral_displacement": ("normalized_body_units", "landing"),
    "landing_balance": ("assessment", "landing"),
}

STATES = {
    "not_assessed",
    "observed",
    "not_visible",
    "insufficient_evidence",
}

ASSESSMENTS = {
    "acceptable",
    "needs_review",
    "detected",
    "not_detected",
    "aligned",
    "misaligned",
}


def _uuid4(value):
    if not isinstance(value, str):
        raise ValueError("Invalid job ID")
    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid job ID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Invalid job ID")
    return value


def new_shot_mechanics(job_id, shot_number, shooting_hand="unknown"):
    _uuid4(job_id)
    if type(shot_number) is not int or shot_number < 1:
        raise ValueError("Shot number must be a positive integer")
    if shooting_hand not in ("left", "right", "unknown"):
        raise ValueError("Invalid shooting hand")

    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "shot_number": shot_number,
        "shooting_hand": shooting_hand,
        "shot_outcome": "unknown",
        "phases": {phase: None for phase in SHOT_PHASES},
        "metrics": {
            name: {
                "state": "not_assessed",
                "value": None,
                "unit": unit,
                "phase": phase,
                "evidence_frames": [],
                "reason": None,
            }
            for name, (unit, phase) in METRICS.items()
        },
    }


def set_phase_frame(report, phase, frame):
    if phase not in SHOT_PHASES:
        raise ValueError("Unknown shot phase")
    if type(frame) is not int or frame < 1:
        raise ValueError("Frame must be a positive integer")
    result = deepcopy(report)
    result["phases"][phase] = frame
    return result


def set_metric(
    report,
    name,
    *,
    state,
    value=None,
    evidence_frames=None,
    reason=None,
):
    if name not in METRICS:
        raise ValueError("Unknown metric")
    if state not in STATES:
        raise ValueError("Invalid observation state")

    frames = [] if evidence_frames is None else evidence_frames
    if not isinstance(frames, list) or any(
        type(frame) is not int or frame < 1 for frame in frames
    ):
        raise ValueError("Invalid evidence frames")

    if state == "observed":
        if not frames:
            raise ValueError("Observed metrics require evidence frames")
        if value is None:
            raise ValueError("Observed metrics require a value")
        if METRICS[name][0] == "assessment":
            if not isinstance(value, str) or value not in ASSESSMENTS:
                raise ValueError("Invalid assessment")
        else:
            if type(value) not in (int, float):
                raise ValueError("Numeric measurement required")
            if not math.isfinite(value):
                raise ValueError("Measurement must be finite")
            if METRICS[name][0] == "degrees" and not 0 <= value <= 180:
                raise ValueError("Angle outside supported range")
    else:
        if value is not None:
            raise ValueError("Unobserved metrics cannot contain values")
        if frames:
            raise ValueError("Unobserved metrics cannot claim evidence")

    if reason is not None and (
        not isinstance(reason, str) or len(reason) > 300
    ):
        raise ValueError("Invalid reason")

    result = deepcopy(report)
    result["metrics"][name].update({
        "state": state,
        "value": value,
        "evidence_frames": list(frames),
        "reason": reason,
    })
    return result