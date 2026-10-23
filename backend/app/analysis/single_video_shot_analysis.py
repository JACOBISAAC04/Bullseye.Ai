"""Combine observed shot mechanics and 2D trajectory evidence.

The output is independent of live API routing. Physical shooting
angles and shot outcomes remain unknown without further evidence.
"""

from app.analysis.observed_ball_arc import analyze_observed_arc
from app.analysis.single_video_evidence_adapter import adapt_shots


def analyze_single_video_shots(job_id, shots, pose_frames, ball_frames):
    if not isinstance(shots, list):
        raise ValueError("Shots must be a list")
    if not isinstance(ball_frames, list):
        raise ValueError("Ball frames must be a list")

    mechanics = adapt_shots(job_id, shots, pose_frames)
    results = []

    for shot, report in zip(shots, mechanics):
        trajectory = analyze_observed_arc(
            ball_frames,
            release_frame=shot["release_frame"],
            end_frame=shot["end_frame"],
        )

        # The schema's degree fields are reserved for a properly
        # calibrated measurement. Do not place a 2D estimate there.
        assert report["metrics"]["release_angle"]["value"] is None
        assert report["metrics"]["ball_arc"]["value"] is None

        results.append({
            "shot_number": report["shot_number"],
            "mechanics": report,
            "trajectory_2d": trajectory,
            "shot_outcome": "unknown",
        })

    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "method": "single_video_observed_evidence",
        "shot_count": len(results),
        "shots": results,
        "limitations": [
            "2D trajectory measurements are camera-dependent.",
            "No physical launch angle is claimed without calibration.",
            "Interpolated ball detections are excluded.",
            "Shot outcomes are not inferred by this adapter.",
            "Hand, finger, grip, and landing assessments require separate evidence.",
        ],
    }