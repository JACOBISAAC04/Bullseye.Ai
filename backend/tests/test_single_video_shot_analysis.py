from uuid import uuid4

import pytest

from app.analysis.single_video_shot_analysis import (
    analyze_single_video_shots,
)


def ball(frame, x, y, confidence=0.95, interpolated=False):
    return {
        "frame": frame,
        "ball_detected": True,
        "interpolated": interpolated,
        "ball": {
            "x": x,
            "y": y,
            "confidence": confidence,
        },
    }


def shot(number, start, release, end, arm="right"):
    return {
        "shot_number": number,
        "start_frame": start,
        "release_frame": release,
        "end_frame": end,
        "arm": arm,
    }


def run(shots, poses=None, balls=None):
    return analyze_single_video_shots(
        str(uuid4()), shots, poses or [], balls or []
    )


def test_one_shot_with_observed_arc():
    result = run(
        [shot(1, 8, 10, 14)],
        [
            {
                "frame": 10,
                "pose_detected": True,
                "right_elbow_angle": 151,
            }
        ],
        [
            ball(10, 100, 300),
            ball(11, 110, 280),
            ball(12, 120, 265),
            ball(13, 130, 255),
            ball(14, 140, 250),
        ],
    )

    item = result["shots"][0]
    assert result["shot_count"] == 1
    assert item["trajectory_2d"]["status"] == "observed_descriptive_only"
    assert item["trajectory_2d"]["release_angle_2d_degrees"] > 0
    assert item["mechanics"]["metrics"]["elbow_angle_release"]["value"] == 151
    assert item["mechanics"]["metrics"]["release_angle"]["value"] is None
    assert item["mechanics"]["metrics"]["ball_arc"]["value"] is None
    assert item["shot_outcome"] == "unknown"


def test_missing_ball_evidence():
    result = run([shot(1, 8, 10, 14)])
    item = result["shots"][0]
    assert item["trajectory_2d"]["status"] == "insufficient_evidence"
    assert item["shot_outcome"] == "unknown"


def test_two_shots_have_independent_windows():
    result = run(
        [
            shot(1, 8, 10, 12),
            shot(2, 18, 20, 22, "left"),
        ],
        balls=[
            ball(10, 100, 300),
            ball(11, 110, 280),
            ball(12, 120, 265),
            ball(20, 300, 400),
            ball(21, 310, 380),
            ball(22, 320, 365),
        ],
    )

    assert result["shot_count"] == 2
    first = result["shots"][0]["trajectory_2d"]["observed_points"]
    second = result["shots"][1]["trajectory_2d"]["observed_points"]
    assert [p["frame"] for p in first] == [10, 11, 12]
    assert [p["frame"] for p in second] == [20, 21, 22]
    assert result["shots"][1]["mechanics"]["shooting_hand"] == "left"


def test_interpolated_points_never_count_as_direct():
    result = run(
        [shot(1, 8, 10, 12)],
        balls=[
            ball(10, 100, 300),
            ball(11, 110, 280, interpolated=True),
            ball(12, 120, 265),
        ],
    )
    trajectory = result["shots"][0]["trajectory_2d"]
    assert trajectory["status"] == "insufficient_evidence"
    assert [p["frame"] for p in trajectory["observed_points"]] == [10, 12]


def test_empty_video_shots():
    result = run([])
    assert result["shot_count"] == 0
    assert result["shots"] == []


def test_duplicate_shots_rejected():
    with pytest.raises(ValueError):
        run([
            shot(1, 8, 10, 12),
            shot(1, 18, 20, 22),
        ])


def test_invalid_shot_window_rejected():
    with pytest.raises(ValueError):
        run([shot(1, 8, 20, 12)])


def test_no_outcome_fabrication():
    result = run(
        [shot(1, 8, 10, 12)],
        balls=[
            ball(10, 100, 300),
            ball(11, 110, 280),
            ball(12, 120, 265),
        ],
    )
    item = result["shots"][0]
    assert item["shot_outcome"] == "unknown"
    assert item["mechanics"]["shot_outcome"] == "unknown"
    assert item["trajectory_2d"]["physical_angle_calibrated"] is False


def test_no_hand_or_landing_inference():
    result = run([shot(1, 8, 10, 12)])
    metrics = result["shots"][0]["mechanics"]["metrics"]
    for name in (
        "support_thumb_thumbing",
        "shooting_finger_alignment",
        "grip_assessment",
        "landing_balance",
        "vertical_jump",
    ):
        assert metrics[name]["state"] == "not_assessed"