from uuid import uuid4

import pytest

from app.analysis.single_video_evidence_adapter import (
    adapt_shot,
    adapt_shots,
)


@pytest.fixture
def job_id():
    return str(uuid4())


@pytest.fixture
def shot():
    return {
        "shot_number": 1,
        "arm": "right",
        "start_frame": 10,
        "release_frame": 20,
        "end_frame": 40,
        "trajectory": [
            {"frame": 20, "x": 100, "y": 100, "interpolated": True}
        ],
    }


def test_empty_pose_does_not_invent_metrics(job_id, shot):
    result = adapt_shot(job_id, shot, [])
    assert result["phases"]["setup"] == 10
    assert result["phases"]["release"] == 20
    assert result["phases"]["follow_through"] is None
    assert result["phases"]["landing"] is None
    assert all(
        item["state"] == "not_assessed"
        for item in result["metrics"].values()
    )
    assert result["shot_outcome"] == "unknown"


def test_observed_elbow_setup_and_release(job_id, shot):
    poses = [
        {
            "frame": 10, "pose_detected": True,
            "right_elbow_angle": 92,
        },
        {
            "frame": 20, "pose_detected": True,
            "right_elbow_angle": 154,
        },
    ]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["elbow_angle_setup"]["value"] == 92
    assert result["metrics"]["elbow_angle_release"]["value"] == 154
    assert result["metrics"]["elbow_angle_release"]["evidence_frames"] == [20]


def test_hip_requires_explicit_angle(job_id, shot):
    poses = [{
        "frame": 20,
        "pose_detected": True,
        "left_hip": {"x": 2, "y": 3},
        "right_hip": {"x": 4, "y": 3},
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["hip_angle_release"]["value"] is None


def test_explicit_hip_angle_supported(job_id, shot):
    poses = [{
        "frame": 20,
        "pose_detected": True,
        "right_hip_angle": 143.5,
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["hip_angle_release"]["value"] == 143.5


def test_pose_not_detected_cannot_supply_angle(job_id, shot):
    poses = [{
        "frame": 20,
        "pose_detected": False,
        "right_elbow_angle": 160,
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["elbow_angle_release"]["value"] is None


@pytest.mark.parametrize("angle", [None, True, -1, 181, float("nan"), float("inf")])
def test_invalid_pose_angle_ignored(job_id, shot, angle):
    poses = [{
        "frame": 20,
        "pose_detected": True,
        "right_elbow_angle": angle,
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["elbow_angle_release"]["value"] is None


def test_left_shooting_arm_selected(job_id, shot):
    shot["arm"] = "left"
    poses = [{
        "frame": 20,
        "pose_detected": True,
        "left_elbow_angle": 147,
        "right_elbow_angle": 98,
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["elbow_angle_release"]["value"] == 147


def test_interpolated_ball_does_not_produce_arc(job_id, shot):
    result = adapt_shot(job_id, shot, [])
    assert result["metrics"]["ball_arc"]["state"] == "not_assessed"
    assert result["metrics"]["release_angle"]["state"] == "not_assessed"


def test_no_grip_or_landing_inference(job_id, shot):
    result = adapt_shot(job_id, shot, [])
    for name in (
        "support_thumb_thumbing",
        "shooting_finger_alignment",
        "support_hand_position",
        "grip_assessment",
        "vertical_jump",
        "landing_balance",
        "landing_forward_displacement",
    ):
        assert result["metrics"][name]["state"] == "not_assessed"


def test_duplicate_pose_frames_rejected(job_id, shot):
    poses = [
        {"frame": 20, "pose_detected": True},
        {"frame": 20, "pose_detected": True},
    ]
    with pytest.raises(ValueError):
        adapt_shot(job_id, shot, poses)


def test_invalid_shot_order_rejected(job_id, shot):
    shot["release_frame"] = 50
    with pytest.raises(ValueError):
        adapt_shot(job_id, shot, [])


def test_unknown_shooting_arm_rejected(job_id, shot):
    shot["arm"] = None
    with pytest.raises(ValueError):
        adapt_shot(job_id, shot, [])


def test_multiple_shots(job_id, shot):
    second = dict(shot, shot_number=2)
    result = adapt_shots(job_id, [shot, second], [])
    assert len(result) == 2
    assert [item["shot_number"] for item in result] == [1, 2]


def test_duplicate_shot_numbers_rejected(job_id, shot):
    with pytest.raises(ValueError):
        adapt_shots(job_id, [shot, dict(shot)], [])


def test_missing_pose_frame_not_fabricated(job_id, shot):
    poses = [{
        "frame": 19,
        "pose_detected": True,
        "right_elbow_angle": 140,
    }]
    result = adapt_shot(job_id, shot, poses)
    assert result["metrics"]["elbow_angle_release"]["value"] is None