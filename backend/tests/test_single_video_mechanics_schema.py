from uuid import uuid4

import pytest

from app.analysis.single_video_mechanics_schema import (
    METRICS,
    SHOT_PHASES,
    new_shot_mechanics,
    set_metric,
    set_phase_frame,
)


@pytest.fixture
def report():
    return new_shot_mechanics(str(uuid4()), 1, "right")


def test_all_required_mechanics_present(report):
    required = {
        "ball_arc",
        "follow_through",
        "support_thumb_thumbing",
        "shooting_finger_alignment",
        "support_hand_position",
        "grip_assessment",
        "elbow_angle_setup",
        "elbow_angle_release",
        "hip_angle_setup",
        "hip_angle_release",
        "vertical_jump",
        "landing_forward_displacement",
        "landing_balance",
    }
    assert required <= set(report["metrics"])
    assert set(report["metrics"]) == set(METRICS)


def test_all_metrics_start_unknown(report):
    assert all(
        item["state"] == "not_assessed"
        and item["value"] is None
        and item["evidence_frames"] == []
        for item in report["metrics"].values()
    )
    assert report["shot_outcome"] == "unknown"


def test_phases_present(report):
    assert set(report["phases"]) == set(SHOT_PHASES)
    assert all(v is None for v in report["phases"].values())


def test_observed_angle_requires_evidence(report):
    with pytest.raises(ValueError):
        set_metric(
            report, "elbow_angle_release",
            state="observed", value=135.0
        )


def test_observed_angle_accepted(report):
    updated = set_metric(
        report, "elbow_angle_release",
        state="observed", value=135.0,
        evidence_frames=[25]
    )
    assert updated["metrics"]["elbow_angle_release"]["value"] == 135.0
    assert updated["metrics"]["elbow_angle_release"]["unit"] == "degrees"
    assert report["metrics"]["elbow_angle_release"]["value"] is None


def test_unseen_thumbing_stays_unknown(report):
    updated = set_metric(
        report, "support_thumb_thumbing",
        state="not_visible",
        reason="Thumb obscured by ball"
    )
    assert updated["metrics"]["support_thumb_thumbing"]["value"] is None


def test_unseen_metric_cannot_claim_pass(report):
    with pytest.raises(ValueError):
        set_metric(
            report, "support_thumb_thumbing",
            state="not_visible", value="not_detected"
        )


def test_observed_finger_alignment(report):
    updated = set_metric(
        report, "shooting_finger_alignment",
        state="observed",
        value="aligned",
        evidence_frames=[31, 32]
    )
    assert updated["metrics"]["shooting_finger_alignment"]["value"] == "aligned"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, 181])
def test_invalid_angles_rejected(report, value):
    with pytest.raises(ValueError):
        set_metric(
            report, "hip_angle_release",
            state="observed",
            value=value,
            evidence_frames=[20]
        )


def test_displacement_has_no_fabricated_centimeters(report):
    updated = set_metric(
        report, "landing_forward_displacement",
        state="observed",
        value=0.25,
        evidence_frames=[40, 52]
    )
    assert (
        updated["metrics"]["landing_forward_displacement"]["unit"]
        == "normalized_body_units"
    )


def test_set_phase_is_immutable(report):
    updated = set_phase_frame(report, "release", 30)
    assert updated["phases"]["release"] == 30
    assert report["phases"]["release"] is None


def test_invalid_phase_rejected(report):
    with pytest.raises(ValueError):
        set_phase_frame(report, "comparison", 10)


def test_invalid_shot_number_rejected():
    with pytest.raises(ValueError):
        new_shot_mechanics(str(uuid4()), 0)


def test_invalid_job_id_rejected():
    with pytest.raises(ValueError):
        new_shot_mechanics("../job", 1)


def test_left_handed_shooter_supported():
    report = new_shot_mechanics(str(uuid4()), 2, "left")
    assert report["shooting_hand"] == "left"


def test_observed_assessment_requires_allowed_value(report):
    with pytest.raises(ValueError):
        set_metric(
            report, "grip_assessment",
            state="observed",
            value="perfect",
            evidence_frames=[15]
        )


def test_invalid_evidence_frame_rejected(report):
    with pytest.raises(ValueError):
        set_metric(
            report, "follow_through",
            state="observed",
            value="needs_review",
            evidence_frames=[0]
        )