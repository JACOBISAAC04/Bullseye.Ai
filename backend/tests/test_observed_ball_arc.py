import pytest

from app.analysis.observed_ball_arc import analyze_observed_arc


def frame(number, x, y, confidence=0.9, interpolated=False):
    return {
        "frame": number,
        "ball_detected": True,
        "interpolated": interpolated,
        "ball": {
            "x": x,
            "y": y,
            "confidence": confidence,
        },
    }


def test_clean_arc_is_descriptive_only():
    frames = [
        frame(10, 100, 300),
        frame(11, 110, 280),
        frame(12, 120, 265),
        frame(13, 130, 255),
        frame(14, 140, 250),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=14
    )
    assert result["status"] == "observed_descriptive_only"
    assert result["release_angle_2d_degrees"] > 0
    assert result["trajectory_fit"] is not None
    assert result["physical_angle_calibrated"] is False


def test_interpolated_points_excluded():
    frames = [
        frame(10, 100, 300),
        frame(11, 110, 280, interpolated=True),
        frame(12, 120, 265),
        frame(13, 130, 255),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=13
    )
    assert [p["frame"] for p in result["observed_points"]] == [10, 12, 13]


def test_interpolated_ball_object_excluded():
    frames = [
        frame(10, 100, 300),
        frame(11, 110, 280),
        frame(12, 120, 265),
    ]
    frames[1]["ball"]["interpolated"] = True
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert result["status"] == "insufficient_evidence"


def test_zero_confidence_excluded():
    frames = [
        frame(10, 100, 300),
        frame(11, 110, 280, confidence=0),
        frame(12, 120, 265),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert result["status"] == "insufficient_evidence"


def test_missing_release_observation_rejected():
    frames = [
        frame(11, 100, 300),
        frame(12, 110, 280),
        frame(13, 120, 265),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=13
    )
    assert result["status"] == "insufficient_evidence"


def test_duplicate_frames_rejected():
    frames = [
        frame(10, 100, 300),
        frame(10, 110, 280),
        frame(12, 120, 265),
    ]
    with pytest.raises(ValueError):
        analyze_observed_arc(
            frames, release_frame=10, end_frame=12
        )


def test_no_made_missed_inference():
    frames = [
        frame(10, 100, 300),
        frame(11, 110, 280),
        frame(12, 120, 265),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert "shot_outcome" not in result
    assert "made" not in result
    assert "missed" not in result


def test_nonfinite_detection_excluded():
    frames = [
        frame(10, 100, 300),
        frame(11, float("nan"), 280),
        frame(12, 120, 265),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert len(result["observed_points"]) == 2


def test_outside_shot_window_excluded():
    frames = [
        frame(9, 90, 320),
        frame(10, 100, 300),
        frame(11, 110, 280),
        frame(12, 120, 265),
        frame(13, 130, 255),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert [p["frame"] for p in result["observed_points"]] == [10, 11, 12]


def test_invalid_frame_window_rejected():
    with pytest.raises(ValueError):
        analyze_observed_arc([], release_frame=20, end_frame=10)


def test_insufficient_points():
    result = analyze_observed_arc(
        [frame(10, 100, 300)],
        release_frame=10,
        end_frame=15,
    )
    assert result["status"] == "insufficient_evidence"


def test_vertical_release_direction():
    frames = [
        frame(10, 100, 300),
        frame(11, 100, 280),
        frame(12, 100, 265),
        frame(13, 100, 255),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=13
    )
    assert result["release_angle_2d_degrees"] == 90.0


def test_stationary_initial_detection():
    frames = [
        frame(10, 100, 300),
        frame(11, 100, 300),
        frame(12, 110, 280),
    ]
    result = analyze_observed_arc(
        frames, release_frame=10, end_frame=12
    )
    assert result["status"] == "insufficient_evidence"