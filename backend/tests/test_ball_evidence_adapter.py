"""Regression tests for joining saved API and accepted ball-observation evidence."""
from __future__ import annotations

from copy import deepcopy
import unittest

from app.coaching.ball_evidence_adapter import BallEvidenceAdapter
from app.coaching.ball_observation_store import build_ball_observation_report


def _ball(x, y=100.0, confidence=0.8):
    return {
        "x": x, "y": y, "x1": x-5, "y1": y-5,
        "x2": x+5, "y2": y+5, "confidence": confidence
    }


def _fixture(*, release_observed=False):
    nframes = 12
    shot = {
        "shot_number": 1, "arm": "right", "start_frame": 3,
        "release_frame": 6, "end_frame": 10,
    }
    frames = []
    for n in range(1, nframes+1):
        f = {"frame": n, "pose_detected": True}
        for joint, x, y in (
            ("right_wrist", 42.0, 70.0 - n),
            ("right_elbow", 46.0, 80.0),
            ("right_shoulder", 50.0, 65.0),
            ("right_hip", 50.0, 95.0),
            ("left_hip", 70.0, 95.0),
            ("left_ankle", 60.0, 150.0),
            ("right_ankle", 50.0, 150.0),
        ):
            f[joint] = {"x": x, "y": y, "z": 0.0}
        frames.append(f)
    api = {
        "analysis": {"frames": frames},
        "metrics": {"shots": [{**shot, "set": {"elbow_angle": 100.0},
                               "release": {"elbow_angle": 110.0}}]},
        "shots": [{**shot, "trajectory_points": 100}],
        "shot_count": 1,
    }
    # Includes a detector rejection at frame 7, never a coaching signal.
    observed = {3, 4, 5, 8, 9}
    if release_observed:
        observed.add(6)
    raw, filtered = [], []
    for n in range(1, nframes+1):
        yes = n in observed or n == 7
        entry = {
            "frame": n, "ball_detected": yes,
            "ball": _ball(20 + n) if yes else None,
        }
        raw.append(entry)
        filtered.append(deepcopy(entry))
    filtered[6]["ball_detected"] = False
    filtered[6]["ball"] = None
    report = build_ball_observation_report(
        raw, filtered, [shot], fps=24.0
    )
    return api, report


class BallEvidenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = BallEvidenceAdapter()
        self.api, self.ball = _fixture()

    def test_observed_ball_series_promoted_without_release(self):
        result = self.adapter.analyze(self.api, self.ball)
        shot = result["shots"][0]
        self.assertTrue(shot["signals"]["ball_positions"]["available"])
        self.assertEqual(
            [p["frame"] for p in shot["signals"]["ball_positions"]["samples"]],
            [3, 4, 5],
        )
        self.assertFalse(shot["signals"]["ball_position"]["available"])
        self.assertEqual(result["intended_form_match"], "not_assessed")

    def test_direct_release_observation_passes_exactly(self):
        api, ball = _fixture(release_observed=True)
        shot = self.adapter.analyze(api, ball)["shots"][0]
        self.assertEqual(shot["signals"]["ball_position"]["sample"]["frame"], 6)

    def test_rejected_outlier_not_promoted(self):
        result = self.adapter.analyze(self.api, self.ball)
        samples = result["shots"][0]["signals"]["ball_positions"]["samples"]
        self.assertNotIn(7, [p["frame"] for p in samples])

    def test_does_not_mutate_either_input(self):
        api, ball = deepcopy(self.api), deepcopy(self.ball)
        self.adapter.analyze(api, ball)
        self.assertEqual(api, self.api)
        self.assertEqual(ball, self.ball)

    def test_mismatched_shot_boundaries_rejected(self):
        self.ball["shots"][0]["release_frame"] = 7
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_mismatched_shooting_arm_rejected(self):
        self.ball["shots"][0]["arm"] = "left"
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_mismatched_video_frame_count_rejected(self):
        self.ball["total_frames"] += 1
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_mismatched_global_and_shot_samples_rejected(self):
        self.ball["shots"][0]["ball_positions"].pop()
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_forged_release_position_rejected(self):
        self.ball["shots"][0]["release_ball_observed"] = True
        self.ball["shots"][0]["release_ball_position"] = {
            **self.ball["ball_positions"][0], "frame": 6
        }
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_interpolated_detection_rejected(self):
        self.ball["ball_positions"][0]["interpolated"] = True
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_outlier_frame_cannot_be_accepted(self):
        b = deepcopy(self.ball["ball_positions"][0])
        b["frame"] = 7
        self.ball["ball_positions"].append(b)
        self.ball["ball_positions"].sort(key=lambda x: x["frame"])
        self.ball["accepted_observation_count"] += 1
        self.ball["detector_observation_count"] += 1
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_provenance_must_be_direct_detector(self):
        self.ball["ball_positions"][0]["provenance"] = "interpolated"
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_invalid_confidence_rejected(self):
        self.ball["ball_positions"][0]["confidence"] = 0.0
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_invalid_bbox_center_rejected(self):
        self.ball["ball_positions"][0]["x"] += 100
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_false_accepted_filter_flag_rejected(self):
        self.ball["ball_positions"][0]["accepted_by_existing_outlier_filter"] = False
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_modified_detector_count_rejected(self):
        self.ball["detector_observation_count"] -= 1
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)

    def test_trajectory_never_used_as_ball_observation(self):
        # No accepted detections inside shot, but dummy trajectory exists.
        self.api["shots"][0]["trajectory"] = [{"frame": 6, "x": 99, "y": 99}]
        result = self.adapter.analyze(self.api, self.ball)
        self.assertFalse(result["shots"][0]["signals"]["ball_position"]["available"])

    def test_metric_shot_boundary_mismatch_rejected(self):
        self.api["metrics"]["shots"][0]["release_frame"] = 7
        with self.assertRaises(ValueError):
            self.adapter.analyze(self.api, self.ball)


    def test_same_frame_ball_body_matches(self):
        shot = self.adapter.analyze(self.api, self.ball)["shots"][0]
        evidence = shot["ball_body_evidence"]
        self.assertEqual(evidence["pre_release_ball_observations"], 3)
        self.assertEqual(evidence["same_frame_pose_ball_matches"], 3)
        self.assertEqual(
            [s["frame"] for s in evidence["samples"]], [3, 4, 5]
        )
        self.assertIn("ball_to_wrist_distance_pixels", evidence["samples"][0])
        self.assertIsNone(evidence["release_frame_match"])

    def test_release_match_only_with_direct_ball_and_pose(self):
        api, ball = _fixture(release_observed=True)
        evidence = self.adapter.analyze(api, ball)["shots"][0]["ball_body_evidence"]
        self.assertEqual(evidence["release_frame_match"]["frame"], 6)

    def test_pose_gap_not_filled_by_ball_observation(self):
        self.api["analysis"]["frames"][3]["pose_detected"] = False  # frame 4
        evidence = self.adapter.analyze(self.api, self.ball)["shots"][0]["ball_body_evidence"]
        self.assertEqual(evidence["same_frame_pose_ball_matches"], 2)
        self.assertEqual(evidence["unmatched_pre_release_ball_observations"], 1)
        self.assertNotIn(4, [m["frame"] for m in evidence["samples"]])

    def test_missing_release_pose_prevents_body_match(self):
        api, ball = _fixture(release_observed=True)
        api["analysis"]["frames"][5]["pose_detected"] = False
        evidence = self.adapter.analyze(api, ball)["shots"][0]["ball_body_evidence"]
        self.assertIsNone(evidence["release_frame_match"])
        # Direct ball detection is still a direct detection; don't fabricate pose.
        self.assertTrue(self.adapter.analyze(api, ball)["shots"][0]["signals"]["ball_position"]["available"])

    def test_invalid_wrist_prevents_ball_body_match(self):
        self.api["analysis"]["frames"][2]["right_wrist"]["x"] = float("nan")
        evidence = self.adapter.analyze(self.api, self.ball)["shots"][0]["ball_body_evidence"]
        self.assertNotIn(3, [m["frame"] for m in evidence["samples"]])

if __name__ == "__main__":
    unittest.main()
