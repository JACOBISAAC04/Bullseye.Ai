"""Safety and schema tests for the standalone observed-mechanics extractor."""
import copy
import math
import unittest

from app.coaching.mechanics_signals import MechanicsSignalExtractor


def frame(n, detected=True, omit=()):
    if not detected:
        return {"frame": n, "time": n / 30, "pose_detected": False}
    result = {"frame": n, "time": n / 30, "pose_detected": True}
    for arm in ("left", "right"):
        for i, joint in enumerate(("shoulder", "elbow", "wrist", "hip", "knee", "ankle")):
            key = f"{arm}_{joint}"
            if key not in omit:
                result[key] = {"x": 100 + i * 5 + n, "y": 70 + i * 9 - n, "z": 0}
    return result


def shot(start=1, release=3, end=6, arm="right", number=1):
    return {
        "shot_number": number, "arm": arm,
        "start_frame": start, "release_frame": release, "end_frame": end,
        "set": {"elbow_angle": 151.5},
        "release": {"elbow_angle": 87.6},
        "follow_through": {"elbow_angle": 164.1},
    }


def payload():
    return {
        "analysis": {"frames": [frame(n) for n in range(1, 7)]},
        "metrics": {"shots": [shot()]},
        "shot_count": 1,
        "shots": [{"trajectory": [
            {"frame": n, "x": n, "y": n} for n in range(1, 6)
        ]}],
    }


def detected_ball(n, confidence=0.8, interpolated=False):
    return {
        "frame": n, "ball_detected": True, "interpolated": interpolated,
        "ball": {"x": 10 + n, "y": 20 + n, "confidence": confidence},
    }


class MechanicsSignalTests(unittest.TestCase):
    def setUp(self):
        self.extractor = MechanicsSignalExtractor()
        self.p = payload()

    def analyze(self):
        return self.extractor.analyze(self.p)

    def test_observed_pose_signal_names(self):
        s = self.analyze()["shots"][0]["signals"]
        for key in ("shoulder_hip_positions", "ankle_positions",
                    "hip_center_series", "wrist_positions",
                    "wrist_position", "elbow_positions", "shoulder_positions",
                    "hip_positions", "shot_metrics"):
            self.assertTrue(s[key]["available"], key)

    def test_shot_trajectory_does_not_claim_ball_detection(self):
        s = self.analyze()["shots"][0]["signals"]
        self.assertFalse(s["ball_positions"]["available"])
        self.assertFalse(s["ball_position"]["available"])

    def test_pose_gap_is_not_interpolated(self):
        self.p["analysis"]["frames"][1] = frame(2, detected=False)
        samples = self.analyze()["shots"][0]["signals"]["wrist_positions"]["samples"]
        self.assertEqual([x["frame"] for x in samples], [1, 3])
        self.assertEqual(self.analyze()["shots"][0]["signals"]["wrist_positions"]["adjacent_observed_pairs"], 0)

    def test_invalid_release_landmark_not_inferred(self):
        self.p["analysis"]["frames"][2] = frame(3, omit=("right_wrist",))
        self.assertFalse(self.analyze()["shots"][0]["signals"]["wrist_position"]["available"])

    def test_negative_or_nonfinite_landmark_ignored(self):
        self.p["analysis"]["frames"][2]["right_wrist"]["x"] = math.nan
        self.assertFalse(self.analyze()["shots"][0]["signals"]["wrist_position"]["available"])

    def test_hip_center_uses_both_sides(self):
        s = self.analyze()["shots"][0]["signals"]["hip_center_series"]
        self.assertEqual(s["observed_frames"], 3)
        self.assertIn("x", s["samples"][0]["value"])

    def test_hip_center_requires_both_sides(self):
        for f in self.p["analysis"]["frames"]:
            f.pop("left_hip", None)
        self.assertFalse(self.analyze()["shots"][0]["signals"]["hip_center_series"]["available"])

    def test_ankles_require_both_sides(self):
        for f in self.p["analysis"]["frames"]:
            f.pop("left_ankle", None)
        self.assertFalse(self.analyze()["shots"][0]["signals"]["ankle_positions"]["available"])

    def test_single_observation_not_a_series(self):
        self.p["analysis"]["frames"] = [frame(1), frame(2, False), frame(3, False)]
        s = self.analyze()["shots"][0]["signals"]["elbow_positions"]
        self.assertFalse(s["available"])
        self.assertEqual(s["observed_frames"], 1)

    def test_ball_positive_confidence_is_only_candidate_observation(self):
        self.p["ball_analysis"] = {"frames": [detected_ball(1), detected_ball(2), detected_ball(3)]}
        s = self.analyze()["shots"][0]["signals"]
        self.assertTrue(s["ball_positions"]["available"])
        self.assertTrue(s["ball_position"]["available"])
        self.assertEqual(s["ball_position"]["sample"]["frame"], 3)

    def test_interpolated_ball_is_ignored(self):
        self.p["ball_analysis"] = {"frames": [
            detected_ball(1), detected_ball(2, interpolated=True),
            detected_ball(3, confidence=0),
        ]}
        s = self.analyze()["shots"][0]["signals"]
        self.assertFalse(s["ball_positions"]["available"])
        self.assertFalse(s["ball_position"]["available"])

    def test_missing_ball_confidence_is_ignored(self):
        b = detected_ball(3)
        del b["ball"]["confidence"]
        self.p["ball_analysis"] = {"frames": [b]}
        self.assertFalse(self.analyze()["shots"][0]["signals"]["ball_position"]["available"])

    def test_ball_release_requires_exact_frame(self):
        self.p["ball_analysis"] = {"frames": [detected_ball(1), detected_ball(2)]}
        s = self.analyze()["shots"][0]["signals"]
        self.assertTrue(s["ball_positions"]["available"])
        self.assertFalse(s["ball_position"]["available"])

    def test_reject_duplicate_pose_frame(self):
        self.p["analysis"]["frames"].append(copy.deepcopy(self.p["analysis"]["frames"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate frame"):
            self.analyze()

    def test_reject_duplicate_ball_frame(self):
        b = detected_ball(1)
        self.p["ball_analysis"] = {"frames": [b, copy.deepcopy(b)]}
        with self.assertRaisesRegex(ValueError, "Duplicate frame"):
            self.analyze()

    def test_reject_invalid_shot_order_and_count(self):
        self.p["metrics"]["shots"][0]["start_frame"] = 4
        with self.assertRaises(ValueError):
            self.analyze()
        self.p["metrics"]["shots"][0]["start_frame"] = 1
        self.p["shot_count"] = 7
        with self.assertRaises(ValueError):
            self.analyze()

    def test_no_form_grading(self):
        report = self.analyze()
        self.assertEqual(report["intended_form_match"], "not_assessed")
        self.assertFalse(report["official_endorsement"])
        self.assertNotIn("correct", report["shots"][0])

    def test_no_pose_data_creates_no_position_samples(self):
        self.p["analysis"]["frames"] = [frame(n, detected=False) for n in range(1, 7)]
        s = self.analyze()["shots"][0]["signals"]
        self.assertFalse(s["wrist_positions"]["available"])
        self.assertFalse(s["wrist_position"]["available"])
        self.assertFalse(s["hip_center_series"]["available"])

    def test_left_arm_is_supported(self):
        self.p["metrics"]["shots"][0]["arm"] = "left"
        sample = self.analyze()["shots"][0]["signals"]["wrist_position"]["sample"]
        self.assertIsNotNone(sample)

    def test_empty_shots(self):
        self.p["metrics"]["shots"] = []
        self.p["shot_count"] = 0
        report = self.analyze()
        self.assertEqual(report["shot_count"], 0)
        self.assertEqual(report["shots"], [])


if __name__ == "__main__":
    unittest.main()
