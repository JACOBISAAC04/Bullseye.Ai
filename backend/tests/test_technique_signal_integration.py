"""Tests for joining observed mechanics signals to the reference validator."""
import copy
import unittest

from app.coaching.mechanics_signals import MechanicsSignalExtractor
from app.coaching.temporal_form_analyzer import TemporalFormAnalyzer
from app.coaching.technique_signal_integration import TechniqueSignalIntegrator


def frame(n, detected=True):
    result = {"frame": n, "time": n / 30, "pose_detected": detected}
    if detected:
        for arm, offset in (("right", 0), ("left", -17)):
            for i, joint in enumerate(("shoulder", "elbow", "wrist", "hip", "knee", "ankle")):
                result[f"{arm}_{joint}"] = {
                    "x": 100 + offset + i * 4 + n / 6,
                    "y": 100 + i * 16 + i * i + n / 4,
                    "z": 0,
                }
    return result


def payload():
    return {
        "analysis": {"frames": [frame(n) for n in range(1, 17)]},
        "metrics": {"shots": [{
            "shot_number": 1, "arm": "right",
            "start_frame": 1, "release_frame": 12, "end_frame": 16,
            "set": {"elbow_angle": 150.0},
            "release": {"elbow_angle": 96.0},
            "follow_through": {"elbow_angle": 145.0},
        }]},
        "shot_count": 1,
        "shots": [{"trajectory": [{"frame": n, "x": n, "y": 10} for n in range(1, 17)]}],
    }


def reports(p=None):
    p = p or payload()
    return TemporalFormAnalyzer().analyze(p), MechanicsSignalExtractor().analyze(p)


def status(output, key):
    matches = [a for a in output["shots"][0]["assessments"] if a["principle_id"] == key]
    assert len(matches) == 1
    return matches[0]


class TechniqueSignalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporal, self.mechanics = reports()
        self.evaluator = TechniqueSignalIntegrator()

    def run_eval(self):
        return self.evaluator.analyze(self.temporal, self.mechanics)

    def test_22_topics_retained_and_never_grades(self):
        out = self.run_eval()
        self.assertEqual(out["reference_count"], 22)
        self.assertEqual(len(out["shots"][0]["assessments"]), 22)
        self.assertEqual(out["intended_form_match"], "not_assessed")
        self.assertFalse(out["official_endorsement"])
        self.assertTrue(all(
            a["intended_form_match"] == "not_assessed"
            for a in out["shots"][0]["assessments"]
        ))

    def test_existing_posture_observation_is_resolved_but_source_provisional(self):
        result = status(self.run_eval(), "posture_setup")
        self.assertEqual(result["status"], "source_verification_required")
        self.assertEqual(result["missing_signals"], [])
        self.assertIn("shoulder_hip_positions", result["newly_available_signals"])
        self.assertIsNone(result["review_question"])

    def test_approved_topic_becomes_descriptive_only(self):
        result = status(self.run_eval(), "individualized_feedback")
        self.assertEqual(result["status"], "descriptive_review_possible")
        self.assertIn("shot_metrics", result["newly_available_signals"])
        self.assertEqual(result["intended_form_match"], "not_assessed")
        self.assertTrue(result["review_question"])

    def test_ball_trajectory_does_not_fill_ball_signals(self):
        for name in ("ball_setup", "ball_lead", "ball_path"):
            item = status(self.run_eval(), name)
            self.assertEqual(item["status"], "missing_measurements")
            self.assertIn("ball_positions", item["missing_signals"])

    def test_release_wrist_does_not_imply_hand_under_ball(self):
        item = status(self.run_eval(), "hand_under_ball")
        self.assertIn("wrist_position", item["newly_available_signals"])
        self.assertIn("ball_position", item["missing_signals"])
        self.assertEqual(item["status"], "missing_measurements")

    def test_guide_hand_has_no_measurement(self):
        item = status(self.run_eval(), "guide_hand_compensation")
        self.assertEqual(item["status"], "missing_measurements")
        self.assertEqual(item["newly_available_signals"], [])

    def test_pose_gap_cannot_pass_timing_quality_gate(self):
        self.temporal["shots"][0]["segments"]["set_to_release"]["quality"] = "insufficient_for_reliable_timing"
        item = status(self.run_eval(), "posture_setup")
        self.assertEqual(item["status"], "insufficient_timing_evidence")
        self.assertFalse(item["review_question"])

    def test_context_based_tip_not_assumed(self):
        item = status(self.run_eval(), "feet_ready_before_catch")
        self.assertEqual(item["status"], "shot_context_not_established")

    def test_shot_release_mismatch_rejected(self):
        self.mechanics["shots"][0]["release_frame"] = 11
        with self.assertRaisesRegex(ValueError, "release_frame"):
            self.run_eval()

    def test_shot_arm_mismatch_rejected(self):
        self.mechanics["shots"][0]["arm"] = "left"
        with self.assertRaisesRegex(ValueError, "arm"):
            self.run_eval()

    def test_bad_mechanics_source_not_promoted(self):
        self.mechanics["shots"][0]["signals"]["wrist_positions"]["source"] = "shots[].trajectory"
        out = self.run_eval()
        self.assertIn("wrist_positions", status(out, "ball_setup")["missing_signals"])

    def test_falsified_availability_flag_not_sufficient(self):
        s = self.mechanics["shots"][0]["signals"]["hip_center_series"]
        s["available"] = True
        s["samples"] = []
        item = status(self.run_eval(), "vertical_motion")
        self.assertEqual(item["status"], "missing_measurements")

    def test_stale_release_wrist_not_promoted(self):
        self.mechanics["shots"][0]["signals"]["wrist_position"]["sample"]["frame"] = 9
        item = status(self.run_eval(), "release_in_sequence")
        self.assertIn("wrist_position", item["missing_signals"])

    def test_detector_observed_ball_can_fill_but_not_grade(self):
        p = payload()
        p["ball_analysis"] = {"frames": [
            {"frame": n, "ball_detected": True,
             "ball": {"x": 25 + n, "y": 60, "confidence": 0.8}}
            for n in range(1, 13)
        ]}
        temporal, mech = reports(p)
        out = self.evaluator.analyze(temporal, mech)
        item = status(out, "ball_path")
        self.assertEqual(item["missing_signals"], [])
        self.assertEqual(item["status"], "source_verification_required")
        self.assertEqual(out["intended_form_match"], "not_assessed")

    def test_interpolated_detections_not_promoted(self):
        p = payload()
        p["ball_analysis"] = {"frames": [
            {"frame": n, "ball_detected": True, "interpolated": True,
             "ball": {"x": n, "y": 30, "confidence": 0.9}}
            for n in range(1, 13)
        ]}
        out = self.evaluator.analyze(*reports(p))
        self.assertIn("ball_positions", status(out, "ball_path")["missing_signals"])

    def test_unknown_ball_source_flag_not_accepted(self):
        s = self.mechanics["shots"][0]["signals"]["ball_positions"]
        s.update({
            "available": True, "source": "ball_analysis.frames",
            "samples": [{"frame": n, "value": {"x": n, "y": n, "confidence": 0.9}}
                        for n in range(1, 13)],
            "observed_frames": 12, "adjacent_observed_pairs": 11,
        })
        out = self.run_eval()
        self.assertIn("ball_positions", status(out, "ball_path")["missing_signals"])

    def test_bad_shot_count_rejected(self):
        self.mechanics["shot_count"] = 3
        with self.assertRaisesRegex(ValueError, "shot_count"):
            self.run_eval()

    def test_no_shots(self):
        out = self.evaluator.analyze(
            {"shot_count": 0, "shots": []},
            {"shot_count": 0, "shots": [], "ball_detection_source_present": False},
        )
        self.assertEqual(out["shot_count"], 0)
        self.assertEqual(out["intended_form_match"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
