"""Run from backend: python -m unittest discover -s tests -p test_coaching_feedback.py -v"""
import copy
import json
import unittest
from pathlib import Path

from app.coaching.coaching_feedback import CoachingFeedbackEngine


def comparison_payload():
    def shot(number, release, elbow, hip, knee):
        return {
            "shot_number": number,
            "release_frame": release,
            "angles_degrees": {
                "set": {"elbow_angle": 153.8, "hip_angle": 139.4, "knee_angle": 160.1},
                "release": {
                    "elbow_angle": elbow,
                    "hip_angle": hip,
                    "knee_angle": knee,
                },
                "follow_through": {},
            },
            "set_torso_lean_degrees": 15.2,
            "follow_through_balance_offset_normalized": 0.112,
        }

    return {
        "official_endorsement": False,
        "shot_count": 2,
        "observed_shots": [
            shot(1, 108, 78.196, 174.113, 154.450),
            shot(2, 396, 106.256, 174.569, 162.061),
        ],
        "release_angle_variation": {
            # The feedback module recomputes variation from observed shots.
            "elbow_angle": {"range_degrees": 999},
        },
    }


class CoachingFeedbackTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = CoachingFeedbackEngine()

    def test_measured_differences_are_descriptive_only(self):
        result = self.engine.analyze(comparison_payload())
        self.assertEqual(result["shot_count"], 2)
        self.assertEqual(result["intended_form_match"], "not_assessed")
        self.assertEqual(result["recommended_form_corrections"], [])
        self.assertAlmostEqual(
            result["release_angle_observations"][0]["range_degrees"], 28.06)
        self.assertEqual(result["official_endorsement"], False)
        self.assertEqual(result["measurement_validation"],
                         "unverified_raw_pose_and_frame_quality")
        for assessment in result["assessments"]:
            self.assertFalse(assessment["form_correction_authorized"])
            self.assertTrue(assessment["references"])
            self.assertNotIn("correct_form", assessment)

    def test_single_shot_cannot_claim_consistency(self):
        payload = comparison_payload()
        payload["observed_shots"] = payload["observed_shots"][:1]
        payload["shot_count"] = 1
        result = self.engine.analyze(payload)
        self.assertTrue(all(
            r["range_degrees"] is None
            for r in result["release_angle_observations"]
        ))
        self.assertIn("at least two", result["first_review_suggestion"])
        self.assertEqual(result["recommended_form_corrections"], [])

    def test_missing_values_not_fabricated(self):
        payload = comparison_payload()
        for shot in payload["observed_shots"]:
            shot["angles_degrees"]["release"] = {}
            shot["follow_through_balance_offset_normalized"] = None
            shot["set_torso_lean_degrees"] = None
            shot["angles_degrees"]["set"] = {}
        result = self.engine.analyze(payload)
        self.assertEqual(result["recommended_form_corrections"], [])
        self.assertTrue(all(
            a["assessment_status"] != "form_correct"
            for a in result["assessments"]
        ))
        self.assertEqual(result["assessments"][3]["assessment_status"],
                         "not_assessable")

    def test_bad_payload_rejected(self):
        for bad in ({}, {"shot_count": 2, "observed_shots": []},
                    {"shot_count": 2, "observed_shots": [],
                     "release_angle_variation": {},
                     "official_endorsement": True}):
            with self.assertRaises(ValueError):
                self.engine.analyze(bad)


if __name__ == "__main__":
    unittest.main()
