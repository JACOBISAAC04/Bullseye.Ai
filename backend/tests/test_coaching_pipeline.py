import copy
import unittest

from app.coaching.coaching_pipeline import CoachingReportPipeline


def point(x, y):
    return {"x": x, "y": y}


def frame(number):
    # Fixed, complete arm/leg geometry; designed for structural tests only.
    return {
        "frame": number, "time": number / 30.0, "pose_detected": True,
        "right_shoulder": point(0, 0),
        "right_elbow": point(0, 1),
        "right_wrist": point(1, 1),
        "right_hip": point(0, 2),
        "right_knee": point(0, 3),
        "right_ankle": point(0, 4),
    }


def payload():
    shot = {
        "shot_number": 1, "arm": "right",
        "start_frame": 1, "release_frame": 7, "end_frame": 13,
        "set": {"elbow_angle": 90.0, "hip_angle": 175.0,
                "knee_angle": 160.0,
                "body_alignment": {"torso_lean_angle": 8.0}},
        "release": {"elbow_angle": 110.0, "hip_angle": 170.0,
                    "knee_angle": 165.0,
                    "ball_velocity_meters_per_second": None},
        "follow_through": {"elbow_angle": 155.0, "hip_angle": 175.0,
                           "knee_angle": 170.0,
                           "balance": {"normalized_offset": 0.1}},
    }
    return {
        "metrics": {"shot_count": 1, "shots": [shot]},
        "analysis": {"frames": [frame(n) for n in range(1, 14)]},
    }


class CoachingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CoachingReportPipeline()

    def test_report_has_no_unsupported_correction(self):
        report = self.pipeline.analyze(payload())
        self.assertEqual(report["shot_count"], 1)
        self.assertEqual(report["intended_form_match"], "not_assessed")
        self.assertFalse(report["official_endorsement"])
        self.assertEqual(report["coaching_feedback"]["recommended_form_corrections"], [])
        self.assertEqual(report["shots"][0]["technique_judgment"], "not_assessed")

    def test_short_preparation_stays_insufficient(self):
        case = payload()
        case["metrics"]["shots"][0]["release_frame"] = 5
        report = self.pipeline.analyze(case)
        quality = report["shots"][0]["timing_evidence"]["set_to_release"]["status"]
        self.assertEqual(quality, "insufficient_for_reliable_timing")

    def test_missing_frames_are_not_fabricated(self):
        case = payload()
        case["analysis"]["frames"] = [
            f for f in case["analysis"]["frames"] if f["frame"] != 4
        ]
        report = self.pipeline.analyze(case)
        quality = report["shots"][0]["timing_evidence"]["set_to_release"]
        self.assertEqual(quality["complete_pose_frames"], 6)
        self.assertEqual(quality["longest_missing_run"], 1)

    def test_missing_pose_data_is_rejected(self):
        case = payload()
        del case["analysis"]
        with self.assertRaises(ValueError):
            self.pipeline.analyze(case)

    def test_empty_shot_list_does_not_claim_form_match(self):
        case = {"metrics": {"shot_count": 0, "shots": []},
                "analysis": {"frames": []}}
        report = self.pipeline.analyze(case)
        self.assertEqual(report["shot_count"], 0)
        self.assertEqual(report["shots"], [])
        self.assertEqual(report["intended_form_match"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
