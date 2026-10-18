"""Synthetic safety and measurement tests for Phase 5.8."""
import copy
import math
import unittest

from app.coaching.measurement_coaching_feedback import MeasurementCoachingFeedback


IDS = (
    "posture_setup",
    "hip_loading",
    "ground_balance",
    "ball_setup",
    "ball_path",
    "individualized_feedback",
)


def fixture(*, ready=True, release_ball=False, shot_number=1):
    samples = []
    for frame, hip, knee in [(10, 145, 155), (11, 150, 160), (12, 155, 168)]:
        samples.append({
            "frame": frame,
            "complete_pose": True,
            "angles_deg": {"hip_angle": hip, "knee_angle": knee, "elbow_angle": 110},
        })
    temporal = {
        "shot_count": 1, "intended_form_match": "not_assessed",
        "shots": [{
            "shot_number": shot_number, "arm": "right",
            "start_frame": 10, "release_frame": 12, "end_frame": 15,
            "segments": {"set_to_release": {
                "quality": "sufficient_for_descriptive_timing" if ready else "insufficient_for_reliable_timing",
                "samples": samples,
            }},
        }]
    }
    ball_samples = [
        {"frame": 10, "value": {"x": 100, "y": 130, "confidence": 0.91}},
        {"frame": 11, "value": {"x": 102, "y": 110, "confidence": 0.81}},
    ]
    if release_ball:
        ball_samples.append({"frame": 12, "value": {"x": 105, "y": 100, "confidence": 0.8}})
    mechanics = {
        "shot_count": 1,
        "ball_evidence_origin": "ball_observations.json: accepted direct BallAnalyzer detections, after existing outlier filtering; re-expressed as ball_analysis.frames",
        "intended_form_match": "not_assessed",
        "shots": [{
            "shot_number": shot_number, "arm": "right",
            "start_frame": 10, "release_frame": 12, "end_frame": 15,
            "signals": {
                "ball_positions": {"source": "ball_analysis.frames", "samples": ball_samples, "available": True},
                "hip_center_series": {
                    "source": "analysis.frames", "samples": [
                        {"frame": 10, "value": {"x": 150, "y": 200}},
                        {"frame": 12, "value": {"x": 155, "y": 190}},
                    ]
                },
                "shot_metrics": {
                    "source": "metrics.shots",
                    "sample": {"phases": {
                        "set": {"hip_angle": 145.0, "knee_angle": 155.0},
                        "release": {"hip_angle": 155, "knee_angle": 168},
                    }},
                },
            },
            "ball_body_evidence": {
                "units": "uncalibrated_2d_pixels",
                "samples": [
                    {"frame": 10, "ball_center": {"x": 100, "y": 130}, "shooting_wrist": {"x": 108, "y": 136}},
                    {"frame": 11, "ball_center": {"x": 102, "y": 110}, "shooting_wrist": {"x": 108, "y": 118}},
                ],
            },
        }],
    }
    review = {
        "shot_count": 1, "intended_form_match": "not_assessed",
        "official_endorsement": False, "corrections_generated": 0,
        "shots": [{"shot_number": shot_number, "review_candidate_count": len(IDS),
                   "review_candidates": [
                       {"principle_id": pid, "title": pid,
                        "review_question": f"Review {pid}",
                        "source_urls": ["https://example.org/source"],
                        "source_support_status": "limited_published_text_support",
                        "review_status": "manual_review_candidate",
                        "original_video_timestamp_verified": False,
                        "intended_form_match": "not_assessed"}
                       for pid in IDS
                   ]}],
    }
    return temporal, mechanics, review


class MeasurementFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.api = MeasurementCoachingFeedback()
        self.data = fixture()

    def run_report(self, data=None):
        return self.api.generate(*(data or self.data))

    def test_six_measurements(self):
        report = self.run_report()
        rows = report["shots"][0]["review_candidates"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["status"] == "measured_descriptive_review" for row in rows))
        self.assertEqual(report["corrections_generated"], 0)
        self.assertEqual(report["intended_form_match"], "not_assessed")

    def test_real_frame_angles(self):
        rows = self.run_report()["shots"][0]["review_candidates"]
        hip = rows[1]["observations"][0]
        self.assertEqual((hip["start_frame"], hip["end_frame"]), (10, 12))
        self.assertEqual(hip["observed_endpoint_difference"], 10.0)

    def test_position_displacement(self):
        obs = self.run_report()["shots"][0]["review_candidates"][2]["observations"][0]
        self.assertEqual(obs["delta_x_pixels"], 5)
        self.assertEqual(obs["delta_y_pixels"], -10)

    def test_matched_ball_and_wrist(self):
        obs = self.run_report()["shots"][0]["review_candidates"][3]["observations"][0]
        self.assertEqual(obs["same_frame_matches"], 2)
        self.assertEqual(obs["first_distance_pixels"], 10)

    def test_ball_path_observed(self):
        obs = self.run_report()["shots"][0]["review_candidates"][4]["observations"][0]
        self.assertEqual(obs["delta_y_pixels"], -20)
        self.assertEqual(obs["end_frame"], 11)

    def test_missing_release_ball_not_inferred(self):
        report = self.run_report()
        ball = report["shots"][0]["review_candidates"][4]["observations"][0]
        self.assertNotEqual(ball["end_frame"], 12)

    def test_short_segment_cannot_claim_measured(self):
        a,b,c = fixture(ready=False)
        report = self.api.generate(a,b,c)
        self.assertTrue(all(r["status"] == "no_supported_measurement" for r in report["shots"][0]["review_candidates"]))

    def test_pose_measurements_not_invented(self):
        a,b,c = fixture()
        a["shots"][0]["segments"]["set_to_release"]["samples"] = []
        report = self.api.generate(a,b,c)
        self.assertEqual(report["shots"][0]["review_candidates"][0]["observations"], [])

    def test_missing_ball_origin_cannot_promote_ball(self):
        a,b,c = fixture()
        del b["ball_evidence_origin"]
        result = self.api.generate(a,b,c)
        self.assertEqual(result["shots"][0]["review_candidates"][4]["observations"], [])

    def test_empty_ball_samples_not_promoted(self):
        a,b,c = fixture()
        b["shots"][0]["signals"]["ball_positions"]["samples"] = []
        result = self.api.generate(a,b,c)
        self.assertEqual(result["shots"][0]["review_candidates"][4]["observations"], [])

    def test_reject_other_source(self):
        a,b,c = fixture()
        c["shots"][0]["review_candidates"][0]["source_support_status"] = "provisional_only"
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_reject_shot_mismatch(self):
        a,b,c = fixture()
        b["shots"][0]["release_frame"] = 13
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_reject_shot_number_mismatch(self):
        a,b,c = fixture()
        c["shots"][0]["shot_number"] = 2
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_reject_auto_grade(self):
        a,b,c = fixture()
        c["intended_form_match"] = "correct"
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_reject_auto_correction(self):
        a,b,c = fixture()
        c["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_does_not_change_input(self):
        before = copy.deepcopy(self.data)
        self.run_report()
        self.assertEqual(self.data, before)

    def test_reject_duplicate_topics(self):
        a,b,c = fixture()
        c["shots"][0]["review_candidates"][1]["principle_id"] = IDS[0]
        with self.assertRaises(ValueError):
            self.api.generate(a,b,c)

    def test_invalid_ball_confidence_ignored(self):
        a,b,c = fixture()
        b["shots"][0]["signals"]["ball_positions"]["samples"][0]["value"]["confidence"] = 0
        result = self.api.generate(a,b,c)
        self.assertEqual(result["shots"][0]["review_candidates"][4]["observations"], [])

    def test_bogus_hand_pair_is_not_accepted(self):
        a,b,c = fixture()
        b["shots"][0]["ball_body_evidence"]["samples"][0]["ball_center"]["x"] = 200
        result = self.api.generate(a,b,c)
        obs = result["shots"][0]["review_candidates"][3]["observations"][0]
        self.assertEqual(obs["same_frame_matches"], 1)

    def test_no_shots(self):
        a,b,c = fixture()
        for obj in (a,b,c):
            obj["shot_count"] = 0
            obj["shots"] = []
        result = self.api.generate(a,b,c)
        self.assertEqual(result["shot_count"], 0)

    def test_no_override_of_source_links(self):
        report = self.run_report()
        self.assertEqual(
            report["shots"][0]["review_candidates"][0]["source_urls"],
            ["https://example.org/source"],
        )


if __name__ == "__main__":
    unittest.main()
