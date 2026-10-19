"""Phase 6.1/6.2: player-facing evidence and replay safety tests.

Run from backend: python -m unittest discover -s tests -p "test_player_feedback.py" -v
"""
import copy
import math
import unittest
from app.coaching.player_feedback import PlayerFeedbackBuilder


URL = "https://www.hustletraining.com/blog/how-to-shoot-with-perfect-form-from-professional-shooting-coach-mike-dunn"
JOB = "9a9333b9-2c79-4a75-9a5c-e8a70b5d83a4"


def _candidate(pid, title, observation):
    return {
        "principle_id": pid,
        "title": title,
        "review_question": f"Review the observed {title.lower()} movement.",
        "source_support_status": "limited_published_text_support",
        "status": "measured_descriptive_review",
        "intended_form_match": "not_assessed",
        "source_urls": [URL],
        "observations": [observation],
    }


def _inputs():
    analysis = {
        "job_id": JOB,
        "video_url": f"/video/results/{JOB}/video",
        "shot_count": 2,
        "shots": [
            {"shot_number": 1, "arm": "right", "start_frame": 68, "release_frame": 108, "end_frame": 133},
            {"shot_number": 2, "arm": "right", "start_frame": 357, "release_frame": 396, "end_frame": 419},
        ],
    }
    hip = {
        "signal": "hip_angle", "unit": "degrees_2d",
        "start_frame": 357, "start_value": 152.594,
        "end_frame": 396, "end_value": 174.569,
        "observed_endpoint_difference": 21.975, "observed_samples": 40,
    }
    knee = {
        "signal": "knee_angle", "unit": "degrees_2d",
        "start_frame": 357, "start_value": 160.354,
        "end_frame": 396, "end_value": 162.061,
        "observed_endpoint_difference": 1.707, "observed_samples": 40,
    }
    balance = {
        "signal": "position_displacement", "joint": "hip_center",
        "unit": "uncalibrated_2d_pixels",
        "start_frame": 357, "end_frame": 396,
        "delta_x_pixels": -8.24, "delta_y_pixels": -4.681,
        "observed_samples": 40,
    }
    wrist = {
        "signal": "same_frame_ball_to_wrist", "unit": "uncalibrated_2d_pixels",
        "first_frame": 357, "first_distance_pixels": 29.055,
        "last_frame": 393, "last_distance_pixels": 11.873,
        "same_frame_matches": 35,
    }
    ball = {
        "signal": "ball_path_endpoints", "unit": "uncalibrated_2d_pixels",
        "start_frame": 357, "start_position": {"x": 550.271, "y": 344.142},
        "start_detector_confidence": 0.8172,
        "end_frame": 393, "end_position": {"x": 562.299, "y": 249.537},
        "end_detector_confidence": 0.6545,
        "delta_x_pixels": 12.028, "delta_y_pixels": -94.605,
        "observed_samples": 35,
    }
    phases = {
        "signal": "phase_angles", "unit": "degrees_2d",
        "observations": [
            {"phase": "set", "angles_degrees_2d": {
                "elbow_angle": 147.285, "hip_angle": 152.594, "knee_angle": 160.354,
            }},
            {"phase": "release", "angles_degrees_2d": {
                "elbow_angle": 106.256, "hip_angle": 174.569, "knee_angle": 162.061,
            }},
        ],
    }
    candidates = [
        _candidate("posture_setup", "Posture before motion", hip),
        _candidate("hip_loading", "Hip loading", hip),
        _candidate("ground_balance", "Balance and connection to ground", balance),
        _candidate("ball_setup", "Organizing hands and ball", wrist),
        _candidate("ball_path", "Pre-release ball pathway", ball),
        _candidate("individualized_feedback", "Player-specific mechanical diagnosis", phases),
    ]
    candidates[1]["observations"].append(knee)
    measured = {
        "shot_count": 2,
        "shots": [
            {"shot_number": 1, "timing_quality": "insufficient_for_reliable_timing",
             "review_candidates": [], "review_candidate_count": 0, "intended_form_match": "not_assessed"},
            {"shot_number": 2, "timing_quality": "sufficient_for_descriptive_timing",
             "review_candidates": candidates, "review_candidate_count": 6, "intended_form_match": "not_assessed"},
        ],
        "intended_form_match": "not_assessed",
        "official_endorsement": False,
        "corrections_generated": 0,
    }
    return analysis, measured


class PlayerFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.analysis, self.measured = _inputs()
        self.builder = PlayerFeedbackBuilder()

    def generate(self, *, fps=30, frame_origin=1):
        return self.builder.generate(self.analysis, self.measured, fps=fps, frame_origin=frame_origin)

    def test_six_cards_video3(self):
        report = self.generate()
        self.assertEqual(report["shot_count"], 2)
        self.assertEqual([s["feedback_count"] for s in report["shots"]], [0, 6])
        self.assertEqual(report["corrections_generated"], 0)

    def test_safe_form_and_endorsement_flags(self):
        report = self.generate()
        self.assertEqual(report["intended_form_match"], "not_assessed")
        self.assertIs(report["official_endorsement"], False)
        self.assertTrue(all(s["intended_form_match"] == "not_assessed" for s in report["shots"]))

    def test_video3_replay_seconds_one_based(self):
        r = self.generate()
        self.assertEqual(r["shots"][0]["replay"]["start_seconds"], round(67/30, 3))
        self.assertEqual(r["shots"][1]["replay"]["release_seconds"], round(395/30, 3))
        self.assertEqual(r["shots"][1]["replay"]["end_seconds"], round(418/30, 3))

    def test_topic_replay_uses_measurement_frames(self):
        c = self.generate()["shots"][1]["feedback_cards"][4]
        self.assertEqual((c["replay"]["start_frame"], c["replay"]["end_frame"]), (357, 393))
        self.assertEqual(c["replay"]["end_seconds"], round(392/30, 3))

    def test_ball_up_is_only_image_plane(self):
        text = self.generate()["shots"][1]["feedback_cards"][4]["observation_text"]
        self.assertIn("94.6 px up", text)
        self.assertIn("12.0 px right", text)
        self.assertNotIn("perfect", text.lower())

    def test_ball_wrist_does_not_assert_grip(self):
        card = self.generate()["shots"][1]["feedback_cards"][3]
        self.assertIn("29.1 image px", card["observation_text"])
        self.assertIn("11.9 image px", card["observation_text"])
        self.assertTrue(any("contact" in note for note in card["limitations"]))

    def test_hip_knee_distinct(self):
        card = self.generate()["shots"][1]["feedback_cards"][1]
        self.assertIn("hip angle", card["observation_text"])
        self.assertIn("knee angle", card["observation_text"])
        self.assertEqual(card["measurement_count"], 2)

    def test_balance_pixels_not_physical(self):
        card = self.generate()["shots"][1]["feedback_cards"][2]
        self.assertIn("8.2 px left", card["observation_text"])
        self.assertIn("4.7 px up", card["observation_text"])
        self.assertTrue(any("balance" in note for note in card["limitations"]))

    def test_phase_angles_do_not_claim_exact_frame(self):
        card = self.generate()["shots"][1]["feedback_cards"][5]
        self.assertIn("did not provide individual frame IDs", card["observation_text"])
        self.assertIn("set sample", card["observation_text"])
        self.assertIn("release sample", card["observation_text"])

    def test_source_url_retained(self):
        self.assertEqual(self.generate()["shots"][1]["feedback_cards"][0]["source_urls"], [URL])

    def test_video_url_retained_if_job_matches(self):
        self.assertEqual(self.generate()["video_url"], f"/video/results/{JOB}/video")

    def test_does_not_mutate_inputs(self):
        a, m = copy.deepcopy((self.analysis, self.measured))
        self.generate()
        self.assertEqual((self.analysis, self.measured), (a, m))

    def test_zero_shots(self):
        a, m = {"shots": [], "shot_count": 0}, {
            "shots": [], "shot_count": 0,
            "intended_form_match": "not_assessed",
            "corrections_generated": 0, "official_endorsement": False,
        }
        report = self.builder.generate(a, m, fps=25)
        self.assertEqual(report["shots"], [])

    def test_video8_no_review(self):
        self.analysis["shots"] = [self.analysis["shots"][0]]
        self.analysis["shot_count"] = 1
        self.measured["shots"] = [self.measured["shots"][0]]
        self.measured["shot_count"] = 1
        self.assertEqual(self.generate()["shots"][0]["feedback_count"], 0)

    def test_manual_fps_required(self):
        with self.assertRaises(ValueError):
            self.generate(fps=0)
        with self.assertRaises(ValueError):
            self.generate(fps=float("nan"))
        with self.assertRaises(ValueError):
            self.generate(fps=True)

    def test_frame_origin_zero_explicitly_supported(self):
        r = self.generate(frame_origin=0)
        self.assertEqual(r["shots"][1]["replay"]["release_seconds"], round(396/30, 3))

    def test_bad_frame_origin_rejected(self):
        with self.assertRaises(ValueError):
            self.generate(frame_origin=2)

    def test_mismatched_shots_rejected(self):
        self.measured["shots"].pop()
        self.measured["shot_count"] -= 1
        with self.assertRaises(ValueError):
            self.generate()

    def test_duplicate_shots_rejected(self):
        self.analysis["shots"][1]["shot_number"] = 1
        with self.assertRaises(ValueError):
            self.generate()

    def test_unsafe_shot_verdict_rejected(self):
        self.measured["shots"][1]["intended_form_match"] = "passed"
        with self.assertRaises(ValueError):
            self.generate()

    def test_unsafe_technique_verdict_rejected(self):
        self.measured["intended_form_match"] = "perfect"
        with self.assertRaises(ValueError):
            self.generate()

    def test_unsafe_correction_rejected(self):
        self.measured["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            self.generate()

    def test_unsafe_endorsement_rejected(self):
        self.measured["official_endorsement"] = True
        with self.assertRaises(ValueError):
            self.generate()

    def test_candidate_needs_qualified_source(self):
        self.measured["shots"][1]["review_candidates"][0]["source_support_status"] = "provisional_only"
        with self.assertRaises(ValueError):
            self.generate()

    def test_candidate_needs_ready_timing(self):
        self.measured["shots"][1]["timing_quality"] = "insufficient_for_reliable_timing"
        with self.assertRaises(ValueError):
            self.generate()

    def test_missing_measurement_stays_hidden(self):
        candidate = self.measured["shots"][1]["review_candidates"][0]
        candidate["status"] = "no_supported_measurement"
        candidate["observations"] = []
        result = self.generate()
        self.assertEqual(result["shots"][1]["feedback_count"], 5)

    def test_duplicate_principle_rejected(self):
        shots = self.measured["shots"][1]["review_candidates"]
        shots[1]["principle_id"] = shots[0]["principle_id"]
        with self.assertRaises(ValueError):
            self.generate()

    def test_bad_endpoints_rejected(self):
        obs = self.measured["shots"][1]["review_candidates"][4]["observations"][0]
        obs["delta_y_pixels"] = 100
        with self.assertRaises(ValueError):
            self.generate()

    def test_bad_angle_units_rejected(self):
        obs = self.measured["shots"][1]["review_candidates"][0]["observations"][0]
        obs["unit"] = "radians"
        with self.assertRaises(ValueError):
            self.generate()

    def test_bad_angle_diff_rejected(self):
        obs = self.measured["shots"][1]["review_candidates"][0]["observations"][0]
        obs["observed_endpoint_difference"] = -21.975
        with self.assertRaises(ValueError):
            self.generate()

    def test_measurement_outside_shot_rejected(self):
        obs = self.measured["shots"][1]["review_candidates"][4]["observations"][0]
        obs["end_frame"] = 397
        with self.assertRaises(ValueError):
            self.generate()

    def test_nonfinite_measurements_rejected(self):
        obs = self.measured["shots"][1]["review_candidates"][0]["observations"][0]
        obs["start_value"] = math.inf
        with self.assertRaises(ValueError):
            self.generate()

    def test_bad_video_url_rejected(self):
        self.analysis["video_url"] = "https://other.example/video"
        with self.assertRaises(ValueError):
            self.generate()

    def test_source_links_must_be_https(self):
        self.measured["shots"][1]["review_candidates"][0]["source_urls"] = ["javascript:alert(1)"]
        with self.assertRaises(ValueError):
            self.generate()

    def test_unrecognized_measurement_is_not_displayed(self):
        candidate = self.measured["shots"][1]["review_candidates"][0]
        candidate["observations"] = [{"signal": "grip_tension", "unit": "invented"}]
        result = self.generate()
        self.assertEqual(result["shots"][1]["feedback_count"], 5)

    def test_extra_observations_do_not_create_false_release_detection(self):
        text = self.generate()["shots"][1]["feedback_cards"][4]["observation_text"]
        self.assertNotIn("release", text.lower())
        self.assertEqual(self.generate()["shots"][1]["feedback_cards"][4]["replay"]["end_frame"], 393)


if __name__ == "__main__":
    unittest.main()
