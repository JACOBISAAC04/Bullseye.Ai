"""Tests for Phase 5.4 ball detector evidence and provenance."""
import copy
import json
import math
import unittest

from app.coaching.ball_observation_store import build_ball_observation_report, save_report


def ball(x=30.0, y=40.0, confidence=0.8):
    return {
        "x": x, "y": y,
        "x1": x - 5, "y1": y - 5,
        "x2": x + 5, "y2": y + 5,
        "confidence": confidence,
    }


def frame(index, item=None):
    return {
        "frame": index, "time": (index - 1) / 30,
        "ball_detected": item is not None, "ball": item,
    }


def fixture():
    raw = [frame(1, ball()), frame(2), frame(3, ball(40, 50)),
           frame(4, ball(300, 300)), frame(5, ball(50, 60))]
    filtered = copy.deepcopy(raw)
    filtered[3]["ball_detected"] = False
    filtered[3]["ball"] = None
    shots = [{
        "shot_number": 1, "arm": "right",
        "start_frame": 1, "release_frame": 3, "end_frame": 5,
    }]
    return raw, filtered, shots


def report_from(raw=None, filtered=None, shots=None, fps=30.0):
    r, f, s = fixture()
    return build_ball_observation_report(
        r if raw is None else raw, f if filtered is None else filtered,
        s if shots is None else shots, fps=fps
    )


class BallObservationStoreTests(unittest.TestCase):
    def test_retains_detector_provenance(self):
        report = report_from()
        self.assertTrue(report["ball_detection_source_present"])
        self.assertEqual(report["ball_positions"][0]["source"], "BallAnalyzer")
        self.assertEqual(report["ball_positions"][0]["provenance"], "direct_detector_output")

    def test_only_accepted_frames_are_observed(self):
        result = report_from()
        self.assertEqual([p["frame"] for p in result["ball_positions"]], [1, 3, 5])
        self.assertEqual(result["rejected_outlier_frames"], [4])
        self.assertEqual(result["detector_observation_count"], 4)
        self.assertEqual(result["accepted_observation_count"], 3)
        self.assertEqual(result["not_detected_frame_count"], 1)

    def test_release_requires_exact_direct_detection(self):
        result = report_from()
        release = result["shots"][0]["release_ball_position"]
        self.assertEqual(release["frame"], 3)
        self.assertTrue(result["shots"][0]["release_ball_observed"])

    def test_release_missing_stays_missing(self):
        raw, filtered, shots = fixture()
        shots[0]["release_frame"] = 2
        result = report_from(raw, filtered, shots)
        self.assertIsNone(result["shots"][0]["release_ball_position"])
        self.assertFalse(result["shots"][0]["ball_position_signal_available"])

    def test_outlier_at_release_stays_missing(self):
        raw, filtered, shots = fixture()
        shots[0]["release_frame"] = 4
        result = report_from(raw, filtered, shots)
        self.assertIsNone(result["shots"][0]["release_ball_position"])

    def test_interpolated_filtered_input_rejected(self):
        raw, filtered, shots = fixture()
        filtered[1] = frame(2, {**ball(), "confidence": 0.0})
        filtered[1]["interpolated"] = True
        with self.assertRaisesRegex(ValueError, "interpolation"):
            report_from(raw, filtered, shots)

    def test_interpolated_raw_input_rejected(self):
        raw, filtered, shots = fixture()
        raw[1]["interpolated"] = True
        with self.assertRaisesRegex(ValueError, "interpolation"):
            report_from(raw, filtered, shots)

    def test_fabricated_filtered_detection_rejected(self):
        raw, filtered, shots = fixture()
        filtered[1] = frame(2, ball())
        with self.assertRaisesRegex(ValueError, "fabricated"):
            report_from(raw, filtered, shots)

    def test_changed_filtered_coordinates_rejected(self):
        raw, filtered, shots = fixture()
        filtered[0]["ball"] = ball(35, 40)
        with self.assertRaisesRegex(ValueError, "modified"):
            report_from(raw, filtered, shots)

    def test_changed_filtered_confidence_rejected(self):
        raw, filtered, shots = fixture()
        filtered[0]["ball"]["confidence"] = 0.45
        with self.assertRaisesRegex(ValueError, "modified"):
            report_from(raw, filtered, shots)

    def test_invalid_confidence_rejected(self):
        raw, filtered, shots = fixture()
        raw[0]["ball"]["confidence"] = 0.0
        with self.assertRaisesRegex(ValueError, "confidence"):
            report_from(raw, filtered, shots)

    def test_nan_and_infinite_coordinates_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            raw, filtered, shots = fixture()
            raw[0]["ball"]["x"] = value
            with self.assertRaises(ValueError):
                report_from(raw, filtered, shots)

    def test_bad_bounding_box_rejected(self):
        raw, filtered, shots = fixture()
        raw[0]["ball"]["x2"] = raw[0]["ball"]["x1"]
        with self.assertRaises(ValueError):
            report_from(raw, filtered, shots)

    def test_missing_ball_object_rejected(self):
        raw, filtered, shots = fixture()
        raw[0]["ball"] = None
        with self.assertRaises(ValueError):
            report_from(raw, filtered, shots)

    def test_frame_number_gap_rejected(self):
        raw, filtered, shots = fixture()
        raw[1]["frame"] = 9
        with self.assertRaisesRegex(ValueError, "consecutive"):
            report_from(raw, filtered, shots)

    def test_frame_count_mismatch_rejected(self):
        raw, filtered, shots = fixture()
        with self.assertRaisesRegex(ValueError, "frame counts"):
            report_from(raw, filtered[:-1], shots)

    def test_invalid_shot_order_rejected(self):
        raw, filtered, shots = fixture()
        shots[0]["release_frame"] = 7
        with self.assertRaisesRegex(ValueError, "invalid frame bounds"):
            report_from(raw, filtered, shots)

    def test_overlapping_shots_rejected(self):
        raw, filtered, shots = fixture()
        shots.append({
            "shot_number": 2, "start_frame": 3,
            "release_frame": 4, "end_frame": 5, "arm": "left"
        })
        with self.assertRaisesRegex(ValueError, "overlapping"):
            report_from(raw, filtered, shots)

    def test_empty_shots_works_without_claims(self):
        result = report_from(shots=[])
        self.assertEqual(result["shot_count"], 0)
        self.assertEqual(result["intended_form_match"], "not_assessed")

    def test_missing_ball_frames_never_interpolated(self):
        result = report_from()
        self.assertNotIn(2, [b["frame"] for b in result["ball_positions"]])
        self.assertTrue(all(b["interpolated"] is False for b in result["ball_positions"]))

    def test_candidate_ball_positions_not_form_grade(self):
        result = report_from()
        self.assertTrue(result["shots"][0]["ball_positions_signal_available"])
        self.assertEqual(result["shots"][0]["assessment"], "detector_observations_only")
        self.assertEqual(result["intended_form_match"], "not_assessed")

    def test_inputs_are_not_modified(self):
        raw, filtered, shots = fixture()
        originals = copy.deepcopy((raw, filtered, shots))
        report_from(raw, filtered, shots)
        self.assertEqual((raw, filtered, shots), originals)

    def test_serializable(self):
        self.assertEqual(
            json.loads(json.dumps(report_from(), allow_nan=False))["shot_count"], 1
        )

    def test_invalid_fps_rejected(self):
        with self.assertRaises(ValueError):
            report_from(fps=0)

    def test_raw_false_with_ball_rejected(self):
        raw, filtered, shots = fixture()
        raw[1]["ball"] = ball()
        with self.assertRaisesRegex(ValueError, "without detection"):
            report_from(raw, filtered, shots)


if __name__ == "__main__":
    unittest.main()
