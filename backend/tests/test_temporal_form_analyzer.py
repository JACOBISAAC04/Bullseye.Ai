"""Regression tests for Bullseye Phase 4.4 temporal measurements."""

import copy
import unittest

from app.coaching.temporal_form_analyzer import TemporalFormAnalyzer


def frame(number, detected=True):
    item = {
        "frame": number,
        "time": (number - 1) / 24,
        "pose_detected": detected,
    }
    if detected:
        shift = number * 0.4
        joints = {
            "shoulder": (100.0, 50.0),
            "elbow": (110.0, 75.0 + shift),
            "wrist": (105.0, 100.0 - shift),
            "hip": (105.0, 110.0),
            "knee": (110.0, 150.0 + shift),
            "ankle": (105.0, 190.0),
        }
        for name, (x, y) in joints.items():
            item[f"right_{name}"] = {"x": x, "y": y, "z": 0.0}
    return item


def payload(frames, start=1, release=8, end=10):
    return {
        "analysis": {"frames": frames},
        "metrics": {"shots": [{
            "shot_number": 1, "arm": "right",
            "start_frame": start, "release_frame": release, "end_frame": end,
        }]},
    }


class TemporalFormTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TemporalFormAnalyzer()

    def test_complete_sequence_descriptive_only(self):
        report = self.analyzer.analyze(payload([frame(i) for i in range(1, 11)]))
        segment = report["shots"][0]["segments"]["set_to_release"]
        self.assertEqual(report["intended_form_match"], "not_assessed")
        self.assertEqual(report["shots"][0]["sequencing_judgment"], "not_assessed")
        self.assertEqual(segment["quality"], "sufficient_for_descriptive_timing")
        self.assertEqual(segment["quality_metrics"]["complete_pose_frames"], 8)
        self.assertEqual(segment["quality_metrics"]["consecutive_complete_pose_pairs"], 7)
        self.assertEqual(segment["quality_metrics"]["longest_missing_run"], 0)

    def test_missing_pose_is_not_interpolated_or_connected(self):
        frames = [frame(i, detected=(i != 4)) for i in range(1, 11)]
        report = self.analyzer.analyze(payload(frames))
        seg = report["shots"][0]["segments"]["set_to_release"]
        self.assertFalse(seg["samples"][3]["complete_pose"])
        self.assertIsNone(seg["samples"][3]["angles_deg"]["elbow_angle"])
        self.assertEqual(seg["quality_metrics"]["consecutive_complete_pose_pairs"], 5)
        self.assertEqual(seg["quality_metrics"]["longest_missing_run"], 1)

    def test_missing_frame_numbers_are_counted_as_missing(self):
        frames = [frame(i) for i in range(1, 11) if i not in (3, 4, 5, 6)]
        report = self.analyzer.analyze(payload(frames))
        seg = report["shots"][0]["segments"]["set_to_release"]
        self.assertEqual(seg["quality"], "insufficient_for_reliable_timing")
        self.assertEqual(seg["quality_metrics"]["longest_missing_run"], 4)
        self.assertIsNone(seg["samples"][2]["angles_deg"]["knee_angle"])

    def test_missing_joint_is_not_inferred(self):
        frames = [frame(i) for i in range(1, 11)]
        frames[4].pop("right_wrist")
        report = self.analyzer.analyze(payload(frames))
        sample = report["shots"][0]["segments"]["set_to_release"]["samples"][4]
        self.assertIsNone(sample["angles_deg"]["elbow_angle"])
        self.assertIsNone(sample["wrist_above_shoulder_px"])
        self.assertIsNotNone(sample["angles_deg"]["knee_angle"])
        self.assertFalse(sample["complete_pose"])

    def test_rejects_invalid_shot_order(self):
        bad = payload([frame(i) for i in range(1, 11)], start=9, release=4)
        with self.assertRaises(ValueError):
            self.analyzer.analyze(bad)

    def test_rejects_duplicate_pose_frame(self):
        frames = [frame(i) for i in range(1, 11)]
        frames.append(copy.deepcopy(frames[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.analyzer.analyze(payload(frames))

    def test_rejects_metrics_only_without_pose_frames(self):
        with self.assertRaisesRegex(ValueError, "analysis.frames"):
            self.analyzer.analyze({"shots": []})


if __name__ == "__main__":
    unittest.main()
