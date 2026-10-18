"""End-to-end tests of Phase 5 evidence-aware measured coaching orchestration."""
from __future__ import annotations

from copy import deepcopy
import unittest

from app.coaching.ball_observation_store import build_ball_observation_report
from app.coaching.measurement_coaching_pipeline import MeasurementCoachingPipeline


def fixture(*, release_detected=False, pose_gap=False):
    total = 24
    shot = {
        "shot_number": 1, "arm": "right",
        "start_frame": 1, "release_frame": 18, "end_frame": total
    }
    frames = []
    for n in range(1, total + 1):
        observed = not (pose_gap and 4 <= n <= 8)
        row = {"frame": n, "time": n / 24.0, "pose_detected": observed}
        if observed:
            for side in ("left", "right"):
                offset = 15 if side == "left" else 0
                for idx, joint in enumerate(
                    ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
                ):
                    row[f"{side}_{joint}"] = {
                        "x": 100.0 + offset + idx * 3 + n * 0.2,
                        "y": 90.0 + idx * 20 - n * 0.5,
                        "z": 0.0
                    }
                row[f"{side}_elbow_angle"] = 110.0
                row[f"{side}_knee_angle"] = 160.0
        frames.append(row)

    payload = {
        "analysis": {"frames": frames},
        "metrics": {"shots": [{
            **shot,
            "set": {"elbow_angle": 150.0, "hip_angle": 140.0, "knee_angle": 155.0},
            "release": {"elbow_angle": 100.0, "hip_angle": 175.0, "knee_angle": 160.0},
            "follow_through": {"elbow_angle": 165.0},
        }]},
        "shot_count": 1,
        "shots": [{**shot, "trajectory_points": 20}],
    }

    raw = []
    for n in range(1, total + 1):
        detected = n % 5 != 0 and (release_detected or n != 18)
        ball = (
            {
                "x": 125.0 + n,
                "y": 240.0 - n * 4,
                "x1": 120.0 + n,
                "y1": 235.0 - n * 4,
                "x2": 130.0 + n,
                "y2": 245.0 - n * 4,
                "confidence": 0.85,
            }
            if detected else None
        )
        raw.append({
            "frame": n, "time": n / 24.0,
            "ball_detected": detected, "ball": ball
        })

    observations = build_ball_observation_report(
        raw, deepcopy(raw), [shot], fps=24.0
    )
    return payload, observations


class MeasurementCoachingPipelineTests(unittest.TestCase):

    def setUp(self):
        self.pipeline = MeasurementCoachingPipeline()
        self.api, self.observations = fixture()

    def test_end_to_end_reports_and_reference_count(self):
        reports = self.pipeline.analyze(self.api, self.observations)
        self.assertEqual(
            list(reports),
            ["temporal", "mechanics", "technique", "sources", "review", "measured"]
        )
        self.assertTrue(all(report["shot_count"] == 1 for report in reports.values()))
        self.assertEqual(reports["technique"]["reference_count"], 22)
        self.assertEqual(reports["sources"]["reference_count"], 22)

    def test_six_measured_review_questions_on_full_synthetic_pose(self):
        measured = self.pipeline.analyze(self.api, self.observations)["measured"]
        shot = measured["shots"][0]
        self.assertEqual(shot["review_candidate_count"], 6)
        self.assertEqual(len(shot["review_candidates"]), 6)
        self.assertEqual(measured["corrections_generated"], 0)

    def test_missing_release_detector_remains_missing(self):
        reports = self.pipeline.analyze(self.api, self.observations)
        self.assertFalse(
            reports["mechanics"]["shots"][0]["signals"]["ball_position"]["available"]
        )
        self.assertNotIn(
            "ball_position",
            reports["technique"]["shots"][0]["available_mechanics_signals"]
        )

    def test_direct_release_detector_is_preserved(self):
        api, ball = fixture(release_detected=True)
        reports = self.pipeline.analyze(api, ball)
        sample = reports["mechanics"]["shots"][0]["signals"]["ball_position"]["sample"]
        self.assertEqual(sample["frame"], 18)
        self.assertIn(
            "ball_position",
            reports["technique"]["shots"][0]["available_mechanics_signals"]
        )

    def test_all_safety_flags_preserved(self):
        reports = self.pipeline.analyze(self.api, self.observations)
        for report in reports.values():
            self.assertEqual(report["intended_form_match"], "not_assessed")
            self.assertIs(report["official_endorsement"], False)
        self.assertEqual(reports["review"]["corrections_generated"], 0)
        self.assertEqual(reports["measured"]["corrections_generated"], 0)

    def test_temporal_gap_blocks_review(self):
        api, ball = fixture(pose_gap=True)
        reports = self.pipeline.analyze(api, ball)
        self.assertEqual(
            reports["temporal"]["shots"][0]["segments"]["set_to_release"]["quality"],
            "insufficient_for_reliable_timing",
        )
        self.assertEqual(reports["measured"]["shots"][0]["review_candidate_count"], 0)

    def test_mismatched_ball_and_api_boundaries_rejected(self):
        broken = deepcopy(self.observations)
        broken["shots"][0]["release_frame"] = 17
        with self.assertRaises(ValueError):
            self.pipeline.analyze(self.api, broken)

    def test_mismatched_api_count_rejected(self):
        api = deepcopy(self.api)
        api["shot_count"] = 2
        with self.assertRaises(ValueError):
            self.pipeline.analyze(api, self.observations)

    def test_reject_nondict_payloads(self):
        with self.assertRaises(ValueError):
            self.pipeline.analyze([], self.observations)
        with self.assertRaises(ValueError):
            self.pipeline.analyze(self.api, [])

    def test_empty_shots_stay_empty(self):
        api = {
            "analysis": {"frames": [
                {"frame": i, "pose_detected": False} for i in range(1, 5)
            ]},
            "metrics": {"shots": []},
            "shots": [],
            "shot_count": 0,
        }
        no_ball = [
            {"frame": i, "ball_detected": False, "ball": None}
            for i in range(1, 5)
        ]
        observation = build_ball_observation_report(
            no_ball, deepcopy(no_ball), [], fps=24.0
        )
        result = self.pipeline.analyze(api, observation)
        for item in result.values():
            self.assertEqual(item["shot_count"], 0)
        self.assertEqual(result["measured"]["corrections_generated"], 0)

    def test_provisional_sources_stay_provisional(self):
        result = self.pipeline.analyze(self.api, self.observations)
        references = result["sources"]["references"]
        provisional = [
            ref for ref in references
            if ref["source_support_status"] == "provisional_only"
        ]
        self.assertTrue(provisional)
        self.assertTrue(all(
            not ref["original_video_timestamp_verified"]
            for ref in provisional
        ))

    def test_inputs_unchanged(self):
        before_api = deepcopy(self.api)
        before_ball = deepcopy(self.observations)
        self.pipeline.analyze(self.api, self.observations)
        self.assertEqual(self.api, before_api)
        self.assertEqual(self.observations, before_ball)


if __name__ == "__main__":
    unittest.main()
