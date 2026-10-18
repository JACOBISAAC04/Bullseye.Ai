"""Tests for evidence-gated, non-grading shooting technique review."""
import copy
import json
import unittest
from pathlib import Path

from app.coaching.technique_validator import TechniqueEvidenceValidator

CATALOG = Path(__file__).resolve().parents[1] / "app/coaching/shooting_coaching_references.json"


def _series(first=160.0, last=175.0, frames=40):
    return {
        "observed_frames": frames,
        "adjacent_observed_pairs": max(0, frames - 1),
        "first_observed": {"frame": 1, "value": first},
        "last_observed": {"frame": frames, "value": last},
        "observed_endpoint_delta": last - first,
    }


def _segment(quality="sufficient_for_descriptive_timing", frames=40):
    return {
        "quality": quality,
        "quality_metrics": {
            "total_frames": frames,
            "complete_pose_frames": frames,
            "longest_missing_run": 0,
            "consecutive_complete_pose_pairs": max(0, frames - 1),
        },
        "observed_angle_series": {
            "elbow_angle": _series(frames=frames),
            "hip_angle": _series(frames=frames),
            "knee_angle": _series(frames=frames),
        },
        "wrist_vertical_position_series": {
            "observed_frames": frames,
            "observed_endpoint_delta": 42.0,
        },
    }


def _report(quality="sufficient_for_descriptive_timing", frames=40):
    return {
        "shot_count": 1,
        "shots": [{
            "shot_number": 1,
            "arm": "right",
            "start_frame": 357,
            "release_frame": 396,
            "segments": {
                "set_to_release": _segment(quality, frames),
                "release_to_shot_end": _segment(frames=24),
            },
        }],
    }


class TechniqueValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    def evaluator(self, catalog=None):
        return TechniqueEvidenceValidator(catalog=copy.deepcopy(catalog or self.catalog))

    def assessment(self, report, pid):
        output = self.evaluator().analyze(report)
        return next(a for a in output["shots"][0]["assessments"] if a["principle_id"] == pid)

    def test_catalog_has_22_entries_and_no_auto_grade(self):
        out = self.evaluator().analyze(_report())
        self.assertEqual(out["reference_count"], 22)
        self.assertEqual(out["intended_form_match"], "not_assessed")
        self.assertFalse(out["official_endorsement"])
        self.assertTrue(all(
            a["intended_form_match"] == "not_assessed"
            for a in out["shots"][0]["assessments"]
        ))

    def test_provisional_masterclass_is_never_passed_off_as_verified(self):
        a = self.assessment(_report(), "hip_loading")
        self.assertEqual(a["status"], "source_verification_required")
        self.assertIsNone(a["review_question"])
        self.assertEqual(a["missing_signals"], [])

    def test_low_quality_blocks_timing_review(self):
        a = self.assessment(_report("insufficient_for_reliable_timing"), "hip_loading")
        self.assertEqual(a["status"], "insufficient_timing_evidence")

    def test_ball_position_is_not_inferred(self):
        a = self.assessment(_report(), "ball_path")
        self.assertEqual(a["status"], "missing_measurements")
        self.assertIn("ball_positions", a["missing_signals"])

    def test_guide_hand_remains_unassessable(self):
        a = self.assessment(_report(), "guide_hand_compensation")
        self.assertEqual(a["status"], "missing_measurements")

    def test_practice_context_is_not_assumed(self):
        a = self.assessment(_report(), "skills_progression")
        self.assertEqual(a["status"], "shot_context_not_established")

    def test_measurements_are_descriptive(self):
        out = self.evaluator().analyze(_report())
        sample = out["shots"][0]["measurements"]["set_to_release"]
        self.assertEqual(
            sample["angle_observations"]["hip_angle"]["observed_endpoint_delta_degrees"],
            15.0,
        )
        self.assertEqual(
            sample["interpretation"],
            "2D descriptive measurements, not a coaching grade",
        )

    def test_approved_source_and_signal_yield_descriptive_only(self):
        catalog = copy.deepcopy(self.catalog)
        p = next(x for x in catalog["principles"] if x["id"] == "hip_loading")
        p["citation"]["source_id"] = "dunn_public_interview"
        p["citation"]["evidence_level"] = "published_primary_interview"
        a = self.evaluator(catalog).analyze(_report())["shots"][0]["assessments"]
        item = next(x for x in a if x["principle_id"] == "hip_loading")
        self.assertEqual(item["status"], "descriptive_review_possible")
        self.assertEqual(item["intended_form_match"], "not_assessed")

    def test_rejects_unsafe_auto_grading(self):
        catalog = copy.deepcopy(self.catalog)
        catalog["principles"][0]["can_auto_grade_correctness"] = True
        with self.assertRaisesRegex(ValueError, "Unsafe auto-grading"):
            self.evaluator(catalog)

    def test_rejects_missing_source(self):
        catalog = copy.deepcopy(self.catalog)
        catalog["principles"][0]["citation"]["source_id"] = "fictional"
        with self.assertRaisesRegex(ValueError, "Missing cited source"):
            self.evaluator(catalog)

    def test_rejects_inconsistent_shot_count(self):
        data = _report()
        data["shot_count"] = 2
        with self.assertRaisesRegex(ValueError, "shot_count"):
            self.evaluator().analyze(data)

    def test_empty_shots_returns_not_assessed(self):
        out = self.evaluator().analyze({"shot_count": 0, "shots": []})
        self.assertEqual(out["shot_count"], 0)
        self.assertEqual(out["intended_form_match"], "not_assessed")

    def test_short_complete_pose_still_insufficient(self):
        a = self.assessment(_report("insufficient_for_reliable_timing", frames=5), "hip_loading")
        self.assertEqual(a["status"], "insufficient_timing_evidence")

    def test_nonlisted_signal_not_marked_available(self):
        a = self.assessment(_report(), "hand_under_ball")
        self.assertIn("ball_position", a["missing_signals"])
        self.assertIn("wrist_position", a["missing_signals"])

    def test_no_shorts_invented(self):
        self.assertEqual(self.catalog["shorts_review_queue"]["items"], [])


if __name__ == "__main__":
    unittest.main()
