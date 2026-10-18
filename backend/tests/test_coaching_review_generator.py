"""Phase 5.7 tests for source-backed, conservative manual review generation."""
import copy
import json
import unittest

from app.coaching.coaching_reference_registry import CoachingReferenceRegistry
from app.coaching.coaching_review_generator import CoachingReviewGenerator


def valid_report():
    registry = CoachingReferenceRegistry()
    refs = registry.catalog["principles"]
    statuses = {
        "posture_setup": "source_verification_required",
        "hip_loading": "source_verification_required",
        "ground_balance": "source_verification_required",
        "ball_setup": "source_verification_required",
        "ball_path": "source_verification_required",
        "sequence": "source_verification_required",
        "avoid_forced_release_height": "source_verification_required",
        "individualized_feedback": "descriptive_review_possible",
        "guide_hand_compensation": "missing_measurements",
        "arm_body_connection": "shot_context_not_established",
        "vertical_motion": "insufficient_timing_evidence",
    }
    integrated = {
        "shot_count": 1,
        "reference_count": len(refs),
        "intended_form_match": "not_assessed",
        "shots": [{
            "shot_number": 2,
            "assessments": [
                {
                    "principle_id": p["id"],
                    "title": p["title"],
                    "status": statuses.get(p["id"], "missing_measurements"),
                    "intended_form_match": "not_assessed",
                }
                for p in refs
            ],
        }],
    }
    return registry.build_report(integrated)


def topic(output, principle_id):
    return next(
        t for t in output["shots"][0]["review_topics"]
        if t["principle_id"] == principle_id
    )


class CoachingReviewGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = CoachingReviewGenerator()
        self.input = valid_report()

    def test_accepts_registry_report_and_retains_22_topics(self):
        out = self.generator.generate(self.input)
        self.assertEqual(out["reference_count"], 22)
        self.assertEqual(out["shot_count"], 1)
        self.assertEqual(len(out["shots"][0]["review_topics"]), 22)

    def test_output_and_shot_are_never_graded(self):
        out = self.generator.generate(self.input)
        self.assertEqual(out["intended_form_match"], "not_assessed")
        self.assertEqual(out["corrections_generated"], 0)
        self.assertFalse(out["official_endorsement"])
        self.assertTrue(all(
            t["intended_form_match"] == "not_assessed"
            for t in out["shots"][0]["review_topics"]
        ))

    def test_published_interview_support_allows_only_manual_review(self):
        out = self.generator.generate(self.input)
        posture = topic(out, "posture_setup")
        self.assertEqual(posture["review_status"], "manual_review_candidate")
        self.assertTrue(posture["review_question"].endswith("?"))
        self.assertEqual(posture["measurement_status"], "source_verification_required")
        self.assertTrue(posture["published_context"])
        self.assertTrue(posture["source_urls"])

    def test_original_masterclass_timestamp_stays_unverified(self):
        out = self.generator.generate(self.input)
        posture = topic(out, "posture_setup")
        self.assertFalse(posture["original_video_timestamp_verified"])
        self.assertFalse(any(
            "youtube.com" in url for url in posture["source_urls"]
        ))

    def test_course_topic_support_does_not_enable_review(self):
        out = self.generator.generate(self.input)
        sequence = topic(out, "sequence")
        self.assertEqual(sequence["review_status"], "not_ready")
        self.assertIsNone(sequence["review_question"])
        self.assertFalse(sequence["published_context"])

    def test_provisional_claims_not_promoted(self):
        out = self.generator.generate(self.input)
        provisional = topic(out, "avoid_forced_release_height")
        self.assertEqual(provisional["review_status"], "not_ready")
        self.assertIsNone(provisional["review_question"])
        self.assertEqual(provisional["source_urls"], [])

    def test_missing_measurements_block_published_context(self):
        out = self.generator.generate(self.input)
        hand = topic(out, "guide_hand_compensation")
        self.assertEqual(hand["review_status"], "not_ready")
        self.assertFalse(hand["published_context"])

    def test_short_or_gappy_timing_blocks_even_published_context(self):
        report = valid_report()
        p = next(
            a for a in report["shots"][0]["assessments"]
            if a["registry_reference_id"] == "posture_setup"
        )
        p["status"] = "insufficient_timing_evidence"
        out = self.generator.generate(report)
        self.assertEqual(topic(out, "posture_setup")["review_status"], "not_ready")

    def test_context_missing_blocks_even_source_supported(self):
        report = valid_report()
        item = next(
            a for a in report["shots"][0]["assessments"]
            if a["registry_reference_id"] == "ball_setup"
        )
        item["status"] = "shot_context_not_established"
        out = self.generator.generate(report)
        self.assertEqual(topic(out, "ball_setup")["review_status"], "not_ready")

    def test_all_input_data_remains_unchanged(self):
        before = copy.deepcopy(self.input)
        self.generator.generate(self.input)
        self.assertEqual(self.input, before)

    def test_forged_source_label_rejected(self):
        data = copy.deepcopy(self.input)
        data["shots"][0]["assessments"][0][
            "registry_source_support_status"
        ] = "verified_youtube"
        with self.assertRaisesRegex(ValueError, "Source-support"):
            self.generator.generate(data)

    def test_forged_original_source_rejected(self):
        data = copy.deepcopy(self.input)
        data["references"][0]["original_citation"]["url"] = "https://fake.example"
        with self.assertRaisesRegex(ValueError, "Registry references differ"):
            self.generator.generate(data)

    def test_missing_or_duplicate_topics_rejected(self):
        data = copy.deepcopy(self.input)
        data["shots"][0]["assessments"][1] = copy.deepcopy(
            data["shots"][0]["assessments"][0]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.generator.generate(data)

    def test_grade_in_input_rejected(self):
        data = copy.deepcopy(self.input)
        data["shots"][0]["assessments"][0]["intended_form_match"] = "correct"
        with self.assertRaisesRegex(ValueError, "Unsafe assessment"):
            self.generator.generate(data)

    def test_empty_shots_supported(self):
        data = self.generator.registry.build_report()
        out = self.generator.generate(data)
        self.assertEqual(out["shot_count"], 0)
        self.assertEqual(out["shots"], [])

    def test_json_serializable(self):
        json.dumps(self.generator.generate(self.input), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
