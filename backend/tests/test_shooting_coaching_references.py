import json
import unittest
from pathlib import Path


LIBRARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "app" / "coaching" / "shooting_coaching_references.json"
)


class ShootingCoachingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))

    def test_every_claim_has_identified_source(self):
        source_ids = {s["id"] for s in self.library["sources"]}
        for entry in self.library["principles"]:
            with self.subTest(entry=entry["id"]):
                self.assertIn(entry["citation"]["source_id"], source_ids)
                self.assertTrue(entry["citation"]["url"].startswith("https://"))

    def test_no_unverified_grading(self):
        self.assertFalse(self.library["use_policy"]["automatic_correctness_grading"])
        self.assertFalse(self.library["official_endorsement"])
        for entry in self.library["principles"]:
            with self.subTest(entry=entry["id"]):
                self.assertFalse(entry["can_auto_grade_correctness"])
                self.assertEqual(entry["safe_output"], "descriptive_or_not_assessed")

    def test_masterclass_timestamp_references_are_marked_provisional(self):
        for entry in self.library["principles"]:
            if entry["citation"]["source_id"] == "youtube_masterclass":
                self.assertEqual(
                    entry["citation"]["evidence_level"],
                    "provisional_third_party_video_summary",
                )
                self.assertIsInstance(entry["citation"]["timestamp_seconds"], int)

    def test_shorts_are_not_fabricated(self):
        shorts = self.library["shorts_review_queue"]
        self.assertEqual(shorts["status"], "awaits_individual_short_urls_and_notes")
        self.assertEqual(shorts["items"], [])
        source = next(
            s for s in self.library["sources"] if s["id"] == "youtube_shorts"
        )
        self.assertEqual(
            source["verification"],
            "collection_only_no_individual_shorts_verified",
        )

    def test_unique_identifiers(self):
        identifiers = [p["id"] for p in self.library["principles"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_no_numeric_shooting_target(self):
        # Numerical source timestamps are allowed, universal scoring targets are not.
        for item in self.library["principles"]:
            for forbidden_key in ("ideal_angle", "minimum_angle", "maximum_angle",
                                  "correctness_score", "target_degrees"):
                self.assertNotIn(forbidden_key, item)


if __name__ == "__main__":
    unittest.main()
