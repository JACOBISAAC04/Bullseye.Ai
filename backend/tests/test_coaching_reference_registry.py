"""Phase 5.6 source registry safety tests. Run from Bullseye/backend."""
import copy
import json
import unittest
from pathlib import Path

from app.coaching.coaching_reference_registry import CoachingReferenceRegistry


HERE = Path(__file__).resolve().parents[1] / "app" / "coaching"
CATALOG = json.loads((HERE / "shooting_coaching_references.json").read_text(encoding="utf-8"))
EVIDENCE = json.loads((HERE / "coaching_source_evidence.json").read_text(encoding="utf-8"))


def fixture():
    return CoachingReferenceRegistry(catalog=copy.deepcopy(CATALOG),
                                     evidence=copy.deepcopy(EVIDENCE))


def integrated():
    return {
        "shot_count": 1,
        "reference_count": len(CATALOG["principles"]),
        "intended_form_match": "not_assessed",
        "shots": [{
            "shot_number": 2,
            "assessments": [{
                "principle_id": p["id"],
                "title": p["title"],
                "status": (
                    "source_verification_required" if p["id"] == "posture_setup"
                    else "insufficient_timing_evidence" if p["id"] == "ground_balance"
                    else "missing_measurements"
                ),
                "missing_signals": [],
                "intended_form_match": "not_assessed",
            } for p in CATALOG["principles"]],
        }],
    }


class CoachingReferenceRegistryTests(unittest.TestCase):
    def test_loads_exact_project_catalog(self):
        r = fixture().build_report()
        self.assertEqual(r["reference_count"], 22)
        self.assertEqual(r["shot_count"], 0)
        self.assertFalse(r["official_endorsement"])
        self.assertEqual(r["intended_form_match"], "not_assessed")

    def test_evidence_has_distinct_ids_and_valid_reference_links(self):
        reg = fixture()
        self.assertEqual(len(reg.evidence["evidence_records"]), 11)
        self.assertEqual(len({x["id"] for x in reg.evidence["evidence_records"]}), 11)
        self.assertTrue(all(reg.support.get("ball_setup") for _ in range(1)))

    def test_original_masterclass_citation_is_never_promoted(self):
        r = fixture().build_report()
        for entry in r["references"]:
            if entry["original_citation"]["source_id"] == "youtube_masterclass":
                self.assertEqual(entry["original_citation_status"], "unverified_specific_claim")
                self.assertFalse(entry["original_video_timestamp_verified"])

    def test_interview_support_is_narrow_and_scoped(self):
        r = fixture().build_report()
        p = next(x for x in r["references"] if x["principle_id"] == "ball_path")
        self.assertEqual(p["source_support_status"], "limited_published_text_support")
        self.assertIn("free-throw", p["supplemental_evidence"][0]["scope_note"].lower())
        self.assertEqual(p["original_citation_status"], "unverified_specific_claim")

    def test_official_course_outline_is_topic_only(self):
        p = next(x for x in fixture().build_report()["references"] if x["principle_id"] == "sequence")
        self.assertEqual(p["source_support_status"], "topic_only_support")
        self.assertEqual(p["supplemental_evidence"][0]["allowed_use"], "topic_only")

    def test_mirrored_post_stays_unverified(self):
        p = next(x for x in fixture().build_report()["references"] if x["principle_id"] == "avoid_forced_release_height")
        self.assertEqual(p["source_support_status"], "provisional_only")

    def test_masterclass_uncorroborated_detail_stays_provisional(self):
        p = next(x for x in fixture().build_report()["references"] if x["principle_id"] == "progressive_motion")
        self.assertEqual(p["source_support_status"], "provisional_only")

    def test_never_generates_correctness_verdict(self):
        r = fixture().build_report(integrated())
        self.assertEqual(r["intended_form_match"], "not_assessed")
        self.assertTrue(all(x["intended_form_match"] == "not_assessed" for x in r["references"]))
        self.assertTrue(all(x["can_auto_grade_correctness"] is False for x in r["references"]))
        self.assertTrue(all(a["intended_form_match"] == "not_assessed" for a in r["shots"][0]["assessments"]))

    def test_preserves_all_assessment_statuses_and_missing_signals(self):
        inp = integrated()
        before = copy.deepcopy(inp)
        result = fixture().build_report(inp)
        self.assertEqual(inp, before)
        self.assertEqual(
            [a["status"] for a in result["shots"][0]["assessments"]],
            [a["status"] for a in before["shots"][0]["assessments"]],
        )
        self.assertEqual(result["shots"][0]["assessments"][0]["missing_signals"], [])

    def test_matches_actual_principle_id(self):
        r = fixture().build_report(integrated())
        a = r["shots"][0]["assessments"][0]
        self.assertEqual(a["registry_reference_id"], "posture_setup")
        self.assertEqual(a["registry_match_strategy"], "principle_id")

    def test_resolves_legacy_title_only_without_guessing(self):
        data = integrated()
        for a in data["shots"][0]["assessments"]:
            a.pop("principle_id")
        r = fixture().build_report(data)
        self.assertTrue(all(a["registry_match_strategy"] == "unique_normalized_title" for a in r["shots"][0]["assessments"]))

    def test_rejects_ambiguous_titles(self):
        c = copy.deepcopy(CATALOG)
        c["principles"][1]["title"] = c["principles"][0]["title"]
        r = CoachingReferenceRegistry(catalog=c, evidence=copy.deepcopy(EVIDENCE))
        data = integrated()
        data["shots"][0]["assessments"][0].pop("principle_id")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            r.build_report(data)

    def test_rejects_id_title_conflict(self):
        data = integrated()
        data["shots"][0]["assessments"][0]["title"] = "Wrong title"
        with self.assertRaisesRegex(ValueError, "conflict"):
            fixture().build_report(data)

    def test_rejects_unknown_explicit_principle_id(self):
        data = integrated()
        data["shots"][0]["assessments"][0]["principle_id"] = "invented"
        with self.assertRaisesRegex(ValueError, "Unknown principle_id"):
            fixture().build_report(data)

    def test_rejects_duplicate_assessment_id(self):
        data = integrated()
        data["shots"][0]["assessments"][1] = copy.deepcopy(data["shots"][0]["assessments"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate assessment"):
            fixture().build_report(data)

    def test_rejects_unsupported_assessment_status(self):
        data = integrated()
        data["shots"][0]["assessments"][0]["status"] = "form_perfect"
        with self.assertRaisesRegex(ValueError, "Unknown assessment status"):
            fixture().build_report(data)

    def test_rejects_graded_input(self):
        data = integrated()
        data["intended_form_match"] = "perfect"
        with self.assertRaisesRegex(ValueError, "graded integrated report"):
            fixture().build_report(data)

    def test_rejects_unknown_evidence_source(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["source_id"] = "youtube_masterclass"
        with self.assertRaisesRegex(ValueError, "Unapproved source"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_forged_video_verification_label(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["source_kind"] = "verified_youtube_video"
        with self.assertRaisesRegex(ValueError, "Wrong source_kind"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_wrong_source_url(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["source_url"] = "https://wrong.example"
        with self.assertRaisesRegex(ValueError, "URL mismatch"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_unreviewed_record(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["reviewed_against_published_text"] = False
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_unlinked_evidence(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["reference_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "Unknown linked"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_unsafe_policy(self):
        e = copy.deepcopy(EVIDENCE)
        e["policy"]["promote_provisional_video_timestamps"] = True
        with self.assertRaisesRegex(ValueError, "Unsafe evidence policy"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_evidence_automatic_form_grading(self):
        e = copy.deepcopy(EVIDENCE)
        e["evidence_records"][0]["can_auto_grade_correctness"] = True
        with self.assertRaisesRegex(ValueError, "Unsafe grading"):
            CoachingReferenceRegistry(catalog=CATALOG, evidence=e)

    def test_rejects_mismatched_source_verification(self):
        c = copy.deepcopy(CATALOG)
        for s in c["sources"]:
            if s["id"] == "dunn_public_interview":
                s["verification"] = "video_verified"
        with self.assertRaisesRegex(ValueError, "verification metadata differs"):
            CoachingReferenceRegistry(catalog=c, evidence=EVIDENCE)

    def test_rejects_missing_original_citation(self):
        c = copy.deepcopy(CATALOG)
        c["principles"][0]["citation"]["source_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown cited source"):
            CoachingReferenceRegistry(catalog=c, evidence=EVIDENCE)

    def test_json_serializable(self):
        json.dumps(fixture().build_report(integrated()))

    def test_does_not_change_existing_library(self):
        c, e = copy.deepcopy(CATALOG), copy.deepcopy(EVIDENCE)
        registry = CoachingReferenceRegistry(catalog=c, evidence=e)
        registry.build_report(integrated())
        self.assertEqual(c, CATALOG)
        self.assertEqual(e, EVIDENCE)


if __name__ == "__main__":
    unittest.main()
