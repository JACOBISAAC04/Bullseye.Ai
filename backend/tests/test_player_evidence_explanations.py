"""Phase 6.3 tests — run with unittest discover from Bullseye/backend."""
import copy
import unittest

from app.coaching.player_feedback import PlayerFeedbackBuilder
from app.coaching.player_evidence_explanations import PlayerEvidenceExplainer
from test_player_feedback import _inputs


def make_report(*, fps=23.976, origin=1):
    analysis, measured = _inputs()
    return PlayerFeedbackBuilder().generate(
        analysis, measured, fps=fps, frame_origin=origin
    )


class PlayerEvidenceExplanationTests(unittest.TestCase):
    def setUp(self):
        self.source = make_report()
        self.builder = PlayerEvidenceExplainer()

    def enrich(self, source=None):
        return self.builder.enrich(self.source if source is None else source)

    def test_two_shots_and_six_cards(self):
        result = self.enrich()
        self.assertEqual(result["shot_count"], 2)
        self.assertEqual([s["feedback_count"] for s in result["shots"]], [0, 6])

    def test_original_input_not_modified(self):
        original = copy.deepcopy(self.source)
        self.enrich()
        self.assertEqual(self.source, original)

    def test_existing_keys_retained(self):
        result = self.enrich()
        self.assertTrue(set(self.source).issubset(result))
        for old, new in zip(self.source["shots"], result["shots"]):
            self.assertTrue(set(old).issubset(new))
            for a, b in zip(old["feedback_cards"], new["feedback_cards"]):
                self.assertTrue(set(a).issubset(b))
                self.assertEqual(a["observation_text"], b["observation_text"])
                self.assertEqual(a["review_question"], b["review_question"])
                self.assertEqual(a["source_urls"], b["source_urls"])

    def test_shot1_explains_insufficient_timing(self):
        x = self.enrich()["shots"][0]["evidence_explanation"]
        self.assertEqual(x["status"], "insufficient_timing_evidence")
        self.assertIn("timing", x["reason"].lower())
        self.assertIn("does not mean", x["reason"])

    def test_shot2_explains_descriptive_evidence(self):
        x = self.enrich()["shots"][1]["evidence_explanation"]
        self.assertEqual(x["status"], "descriptive_reviews_available")
        self.assertIn("6", x["headline"])

    def test_no_cards_with_sufficient_quality_neutral_reason(self):
        report = make_report()
        shot = report["shots"][1]
        shot["feedback_cards"] = []
        shot["feedback_count"] = 0
        x = self.enrich(report)["shots"][1]["evidence_explanation"]
        self.assertEqual(x["status"], "no_eligible_measured_reviews")
        self.assertIn("does not identify", x["reason"])

    def test_no_form_grade_or_corrections(self):
        r = self.enrich()
        self.assertEqual(r["intended_form_match"], "not_assessed")
        self.assertEqual(r["corrections_generated"], 0)
        self.assertIs(r["official_endorsement"], False)
        for card in r["shots"][1]["feedback_cards"]:
            self.assertIs(card["evidence_explanation"]["can_grade_form"], False)
            self.assertIs(card["evidence_explanation"]["can_recommend_correction"], False)

    def test_no_numeric_confidence_fabricated(self):
        r = self.enrich()
        self.assertEqual(r["evidence_confidence_scoring"], "not_available")
        for card in r["shots"][1]["feedback_cards"]:
            self.assertEqual(card["evidence_explanation"]["confidence"]["kind"], "not_scored")
            self.assertIn("No calibrated", card["evidence_explanation"]["confidence"]["label"])

    def test_frame_and_replay_retained(self):
        r = self.enrich()
        old = self.source["shots"][1]["feedback_cards"][4]["replay"]
        new = r["shots"][1]["feedback_cards"][4]
        self.assertEqual(new["replay"], old)
        self.assertEqual(new["evidence_explanation"]["frame_range"],
                         {"start_frame": 357, "end_frame": 393})

    def test_source_context_is_qualified_not_verified(self):
        x = self.enrich()["shots"][1]["feedback_cards"][0]["evidence_explanation"]
        self.assertIn("not an endorsement", x["source_context"])
        self.assertIn("not", x["source_context"])

    def test_source_urls_stay_same(self):
        old = self.source["shots"][1]["feedback_cards"][0]["source_urls"]
        actual = self.enrich()["shots"][1]["feedback_cards"][0]["source_urls"]
        self.assertEqual(old, actual)

    def test_posture_display_title_not_torso_diagnosis(self):
        x = self.enrich()["shots"][1]["feedback_cards"][0]
        self.assertEqual(x["display_title"], "Hip-angle observation (posture context)")
        self.assertIn("does not establish torso posture", x["player_review_question"])

    def test_balance_title_not_physical_balance(self):
        x = self.enrich()["shots"][1]["feedback_cards"][2]
        self.assertEqual(x["display_title"], "Hip-center movement in the image")
        self.assertIn("cannot establish physical balance", x["player_review_question"])

    def test_player_specific_diagnosis_renamed(self):
        x = self.enrich()["shots"][1]["feedback_cards"][5]
        self.assertEqual(x["display_title"], "Joint-angle observations")

    def test_ball_wrist_contact_not_claimed(self):
        x = self.enrich()["shots"][1]["feedback_cards"][3]
        self.assertIn("does not prove hand contact", x["player_review_question"])

    def test_ball_path_not_claimed_continuous(self):
        x = self.enrich()["shots"][1]["feedback_cards"][4]
        self.assertIn("Endpoints do not establish", x["player_review_question"])

    def test_old_title_and_question_preserved(self):
        a = self.source["shots"][1]["feedback_cards"][5]
        b = self.enrich()["shots"][1]["feedback_cards"][5]
        self.assertEqual(a["title"], b["title"])
        self.assertEqual(a["review_question"], b["review_question"])

    def test_reject_unsafe_form_verdict(self):
        self.source["intended_form_match"] = "perfect"
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_unsafe_shot_verdict(self):
        self.source["shots"][1]["intended_form_match"] = "correct"
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_auto_corrections(self):
        self.source["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_endorsement(self):
        self.source["official_endorsement"] = True
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_feedback_card_bypass_timing(self):
        self.source["shots"][1]["timing_quality"] = "insufficient_for_reliable_timing"
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_unknown_timing_quality(self):
        self.source["shots"][0]["timing_quality"] = "perfect"
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_unsafe_evidence_status(self):
        self.source["shots"][1]["feedback_cards"][0]["evidence_status"] = "perfect"
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_bad_card_sample_count(self):
        self.source["shots"][1]["feedback_cards"][0]["measurement_count"] = -1
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_missing_card_limitations(self):
        self.source["shots"][1]["feedback_cards"][0]["limitations"] = []
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_duplicate_cards(self):
        cards = self.source["shots"][1]["feedback_cards"]
        cards.append(copy.deepcopy(cards[0]))
        self.source["shots"][1]["feedback_count"] += 1
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_duplicate_shots(self):
        self.source["shots"][1]["shot_number"] = 1
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_mismatched_shot_count(self):
        self.source["shot_count"] = 3
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_mismatched_feedback_count(self):
        self.source["shots"][1]["feedback_count"] = 100
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_replay_outside_shot(self):
        c = self.source["shots"][1]["feedback_cards"][0]
        c["replay"]["start_frame"] = 1
        c["replay"]["start_seconds"] = 0
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_replay_timestamp_mismatch(self):
        self.source["shots"][1]["feedback_cards"][0]["replay"]["end_seconds"] = 777
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_non_https_source(self):
        self.source["shots"][1]["feedback_cards"][0]["source_urls"] = ["http://example.org"]
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_re_enrichment(self):
        with self.assertRaises(ValueError):
            self.enrich(self.enrich())

    def test_zero_shots(self):
        x = make_report()
        x["shot_count"] = 0
        x["shots"] = []
        r = self.enrich(x)
        self.assertEqual(r["shots"], [])
        self.assertEqual(r["evidence_confidence_scoring"], "not_available")

    def test_zero_based_frames(self):
        analysis, measured = _inputs()
        zero = PlayerFeedbackBuilder().generate(
            analysis, measured, fps=23.976, frame_origin=0
        )
        r = self.enrich(zero)
        self.assertEqual(r["frame_origin"], 0)

    def test_unknown_principle_retains_existing_title(self):
        card = self.source["shots"][1]["feedback_cards"][0]
        card["principle_id"] = "other_measured_topic"
        x = self.enrich()["shots"][1]["feedback_cards"][0]
        self.assertEqual(x["display_title"], card["title"])

    def test_reject_non_object_input(self):
        with self.assertRaises(ValueError):
            self.enrich([])

    def test_reject_missing_fps(self):
        del self.source["fps"]
        with self.assertRaises(ValueError):
            self.enrich()

    def test_reject_second_enrichment_fields_in_card(self):
        self.source["shots"][1]["feedback_cards"][0]["display_title"] = "claim"
        with self.assertRaises(ValueError):
            self.enrich()


if __name__ == "__main__":
    unittest.main()
