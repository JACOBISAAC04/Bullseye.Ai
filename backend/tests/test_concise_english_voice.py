"""Tests for optional concise English speech narration (no Windows synthesis needed)."""
import copy
import unittest

from app.coaching.concise_english_voice import ConciseEnglishVoice

URL = "https://example.com/verified-interview"

HIP = (
    "Estimated hip angle changed from 152.6° (frame 357) to 174.6° "
    "(frame 396); an endpoint difference of +22.0° across 40 observed samples. "
    "Estimated knee angle changed from 160.4° (frame 357) to 162.1° "
    "(frame 396); an endpoint difference of +1.7° across 40 observed samples."
)
BALL = (
    "Accepted ball-detector endpoint moved 12.0 px right and 94.6 px up in "
    "the image from frame 357 to 393 (35 observed detector frames)."
)
WRIST = (
    "Detected ball center was 29.1 image px from the wrist at frame 357 "
    "and 11.9 image px at frame 393 (35 frames with both observations)."
)

def make_card(topic, text):
    return {
        "principle_id": topic,
        "title": topic,
        "observation_text": text,
        "review_question": "Review the observed mechanics.",
        "source_urls": [URL],
        "limitations": ["Camera projection limits accuracy."],
        "evidence_status": "observed_2d_descriptive",
        "measurement_count": 2,
        "replay": {
            "start_frame": 357, "end_frame": 396,
            "start_seconds": 14.24, "end_seconds": 15.8,
        },
    }

def fixture():
    cards = [
        make_card("posture_setup", "Estimated hip angle changed from 152.6° to 174.6°."),
        make_card("hip_loading", HIP),
        make_card("ball_path", BALL),
        make_card("ball_setup", WRIST),
    ]
    return {
        "fps": 25.0,
        "frame_origin": 1,
        "shot_count": 2,
        "official_endorsement": False,
        "intended_form_match": "not_assessed",
        "corrections_generated": 0,
        "shots": [
            {
                "shot_number": 1,
                "timing_quality": "insufficient_for_reliable_timing",
                "intended_form_match": "not_assessed",
                "feedback_cards": [], "feedback_count": 0,
                "replay": {
                    "start_frame": 68, "end_frame": 133,
                    "start_seconds": 2.68, "end_seconds": 5.28,
                },
            },
            {
                "shot_number": 2,
                "timing_quality": "sufficient_for_descriptive_timing",
                "intended_form_match": "not_assessed",
                "feedback_cards": cards, "feedback_count": len(cards),
                "replay": {
                    "start_frame": 357, "end_frame": 419,
                    "start_seconds": 14.24, "end_seconds": 16.72,
                },
            },
        ],
    }

class ConciseEnglishVoiceTests(unittest.TestCase):
    def setUp(self):
        self.report = fixture()
        self.builder = ConciseEnglishVoice()

    def get_shot(self, index=1):
        return self.builder.build(self.report)["shots"][index]

    def test_uses_english(self):
        self.assertEqual(self.builder.build(self.report)["language"], "en")

    def test_two_shots(self):
        self.assertEqual(len(self.builder.build(self.report)["shots"]), 2)

    def test_short_scripts(self):
        for shot in self.builder.build(self.report)["shots"]:
            self.assertLessEqual(shot["spoken_word_count"], 75)

    def test_no_card_message(self):
        text = self.get_shot(0)["narration_text"]
        self.assertIn("not enough reliable", text)
        self.assertIn("does not mean your shooting form was wrong", text)

    def test_hip_value_comes_from_endpoints(self):
        text = self.get_shot()["narration_text"]
        self.assertIn("22 degrees", text)
        self.assertIn("2 degrees", text)

    def test_ball_value_comes_from_endpoints(self):
        self.assertIn("95 pixels up", self.get_shot()["narration_text"])

    def test_topics_prioritized(self):
        self.assertEqual(self.get_shot()["spoken_topics"], ["hip_loading", "ball_path"])

    def test_no_duplicate_hip_observation(self):
        self.assertEqual(self.get_shot()["narration_text"].count("estimated hip angle"), 1)

    def test_technical_details_not_read(self):
        speech = self.get_shot()["narration_text"]
        for detail in ("frame 357", "40 observed samples", "release sample", "end point", URL):
            self.assertNotIn(detail, speech)

    def test_source_urls_preserved(self):
        self.assertEqual(self.get_shot()["source_urls"], [URL])

    def test_replay_preserved(self):
        self.assertEqual(self.get_shot()["replay"], self.report["shots"][1]["replay"])

    def test_all_card_count_preserved(self):
        self.assertEqual(self.get_shot()["feedback_count"], 4)

    def test_safety_flags(self):
        manifest = self.builder.build(self.report)
        self.assertEqual(manifest["intended_form_match"], "not_assessed")
        self.assertEqual(manifest["corrections_generated"], 0)
        self.assertEqual(manifest["made_shots"], "not_assessed")
        self.assertEqual(manifest["missed_shots"], "not_assessed")

    def test_no_input_mutation(self):
        before = copy.deepcopy(self.report)
        self.builder.build(self.report)
        self.assertEqual(before, self.report)

    def test_no_unsupported_technical_correction(self):
        t = self.get_shot()["narration_text"].lower()
        self.assertNotIn("incorrect", t)
        self.assertNotIn("perfect", t)
        self.assertNotIn("should be at least", t)
        self.assertIn("do not prove a shooting mistake", t)

    def test_reject_grade(self):
        self.report["intended_form_match"] = "ideal"
        with self.assertRaises(ValueError):
            self.builder.build(self.report)

    def test_reject_corrections(self):
        self.report["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            self.builder.build(self.report)

    def test_reject_unqualified_card(self):
        self.report["shots"][1]["timing_quality"] = "insufficient_for_reliable_timing"
        with self.assertRaises(ValueError):
            self.builder.build(self.report)

    def test_fallback_when_no_parsable_measurements(self):
        for card in self.report["shots"][1]["feedback_cards"]:
            card["observation_text"] = "Observed position measurements are available."
        text = self.get_shot()["narration_text"]
        self.assertIn("not enough suitable numerical detail", text)
        self.assertEqual(self.get_shot()["spoken_topics"], [])

    def test_wrist_is_fallback(self):
        self.report["shots"][1]["feedback_cards"] = [make_card("ball_setup", WRIST)]
        self.report["shots"][1]["feedback_count"] = 1
        shot = self.get_shot()
        self.assertEqual(shot["spoken_topics"], ["ball_setup"])
        self.assertIn("29 to 12 pixels", shot["narration_text"])

    def test_angle_falling(self):
        self.report["shots"][1]["feedback_cards"] = [
            make_card("hip_loading", "Estimated hip angle changed from 150.0° to 120.0°.")
        ]
        self.report["shots"][1]["feedback_count"] = 1
        self.assertIn("decreased by about 30 degrees", self.get_shot()["narration_text"])

    def test_reject_out_of_range_angle_without_fabrication(self):
        self.report["shots"][1]["feedback_cards"] = [
            make_card("hip_loading", "Estimated hip angle changed from 300.0° to 120.0°.")
        ]
        self.report["shots"][1]["feedback_count"] = 1
        self.assertEqual(self.get_shot()["spoken_topics"], [])

    def test_filename_is_separate(self):
        self.assertEqual(self.get_shot()["audio_filename"], "shot_002_english_concise.wav")
        self.assertEqual(self.get_shot()["audio_status"], "script_only")

    def test_missing_cards_cannot_fake_coaching(self):
        self.report["shots"][1]["feedback_cards"] = []
        self.report["shots"][1]["feedback_count"] = 0
        self.assertEqual(self.get_shot()["spoken_topics"], [])

if __name__ == "__main__":
    unittest.main()
