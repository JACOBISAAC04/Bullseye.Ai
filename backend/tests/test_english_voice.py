"""Phase 6 English-only voice narration tests (speech synthesis mocked)."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.coaching.english_voice import (
    EnglishVoiceCoaching, _english_readable, synthesize_windows_wav,
)


SOURCE = "https://example.com/coach-interview"


def fixture():
    shot1_replay = {
        "start_frame": 68, "end_frame": 133,
        "start_seconds": 2.68, "end_seconds": 5.28,
    }
    shot2_replay = {
        "start_frame": 357, "end_frame": 419,
        "start_seconds": 14.24, "end_seconds": 16.72,
    }
    card_replay = {
        "start_frame": 357, "end_frame": 396,
        "start_seconds": 14.24, "end_seconds": 15.8,
    }
    return {
        "fps": 25.0,
        "frame_origin": 1,
        "shot_count": 2,
        "intended_form_match": "not_assessed",
        "official_endorsement": False,
        "corrections_generated": 0,
        "shots": [
            {
                "shot_number": 1,
                "intended_form_match": "not_assessed",
                "timing_quality": "insufficient_for_reliable_timing",
                "feedback_count": 0,
                "replay": shot1_replay,
                "feedback_cards": [],
            },
            {
                "shot_number": 2,
                "intended_form_match": "not_assessed",
                "timing_quality": "sufficient_for_descriptive_timing",
                "feedback_count": 1,
                "replay": shot2_replay,
                "feedback_cards": [
                    {
                        "principle_id": "hip_loading",
                        "title": "Hip loading",
                        "display_title": "Hip and knee movement",
                        "review_question": "Review hip and knee movement.",
                        "player_review_question": "How do your hip and knee angles change?",
                        "observation_text": "Hip angle changed by 21.9° over 40 frames.",
                        "source_urls": [SOURCE],
                        "limitations": ["Uncalibrated 2D movement."],
                        "evidence_status": "observed_2d_descriptive",
                        "measurement_count": 2,
                        "replay": card_replay,
                        "evidence_explanation": {
                            "status": "measured_descriptive_only",
                            "can_grade_form": False,
                            "can_recommend_correction": False,
                            "confidence": {"kind": "not_scored"},
                        },
                    }
                ],
            },
        ],
    }


class EnglishVoiceTests(unittest.TestCase):
    def setUp(self):
        self.builder = EnglishVoiceCoaching()
        self.data = fixture()

    def test_two_shots(self):
        result = self.builder.build(self.data)
        self.assertEqual(len(result["shots"]), 2)

    def test_output_in_english(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["language"], "en")
        self.assertIn("Shot 2.", result["shots"][1]["narration_text"])

    def test_speech_safety(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["intended_form_match"], "not_assessed")
        self.assertEqual(result["corrections_generated"], 0)
        self.assertEqual(result["made_shots"], "not_assessed")

    def test_no_form_grade(self):
        result = self.builder.build(self.data)
        self.assertIn("does not assign a shooting form grade", result["shots"][1]["narration_text"])

    def test_zero_card_shot_does_not_claim_failure(self):
        result = self.builder.build(self.data)
        narration = result["shots"][0]["narration_text"]
        self.assertIn("does not mean your shooting form was wrong", narration)

    def test_short_timing_is_explained(self):
        result = self.builder.build(self.data)
        self.assertIn("timing was too limited", result["shots"][0]["narration_text"])

    def test_one_card_read(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["shots"][1]["feedback_count"], 1)
        self.assertIn("Hip and knee movement", result["shots"][1]["narration_text"])

    def test_measurement_is_read(self):
        result = self.builder.build(self.data)
        self.assertIn("21.9 degrees", result["shots"][1]["narration_text"])

    def test_enriched_question_is_read(self):
        result = self.builder.build(self.data)
        self.assertIn("How do your hip and knee angles change?", result["shots"][1]["narration_text"])

    def test_plain_player_report_works(self):
        for card in self.data["shots"][1]["feedback_cards"]:
            card.pop("display_title")
            card.pop("player_review_question")
            card.pop("evidence_explanation")
        result = self.builder.build(self.data)
        self.assertIn("Hip loading", result["shots"][1]["narration_text"])

    def test_preserves_source_urls(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["shots"][1]["source_urls"], [SOURCE])
        self.assertNotIn(SOURCE, result["shots"][1]["narration_text"])

    def test_preserves_replay(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["shots"][1]["replay"], self.data["shots"][1]["replay"])

    def test_no_mutation(self):
        before = copy.deepcopy(self.data)
        self.builder.build(self.data)
        self.assertEqual(self.data, before)

    def test_grade_rejected(self):
        self.data["intended_form_match"] = "perfect"
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_corrections_rejected(self):
        self.data["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_endorsement_rejected(self):
        self.data["official_endorsement"] = True
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_shot_grade_rejected(self):
        self.data["shots"][0]["intended_form_match"] = "correct"
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_bad_card_evidence_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["evidence_status"] = "inferred_correct"
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_unsafe_explanation_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["evidence_explanation"][
            "can_recommend_correction"
        ] = True
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_missing_source_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["source_urls"] = []
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_insecure_source_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["source_urls"] = ["http://example.com"]
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_unqualified_timing_rejected(self):
        self.data["shots"][1]["timing_quality"] = "insufficient_for_reliable_timing"
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_card_replay_outside_shot_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["replay"]["start_frame"] = 350
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_card_timestamp_mismatch_rejected(self):
        self.data["shots"][1]["feedback_cards"][0]["replay"]["start_seconds"] = 15
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_duplicate_topic_rejected(self):
        shot = self.data["shots"][1]
        shot["feedback_cards"].append(copy.deepcopy(shot["feedback_cards"][0]))
        shot["feedback_count"] = 2
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_report_count_mismatch_rejected(self):
        self.data["shot_count"] = 1
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_missing_fps_rejected(self):
        self.data["fps"] = 0
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_nonfinite_fps_rejected(self):
        self.data["fps"] = math.inf
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_frame_origin_rejected(self):
        self.data["frame_origin"] = -1
        with self.assertRaises(ValueError):
            self.builder.build(self.data)

    def test_zero_shots(self):
        self.data["shots"] = []
        self.data["shot_count"] = 0
        result = self.builder.build(self.data)
        self.assertEqual(result["shots"], [])

    def test_replay_readability(self):
        self.assertEqual(
            _english_readable("Moving 2D position by 12 px and 22°"),
            "Moving two dimensional position by 12 pixels and 22 degrees",
        )

    def test_audio_filename_english(self):
        result = self.builder.build(self.data)
        self.assertEqual(result["shots"][1]["audio_filename"], "shot_002_english.wav")

    def test_rate_validation(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                synthesize_windows_wav("Hello", Path(td)/"a.wav", rate=11)

    def test_no_file_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"a.wav"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                synthesize_windows_wav("Hello", path)

    def test_mocked_windows_sapi(self):
        # The real Windows SAPI engine is not available on Linux CI.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"a.wav"
            def fake_run(args, **kwargs):
                self.assertEqual(args[0], "powershell.exe")
                self.assertEqual(args[-2:], ["-Rate", "-1"])
                path.write_bytes(b"RIFF" + b"\x00"*50)
                return mock.Mock(returncode=0)
            with mock.patch(
                "app.coaching.english_voice.subprocess.run", side_effect=fake_run
            ):
                synthesize_windows_wav("Hello", path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".narration.txt").exists())


if __name__ == "__main__":
    unittest.main()
