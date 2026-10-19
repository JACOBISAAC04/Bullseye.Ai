"""Regression tests: different evidence -> different narration, no invented errors."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
import wave
import io
from pathlib import Path

from app.coaching.coaching_explainer_v4 import (
    ShotSpecificCoachingExplainer, extract_observations
)
from app.coaching.coaching_explanation_voice_v4 import create_focus_voice
from test_coaching_decision_engine import evidence_report


def sample_two_qualified_shots():
    report = evidence_report()
    report["shots"][0] = copy.deepcopy(report["shots"][1])
    report["shots"][0]["shot_number"] = 1
    # Hypothetical source report for tests only: consistent displayed 2D numbers.
    for card in report["shots"][0]["feedback_cards"]:
        if card["principle_id"] in ("hip_loading", "posture_setup"):
            card["observation_text"] = (
                card["observation_text"]
                .replace("174.6°", "158.6°")
                .replace("+22.0°", "+6.0°")
            )
        if card["principle_id"] == "ball_setup":
            card["observation_text"] = (
                card["observation_text"].replace("11.9 image px", "40.9 image px")
            )
        if card["principle_id"] == "ball_path":
            card["observation_text"] = (
                card["observation_text"].replace("12.0 px right", "12.0 px left")
                .replace("94.6 px up", "94.6 px down")
            )
    return report


def fake_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\\0\\0" * 8000)
    return buffer.getvalue()


class FakeGeminiTTS:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("TTS offline")
        return fake_wav()


def fake_windows(text, path, *, rate):
    Path(path).write_bytes(fake_wav())


class ShotSpecificCoachV4Tests(unittest.TestCase):
    def setUp(self):
        self.report = evidence_report()
        self.explainer = ShotSpecificCoachingExplainer()

    def test_existing_video3_distinct_evidence_availability(self):
        shots = self.explainer.build(self.report)["shots"]
        self.assertEqual(shots[0]["shot_specific_observation_count"], 0)
        self.assertEqual(shots[1]["shot_specific_observation_count"], 3)
        self.assertIn("reliable technique assessment", shots[0]["voice_script"])
        self.assertIn("hip", shots[1]["voice_script"])

    def test_diff_shots_same_topics_not_same_script(self):
        result = self.explainer.build(sample_two_qualified_shots())
        one, two = (shot["voice_script"] for shot in result["shots"])
        self.assertNotEqual(one.replace("shot 1", "shot N"),
                            two.replace("shot 2", "shot N"))
        self.assertIn("farther from", one)
        self.assertIn("closer to", two)
        self.assertIn("downward", one)
        self.assertIn("upward", two)

    def test_identical_evidence_does_not_invent_a_difference(self):
        report = sample_two_qualified_shots()
        report["shots"][0]["feedback_cards"] = copy.deepcopy(
            report["shots"][1]["feedback_cards"]
        )
        one, two = (shot["voice_script"] for shot in self.explainer.build(report)["shots"])
        self.assertNotEqual(one, two)
        self.assertIn("same broad movement descriptions", two)
        self.assertIn("don't support a distinct personal correction", two)
        self.assertNotIn("set-and-hold reps", two)

    def test_no_biomechanical_grade(self):
        r = self.explainer.build(sample_two_qualified_shots())
        self.assertEqual(r["confirmed_mistakes"], 0)
        self.assertEqual(r["intended_form_match"], "not_assessed")
        self.assertTrue(all(s["confirmed_mistakes"] == [] for s in r["shots"]))
        for shot in r["shots"]:
            self.assertNotIn("your form is wrong", shot["voice_script"].lower())

    def test_descriptive_evidence_and_drills(self):
        r = self.explainer.build(self.report)["shots"][1]
        self.assertIn("set-to-rise", r["voice_script"])
        self.assertIn("set point", r["voice_script"])
        self.assertIn("same angle", r["voice_script"])
        self.assertNotIn("22 degrees", r["voice_script"])
        self.assertNotIn("94 pixels", r["voice_script"])

    def test_underlying_cards_unchanged(self):
        source = copy.deepcopy(self.report)
        self.explainer.build(self.report)
        self.assertEqual(self.report, source)

    def test_voice_limit(self):
        result = self.explainer.build(self.report)
        self.assertTrue(all(len(s["voice_script"]) <= 1100 for s in result["shots"]))

    def test_missing_parsable_measurements_not_replaced_by_generic_drills(self):
        report = copy.deepcopy(self.report)
        for card in report["shots"][1]["feedback_cards"]:
            card["observation_text"] = "Unknown summary of an observation."
        shot = self.explainer.build(report)["shots"][1]
        self.assertEqual(shot["shot_specific_observation_count"], 0)
        self.assertEqual(shot["narration_status"], "insufficient_distinct_evidence")
        self.assertNotIn("set-and-hold reps", shot["voice_script"])

    def test_untrusted_made_is_rejected(self):
        with self.assertRaises(ValueError):
            self.explainer.build(self.report, verified_outcomes={
                2: {"outcome":"made", "source":"user_claim"}
            })

    def test_verified_made_still_reviewed(self):
        result = self.explainer.build(self.report, verified_outcomes={
            2: {"outcome":"made", "source":"trusted_hoop_outcome_evaluator",
                "evidence_frames":[417]}
        })
        shot = result["shots"][1]
        self.assertIn("went in", shot["voice_script"])
        self.assertEqual(shot["shot_specific_observation_count"], 3)
        self.assertEqual(shot["confirmed_mistakes"], [])

    def test_angle_extractor_checks_consistency(self):
        angle = extract_observations([{
            "principle_id":"hip_loading",
            "evidence_status":"observed_2d_descriptive",
            "observation_text": (
                "Estimated hip angle changed from 152.6° (frame 20) "
                "to 174.6° (frame 59); an endpoint difference of +44.0°"
            )
        }])
        self.assertNotIn("angles", angle)

    def test_raw_user_provided_grade_refused(self):
        data = copy.deepcopy(self.report)
        data["intended_form_match"] = "perfect"
        with self.assertRaises(ValueError):
            self.explainer.build(data)

    def test_voice_script_only(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "one"
            voice = create_focus_voice(self.report, dest, script_only=True)
            self.assertEqual(voice["shots"][1]["tts_status"], "script_only")
            self.assertEqual(voice["shots"][1]["narration_status"], "per_shot_observations")
            self.assertFalse((dest / "shot_002_english_focus.wav").exists())

    def test_voice_gemini_generated(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "two"
            fake = FakeGeminiTTS()
            voice = create_focus_voice(
                self.report, dest, tts="gemini",
                tts_client=fake, windows_synth=fake_windows
            )
            self.assertEqual(voice["shots"][1]["tts_status"], "gemini_generated")
            self.assertEqual(len(fake.calls), 1)
            self.assertTrue((dest / "shot_002_english_focus.wav").exists())

    def test_voice_gemini_failure_does_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "three"
            windows_calls = []

            def track_windows(text, path, *, rate):
                windows_calls.append(text)
                fake_windows(text, path, rate=rate)

            with self.assertRaisesRegex(RuntimeError, "Gemini TTS failed"):
                create_focus_voice(
                    self.report, dest, tts="gemini",
                    tts_client=FakeGeminiTTS(True),
                    windows_synth=track_windows,
                )

            self.assertEqual(len(windows_calls), 1)
            self.assertFalse(dest.exists())


    def test_gemini_checkpoint_resumes_without_regenerating_shot_one(self):
        report = sample_two_qualified_shots()

        class InterruptOnSecondShot:
            def __init__(self):
                self.calls = []

            def synthesize(self, text):
                self.calls.append(text)
                if len(self.calls) == 2:
                    raise RuntimeError("SIMULATED_QUOTA_EXHAUSTED")
                return fake_wav()

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "coach_voice_v4"
            checkpoint = Path(folder) / "coach_voice_v4_checkpoint"

            first_client = InterruptOnSecondShot()

            with self.assertRaisesRegex(
                RuntimeError, "SIMULATED_QUOTA_EXHAUSTED"
            ):
                create_focus_voice(
                    report,
                    destination,
                    tts="gemini",
                    tts_client=first_client,
                )

            self.assertEqual(len(first_client.calls), 2)
            self.assertFalse(destination.exists())
            self.assertTrue(checkpoint.is_dir())

            first_wav = checkpoint / "shot_001_english_focus.wav"
            first_cache = checkpoint / "shot_001_english_focus.wav.cache.json"

            self.assertTrue(first_wav.is_file())
            self.assertTrue(first_cache.is_file())
            original_audio = first_wav.read_bytes()

            second_client = FakeGeminiTTS()
            result = create_focus_voice(
                report,
                destination,
                tts="gemini",
                tts_client=second_client,
            )

            self.assertEqual(len(second_client.calls), 1)
            self.assertEqual(second_client.calls[0], first_client.calls[1])

            self.assertFalse(checkpoint.exists())
            self.assertTrue(destination.is_dir())
            self.assertEqual(
                (destination / "shot_001_english_focus.wav").read_bytes(),
                original_audio,
            )
            self.assertTrue(
                (destination / "shot_002_english_focus.wav").is_file()
            )
            self.assertTrue(
                (destination / "voice_manifest.json").is_file()
            )
            self.assertEqual(
                [shot["tts_status"] for shot in result["shots"]],
                ["gemini_generated", "gemini_generated"],
            )

    def test_existing_folder_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "one"
            dest.mkdir()
            with self.assertRaises(FileExistsError):
                create_focus_voice(self.report, dest, script_only=True)

    def test_one_shot_only(self):
        data = copy.deepcopy(self.report)
        data["shot_count"] = 1
        data["shots"] = data["shots"][1:]
        data["shots"][0]["shot_number"] = 1
        result = self.explainer.build(data)
        self.assertEqual(result["shot_count"], 1)


if __name__ == "__main__":
    unittest.main()
