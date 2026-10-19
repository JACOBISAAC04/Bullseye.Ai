"""Phase 6: Gemini-assisted, evidence-constrained voice narration tests.

Run: python -m unittest discover -s tests -p "test_gemini_coaching.py" -v
Tests never call Gemini or Windows speech.
"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.coaching.coaching_script_validator import compose_script, validate_plan
from app.coaching.gemini_coaching_writer import (
    DEFAULT_MODEL, GeminiCoachingWriter, GeminiRestClient,
)
from app.coaching.coaching_audio_pipeline import run_pipeline


def card(topic, text):
    return {
        "principle_id": topic,
        "title": topic,
        "observation_text": text,
        "review_question": "Review measured evidence.",
        "source_urls": ["https://example.org/real-source"],
        "limitations": ["Camera angle limits confidence."],
        "evidence_status": "observed_2d_descriptive",
        "measurement_count": 2,
        "replay": {
            "start_frame": 357, "end_frame": 396,
            "start_seconds": 14.24, "end_seconds": 15.8,
        },
    }


def fixture():
    return {
        "fps": 25.0,
        "frame_origin": 1,
        "shot_count": 2,
        "official_endorsement": False,
        "intended_form_match": "not_assessed",
        "corrections_generated": 0,
        "shots": [
            {
                "shot_number": 1, "arm": "right",
                "timing_quality": "insufficient_for_reliable_timing",
                "intended_form_match": "not_assessed",
                "feedback_cards": [], "feedback_count": 0,
                "replay": {
                    "start_frame": 68, "end_frame": 133,
                    "start_seconds": 2.68, "end_seconds": 5.28,
                },
            },
            {
                "shot_number": 2, "arm": "right",
                "timing_quality": "sufficient_for_descriptive_timing",
                "intended_form_match": "not_assessed",
                "feedback_cards": [
                    card("hip_loading",
                         "Estimated hip angle changed from 152.6° (frame 357) "
                         "to 174.6° (frame 396); an endpoint difference of +22.0° "
                         "across 40 observed samples. Estimated knee angle changed "
                         "from 160.4° (frame 357) to 162.1° (frame 396); "
                         "an endpoint difference of +1.7° across 40 observed samples."),
                    card("ball_path",
                         "Accepted ball-detector endpoint moved 12.0 px right "
                         "and 94.6 px up in the image from frame 357 to 393 "
                         "(35 observed detector frames)."),
                ],
                "feedback_count": 2,
                "replay": {
                    "start_frame": 357, "end_frame": 419,
                    "start_seconds": 14.24, "end_seconds": 16.72,
                },
            },
        ],
    }


class FakeClient:
    def __init__(self, plan=None, error=None):
        self.plan = plan or {
            "style": "friendly",
            "evidence_ids": ["ball_path", "hip_loading"],
        }
        self.error = error
        self.calls = []

    def select_plan(self, number, facts):
        self.calls.append((number, copy.deepcopy(facts)))
        if self.error:
            raise self.error
        return copy.deepcopy(self.plan)


class FakeHTTPResponse:
    def __init__(self, result):
        self.data = json.dumps(result).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def read(self, size):
        return self.data[:size]


class GeminiCoachingTests(unittest.TestCase):

    def setUp(self):
        self.data = fixture()

    def test_local_build_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            m = GeminiCoachingWriter().build(self.data)
        self.assertEqual(m["shots"][1]["gemini_status"], "not_requested")
        self.assertEqual(m["shots"][1]["script_source"], "local_approved_sentences")

    def test_local_script_under_85_words(self):
        m = GeminiCoachingWriter().build(self.data)
        self.assertLessEqual(m["shots"][1]["spoken_word_count"], 85)

    def test_script_contains_grounded_hip(self):
        s = GeminiCoachingWriter().build(self.data)["shots"][1]["narration_text"]
        self.assertIn("22 degrees", s)
        self.assertIn("2 degrees", s)

    def test_script_contains_grounded_ball(self):
        s = GeminiCoachingWriter().build(self.data)["shots"][1]["narration_text"]
        self.assertIn("95 pixels up", s)

    def test_no_duplicate_observations(self):
        s = GeminiCoachingWriter().build(self.data)["shots"][1]["narration_text"]
        self.assertEqual(s.count("estimated hip angle"), 1)

    def test_no_raw_frame_numbers(self):
        s = GeminiCoachingWriter().build(self.data)["shots"][1]["narration_text"]
        self.assertNotIn("frame 357", s)
        self.assertNotIn("40 observed samples", s)

    def test_successful_gemini_selection(self):
        fake = FakeClient()
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        shot = m["shots"][1]
        self.assertEqual(shot["gemini_status"], "accepted")
        self.assertEqual(shot["script_source"], "gemini_selected_approved_sentences")
        self.assertEqual(shot["spoken_topics"], ["ball_path", "hip_loading"])
        self.assertTrue(shot["narration_text"].startswith("Let's review shot 2"))
        self.assertEqual(len(fake.calls), 1)

    def test_gemini_cannot_add_custom_text(self):
        fake = FakeClient({
            "style": "friendly",
            "evidence_ids": ["hip_loading", "ball_path"],
            "narration_text": "Your release angle is wrong!",
        })
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][1]["gemini_status"], "invalid_or_unavailable")
        self.assertNotIn("wrong", m["shots"][1]["narration_text"])

    def test_gemini_cannot_invent_extra_topic(self):
        fake = FakeClient({
            "style": "direct",
            "evidence_ids": ["hip_loading", "ball_path", "shooting_mistake"],
        })
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][1]["gemini_status"], "invalid_or_unavailable")

    def test_gemini_cannot_omit_disclaimer(self):
        fake = FakeClient()
        s = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)["shots"][1]["narration_text"]
        self.assertIn("do not confirm a shooting mistake", s)

    def test_gemini_cannot_omit_measurement(self):
        fake = FakeClient({"style": "focused", "evidence_ids": ["hip_loading"]})
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][1]["gemini_status"], "invalid_or_unavailable")
        self.assertIn("95 pixels up", m["shots"][1]["narration_text"])

    def test_unknown_style_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan(
                {"style": "perfect-shooting", "evidence_ids": ["a"]}, ["a"]
            )

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan(
                {"style": "focused", "evidence_ids": ["a", "a"]}, ["a", "b"]
            )

    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan(None, ["a"])

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan(
                {"style": "focused", "evidence_ids": ["b"]}, ["a"]
            )

    def test_custom_script_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan(
                {"style": "focused", "evidence_ids": ["a"],
                 "script": "You missed."}, ["a"]
            )

    def test_no_gemini_call_on_insufficient_shot(self):
        fake = FakeClient()
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][0]["gemini_status"], "not_needed")
        self.assertEqual(len(fake.calls), 1)

    def test_missing_key_falls_back(self):
        with patch.dict(os.environ, {}, clear=True):
            m = GeminiCoachingWriter().build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][1]["gemini_status"], "api_key_missing")

    def test_remote_exception_falls_back(self):
        fake = FakeClient(error=RuntimeError("Network broken"))
        m = GeminiCoachingWriter(fake).build(self.data, use_gemini=True)
        self.assertEqual(m["shots"][1]["gemini_status"], "invalid_or_unavailable")

    def test_no_raw_video_sent_in_payload(self):
        obj = GeminiRestClient._payload(2, [
            ("hip_loading", "Your estimated hip angle changed.")
        ])
        payload = json.dumps(obj)
        self.assertNotIn("video.mp4", payload)
        self.assertNotIn("example.org", payload)

    def test_api_key_in_header_only(self):
        client = GeminiRestClient("test-private-token")
        result = {"candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{
                "text": json.dumps({
                    "style": "focused", "evidence_ids": ["hip_loading"],
                }),
            }]}
        }]}
        with patch("app.coaching.gemini_coaching_writer.urlopen",
                   return_value=FakeHTTPResponse(result)) as http:
            plan = client.select_plan(2, [("hip_loading", "Verified fact.")])
        req = http.call_args[0][0]
        self.assertEqual(plan["style"], "focused")
        self.assertEqual(req.get_header("X-goog-api-key"), "test-private-token")
        self.assertNotIn("test-private-token", req.data.decode())
        self.assertNotIn("test-private-token", req.full_url)

    def test_api_invalid_json_fails_safely(self):
        client = GeminiRestClient("dummy")
        result = {"candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": "INVALID"}]}
        }]}
        with patch("app.coaching.gemini_coaching_writer.urlopen",
                   return_value=FakeHTTPResponse(result)):
            with self.assertRaises(ValueError):
                client.select_plan(2, [("hip_loading", "Observed.")])

    def test_api_blocked_response_fails(self):
        client = GeminiRestClient("dummy")
        result = {"candidates": [{
            "finishReason": "SAFETY",
            "content": {"parts": [{"text": "{}"}]}
        }]}
        with patch("app.coaching.gemini_coaching_writer.urlopen",
                   return_value=FakeHTTPResponse(result)):
            with self.assertRaises(ValueError):
                client.select_plan(2, [("hip_loading", "Observed.")])

    def test_model_identifier_validation(self):
        for name in ("http://localhost", "gemini/../../", " ", ""):
            with self.assertRaises(ValueError):
                GeminiRestClient("key", model=name)

    def test_unsafe_timeout_rejected(self):
        with self.assertRaises(ValueError):
            GeminiRestClient("key", timeout=0)

    def test_replay_preserved(self):
        m = GeminiCoachingWriter().build(self.data)
        self.assertEqual(m["shots"][1]["replay"], self.data["shots"][1]["replay"])

    def test_source_urls_preserved(self):
        m = GeminiCoachingWriter().build(self.data)
        self.assertEqual(
            m["shots"][1]["source_urls"], ["https://example.org/real-source"]
        )

    def test_not_graded(self):
        m = GeminiCoachingWriter().build(self.data)
        self.assertEqual(m["intended_form_match"], "not_assessed")
        self.assertEqual(m["corrections_generated"], 0)
        self.assertEqual(m["made_shots"], "not_assessed")
        self.assertEqual(m["missed_shots"], "not_assessed")

    def test_input_unchanged(self):
        before = copy.deepcopy(self.data)
        GeminiCoachingWriter().build(self.data)
        self.assertEqual(before, self.data)

    def test_reject_grade_input(self):
        self.data["intended_form_match"] = "perfect"
        with self.assertRaises(ValueError):
            GeminiCoachingWriter().build(self.data)

    def test_reject_corrections_input(self):
        self.data["corrections_generated"] = 1
        with self.assertRaises(ValueError):
            GeminiCoachingWriter().build(self.data)

    def test_reject_unqualified_timing(self):
        self.data["shots"][1]["timing_quality"] = "insufficient_for_reliable_timing"
        with self.assertRaises(ValueError):
            GeminiCoachingWriter().build(self.data)

    def test_empty_shots(self):
        self.data["shots"] = []
        self.data["shot_count"] = 0
        m = GeminiCoachingWriter().build(self.data)
        self.assertEqual(m["shots"], [])

    def test_null_byte_in_approved_sentence_rejected(self):
        with self.assertRaises(ValueError):
            compose_script(2, [("hip_loading", "Bad\x00phrase.")], {
                "style": "focused", "evidence_ids": ["hip_loading"],
            })

    def test_one_fact_works(self):
        facts = [("hip_loading", "Your hip angle increased.")]
        s = compose_script(2, facts, {
            "style": "focused", "evidence_ids": ["hip_loading"],
        })
        self.assertIn("Your hip angle increased.", s)

    def test_pipeline_script_only_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice"
            report = run_pipeline(self.data, path, script_only=True)
            self.assertTrue((path / "voice_manifest.json").exists())
            self.assertEqual(report["shots"][1]["audio_status"], "script_only")
            self.assertFalse((path / "shot_002_english_ai_voice.wav").exists())

    def test_pipeline_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice"
            run_pipeline(self.data, path, script_only=True)
            with self.assertRaises(FileExistsError):
                run_pipeline(self.data, path, script_only=True)

    def test_pipeline_reject_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_pipeline(self.data, Path(tmp), rate=100, script_only=True)

    def test_pipeline_mock_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice"
            with patch("app.coaching.coaching_audio_pipeline.synthesize_windows_wav") as synth:
                m = run_pipeline(self.data, path)
            self.assertEqual(synth.call_count, 2)
            self.assertEqual(m["shots"][1]["audio_status"], "generated")

    def test_cannot_inject_scoring_via_style(self):
        with self.assertRaises(ValueError):
            validate_plan({
                "style": "You have perfect form",
                "evidence_ids": ["hip_loading"],
            }, ["hip_loading"])


if __name__ == "__main__":
    unittest.main()
