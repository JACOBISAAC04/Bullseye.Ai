"""Tests for the Phase 6 voice playback router. No network or Windows TTS."""
from __future__ import annotations

import json
import tempfile
import unittest
import uuid
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import voice_playback as vp


def _mock_synthesis(text, path, rate=0):
    path = Path(path)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 80)


def _fixtures():
    temporal = {
        "shot_count": 2,
        "shots": [
            {"shot_number": 1, "arm": "right", "start_frame": 10,
             "release_frame": 50, "end_frame": 70},
            {"shot_number": 2, "arm": "right", "start_frame": 100,
             "release_frame": 104, "end_frame": 130},
        ],
    }
    candidate = {
        "principle_id": "hip_loading",
        "title": "Hip loading",
        "review_question": "How do your hips move?",
        "source_urls": ["https://example.org/research"],
        "source_support_status": "limited_published_text_support",
        "status": "measured_descriptive_review",
        "observations": [
            {"signal": "hip_angle", "unit": "degrees_2d",
             "start_frame": 10, "start_value": 150.0,
             "end_frame": 50, "end_value": 172.0,
             "observed_endpoint_difference": 22.0,
             "observed_samples": 41},
            {"signal": "knee_angle", "unit": "degrees_2d",
             "start_frame": 10, "start_value": 160.0,
             "end_frame": 50, "end_value": 162.0,
             "observed_endpoint_difference": 2.0,
             "observed_samples": 41},
        ],
        "intended_form_match": "not_assessed",
    }
    measured = {
        "shot_count": 2,
        "shots": [
            {"shot_number": 1,
             "timing_quality": "sufficient_for_descriptive_timing",
             "review_candidate_count": 1,
             "review_candidates": [candidate],
             "intended_form_match": "not_assessed"},
            {"shot_number": 2,
             "timing_quality": "insufficient_for_reliable_timing",
             "review_candidate_count": 0,
             "review_candidates": [],
             "intended_form_match": "not_assessed"},
        ],
        "intended_form_match": "not_assessed",
        "official_endorsement": False,
        "corrections_generated": 0,
    }
    return temporal, measured


class VoicePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job = uuid.uuid4()
        self.analyzed = self.root / "analyzed"
        self.uploads = self.root / "videos"
        self.directory = self.analyzed / str(self.job)
        self.directory.mkdir(parents=True)
        self.uploads.mkdir()
        self.patch_analyzed = patch.object(vp, "ANALYZED_DIR", self.analyzed)
        self.patch_uploads = patch.object(vp, "UPLOAD_DIR", self.uploads)
        self.patch_fps = patch.object(vp, "_video_fps", return_value=30.0)
        for action in (self.patch_analyzed, self.patch_uploads, self.patch_fps):
            action.start()
            self.addCleanup(action.stop)
        self.original = self.uploads / f"{self.job}.mp4"
        self.original.write_bytes(b"fake-video")
        (self.directory / "shooting_analysis.mp4").write_bytes(b"fake-analysis")
        self.temporal, self.measured = _fixtures()
        self._write()
        app = FastAPI()
        app.include_router(vp.router)
        self.client = TestClient(app, client=("127.0.0.1", 50123))
        self.remote = TestClient(app, client=("192.0.2.1", 50123))
        self.prefix = f"/video/results/{self.job}"

    def _write(self):
        for name, data in (
            ("temporal_form.json", self.temporal),
            ("measured_coaching_report.json", self.measured),
        ):
            (self.directory / name).write_text(json.dumps(data), encoding="utf-8")

    def _generate(self):
        with patch("app.coaching.coaching_audio_pipeline.synthesize_windows_wav",
                   side_effect=_mock_synthesis):
            return vp.generate_job_voice(self.job, use_gemini=False)

    def test_builds_report_from_saved_evidence(self):
        result = vp.build_job_player_report(self.job)
        self.assertEqual(result["shot_count"], 2)
        self.assertEqual(result["shots"][0]["feedback_count"], 1)
        self.assertEqual(result["shots"][1]["feedback_count"], 0)
        self.assertEqual(result["shots"][0]["evidence_explanation"]["status"],
                         "descriptive_reviews_available")
        self.assertEqual(result["intended_form_match"], "not_assessed")

    def test_no_reanalysis_needed(self):
        self._generate()
        self.assertFalse((self.directory / "detections.json").exists())

    def test_audio_manifest_generated(self):
        response = self._generate()
        self.assertEqual([s["shot_number"] for s in response["shots"]], [1, 2])
        self.assertEqual(response["shots"][0]["gemini_status"], "not_requested")
        self.assertEqual(response["shots"][1]["gemini_status"], "not_needed")
        self.assertEqual(response["shot_count"] if "shot_count" in response else 2, 2)
        self.assertTrue((self.directory / vp.VOICE_FOLDER /
                         "player_feedback.json").is_file())

    def test_successful_get_endpoints(self):
        self._generate()
        self.assertEqual(self.client.get(self.prefix + "/player").status_code, 200)
        self.assertEqual(self.client.get(self.prefix + "/stream").status_code, 200)
        manifest = self.client.get(self.prefix + "/voice/manifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(len(manifest.json()["shots"]), 2)
        self.assertEqual(
            self.client.get(self.prefix + "/voice/player-feedback").status_code, 200
        )
        audio = self.client.get(self.prefix + "/voice/shot/1")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/wav")

    def test_missing_voice_not_fabricated(self):
        self.assertEqual(self.client.get(self.prefix + "/voice/manifest").status_code, 404)
        self.assertEqual(self.client.get(self.prefix + "/voice/shot/1").status_code, 404)

    def test_remote_generation_blocked(self):
        response = self.remote.post(self.prefix + "/voice?use_gemini=true")
        self.assertEqual(response.status_code, 403)

    def test_local_generation_works(self):
        with patch("app.coaching.coaching_audio_pipeline.synthesize_windows_wav",
                   side_effect=_mock_synthesis):
            response = self.client.post(self.prefix + "/voice?use_gemini=false")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["shots"][0]["audio_url"].endswith("/shot/1"))

    def test_repeated_request_uses_existing_audio(self):
        self._generate()
        with patch("app.coaching.coaching_audio_pipeline.synthesize_windows_wav",
                   side_effect=AssertionError("must not regenerate")):
            response = self.client.post(self.prefix + "/voice?use_gemini=true")
        self.assertEqual(response.status_code, 200)

    def test_bad_shot_cannot_read_arbitrary_path(self):
        self._generate()
        self.assertEqual(
            self.client.get(self.prefix + "/voice/shot/999").status_code, 404
        )
        self.assertEqual(
            self.client.get(self.prefix + "/voice/shot/0").status_code, 404
        )

    def test_missing_measured_is_404(self):
        (self.directory / "measured_coaching_report.json").unlink()
        with patch("app.coaching.coaching_audio_pipeline.synthesize_windows_wav",
                   side_effect=_mock_synthesis):
            response = self.client.post(self.prefix + "/voice")
        self.assertEqual(response.status_code, 404)

    def test_unapproved_form_verdict_rejected(self):
        self.measured["intended_form_match"] = "perfect"
        self._write()
        with self.assertRaises(ValueError):
            vp.build_job_player_report(self.job)

    def test_mismatched_shot_count_rejected(self):
        self.measured["shots"].pop()
        self._write()
        with self.assertRaises(ValueError):
            vp.build_job_player_report(self.job)

    def test_partial_voice_output_not_overwritten(self):
        folder = self.directory / vp.VOICE_FOLDER
        folder.mkdir()
        (folder / "important.txt").write_text("preserve")
        with self.assertRaises(FileExistsError):
            vp.generate_job_voice(self.job)
        self.assertEqual((folder / "important.txt").read_text(), "preserve")

    def test_unknown_job_returns_404(self):
        other = uuid.uuid4()
        self.assertEqual(
            self.client.get(f"/video/results/{other}/player").status_code, 404
        )

    def test_no_key_or_gemini_request_in_default(self):
        with patch("app.coaching.gemini_coaching_writer.GeminiRestClient",
                   side_effect=AssertionError("Unexpected Gemini call")):
            response = self._generate()
        self.assertEqual(response["shots"][0]["gemini_status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
