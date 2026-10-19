"""New coach-style playback route; no real Gemini requests or model loading."""
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

from app.api import coach_voice_browser as coach


class CoachVoiceBrowserTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.job = uuid.uuid4()
        self.job_folder = self.folder / str(self.job)
        self.job_folder.mkdir()
        (self.job_folder / "shooting_analysis.mp4").write_bytes(b"video")
        self.previous = self.job_folder / "voice_playback"
        self.previous.mkdir()
        (self.previous / "shot_001_english_ai_voice.wav").write_bytes(b"old")
        self.report = {"shot_count": 1, "shots": [{"shot_number": 1}]}
        p1 = patch.object(coach, "_job_folder", side_effect=lambda job: self.folder / str(job))
        p2 = patch.object(coach, "build_job_player_report", return_value=self.report)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        app = FastAPI()
        app.include_router(coach.router)
        self.client = TestClient(app, client=("127.0.0.1", 50001))
        self.remote = TestClient(app, client=("192.0.2.1", 50001))
        self.url = f"/video/results/{self.job}/coach-voice"
        self.calls = []

    def mock_generation(self, report, destination, *, tts, voice):
        self.calls.append((report, tts, voice))
        destination.mkdir()
        with wave.open(str(destination / "shot_001_english_focus.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\\0\\0" * 800)
        manifest = {
            "shot_count": 1,
            "shots": [{
                "shot_number": 1,
                "voice_script": "Let's review your setup. Check your shooting-hand support.",
                "focus_count": 1,
                "replay": {"start_seconds": 2.2, "end_seconds": 3.3},
                "tts_status": "gemini_generated" if tts == "gemini" else "windows_generated",
                "audio_filename": "shot_001_english_focus.wav",
            }],
        }
        (destination / "voice_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_new_endpoint_speaks_coach_focus(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            response = self.client.post(self.url + "?use_gemini=true")
        self.assertEqual(response.status_code, 200, response.text)
        shot = response.json()["shots"][0]
        self.assertEqual(shot["tts_status"], "gemini_generated")
        self.assertIn("shooting-hand support", shot["narration_text"])
        self.assertNotIn("degrees", shot["narration_text"])
        self.assertEqual(shot["focus_count"], 1)
        self.assertEqual(self.calls[0][1:], ("gemini", "Kore"))

    def test_old_voice_is_preserved(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            self.client.post(self.url)
        self.assertEqual(
            (self.previous / "shot_001_english_ai_voice.wav").read_bytes(), b"old"
        )

    def test_audio_get(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            self.client.post(self.url)
        response = self.client.get(self.url + "/shot/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")

    def test_cache_does_not_generate_again(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            self.client.post(self.url)
            self.client.post(self.url)
        self.assertEqual(len(self.calls), 1)

    def test_empty_job_404(self):
        self.assertEqual(self.client.get(self.url + "/manifest").status_code, 404)

    def test_remote_generation_denied(self):
        self.assertEqual(self.remote.post(self.url).status_code, 403)

    def test_nonexistent_shot_denied(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            self.client.post(self.url)
        self.assertEqual(self.client.get(self.url + "/shot/99").status_code, 404)
        self.assertEqual(self.client.get(self.url + "/shot/0").status_code, 404)

    def test_existing_partial_folder_not_overwritten(self):
        folder = self.job_folder / coach.VOICE_FOLDER
        folder.mkdir()
        (folder / "important.txt").write_text("keep")
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            res = self.client.post(self.url)
        self.assertEqual(res.status_code, 409)
        self.assertEqual((folder / "important.txt").read_text(), "keep")

    def test_invalid_audio_filename_rejected(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            self.client.post(self.url)
        path = self.job_folder / coach.VOICE_FOLDER / "voice_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["shots"][0]["audio_filename"] = "../unsafe.wav"
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.client.get(self.url + "/manifest").status_code, 500)

    def test_local_windows_option(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            response = self.client.post(self.url + "?use_gemini=false")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["shots"][0]["tts_status"], "windows_generated")


    def test_cached_windows_voice_not_served_as_gemini(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            first = self.client.post(self.url + "?use_gemini=false")
            second = self.client.post(self.url + "?use_gemini=true")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(len(self.calls), 1)

        manifest = self.client.get(self.url + "/manifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(
            manifest.json()["shots"][0]["tts_status"],
            "windows_generated",
        )

    def test_cached_gemini_voice_not_served_as_windows(self):
        with patch.object(coach, "create_focus_voice", side_effect=self.mock_generation):
            first = self.client.post(self.url + "?use_gemini=true")
            second = self.client.post(self.url + "?use_gemini=false")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(len(self.calls), 1)

    def test_gemini_quota_error_returns_429(self):
        import io
        from urllib.error import HTTPError

        body = (
            b'{"error":{"code":429,"message":'
            b'"Resource exhausted. Please retry in 13h21m6s."}}'
        )

        def fail_generation(*args, **kwargs):
            error = HTTPError(
                "https://example.invalid/tts",
                429,
                "Too Many Requests",
                {},
                io.BytesIO(body),
            )
            raise RuntimeError("Gemini TTS failed") from error

        with patch.object(coach, "create_focus_voice", side_effect=fail_generation):
            response = self.client.post(self.url + "?use_gemini=true")

        self.assertEqual(response.status_code, 429, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "gemini_quota_exhausted")
        self.assertEqual(detail["retry_after"], "13h21m6s")


if __name__ == "__main__":
    unittest.main()
