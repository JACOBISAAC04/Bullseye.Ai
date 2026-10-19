"""Tests for app.analysis.browser_video (local FFmpeg; no remote APIs)."""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.analysis.browser_video import (
    BROWSER_FILENAME,
    BrowserVideoError,
    VideoProperties,
    _preserves_timeline,
    ensure_browser_video,
    inspect_video,
)


class BrowserVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import imageio_ffmpeg
        cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.source = Path(cls.fixture_dir.name) / "source.mp4"
        run = subprocess.run(
            [
                cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x96:r=12",
                "-frames:v", "12", "-c:v", "mpeg4",
                "-q:v", "4", str(cls.source),
            ], capture_output=True, text=True, timeout=20
        )
        if run.returncode != 0:
            raise AssertionError("Could not build local test fixture: " + run.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.original = self.root / "shooting_analysis.mp4"
        shutil.copyfile(self.source, self.original)

    def _convert(self):
        return ensure_browser_video(self.original, ffmpeg_exe=self.ffmpeg)

    def test_fixture_is_not_h264(self):
        self.assertFalse(inspect_video(self.original).browser_compatible)

    def test_conversion_creates_named_h264_file(self):
        result = self._convert()
        self.assertTrue(result.created)
        self.assertEqual(result.path.name, BROWSER_FILENAME)
        self.assertIn(result.properties.codec.lower(), {"avc1", "h264", "x264"})

    def test_preserves_timeline(self):
        original = inspect_video(self.original)
        converted = self._convert().properties
        self.assertTrue(_preserves_timeline(original, converted))
        self.assertEqual(original.frames, converted.frames)
        self.assertEqual((original.width, original.height),
                         (converted.width, converted.height))

    def test_original_bytes_unchanged(self):
        before = self.original.read_bytes()
        self._convert()
        self.assertEqual(self.original.read_bytes(), before)

    def test_reuses_existing_h264(self):
        first = self._convert()
        before = first.path.read_bytes()
        with patch("app.analysis.browser_video.subprocess.run") as mocked:
            second = self._convert()
        mocked.assert_not_called()
        self.assertFalse(second.created)
        self.assertEqual(second.path.read_bytes(), before)

    def test_replaces_corrupt_browser_copy(self):
        target = self.root / BROWSER_FILENAME
        target.write_bytes(b"not a video")
        result = self._convert()
        self.assertTrue(result.created)
        self.assertTrue(inspect_video(target).browser_compatible)

    def test_custom_destination(self):
        path = self.root / "other.mp4"
        result = ensure_browser_video(
            self.original, destination=path, ffmpeg_exe=self.ffmpeg
        )
        self.assertEqual(result.path, path)
        self.assertTrue(path.is_file())

    def test_missing_source(self):
        with self.assertRaises(BrowserVideoError):
            ensure_browser_video(self.root / "absent.mp4", ffmpeg_exe=self.ffmpeg)

    def test_empty_source(self):
        self.original.write_bytes(b"")
        with self.assertRaises(BrowserVideoError):
            self._convert()

    def test_same_destination_rejected(self):
        with self.assertRaises(ValueError):
            ensure_browser_video(self.original, destination=self.original)

    def test_invalid_timeouts(self):
        for timeout in (-1, 0, math.inf, math.nan, True, 601):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    ensure_browser_video(self.original, timeout_seconds=timeout)

    def test_missing_ffmpeg_keeps_original(self):
        before = self.original.read_bytes()
        with self.assertRaises(BrowserVideoError):
            ensure_browser_video(
                self.original,
                ffmpeg_exe=self.root / "nonexistent_ffmpeg",
            )
        self.assertEqual(self.original.read_bytes(), before)
        self.assertFalse((self.root / BROWSER_FILENAME).exists())

    def test_encoder_failure_does_not_publish_partial_copy(self):
        fake_result = subprocess.CompletedProcess([], 1, "", "encoder failed")
        with patch(
            "app.analysis.browser_video.subprocess.run", return_value=fake_result
        ):
            with self.assertRaisesRegex(BrowserVideoError, "exit 1"):
                self._convert()
        self.assertFalse((self.root / BROWSER_FILENAME).exists())
        self.assertEqual(
            list(self.root.glob(".bullseye_h264_*.mp4")), []
        )

    def test_encoder_timeout_cleans_temporary(self):
        with patch(
            "app.analysis.browser_video.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ffmpeg"], 2),
        ):
            with self.assertRaisesRegex(BrowserVideoError, "timed out"):
                self._convert()
        self.assertEqual(list(self.root.glob(".bullseye_h264_*.mp4")), [])

    def test_nonh264_output_rejected(self):
        def fake_run(args, **kwargs):
            shutil.copyfile(self.original, args[-1])
            return subprocess.CompletedProcess(args, 0, "", "")
        with patch("app.analysis.browser_video.subprocess.run", side_effect=fake_run):
            with self.assertRaisesRegex(BrowserVideoError, "Expected H.264"):
                self._convert()
        self.assertFalse((self.root / BROWSER_FILENAME).exists())

    def test_reject_changed_dimensions(self):
        a = VideoProperties("FMP4", 24, 100, 768, 432)
        b = VideoProperties("avc1", 24, 100, 640, 360)
        self.assertFalse(_preserves_timeline(a, b))

    def test_reject_changed_frame_count(self):
        a = VideoProperties("FMP4", 24, 100, 768, 432)
        b = VideoProperties("avc1", 24, 97, 768, 432)
        self.assertFalse(_preserves_timeline(a, b))

    def test_reject_changed_frame_rate(self):
        a = VideoProperties("FMP4", 23.976, 502, 768, 432)
        b = VideoProperties("avc1", 30, 502, 768, 432)
        self.assertFalse(_preserves_timeline(a, b))

    def test_unrelated_job_files_remain_untouched(self):
        other = self.root / "measured_coaching_report.json"
        other.write_text('{"shot_count":2}', encoding="utf-8")
        self._convert()
        self.assertEqual(
            other.read_text(encoding="utf-8"), '{"shot_count":2}'
        )


if __name__ == "__main__":
    unittest.main()
