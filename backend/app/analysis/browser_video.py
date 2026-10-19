"""Make Bullseye's analyzed MP4 playable in HTML5 browsers.

Original shooting_analysis.mp4 is never modified. The browser copy is created
in the same job folder as shooting_analysis_browser.mp4.

Usage from backend:
    python -m app.analysis.browser_video --input data/analyzed/JOB/shooting_analysis.mp4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


BROWSER_FILENAME = "shooting_analysis_browser.mp4"
_BROWSER_FOURCC = frozenset({"avc1", "h264", "x264"})


class BrowserVideoError(RuntimeError):
    """Browser video could not be created or verified."""


@dataclass(frozen=True)
class VideoProperties:
    codec: str
    fps: float
    frames: int
    width: int
    height: int

    @property
    def browser_compatible(self) -> bool:
        return self.codec.lower() in _BROWSER_FOURCC


@dataclass(frozen=True)
class BrowserVideoResult:
    path: Path
    created: bool
    properties: VideoProperties


def inspect_video(path: Path) -> VideoProperties:
    """Read codec and frame metadata; reject missing or unreadable files."""
    import cv2

    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise BrowserVideoError(f"Video missing or empty: {path.name}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise BrowserVideoError(f"Could not read video: {path.name}")
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * n)) & 255) for n in range(4))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (
            not math.isfinite(fps) or fps <= 0
            or frames <= 0 or width <= 0 or height <= 0
        ):
            raise BrowserVideoError(f"Invalid video metadata: {path.name}")

        return VideoProperties(
            codec=codec, fps=fps, frames=frames, width=width, height=height
        )
    finally:
        capture.release()


def _preserves_timeline(
    source: VideoProperties, converted: VideoProperties
) -> bool:
    # One frame tolerance covers rare container duration rounding.
    return (
        abs(source.frames - converted.frames) <= 1
        and abs(source.fps - converted.fps) <= max(0.05, source.fps * 0.001)
        and source.width == converted.width
        and source.height == converted.height
    )


def _ffmpeg_executable(explicit: str | Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError) as exc:
        raise BrowserVideoError(
            "FFmpeg unavailable; install with: python -m pip install imageio-ffmpeg"
        ) from exc


def ensure_browser_video(
    source: Path,
    *,
    destination: Path | None = None,
    ffmpeg_exe: str | Path | None = None,
    timeout_seconds: float = 180.0,
) -> BrowserVideoResult:
    """Create a verified H.264/yuv420p MP4 atomically, or reuse the valid copy.

    Raises BrowserVideoError on failure. Callers should catch it when browser
    playback is a noncritical enrichment of a completed basketball analysis.
    """
    source = Path(source).resolve()
    destination = (
        Path(destination).resolve()
        if destination is not None
        else source.with_name(BROWSER_FILENAME)
    )
    if source == destination:
        raise ValueError("Browser output cannot overwrite the analyzed video")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 1 <= timeout_seconds <= 600
    ):
        raise ValueError("Timeout must be between 1 and 600 seconds")

    source_meta = inspect_video(source)

    if destination.is_file():
        try:
            existing_meta = inspect_video(destination)
            if existing_meta.browser_compatible and _preserves_timeline(
                source_meta, existing_meta
            ):
                return BrowserVideoResult(destination, False, existing_meta)
        except BrowserVideoError:
            pass  # Regenerate an invalid or truncated browser copy.

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".bullseye_h264_", suffix=".mp4", dir=destination.parent
    )
    import os
    os.close(fd)
    temporary = Path(temporary_name)

    try:
        command = [
            _ffmpeg_executable(ffmpeg_exe),
            "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y",
            "-i", str(source),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=float(timeout_seconds), check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BrowserVideoError("H.264 conversion timed out") from exc
        except OSError as exc:
            raise BrowserVideoError("Unable to launch FFmpeg") from exc

        if completed.returncode != 0:
            # Keep the error concise. Do not include arbitrary FFmpeg output
            # or platform-specific local paths in client-facing API results.
            raise BrowserVideoError(
                f"FFmpeg H.264 conversion failed (exit {completed.returncode})"
            )

        converted_meta = inspect_video(temporary)
        if not converted_meta.browser_compatible:
            raise BrowserVideoError(
                f"Expected H.264 output; got {converted_meta.codec}"
            )
        if not _preserves_timeline(source_meta, converted_meta):
            raise BrowserVideoError(
                "Converted video frame count, FPS or dimensions differ"
            )

        # Only publish a fully verified MP4, never a partial file.
        temporary.replace(destination)
        return BrowserVideoResult(destination, True, converted_meta)

    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    try:
        result = ensure_browser_video(
            args.input, destination=args.output, timeout_seconds=args.timeout
        )
    except (ValueError, OSError, BrowserVideoError) as exc:
        parser.error(str(exc))
    print("Browser video:", result.path)
    print("Status:", "created" if result.created else "already_available")
    print(
        f"Codec: {result.properties.codec} | FPS: {result.properties.fps:.3f}"
        f" | Frames: {result.properties.frames} | "
        f"Size: {result.properties.width}x{result.properties.height}"
    )


if __name__ == "__main__":
    main()
