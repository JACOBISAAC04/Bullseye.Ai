r"""
Import a free-Colab YOLO26 detection ZIP for a specific original video.

Run from anywhere:
    python tools/import_yolo26.py data/videos/video3.mp4 "C:\Downloads\Bullseye_YOLO26_video3_results.zip" --run

No local YOLO26 installation is required; this only reads the exported JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 500 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def resolve_from_backend(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = BACKEND_DIR / p
    return p.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open original video: {path}")
        info = {
            "filename": path.name,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        if min(info["width"], info["height"], info["frame_count"]) <= 0:
            raise ValueError("Video metadata is incomplete.")
        return info
    finally:
        capture.release()


def validate_report(report: dict, video: Path) -> tuple[dict, int]:
    if not isinstance(report, dict) or report.get("schema_version") not in (1, 2):
        raise ValueError("Unrecognized YOLO26 detections schema.")
    if not isinstance(report.get("video"), dict):
        raise ValueError("Missing detections.video metadata.")
    exported = report["video"]
    actual = probe_video(video)
    for field in ("filename", "width", "height", "frame_count"):
        if exported.get(field) != actual[field]:
            raise ValueError(
                f"Wrong video: {field} exported={exported.get(field)!r}; "
                f"actual={actual[field]!r}."
            )

    exported_fps = float(exported.get("fps", 0))
    if not math.isfinite(exported_fps) or abs(exported_fps - actual["fps"]) > 0.05:
        raise ValueError(
            f"FPS mismatch: exported={exported_fps}, actual={actual['fps']:.5f}"
        )

    expected_hash = exported.get("sha256")
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("Malformed video SHA-256 in YOLO26 report.")
        real_hash = sha256_file(video)
        if real_hash.lower() != expected_hash.lower():
            raise ValueError(
                "Video SHA-256 differs: detections belong to another video or version."
            )
        print("Video SHA-256 verified.")
    else:
        print(
            "NOTE: Legacy export has no SHA-256. "
            "Filename/dimensions/frame count/FPS match, but video identity is not cryptographically verified."
        )

    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != actual["frame_count"]:
        raise ValueError("Detections must contain one entry per original video frame.")
    person_frames = 0
    for index, entry in enumerate(frames, start=1):
        if not isinstance(entry, dict) or entry.get("frame") != index:
            raise ValueError(f"Frame sequence is invalid at entry {index}.")
        detections = entry.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"Detections for frame {index} must be a list.")
        any_person = False
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError(f"Malformed detection in frame {index}.")
            bbox = detection.get("bbox_xyxy")
            confidence = detection.get("confidence")
            cid = detection.get("class_id")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in bbox)
                or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or cid not in (0, 32)
            ):
                raise ValueError(f"Invalid detection fields in frame {index}.")
            any_person |= cid == 0
        person_frames += any_person
    return actual, person_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate/import YOLO26 Colab results and optionally analyze in Bullseye."
    )
    parser.add_argument("video", help="Original video path (from backend or absolute)")
    parser.add_argument("zip_file", help="YOLO26 Colab result ZIP path")
    parser.add_argument(
        "--run", action="store_true",
        help="Run existing test_shooting_analysis_video.py after successful import",
    )
    parser.add_argument(
        "--expected-shots", type=int,
        help="Optional regression assertion; valid only with --run",
    )
    args = parser.parse_args()

    if args.expected_shots is not None and not args.run:
        parser.error("--expected-shots requires --run")
    if args.expected_shots is not None and args.expected_shots < 0:
        parser.error("--expected-shots must be non-negative")

    video = resolve_from_backend(args.video)
    zip_path = resolve_from_backend(args.zip_file)
    if not video.is_file() or not zip_path.is_file():
        parser.error(f"Video or ZIP not found: {video}, {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        if "detections.json" not in archive.namelist():
            raise ValueError("ZIP has no top-level detections.json.")
        info = archive.getinfo("detections.json")
        if info.file_size > MAX_JSON_BYTES:
            raise ValueError("detections.json exceeds permitted size.")
        report = json.loads(archive.read(info).decode("utf-8"))
        metadata, person_frames = validate_report(report, video)
        output_dir = BACKEND_DIR / "data" / "analyzed" / f"yolo26_{video.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "detections.json"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if "summary.json" in archive.namelist():
            summary_info = archive.getinfo("summary.json")
            if summary_info.file_size <= 1024 * 1024:
                summary_data = json.loads(archive.read(summary_info).decode("utf-8"))
                (output_dir / "summary.json").write_text(
                    json.dumps(summary_data, indent=2), encoding="utf-8"
                )
        # Only known sample-image basenames are allowed; never extract archive paths.
        for member in archive.infolist():
            if (
                re.fullmatch(r"frame_[0-9]+\.jpg", member.filename)
                and member.file_size <= MAX_IMAGE_BYTES
            ):
                (output_dir / member.filename).write_bytes(archive.read(member))

    print(f"Imported: {json_path}")
    print(f"Original video: {video.name} ({metadata['frame_count']} frames)")
    print(f"Person candidates: {person_frames}/{metadata['frame_count']} frames")
    print("Local YOLO26 inference: not needed.")

    if args.run:
        command = [
            sys.executable,
            str(BACKEND_DIR / "test_shooting_analysis_video.py"),
            str(video),
            str(json_path),
        ]
        print("Running Bullseye shooting analysis...", flush=True)
        completed = subprocess.run(
            command, cwd=str(BACKEND_DIR),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if args.expected_shots is not None:
            matches = re.findall(r"(?m)^Shots:\s*(\d+)\s*$", completed.stdout)
            if not matches:
                raise RuntimeError("Could not locate the final 'Shots:' count.")
            actual_shots = int(matches[-1])
            if actual_shots != args.expected_shots:
                raise RuntimeError(
                    f"REGRESSION FAILURE: expected {args.expected_shots} shots, "
                    f"found {actual_shots}."
                )
            print(f"PASS: {actual_shots} expected shots.")


if __name__ == "__main__":
    main()
