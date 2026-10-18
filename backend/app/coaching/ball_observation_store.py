"""Phase 5.4: preserve *detector-observed* basketball positions.

Call with two independent snapshots:
  raw_frames: BallAnalyzer frames BEFORE remove_ball_outliers()
  filtered_frames: frames AFTER remove_ball_outliers(), BEFORE interpolation
  shots: ShotTracker result["shots"] (may be computed after interpolation)

No output is a technique grade or a validated ball/hand contact measurement.
This module deliberately refuses already-interpolated input.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


_REQUIRED_BALL_FIELDS = ("x", "y", "x1", "y1", "x2", "y2", "confidence")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_ball(ball: Any, label: str) -> dict:
    if not isinstance(ball, dict):
        raise ValueError(f"{label} must be a ball detection object")
    values = {field: _finite_number(ball.get(field), f"{label}.{field}")
              for field in _REQUIRED_BALL_FIELDS}

    if not (0.0 < values["confidence"] <= 1.0):
        raise ValueError(f"{label}.confidence must be in (0, 1]")
    if not (values["x1"] < values["x2"] and values["y1"] < values["y2"]):
        raise ValueError(f"{label} has an invalid bounding box")
    if values["x"] < 0 or values["y"] < 0:
        raise ValueError(f"{label} center cannot be negative")
    if abs(values["x"] - (values["x1"] + values["x2"]) / 2) > 1e-4:
        raise ValueError(f"{label}.x does not match bounding box center")
    if abs(values["y"] - (values["y1"] + values["y2"]) / 2) > 1e-4:
        raise ValueError(f"{label}.y does not match bounding box center")
    return values


def _validated_frames(frames: Any, label: str) -> list[dict]:
    if not isinstance(frames, list):
        raise ValueError(f"{label} must be a list")
    result = []
    for index, item in enumerate(frames, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        if _positive_integer(item.get("frame"), f"{label}[{index}].frame") != index:
            raise ValueError(f"{label} must have consecutive 1-based frame numbers")
        # A false interpolated flag is allowed; a true one never is.
        if item.get("interpolated") is True:
            raise ValueError(
                f"{label} frame {index} contains interpolation; "
                "capture snapshots before interpolate_ball_gaps()"
            )
        detected = item.get("ball_detected")
        if not isinstance(detected, bool):
            raise ValueError(f"{label} frame {index} ball_detected must be boolean")
        ball = item.get("ball")
        if detected:
            validated = _validated_ball(ball, f"{label} frame {index} ball")
        else:
            if ball is not None:
                raise ValueError(f"{label} frame {index} has ball without detection")
            validated = None
        result.append({"frame": index, "ball_detected": detected, "ball": validated})
    return result


def _validated_shots(shots: Any, total_frames: int) -> list[dict]:
    if not isinstance(shots, list):
        raise ValueError("shots must be a list")
    validated = []
    numbers = set()
    ranges = []
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise ValueError(f"shots[{index}] must be an object")
        number = _positive_integer(shot.get("shot_number"), "shot_number")
        start = _positive_integer(shot.get("start_frame"), "start_frame")
        release = _positive_integer(shot.get("release_frame"), "release_frame")
        end = _positive_integer(shot.get("end_frame"), "end_frame")
        if not start <= release <= end <= total_frames:
            raise ValueError(f"shot {number} has invalid frame bounds")
        if number in numbers:
            raise ValueError(f"duplicate shot number {number}")
        if any(start <= prior_end and prior_start <= end
               for prior_start, prior_end in ranges):
            raise ValueError("overlapping shot windows would duplicate ball evidence")
        numbers.add(number)
        ranges.append((start, end))
        validated.append({
            "shot_number": number,
            "arm": shot.get("arm"),
            "start_frame": start,
            "release_frame": release,
            "end_frame": end,
        })
    return validated


def build_ball_observation_report(
    raw_frames: list[dict],
    filtered_frames: list[dict],
    shots: list[dict],
    *,
    fps: float,
) -> dict:
    """Build a JSON-serializable record from raw and outlier-filtered frames.

    Only detections appearing unchanged in BOTH snapshots become coaching
    candidate observations. Raw detections rejected by the existing filter
    are counted but never used as candidate ball coordinates.
    """
    fps = _finite_number(fps, "fps")
    if fps <= 0:
        raise ValueError("fps must be positive")
    raw = _validated_frames(raw_frames, "raw_frames")
    filtered = _validated_frames(filtered_frames, "filtered_frames")
    if len(raw) != len(filtered):
        raise ValueError("raw/filtered frame counts disagree")
    checked_shots = _validated_shots(shots, len(raw))

    observations = []
    rejected_frames = []
    raw_detected = 0
    for original, cleaned in zip(raw, filtered):
        frame = original["frame"]
        raw_yes = original["ball_detected"]
        cleaned_yes = cleaned["ball_detected"]
        if cleaned_yes and not raw_yes:
            raise ValueError(f"filtered frame {frame} fabricated a detection")
        if raw_yes:
            raw_detected += 1
        if raw_yes and not cleaned_yes:
            rejected_frames.append(frame)
        if cleaned_yes:
            if cleaned["ball"] != original["ball"]:
                raise ValueError(f"filtered frame {frame} modified detector data")
            ball = original["ball"]
            observations.append({
                "frame": frame,
                "time_seconds": round((frame - 1) / fps, 6),
                "x": ball["x"],
                "y": ball["y"],
                "x1": ball["x1"],
                "y1": ball["y1"],
                "x2": ball["x2"],
                "y2": ball["y2"],
                "confidence": ball["confidence"],
                "source": "BallAnalyzer",
                "provenance": "direct_detector_output",
                "accepted_by_existing_outlier_filter": True,
                "interpolated": False,
            })

    per_shot = []
    for shot in checked_shots:
        candidates = [
            deepcopy(observation) for observation in observations
            if shot["start_frame"] <= observation["frame"] <= shot["end_frame"]
        ]
        release = next(
            (deepcopy(observation) for observation in candidates
             if observation["frame"] == shot["release_frame"]),
            None,
        )
        before_release = [
            observation for observation in candidates
            if observation["frame"] <= shot["release_frame"]
        ]
        per_shot.append({
            **shot,
            "ball_positions": candidates,
            "observed_ball_frame_count": len(candidates),
            "observed_pre_release_ball_frame_count": len(before_release),
            "ball_positions_signal_available": len(before_release) >= 2,
            "release_ball_position": release,
            "ball_position_signal_available": release is not None,
            "release_ball_observed": release is not None,
            "rejected_outlier_frames": [
                frame for frame in rejected_frames
                if shot["start_frame"] <= frame <= shot["end_frame"]
            ],
            "assessment": "detector_observations_only",
        })

    return {
        "schema_version": "1.0",
        "method": "raw_and_outlier_filtered_detector_snapshot_comparison",
        "source": "BallAnalyzer",
        "fps": fps,
        "total_frames": len(raw),
        "shot_count": len(per_shot),
        "detector_observation_count": raw_detected,
        "accepted_observation_count": len(observations),
        "rejected_outlier_count": len(rejected_frames),
        "rejected_outlier_frames": rejected_frames,
        "not_detected_frame_count": len(raw) - raw_detected,
        "ball_detection_source_present": True,
        "ball_positions": observations,
        "shots": per_shot,
        "intended_form_match": "not_assessed",
        "limitations": [
            "A detector output is a candidate observation, not proof of a correct detection.",
            "Rejected outliers are not used as coaching ball positions.",
            "Interpolated positions are never accepted as detector evidence.",
            "Frame coverage and confidence do not establish ball-hand contact or shooting-form correctness.",
            "Shot windows are shot-tracker boundaries, not validated biomechanics phases.",
        ],
    }


def save_report(report: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


__all__ = ["build_ball_observation_report", "save_report"]
