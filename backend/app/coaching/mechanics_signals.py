"""Extract observed 2D shooting-mechanics signals from saved Bullseye API JSON.

Run from backend:
    python app/coaching/mechanics_signals.py data/analyzed/hoop5_video3.json \
        --output data/analyzed/phase5_mechanics_video3.json

No inference, no interpolation, no technique grading, and no new video processing.
Only explicitly observed landmarks and (when provided) detector-origin ball frames
are used. Shot trajectories are intentionally NOT treated as raw ball detections.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SERIES_JOINTS = {
    "wrist_positions": ("wrist",),
    "elbow_positions": ("elbow",),
    "shoulder_positions": ("shoulder",),
    "hip_positions": ("hip",),
    "shoulder_hip_positions": ("shoulder", "hip"),
    "ankle_positions": ("left_ankle", "right_ankle"),
    "hip_center_series": ("left_hip", "right_hip"),
}
SIGNAL_NAMES = (
    "wrist_positions", "wrist_position", "elbow_positions",
    "shoulder_positions", "shoulder_hip_positions", "hip_positions",
    "hip_center_series", "ankle_positions", "ball_positions",
    "ball_position", "shot_metrics",
)
MIN_ADJACENT_PAIRS = 1   # Visibility condition only; NOT a timing-quality gate.


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _int_frame(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _point(frame, name):
    point = frame.get(name)
    if not isinstance(point, dict):
        return None
    x, y = _finite(point.get("x")), _finite(point.get("y"))
    return {"x": round(x, 3), "y": round(y, 3)} if x is not None and y is not None else None


def _index_frames(frames, name):
    if not isinstance(frames, list):
        raise ValueError(f"{name} must be a list")
    result = {}
    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError(f"{name} entries must be objects")
        n = _int_frame(entry.get("frame"), f"{name}[].frame")
        if n in result:
            raise ValueError(f"Duplicate frame {n} in {name}")
        result[n] = entry
    return result


def _observed_ball(frame):
    """A candidate detector observation, never an interpolated tracker point."""
    if not isinstance(frame, dict):
        return None
    if frame.get("ball_detected") is not True:
        return None
    if frame.get("interpolated") is True:
        return None
    ball = frame.get("ball")
    if not isinstance(ball, dict) or ball.get("interpolated") is True:
        return None
    confidence = _finite(ball.get("confidence"))
    p = _point(frame, "ball")
    # No confidence or zero confidence cannot establish direct detection.
    if confidence is None or not (0.0 < confidence <= 1.0) or p is None:
        return None
    return {**p, "confidence": round(confidence, 4)}


def _series(samples, source, description):
    adjacency = sum(b["frame"] == a["frame"] + 1 for a, b in zip(samples, samples[1:]))
    present = len(samples) >= 2 and adjacency >= MIN_ADJACENT_PAIRS
    return {
        "status": "observed_descriptive_only" if present else "not_observed_enough",
        "available": present,
        "source": source,
        "definition": description,
        "observed_frames": len(samples),
        "adjacent_observed_pairs": adjacency,
        "samples": samples,
    }


def _single(sample, source, description):
    return {
        "status": "observed_descriptive_only" if sample else "not_observed",
        "available": sample is not None,
        "source": source if sample else None,
        "definition": description,
        "sample": sample,
    }


def _joint_samples(pose, start, release, arm, name):
    joints = SERIES_JOINTS[name]
    collected = []
    for n in range(start, release + 1):
        frame = pose.get(n)
        if frame is None or frame.get("pose_detected") is not True:
            continue
        point_map = {}
        for joint in joints:
            identifier = joint if joint.startswith(("left_", "right_")) else f"{arm}_{joint}"
            p = _point(frame, identifier)
            if p is None:
                break
            point_map[identifier] = p
        if len(point_map) != len(joints):
            continue
        if name == "hip_center_series":
            left, right = point_map["left_hip"], point_map["right_hip"]
            value = {
                "x": round((left["x"] + right["x"]) / 2, 3),
                "y": round((left["y"] + right["y"]) / 2, 3),
            }
        else:
            value = point_map
        collected.append({"frame": n, "value": value})
    return collected


def _metrics_observation(shot):
    """Only a small, explicitly available subset, not an evaluation of quality."""
    observed = {}
    for phase in ("set", "release", "follow_through"):
        obj = shot.get(phase)
        if not isinstance(obj, dict):
            continue
        angles = {}
        for key in ("elbow_angle", "hip_angle", "knee_angle"):
            n = _finite(obj.get(key))
            if n is not None:
                angles[key] = round(n, 3)
        if angles:
            observed[phase] = angles
    return observed or None


class MechanicsSignalExtractor:
    def analyze(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a saved API response JSON object")
        analysis = payload.get("analysis")
        metrics = payload.get("metrics")
        if not isinstance(analysis, dict) or not isinstance(metrics, dict):
            raise ValueError("Expected analysis and metrics objects")
        pose = _index_frames(analysis.get("frames"), "analysis.frames")
        shots = metrics.get("shots")
        if not isinstance(shots, list):
            raise ValueError("Expected metrics.shots list")
        if "shot_count" in payload and payload["shot_count"] != len(shots):
            raise ValueError("Top-level shot_count does not match metrics.shots")

        # Raw ball detections are optional and generally absent from saved API JSON.
        # NEVER fall back to shots[].trajectory (which may contain interpolated points).
        ball_source = "ball_analysis.frames"
        ball_input = payload.get("ball_analysis")
        if ball_input is None:
            ball_index = {}
            raw_ball_present = False
        elif isinstance(ball_input, dict):
            ball_index = _index_frames(ball_input.get("frames"), ball_source)
            raw_ball_present = True
        else:
            raise ValueError("ball_analysis must be an object if supplied")

        result_shots = []
        seen_shot_numbers = set()
        for i, shot in enumerate(shots, 1):
            if not isinstance(shot, dict):
                raise ValueError("metrics.shots entries must be objects")
            arm = shot.get("arm")
            if arm not in ("left", "right"):
                raise ValueError("Each shot must declare arm='left' or 'right'")
            start = _int_frame(shot.get("start_frame"), "start_frame")
            release = _int_frame(shot.get("release_frame"), "release_frame")
            end = _int_frame(shot.get("end_frame"), "end_frame")
            if start > release or release > end:
                raise ValueError("Shot frames must satisfy start <= release <= end")
            number = _int_frame(shot.get("shot_number", i), "shot_number")
            if number in seen_shot_numbers:
                raise ValueError("Duplicate shot_number")
            seen_shot_numbers.add(number)

            signals = {}
            for name in SERIES_JOINTS:
                samples = _joint_samples(pose, start, release, arm, name)
                signals[name] = _series(
                    samples, "analysis.frames",
                    f"Observed 2D pixel coordinates during set-to-release: {name}; "
                    "not calibrated 3D movement or a biomechanical score",
                )

            release_frame = pose.get(release)
            wrist = None
            if release_frame is not None and release_frame.get("pose_detected") is True:
                wrist = _point(release_frame, f"{arm}_wrist")
            signals["wrist_position"] = _single(
                {"frame": release, **wrist} if wrist is not None else None,
                "analysis.frames",
                "Shooting-side wrist coordinates at the exact tracked release frame, 2D pixels",
            )

            ball_samples = []
            for n in range(start, release + 1):
                obs = _observed_ball(ball_index.get(n))
                if obs is not None:
                    ball_samples.append({"frame": n, "value": obs})
            signals["ball_positions"] = _series(
                ball_samples, ball_source if raw_ball_present else None,
                "Detector-observed ball centers only, set-to-release; "
                "no interpolation or assumed possession/contact",
            )
            release_ball = _observed_ball(ball_index.get(release))
            signals["ball_position"] = _single(
                {"frame": release, **release_ball} if release_ball else None,
                ball_source,
                "Detector-observed ball center exactly at tracked release frame; "
                "not proof of hand-ball separation",
            )

            observed_metrics = _metrics_observation(shot)
            signals["shot_metrics"] = _single(
                {"phases": observed_metrics} if observed_metrics else None,
                "metrics.shots",
                "Existing measured phase angles (2D) for contextual descriptive review only",
            )

            result_shots.append({
                "shot_number": number,
                "arm": arm,
                "start_frame": start,
                "release_frame": release,
                "end_frame": end,
                "signals": signals,
                "available_signals": [name for name, s in signals.items() if s["available"]],
                "intended_form_match": "not_assessed",
            })

        return {
            "schema_version": "1.0",
            "method": "observed_2d_mechanics_signal_extraction",
            "shot_count": len(result_shots),
            "shots": result_shots,
            "ball_detection_source_present": raw_ball_present,
            "official_endorsement": False,
            "intended_form_match": "not_assessed",
            "limitations": [
                "2D image coordinates are perspective-dependent and lack calibration.",
                "Missing pose points are not interpolated, smoothed, or inferred.",
                "At least two adjacent observed frames establish series availability only, not timing reliability.",
                "The Phase 4 temporal quality gate remains necessary for sequencing interpretations.",
                "Ball detections are accepted only from explicit ball_analysis.frames with positive confidence.",
                "Shots[].trajectory is ignored: tracked trajectories may contain interpolated ball positions.",
                "Even detector observations may be false positives; no tracking-confidence calibration is implied.",
                "No guide-hand contact, foot-ground contact, ball grip, or biomechanical correctness is assessed.",
                "Tracked release and end frames are algorithm outputs, not biomechanically certified events.",
                "No form assessment or corrective advice is produced.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    report = MechanicsSignalExtractor().analyze(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Shots analyzed:", report["shot_count"])
    print("Raw ball detections available:", report["ball_detection_source_present"])
    for shot in report["shots"]:
        print(f"Shot {shot['shot_number']} ({shot['arm']}): "
              f"{len(shot['available_signals'])}/{len(SIGNAL_NAMES)} signals observed")
        print("  Available:", ", ".join(shot["available_signals"]) or "none")
    print("Intended-form match:", report["intended_form_match"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
