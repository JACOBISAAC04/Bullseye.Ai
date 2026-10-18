"""Bullseye Phase 4.4: conservative temporal shooting-form measurements.

Input: saved Phase 3 POST /video/analyze response (analysis.frames + metrics.shots).
No model inference, ideal-angle targets, or unverified coaching judgments.

Run from backend:
    python app/coaching/temporal_form_analyzer.py \
        data/analyzed/hoop5_video3.json \
        --output data/analyzed/phase4_temporal_form.json

All angles are 2D pixel-space interior angles. Missing landmark data is NEVER
interpolated. A shot's end_frame is the tracker end, not a certified instant
of a player's biomechanical follow-through.
"""

import argparse
import json
import math
from pathlib import Path


JOINT_SETS = {
    "elbow_angle": ("shoulder", "elbow", "wrist"),
    "hip_angle": ("shoulder", "hip", "knee"),
    "knee_angle": ("hip", "knee", "ankle"),
}
REQUIRED_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")

# Engineering data-quality gates ONLY; these are not shooting-form standards.
MIN_COVERAGE = 0.80
MAX_MISSING_RUN = 3
MIN_CONSECUTIVE_PAIRS = 5


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(frame, arm, joint):
    value = frame.get(f"{arm}_{joint}")
    if not isinstance(value, dict):
        return None
    x, y = _finite(value.get("x")), _finite(value.get("y"))
    return (x, y) if x is not None and y is not None else None


def _angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denominator = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denominator <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denominator))
    return round(math.degrees(math.acos(cosine)), 3)


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _longest_missing_run(samples):
    longest = current = 0
    for sample in samples:
        if not sample["complete_pose"]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _build_sample(frame_number, frame, arm):
    if not isinstance(frame, dict) or not frame.get("pose_detected", False):
        return {
            "frame": frame_number,
            "time_seconds": None,
            "complete_pose": False,
            "angles_deg": {name: None for name in JOINT_SETS},
            "wrist_above_shoulder_px": None,
        }

    points = {joint: _point(frame, arm, joint) for joint in REQUIRED_JOINTS}
    angles = {
        name: _angle(*(points[joint] for joint in joints))
        for name, joints in JOINT_SETS.items()
    }
    shoulder, wrist = points["shoulder"], points["wrist"]
    wrist_height = (
        round(shoulder[1] - wrist[1], 3)
        if shoulder is not None and wrist is not None else None
    )
    stamp = _finite(frame.get("time"))

    return {
        "frame": frame_number,
        "time_seconds": round(stamp, 5) if stamp is not None else None,
        "complete_pose": (
            all(points.values()) and all(angle is not None for angle in angles.values())
        ),
        "angles_deg": angles,
        "wrist_above_shoulder_px": wrist_height,
    }


def _measure_series(samples, key, label):
    valid = [
        (sample["frame"], sample[label][key])
        for sample in samples if sample[label][key] is not None
    ]
    if not valid:
        return {
            "observed_frames": 0,
            "first_observed": None,
            "last_observed": None,
            "observed_endpoint_delta": None,
            "minimum": None,
            "maximum": None,
            "adjacent_observed_pairs": 0,
        }

    adjacent_pairs = sum(
        b[0] == a[0] + 1
        for a, b in zip(valid, valid[1:])
    )
    return {
        "observed_frames": len(valid),
        "first_observed": {"frame": valid[0][0], "value": valid[0][1]},
        "last_observed": {"frame": valid[-1][0], "value": valid[-1][1]},
        "observed_endpoint_delta": round(valid[-1][1] - valid[0][1], 3),
        "minimum": round(min(v for _, v in valid), 3),
        "maximum": round(max(v for _, v in valid), 3),
        "adjacent_observed_pairs": adjacent_pairs,
    }


def _phase_report(frame_by_number, start, end, arm):
    samples = [
        _build_sample(number, frame_by_number.get(number), arm)
        for number in range(start, end + 1)
    ]
    valid = sum(bool(sample["complete_pose"]) for sample in samples)
    consecutive_pairs = sum(
        a["complete_pose"] and b["complete_pose"]
        and b["frame"] == a["frame"] + 1
        for a, b in zip(samples, samples[1:])
    )
    coverage = valid / len(samples)
    gap = _longest_missing_run(samples)

    quality = (
        "sufficient_for_descriptive_timing"
        if coverage >= MIN_COVERAGE and gap <= MAX_MISSING_RUN
        and consecutive_pairs >= MIN_CONSECUTIVE_PAIRS
        else "insufficient_for_reliable_timing"
    )

    return {
        "start_frame": start,
        "end_frame": end,
        "quality": quality,
        "quality_metrics": {
            "total_frames": len(samples),
            "complete_pose_frames": valid,
            "complete_pose_coverage": round(coverage, 3),
            "longest_missing_run": gap,
            "consecutive_complete_pose_pairs": consecutive_pairs,
            "quality_gate_is_not_coaching_standard": True,
        },
        "observed_angle_series": {
            name: _measure_series(samples, name, "angles_deg")
            for name in JOINT_SETS
        },
        "wrist_vertical_position_series": _wrist_series(samples),
        "samples": samples,
    }


def _wrist_series(samples):
    wrapper = [
        {
            "frame": sample["frame"],
            "height": {"value": sample["wrist_above_shoulder_px"]},
        }
        for sample in samples
    ]
    return _measure_series(wrapper, "value", "height")


class TemporalFormAnalyzer:
    def analyze(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")

        analysis = payload.get("analysis")
        metrics = payload.get("metrics")
        if not isinstance(analysis, dict) or not isinstance(
            analysis.get("frames"), list
        ):
            raise ValueError("Expected analysis.frames from a saved Phase 3 API response")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("shots"), list):
            raise ValueError("Expected metrics.shots from a saved Phase 3 API response")

        frame_by_number = {}
        for frame in analysis["frames"]:
            if not isinstance(frame, dict):
                raise ValueError("Each pose frame must be an object")
            number = _positive_int(frame.get("frame"), "analysis.frames[].frame")
            if number in frame_by_number:
                raise ValueError(f"Duplicate pose frame number {number}")
            frame_by_number[number] = frame

        shot_reports = []
        for index, shot in enumerate(metrics["shots"], 1):
            if not isinstance(shot, dict):
                raise ValueError("Each shot must be an object")
            start = _positive_int(shot.get("start_frame"), "start_frame")
            release = _positive_int(shot.get("release_frame"), "release_frame")
            end = _positive_int(shot.get("end_frame"), "end_frame")
            if not start <= release <= end:
                raise ValueError(f"Shot {index}: invalid start/release/end order")
            arm = shot.get("arm")
            if arm not in ("left", "right"):
                raise ValueError(f"Shot {index}: missing or invalid shooting arm")

            segments = {
                "set_to_release": _phase_report(
                    frame_by_number, start, release, arm
                ),
                "release_to_shot_end": _phase_report(
                    frame_by_number, release, end, arm
                ),
            }
            shot_reports.append({
                "shot_number": shot.get("shot_number", index),
                "arm": arm,
                "start_frame": start,
                "release_frame": release,
                "end_frame": end,
                "segments": segments,
                "intended_form_match": "not_assessed",
                "sequencing_judgment": "not_assessed",
            })

        return {
            "method": "observed_2d_pose_temporal_descriptive_v1",
            "shot_count": len(shot_reports),
            "shots": shot_reports,
            "intended_form_match": "not_assessed",
            "official_endorsement": False,
            "limitations": [
                "2D single-camera angles depend on camera perspective and pose tracking.",
                "Missing landmarks and missing frames are not interpolated.",
                "Only adjacent observed frames count as consecutive movement evidence.",
                "Shot end_frame is a tracker boundary, not a validated biomechanical follow-through endpoint.",
                "Data-quality gates are engineering heuristics, not SeeMikeDunn angle targets.",
                "No coaching correctness or energy-transfer-sequencing claim is made.",
                "Interpreting technique requires source-verified coaching criteria and human validation.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Saved /video/analyze response JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = TemporalFormAnalyzer().analyze(payload)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Shots analyzed: {report['shot_count']}")
    for shot in report["shots"]:
        print(f"Shot {shot['shot_number']} ({shot['arm']}):")
        for name, segment in shot["segments"].items():
            q = segment["quality_metrics"]
            print(
                f"  {name}: {segment['quality']}, "
                f"coverage={q['complete_pose_frames']}/{q['total_frames']}, "
                f"longest_missing_run={q['longest_missing_run']}, "
                f"adjacent_pairs={q['consecutive_complete_pose_pairs']}"
            )
    print(f"Intended-form match: {report['intended_form_match']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
