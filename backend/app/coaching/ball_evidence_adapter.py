"""Join a saved Bullseye API result to its *matching* ball-observation report.

No detection, interpolation, pose inference, coaching grade, or API modification.
From Bullseye/backend:
    python -m app.coaching.ball_evidence_adapter \
      data/analyzed/phase5_api_video3.json \
      data/analyzed/phase5_ball_observations_video3.json \
      --output data/analyzed/phase5_mechanics_with_ball_video3.json

The internal ball_analysis.frames representation is reconstructed ONLY from
accepted, direct detector outputs, not from trajectories or interpolated data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from app.coaching.mechanics_signals import MechanicsSignalExtractor

_BALL_COORDS = ("x", "y", "x1", "y1", "x2", "y2", "confidence")
_SHOT_FIELDS = ("shot_number", "arm", "start_frame", "release_frame", "end_frame")


def _int(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _shots(report, label):
    if not isinstance(report, dict) or not isinstance(report.get("shots"), list):
        raise ValueError(f"{label}.shots must be a list")
    shots = report["shots"]
    if report.get("shot_count") != len(shots):
        raise ValueError(f"{label}.shot_count disagrees with shots")
    result = {}
    for s in shots:
        if not isinstance(s, dict):
            raise ValueError(f"{label}.shots must contain objects")
        n = _int(s.get("shot_number"), f"{label}.shot_number", minimum=1)
        if n in result:
            raise ValueError(f"{label} has duplicate shot_number {n}")
        if s.get("arm") not in ("left", "right"):
            raise ValueError(f"{label} has invalid shooting arm")
        a, b, c = (
            _int(s.get(k), f"{label}.{k}", minimum=1)
            for k in ("start_frame", "release_frame", "end_frame")
        )
        if not a <= b <= c:
            raise ValueError(f"{label}: invalid shot boundaries")
        result[n] = s
    return result


def _validate_observation(item, total, label):
    if not isinstance(item, dict):
        raise ValueError(f"{label} must be an object")
    n = _int(item.get("frame"), f"{label}.frame", minimum=1)
    if n > total:
        raise ValueError(f"{label}.frame exceeds total_frames")
    if (
        item.get("source") != "BallAnalyzer"
        or item.get("provenance") != "direct_detector_output"
        or item.get("accepted_by_existing_outlier_filter") is not True
        or item.get("interpolated") is not False
    ):
        raise ValueError(f"{label} is not an accepted direct detector observation")
    coords = {k: _number(item.get(k), f"{label}.{k}") for k in _BALL_COORDS}
    if not 0 < coords["confidence"] <= 1:
        raise ValueError(f"{label}: invalid confidence")
    if (
        coords["x1"] >= coords["x2"] or coords["y1"] >= coords["y2"]
        or coords["x"] < 0 or coords["y"] < 0
        or abs(coords["x"] - (coords["x1"] + coords["x2"]) / 2) > 1e-4
        or abs(coords["y"] - (coords["y1"] + coords["y2"]) / 2) > 1e-4
    ):
        raise ValueError(f"{label}: invalid bounding box and center")
    return n, coords


class BallEvidenceAdapter:
    """Safely merge direct ball evidence into the existing mechanics extractor."""

    def analyze(self, api_result: dict, observation_report: dict) -> dict:
        if not isinstance(api_result, dict) or not isinstance(observation_report, dict):
            raise ValueError("Both inputs must be JSON objects")
        pose = api_result.get("analysis")
        metrics = api_result.get("metrics")
        if not isinstance(pose, dict) or not isinstance(metrics, dict):
            raise ValueError("API report requires analysis and metrics")
        if not isinstance(pose.get("frames"), list):
            raise ValueError("API analysis.frames must be a list")

        total = _int(observation_report.get("total_frames"), "total_frames", minimum=1)
        if len(pose["frames"]) != total:
            raise ValueError("Video frame count differs between reports")
        if observation_report.get("source") != "BallAnalyzer":
            raise ValueError("Ball report must have BallAnalyzer source")
        if observation_report.get("method") != "raw_and_outlier_filtered_detector_snapshot_comparison":
            raise ValueError("Ball report must originate from raw-and-filtered snapshots")
        if observation_report.get("ball_detection_source_present") is not True:
            raise ValueError("Missing detector source evidence")
        fps = _number(observation_report.get("fps"), "fps")
        if fps <= 0:
            raise ValueError("fps must be positive")

        # Require the tracked shot identifiers, arms and boundaries to match
        # BOTH the saved API summary and saved shooting metrics.
        api_shots = _shots(api_result, "api")
        metric_shots = _shots(
            {"shots": metrics.get("shots"), "shot_count": len(metrics.get("shots", []))
             if isinstance(metrics.get("shots"), list) else None},
            "metrics",
        )
        observation_shots = _shots(observation_report, "ball")
        if not (api_shots.keys() == metric_shots.keys() == observation_shots.keys()):
            raise ValueError("Shot identifiers differ between reports")
        for n in api_shots:
            for key in _SHOT_FIELDS:
                vals = [r[n].get(key) for r in (api_shots, metric_shots, observation_shots)]
                if vals[0] != vals[1] or vals[0] != vals[2]:
                    raise ValueError(f"Shot {n}: mismatched {key}")
            if api_shots[n]["end_frame"] > total:
                raise ValueError("Shot ends outside the observation frames")

        accepted = observation_report.get("ball_positions")
        if not isinstance(accepted, list):
            raise ValueError("Ball report requires ball_positions list")
        if _int(observation_report.get("accepted_observation_count"),
                "accepted_observation_count") != len(accepted):
            raise ValueError("Accepted observation count differs from observations")
        rejected = observation_report.get("rejected_outlier_frames")
        if not isinstance(rejected, list):
            raise ValueError("Rejected frame list is required")
        rejected_set = set()
        for frame in rejected:
            n = _int(frame, "rejected_outlier_frames[]", minimum=1)
            if n > total or n in rejected_set:
                raise ValueError("Rejected frames contain invalid or repeated frame")
            rejected_set.add(n)
        if _int(observation_report.get("rejected_outlier_count"), "rejected_outlier_count") != len(rejected):
            raise ValueError("Rejected outlier count differs from frame list")
        raw_count = _int(observation_report.get("detector_observation_count"), "detector_observation_count")
        if raw_count != len(accepted) + len(rejected):
            raise ValueError("Raw count differs from accepted plus rejected")
        not_detected = _int(observation_report.get("not_detected_frame_count"), "not_detected_frame_count")
        if not_detected != total - raw_count:
            raise ValueError("Not-detected count conflicts with detector count")

        observations = {}
        previous = 0
        for item in accepted:
            n, coords = _validate_observation(item, total, "ball_positions[]")
            if n <= previous or n in rejected_set:
                raise ValueError("Accepted frames must be sorted, unique and not rejected")
            observations[n] = (item, coords)
            previous = n

        # Verify per-shot ball observations against the GLOBAL accepted list.
        # This prevents independently fabricated per-shot release detections.
        for number, shot in observation_shots.items():
            start, release, end = (shot[k] for k in
                                   ("start_frame", "release_frame", "end_frame"))
            expected = [
                item for n, (item, _) in observations.items()
                if start <= n <= end
            ]
            if shot.get("ball_positions") != expected:
                raise ValueError(f"Shot {number}: ball samples differ from global report")
            if shot.get("observed_ball_frame_count") != len(expected):
                raise ValueError(f"Shot {number}: ball observation count mismatch")
            pre = sum(item["frame"] <= release for item in expected)
            if shot.get("observed_pre_release_ball_frame_count") != pre:
                raise ValueError(f"Shot {number}: pre-release count mismatch")
            at_release = observations.get(release)
            expected_release = at_release[0] if at_release else None
            if shot.get("release_ball_position") != expected_release:
                raise ValueError(f"Shot {number}: release observation mismatch")
            if shot.get("release_ball_observed") is not (at_release is not None):
                raise ValueError(f"Shot {number}: release flag mismatch")
            if shot.get("ball_position_signal_available") is not (at_release is not None):
                raise ValueError(f"Shot {number}: release availability mismatch")
            if shot.get("ball_positions_signal_available") is not (pre >= 2):
                raise ValueError(f"Shot {number}: pre-release availability mismatch")
            expected_rejected = sorted(n for n in rejected_set if start <= n <= end)
            if sorted(shot.get("rejected_outlier_frames", [])) != expected_rejected:
                raise ValueError(f"Shot {number}: rejected frames mismatch")

        # Convert only accepted direct observations. Missing/rejected/interpolated
        # frames are not filled. This is an input contract for MechanicsSignalExtractor,
        # not a claim of independently verified ball positions.
        detector_frames = [
            {"frame": n, "ball_detected": n in observations,
             "interpolated": False,
             "ball": deepcopy(observations[n][1]) if n in observations else None}
            for n in range(1, total + 1)
        ]
        extractor_input = {
            "analysis": deepcopy(pose),
            "metrics": deepcopy(metrics),
            "shot_count": len(api_shots),
            "ball_analysis": {"frames": detector_frames},
        }
        result = MechanicsSignalExtractor().analyze(extractor_input)

        # Match accepted ball detections to observed landmarks ONLY at the
        # same video frame. All reported distances are 2D pixel quantities.
        pose_by_frame = {}
        for item in pose["frames"]:
            if not isinstance(item, dict):
                raise ValueError("Invalid pose frame")
            n = _int(item.get("frame"), "analysis.frames[].frame", minimum=1)
            if n in pose_by_frame:
                raise ValueError("Duplicate pose frame")
            pose_by_frame[n] = item

        for shot in result["shots"]:
            original = observation_shots[shot["shot_number"]]
            matches = []
            pre_release = [
                obs for obs in original["ball_positions"]
                if obs["frame"] <= shot["release_frame"]
            ]
            for obs in pre_release:
                frame = obs["frame"]
                pose_frame = pose_by_frame.get(frame)
                if not isinstance(pose_frame, dict) or pose_frame.get("pose_detected") is not True:
                    continue
                arm = shot["arm"]
                points = {}
                for joint in ("wrist", "shoulder", "hip"):
                    pt = pose_frame.get(f"{arm}_{joint}")
                    if not isinstance(pt, dict):
                        break
                    x, y = pt.get("x"), pt.get("y")
                    if (type(x) not in (int, float) or type(y) not in (int, float)
                            or not math.isfinite(x) or not math.isfinite(y)
                            or x < 0 or y < 0):
                        break
                    points[joint] = {"x": float(x), "y": float(y)}
                if len(points) != 3:
                    continue
                distance = math.hypot(
                    obs["x"] - points["wrist"]["x"],
                    obs["y"] - points["wrist"]["y"],
                )
                matches.append({
                    "frame": frame,
                    "ball_confidence": obs["confidence"],
                    "ball_center": {"x": obs["x"], "y": obs["y"]},
                    "shooting_wrist": points["wrist"],
                    "shooting_shoulder": points["shoulder"],
                    "shooting_hip": points["hip"],
                    "ball_to_wrist_distance_pixels": round(distance, 3),
                    "ball_minus_shoulder_y_pixels": round(
                        obs["y"] - points["shoulder"]["y"], 3
                    ),
                    "ball_minus_hip_y_pixels": round(
                        obs["y"] - points["hip"]["y"], 3
                    ),
                })
            exact_release = next(
                (m for m in matches if m["frame"] == shot["release_frame"]),
                None,
            )
            shot["ball_body_evidence"] = {
                "status": (
                    "observed_descriptive_only" if matches else "not_observed"
                ),
                "pre_release_ball_observations": len(pre_release),
                "same_frame_pose_ball_matches": len(matches),
                "unmatched_pre_release_ball_observations": len(pre_release) - len(matches),
                "release_frame_match": exact_release,
                "samples": matches,
                "units": "uncalibrated_2d_pixels",
                "interpretation": (
                    "Observed pixel relationships only; not hand contact, "
                    "shooting correctness, or a validated release event."
                ),
            }

        result["ball_evidence_origin"] = (
            "ball_observations.json: accepted direct BallAnalyzer detections, "
            "after existing outlier filtering; re-expressed as ball_analysis.frames"
        )
        result["limitations"].extend([
            "A source/provenance marker is reported metadata, not a guarantee every detector prediction is correct.",
            "Missing exact-release ball detections remain missing, even where tracked trajectories are interpolated.",
            "The 2D ball path is not a physical trajectory calibration or proof of hand-ball contact.",
        ])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_json", type=Path)
    parser.add_argument("ball_observations_json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    api = json.loads(args.api_json.read_text(encoding="utf-8"))
    ball = json.loads(args.ball_observations_json.read_text(encoding="utf-8"))
    result = BallEvidenceAdapter().analyze(api, ball)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Shots:", result["shot_count"])
    for shot in result["shots"]:
        pre = shot["signals"]["ball_positions"]
        release = shot["signals"]["ball_position"]
        print(
            f"Shot {shot['shot_number']}: "
            f"{pre['observed_frames']} pre-release detector observations, "
            f"ball_positions available={pre['available']}, "
            f"exact release observed={release['available']}"
        )
    print("Form verdict:", result["intended_form_match"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
