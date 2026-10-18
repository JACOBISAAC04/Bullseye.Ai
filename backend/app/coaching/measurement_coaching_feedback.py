"""Phase 5.8: attach observed, frame-numbered measurements to safe review prompts.

Usage from Bullseye/backend:
    python -m app.coaching.measurement_coaching_feedback \
      --temporal data/analyzed/phase4_temporal_form.json \
      --mechanics data/analyzed/phase5_mechanics_with_ball_video3.json \
      --review data/analyzed/phase5_coaching_review_video3.json \
      --output data/analyzed/phase5_measured_review_video3.json

This module NEVER determines whether a shooter follows an ideal form.
No angle targets, drills, forced judgments, or interpolation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value):
    return (
        float(value)
        if type(value) in (int, float) and math.isfinite(value)
        else None
    )


def _frame(value):
    return type(value) is int and value > 0


def _index(report, label):
    if not isinstance(report, dict) or not isinstance(report.get("shots"), list):
        raise ValueError(f"{label} needs shots list")
    shots = report["shots"]
    if type(report.get("shot_count")) is not int or report["shot_count"] != len(shots):
        raise ValueError(f"{label} shot_count mismatch")
    result = {}
    for shot in shots:
        if not isinstance(shot, dict) or not _frame(shot.get("shot_number")):
            raise ValueError(f"{label} has invalid shot number")
        number = shot["shot_number"]
        if number in result:
            raise ValueError(f"{label} duplicate shot number")
        result[number] = shot
    return result


def _point(value):
    if not isinstance(value, dict):
        return None
    x, y = _finite(value.get("x")), _finite(value.get("y"))
    if x is None or y is None or x < 0 or y < 0:
        return None
    return {"x": round(x, 3), "y": round(y, 3)}


def _observed_series(signal, source, lo, hi):
    """Use data samples, never an availability flag alone."""
    if not isinstance(signal, dict) or signal.get("source") != source:
        return []
    samples = signal.get("samples")
    if not isinstance(samples, list):
        return []
    result = []
    used = set()
    for item in samples:
        if not isinstance(item, dict) or not _frame(item.get("frame")):
            continue
        frame = item["frame"]
        if frame in used or not lo <= frame <= hi:
            continue
        used.add(frame)
        result.append(item)
    return sorted(result, key=lambda entry: entry["frame"])


def _temporal_angle(segment, key):
    if not isinstance(segment, dict) or segment.get("quality") != "sufficient_for_descriptive_timing":
        return None
    samples = segment.get("samples")
    if not isinstance(samples, list):
        return None
    vals, seen = [], set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("complete_pose") is not True:
            continue
        frame = sample.get("frame")
        if not _frame(frame) or frame in seen:
            continue
        seen.add(frame)
        angle = sample.get("angles_deg")
        value = _finite(angle.get(key)) if isinstance(angle, dict) else None
        if value is not None and 0 <= value <= 180:
            vals.append((frame, value))
    vals.sort()
    if len(vals) < 2:
        return None
    a, b = vals[0], vals[-1]
    return {
        "signal": key,
        "unit": "degrees_2d",
        "start_frame": a[0],
        "start_value": round(a[1], 3),
        "end_frame": b[0],
        "end_value": round(b[1], 3),
        "observed_endpoint_difference": round(b[1] - a[1], 3),
        "observed_samples": len(vals),
        "interpretation": "Observed 2D endpoint change; not an ideal-angle comparison or proof of sequencing.",
    }


def _position_displacement(signal, joint, lo, hi):
    samples = _observed_series(signal, "analysis.frames", lo, hi)
    valid = []
    for item in samples:
        value = item.get("value")
        p = _point(value.get(joint)) if isinstance(value, dict) and joint else _point(value)
        if p is not None:
            valid.append((item["frame"], p))
    if len(valid) < 2:
        return None
    (fa, a), (fb, b) = valid[0], valid[-1]
    return {
        "signal": "position_displacement",
        "joint": joint or "hip_center",
        "unit": "uncalibrated_2d_pixels",
        "start_frame": fa,
        "end_frame": fb,
        "delta_x_pixels": round(b["x"] - a["x"], 3),
        "delta_y_pixels": round(b["y"] - a["y"], 3),
        "observed_samples": len(valid),
        "interpretation": "Image-plane displacement only; cannot establish physical balance or body lean.",
    }


def _ball_path(mechanics, lo, hi):
    signals = mechanics.get("signals", {})
    if not isinstance(signals, dict):
        return None
    ball = signals.get("ball_positions")
    if "accepted direct BallAnalyzer detections" not in mechanics.get("_ball_origin", ""):
        return None
    samples = _observed_series(ball, "ball_analysis.frames", lo, hi)
    vals = []
    for item in samples:
        v = item.get("value")
        p = _point(v)
        confidence = _finite(v.get("confidence")) if isinstance(v, dict) else None
        if p and confidence is not None and 0 < confidence <= 1:
            vals.append((item["frame"], p, confidence))
    if len(vals) < 2:
        return None
    a, b = vals[0], vals[-1]
    return {
        "signal": "ball_path_endpoints",
        "unit": "uncalibrated_2d_pixels",
        "start_frame": a[0],
        "start_position": a[1],
        "start_detector_confidence": round(a[2], 4),
        "end_frame": b[0],
        "end_position": b[1],
        "end_detector_confidence": round(b[2], 4),
        "delta_x_pixels": round(b[1]["x"] - a[1]["x"], 3),
        "delta_y_pixels": round(b[1]["y"] - a[1]["y"], 3),
        "observed_samples": len(vals),
        "interpretation": "Accepted detector centers; no missing ball frames were filled. Not a calibrated ball path or release determination.",
    }


def _ball_wrist(mechanics, lo, hi):
    evidence = mechanics.get("ball_body_evidence")
    if not isinstance(evidence, dict) or evidence.get("units") != "uncalibrated_2d_pixels":
        return None
    if "accepted direct BallAnalyzer detections" not in mechanics.get("_ball_origin", ""):
        return None
    ball_signal = mechanics.get("signals", {}).get("ball_positions", {})
    approved = {}
    for item in _observed_series(ball_signal, "ball_analysis.frames", lo, hi):
        v = item.get("value")
        p = _point(v)
        c = _finite(v.get("confidence")) if isinstance(v, dict) else None
        if p is not None and c is not None and 0 < c <= 1:
            approved[item["frame"]] = p
    vals = []
    for item in evidence.get("samples", []):
        if not isinstance(item, dict) or not _frame(item.get("frame")):
            continue
        n = item["frame"]
        ball = _point(item.get("ball_center"))
        wrist = _point(item.get("shooting_wrist"))
        if n not in approved or ball != approved[n] or wrist is None:
            continue
        d = math.hypot(ball["x"] - wrist["x"], ball["y"] - wrist["y"])
        vals.append((n, round(d, 3)))
    vals.sort()
    if not vals:
        return None
    # Duplicate frames may not be counted multiple times.
    vals = list({frame: (frame, dist) for frame, dist in vals}.values())
    vals.sort()
    return {
        "signal": "same_frame_ball_to_wrist",
        "unit": "uncalibrated_2d_pixels",
        "first_frame": vals[0][0],
        "first_distance_pixels": vals[0][1],
        "last_frame": vals[-1][0],
        "last_distance_pixels": vals[-1][1],
        "same_frame_matches": len(vals),
        "interpretation": "Observed ball center and wrist landmark at the same frame; cannot establish hand contact or grip.",
    }


def _phase_angles(mechanics):
    signal = mechanics.get("signals", {}).get("shot_metrics")
    if not isinstance(signal, dict) or signal.get("source") != "metrics.shots":
        return None
    sample = signal.get("sample")
    phases = sample.get("phases") if isinstance(sample, dict) else None
    if not isinstance(phases, dict):
        return None
    rows = []
    for phase in ("set", "release"):
        values = phases.get(phase)
        if not isinstance(values, dict):
            continue
        clean = {
            k: round(v, 3)
            for k in ("elbow_angle", "hip_angle", "knee_angle")
            if (v := _finite(values.get(k))) is not None and 0 <= v <= 180
        }
        if clean:
            rows.append({"phase": phase, "angles_degrees_2d": clean})
    if not rows:
        return None
    return {
        "signal": "phase_angles",
        "unit": "degrees_2d",
        "observations": rows,
        "interpretation": "Measured joint angles, not a diagnosis or a pass/fail threshold.",
    }


class MeasurementCoachingFeedback:
    """Join three saved reports; never produce an automated technique correction."""

    def generate(self, temporal: dict, mechanics: dict, review: dict) -> dict:
        ts = _index(temporal, "temporal")
        ms = _index(mechanics, "mechanics")
        rs = _index(review, "review")
        if not (ts.keys() == ms.keys() == rs.keys()):
            raise ValueError("Shot identifiers do not match")
        for label, report in (("temporal", temporal), ("mechanics", mechanics), ("review", review)):
            if report.get("intended_form_match") != "not_assessed":
                raise ValueError(f"{label}: technique judgments are not allowed")
        if review.get("official_endorsement") is not False or review.get("corrections_generated") != 0:
            raise ValueError("Review must be non-endorsed and non-corrective")
        if mechanics.get("ball_evidence_origin") is not None:
            origin = mechanics["ball_evidence_origin"]
        else:
            origin = ""
        result = []
        for number in sorted(ts):
            t, m, r = ts[number], ms[number], rs[number]
            for key in ("arm", "start_frame", "release_frame", "end_frame"):
                if t.get(key) != m.get(key):
                    raise ValueError(f"Shot {number}: {key} mismatch")
            lo, release, end = (t[k] for k in ("start_frame", "release_frame", "end_frame"))
            if any(not _frame(n) for n in (lo, release, end)) or not lo <= release <= end:
                raise ValueError("Invalid shot boundaries")
            if not isinstance(r.get("review_candidates"), list) or r.get("review_candidate_count") != len(r["review_candidates"]):
                raise ValueError("Review candidate count mismatch")
            segment = t.get("segments", {}).get("set_to_release")
            ready = isinstance(segment, dict) and segment.get("quality") == "sufficient_for_descriptive_timing"
            valid_ids = set()
            candidates = []
            for candidate in r["review_candidates"]:
                if not isinstance(candidate, dict) or candidate.get("review_status") != "manual_review_candidate":
                    raise ValueError("Only manual-review candidates are accepted")
                pid = candidate.get("principle_id")
                if not isinstance(pid, str) or pid in valid_ids:
                    raise ValueError("Invalid or duplicated coaching principle")
                valid_ids.add(pid)
                if candidate.get("intended_form_match") != "not_assessed":
                    raise ValueError("Candidate attempts to grade technique")
                if candidate.get("source_support_status") != "limited_published_text_support":
                    raise ValueError("Review candidate lacks qualified published-text source support")
                if candidate.get("original_video_timestamp_verified") is True:
                    raise ValueError("Unverified video timestamps cannot be upgraded here")
                observations = []
                enriched = {**m, "_ball_origin": origin}
                if ready:
                    if pid == "posture_setup":
                        obs = _temporal_angle(segment, "hip_angle")
                        if obs:
                            observations.append(obs)
                    elif pid == "hip_loading":
                        for joint in ("hip_angle", "knee_angle"):
                            obs = _temporal_angle(segment, joint)
                            if obs:
                                observations.append(obs)
                    elif pid == "ground_balance":
                        obs = _position_displacement(
                            m.get("signals", {}).get("hip_center_series"),
                            None, lo, release
                        )
                        if obs:
                            observations.append(obs)
                    elif pid == "ball_setup":
                        obs = _ball_wrist(enriched, lo, release)
                        if obs:
                            observations.append(obs)
                    elif pid == "ball_path":
                        obs = _ball_path(enriched, lo, release)
                        if obs:
                            observations.append(obs)
                    elif pid == "individualized_feedback":
                        obs = _phase_angles(m)
                        if obs:
                            observations.append(obs)
                candidates.append({
                    "principle_id": pid,
                    "title": candidate.get("title"),
                    "review_question": candidate.get("review_question"),
                    "source_urls": candidate.get("source_urls", []),
                    "source_support_status": candidate.get("source_support_status"),
                    "status": (
                        "measured_descriptive_review" if observations else
                        "no_supported_measurement"
                    ),
                    "observations": observations,
                    "notes": (
                        "These observations are 2D detector/pose estimates. "
                        "They are descriptive and do not establish shooting correctness."
                    ),
                    "intended_form_match": "not_assessed",
                })
            result.append({
                "shot_number": number,
                "timing_quality": segment.get("quality") if isinstance(segment, dict) else "missing",
                "review_candidate_count": len(candidates),
                "review_candidates": candidates,
                "intended_form_match": "not_assessed",
            })
        return {
            "schema_version": "1.0",
            "method": "observed_measurements_with_source_linked_review_prompts",
            "shot_count": len(result),
            "shots": result,
            "intended_form_match": "not_assessed",
            "official_endorsement": False,
            "corrections_generated": 0,
            "limitations": [
                "Measures are perspective-dependent 2D estimates, not physical biomechanics.",
                "Measured deltas between observed endpoints do not establish a continuous movement sequence.",
                "Missing pose and ball frames are never filled.",
                "These source-linked review questions do not prove adherence to a coaching method.",
                "No universal ideal angles, drills, or automatic form-correction verdicts are generated.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.temporal, args.mechanics, args.review)
    ]
    result = MeasurementCoachingFeedback().generate(*data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Shots:", result["shot_count"])
    for shot in result["shots"]:
        print(
            f"Shot {shot['shot_number']}: {sum(c['status'] == 'measured_descriptive_review' for c in shot['review_candidates'])}"
            f"/{shot['review_candidate_count']} prompts include measured evidence"
        )
    print("Intended-form match:", result["intended_form_match"])
    print("Corrections generated:", result["corrections_generated"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
