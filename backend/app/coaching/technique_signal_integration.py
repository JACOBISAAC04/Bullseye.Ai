"""Integrate observed mechanics signals into source-linked technique evidence reviews.

Input: Phase 4 temporal JSON + Phase 5 mechanics JSON from the SAME analyzed video.
No video inference; no model calls; no corrections or automatic form grading.
The existing validator remains unchanged and is used for catalog/source checks.

Run from Bullseye/backend:
  python app/coaching/technique_signal_integration.py \\
      data/analyzed/phase4_temporal_form.json \\
      data/analyzed/phase5_mechanics_video3.json \\
      --output data/analyzed/phase5_integrated_video3.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from app.coaching.technique_validator import TechniqueEvidenceValidator

TIMING_OK = "sufficient_for_descriptive_timing"
CONTEXT_PHASES = frozenset({
    "training", "practice", "moving_shot", "catch_and_shoot", "free_throw"
})
MECHANICS_SERIES = {
    "wrist_positions": ("wrist",),
    "elbow_positions": ("elbow",),
    "shoulder_positions": ("shoulder",),
    "hip_positions": ("hip",),
    "shoulder_hip_positions": ("shoulder", "hip"),
    "ankle_positions": ("left_ankle", "right_ankle"),
    "hip_center_series": (),  # Direct midpoint x/y sample
    "ball_positions": (),    # Detector-observed ball center with confidence
}
MECHANICS_SINGLE = frozenset({"wrist_position", "ball_position", "shot_metrics"})
MECHANICS_NAMES = frozenset(MECHANICS_SERIES) | MECHANICS_SINGLE


def _finite(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _xy(point):
    return (
        isinstance(point, dict)
        and _finite(point.get("x"))
        and _finite(point.get("y"))
    )


def _sample_is_valid(signal_name, item, arm):
    if not isinstance(item, dict) or not isinstance(item.get("value"), dict):
        return False
    value = item["value"]
    if signal_name == "hip_center_series":
        return _xy(value)
    if signal_name == "ball_positions":
        confidence = value.get("confidence")
        return _xy(value) and _finite(confidence) and 0 < confidence <= 1
    if signal_name == "ankle_positions":
        expected = ("left_ankle", "right_ankle")
    else:
        expected = tuple(f"{arm}_{part}" for part in MECHANICS_SERIES[signal_name])
    return all(_xy(value.get(key)) for key in expected)


def _mechanics_available(name, shot, ball_source_present):
    """Validate detector/pose provenance and observation shape, not just `available`."""
    signal = shot.get("signals", {}).get(name)
    if not isinstance(signal, dict) or signal.get("available") is not True:
        return False

    arm = shot["arm"]
    start, release = shot["start_frame"], shot["release_frame"]

    if name in MECHANICS_SERIES:
        expected_source = (
            "ball_analysis.frames" if name == "ball_positions" else "analysis.frames"
        )
        if signal.get("source") != expected_source:
            return False
        if name == "ball_positions" and not ball_source_present:
            return False
        samples = signal.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            return False
        frames = [s.get("frame") for s in samples if isinstance(s, dict)]
        if (
            len(frames) != len(samples)
            or any(type(n) is not int or n < start or n > release for n in frames)
            or frames != sorted(set(frames))
        ):
            return False
        adjacent = sum(b == a + 1 for a, b in zip(frames, frames[1:]))
        if (
            adjacent < 1
            or signal.get("observed_frames") != len(samples)
            or signal.get("adjacent_observed_pairs") != adjacent
        ):
            return False
        return all(_sample_is_valid(name, sample, arm) for sample in samples)

    sample = signal.get("sample")
    if not isinstance(sample, dict):
        return False
    if name == "wrist_position":
        return (
            signal.get("source") == "analysis.frames"
            and sample.get("frame") == release
            and _xy(sample)
        )
    if name == "ball_position":
        conf = sample.get("confidence")
        return (
            ball_source_present
            and signal.get("source") == "ball_analysis.frames"
            and sample.get("frame") == release
            and _xy(sample)
            and _finite(conf) and 0 < conf <= 1
        )
    if name == "shot_metrics":
        phases = sample.get("phases")
        if signal.get("source") != "metrics.shots" or not isinstance(phases, dict):
            return False
        return any(
            isinstance(values, dict)
            and any(_finite(values.get(a)) for a in ("elbow_angle", "hip_angle", "knee_angle"))
            for values in phases.values()
        )
    return False


def _indexed_shots(report, label):
    shots = report.get("shots")
    if not isinstance(shots, list) or report.get("shot_count") != len(shots):
        raise ValueError(f"{label}: shot_count must match shots array")
    out = {}
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError(f"{label}: each shot must be an object")
        n = shot.get("shot_number")
        if type(n) is not int or n <= 0 or n in out:
            raise ValueError(f"{label}: shot_number must be unique positive integer")
        if shot.get("arm") not in ("left", "right"):
            raise ValueError(f"{label}: shooting arm must be left or right")
        a, b, c = (shot.get(k) for k in ("start_frame", "release_frame", "end_frame"))
        if any(type(v) is not int or v < 1 for v in (a, b, c)) or not a <= b <= c:
            raise ValueError(f"{label}: invalid shot frame boundaries")
        out[n] = shot
    return out


class TechniqueSignalIntegrator:
    """Combine two saved reports conservatively; never grade technique."""

    def __init__(self, catalog_path=None, catalog=None):
        self.validator = TechniqueEvidenceValidator(catalog_path=catalog_path, catalog=catalog)
        self.by_principle = {p["id"]: p for p in self.validator.catalog["principles"]}

    def analyze(self, temporal_report, mechanics_report):
        if not isinstance(temporal_report, dict) or not isinstance(mechanics_report, dict):
            raise ValueError("Both input reports must be JSON objects")

        temporal = _indexed_shots(temporal_report, "temporal")
        mechanics = _indexed_shots(mechanics_report, "mechanics")
        if temporal.keys() != mechanics.keys():
            raise ValueError("Shot numbers differ between temporal and mechanics reports")
        for n in temporal:
            t, m = temporal[n], mechanics[n]
            for key in ("arm", "start_frame", "release_frame", "end_frame"):
                if t[key] != m[key]:
                    raise ValueError(f"Shot {n}: {key} differs between input reports")
            if not isinstance(m.get("signals"), dict):
                raise ValueError(f"Shot {n}: missing mechanics signals object")

        ball_source_present = mechanics_report.get("ball_detection_source_present")
        if type(ball_source_present) is not bool:
            raise ValueError("Mechanics report must declare ball_detection_source_present")

        baseline = self.validator.analyze(temporal_report)
        result_shots = []
        for shot_result in baseline["shots"]:
            n = shot_result["shot_number"]
            mechanical = mechanics[n]
            segment = temporal[n]["segments"]["set_to_release"]
            quality = segment["quality"]
            assessments = []
            for assessment in shot_result["assessments"]:
                entry = self.by_principle[assessment["principle_id"]]
                # The original validator has already checked temporal signals
                # and validated the source catalog. Only resolve missing
                # signals that are actually available in mechanics.
                previous_missing = assessment["missing_signals"]
                missing = [
                    signal for signal in previous_missing
                    if signal not in MECHANICS_NAMES
                    or not _mechanics_available(
                        signal, mechanical, ball_source_present
                    )
                ]
                resolved = [s for s in previous_missing if s not in missing]
                source_reviewed = assessment["source_verified_for_review"]
                context_unknown = entry["shot_phase"] in CONTEXT_PHASES

                if context_unknown:
                    status = "shot_context_not_established"
                elif not assessment["required_signals"] or missing:
                    status = "missing_measurements"
                elif quality != TIMING_OK:
                    status = "insufficient_timing_evidence"
                elif not source_reviewed:
                    status = "source_verification_required"
                else:
                    status = "descriptive_review_possible"

                updated = dict(assessment)
                updated.update({
                    "status": status,
                    "missing_signals": missing,
                    "newly_available_signals": resolved,
                    "review_question": (
                        entry.get("review_question")
                        if status == "descriptive_review_possible" else None
                    ),
                    "explanation": {
                        "descriptive_review_possible":
                            "Observed signals and an eligible source support descriptive review, not a form grade.",
                        "source_verification_required":
                            "Observed signals exist, but source details need direct verification.",
                        "insufficient_timing_evidence":
                            "The set-to-release segment fails the existing temporal evidence gate.",
                        "missing_measurements":
                            "Required signals are missing or not reliably observed; no data was inferred.",
                        "shot_context_not_established":
                            "The video's shooting or drill context is unknown.",
                    }[status],
                })
                assessments.append(updated)

            result_shots.append({
                **{k: shot_result[k] for k in ("shot_number", "arm", "measurements")},
                "available_mechanics_signals": [
                    s for s in sorted(MECHANICS_NAMES)
                    if _mechanics_available(s, mechanical, ball_source_present)
                ],
                "assessments": assessments,
                "status_counts": dict(Counter(a["status"] for a in assessments)),
                "intended_form_match": "not_assessed",
            })

        return {
            "schema_version": "1.0",
            "method": "joined_temporal_and_observed_mechanics_reference_review",
            "shot_count": len(result_shots),
            "reference_count": baseline["reference_count"],
            "shots": result_shots,
            "official_endorsement": False,
            "intended_form_match": "not_assessed",
            "limitations": [
                "Source-backed coaching topics are not validated biomechanical form standards.",
                "Sufficient pose visibility is not proof of accurate landmarks or correct technique.",
                "Series signals are observed during set-to-release; no later movement is inferred.",
                "Only detector-origin ball_analysis.frames can count as ball observations.",
                "Tracked trajectories may contain interpolation and are never promoted to raw detections.",
                "Context-sensitive drills require independently documented clip context.",
                "Provisional masterclass summaries and Shorts are not treated as verified instruction.",
                "Guide-hand contact and hand tension remain unmeasured.",
                "No automatic ideal-angle, correctness, or score claims are produced.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temporal_json", type=Path)
    parser.add_argument("mechanics_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    t = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    m = json.loads(args.mechanics_json.read_text(encoding="utf-8"))
    out = TechniqueSignalIntegrator().analyze(t, m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Shots: {out['shot_count']} | References: {out['reference_count']}")
    for shot in out["shots"]:
        print(f"Shot {shot['shot_number']}: {shot['status_counts']}")
        print("  Observed mechanics signals:", ", ".join(shot["available_mechanics_signals"]))
    print("Intended-form match:", out["intended_form_match"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
