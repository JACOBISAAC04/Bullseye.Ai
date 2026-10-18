"""Phase 4: descriptive shooting-form comparisons using existing Bullseye metrics.

Does not assign ideal joint angles or judge correctness from single-camera poses.
Run from backend:
    python app/coaching/form_comparison.py data/analyzed/hoop5_video3.json \\
        --output data/analyzed/phase4_form_comparison.json
"""
import argparse
import json
import math
from pathlib import Path


ANGLE_KEYS = ("elbow_angle", "hip_angle", "knee_angle")
PHASES = ("set", "release", "follow_through")


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 3) if math.isfinite(value) else None


def phase_angles(shot, phase):
    section = shot.get(phase) or {}
    if not isinstance(section, dict):
        section = {}
    return {key: finite_number(section.get(key)) for key in ANGLE_KEYS}


class FormComparisonAnalyzer:
    def __init__(self, reference_path=None):
        path = Path(reference_path) if reference_path else Path(__file__).with_name(
            "reference_principles.json"
        )
        self.reference = json.loads(path.read_text(encoding="utf-8"))

    def analyze(self, payload):
        # Accept either the full /video/analyze response or metrics-only JSON.
        metrics = payload.get("metrics", payload)
        if not isinstance(metrics, dict) or not isinstance(metrics.get("shots"), list):
            raise ValueError("Expected metrics object containing a 'shots' list")

        observed_shots = []
        for index, shot in enumerate(metrics["shots"], start=1):
            observed_shots.append({
                "shot_number": shot.get("shot_number", index),
                "arm": shot.get("arm"),
                "start_frame": shot.get("start_frame"),
                "release_frame": shot.get("release_frame"),
                "end_frame": shot.get("end_frame"),
                "angles_degrees": {
                    phase: phase_angles(shot, phase) for phase in PHASES
                },
                "set_torso_lean_degrees": finite_number(
                    ((shot.get("set") or {}).get("body_alignment") or {}).get(
                        "torso_lean_angle"
                    )
                ),
                "follow_through_balance_offset_normalized": finite_number(
                    ((shot.get("follow_through") or {}).get("balance") or {}).get(
                        "normalized_offset"
                    )
                ),
                "release_speed_mps": finite_number(
                    (shot.get("release") or {}).get("ball_velocity_meters_per_second")
                ),
            })

        release_variation = {}
        for key in ANGLE_KEYS:
            pairs = [
                (shot["shot_number"], shot["angles_degrees"]["release"][key])
                for shot in observed_shots
                if shot["angles_degrees"]["release"][key] is not None
            ]
            if len(pairs) >= 2:
                values = [value for _, value in pairs]
                release_variation[key] = {
                    "range_degrees": round(max(values) - min(values), 3),
                    "shot_values": [
                        {"shot_number": number, "degrees": value}
                        for number, value in pairs
                    ],
                    "interpretation": "Descriptive difference only; no ideal threshold."
                }
            else:
                release_variation[key] = None

        readings = {
            "preparation": {
                "type": "descriptive_only",
                "observed": [
                    {
                        "shot_number": s["shot_number"],
                        "set_angles_degrees": s["angles_degrees"]["set"],
                        "torso_lean_degrees": s["set_torso_lean_degrees"],
                    } for s in observed_shots
                ],
                "message": "Measured set positions; no reference target angles available."
            },
            "connection": {
                "type": "needs_temporal_analysis",
                "observed": {"release_angle_variation": release_variation},
                "message": "Release-to-release differences are measurable, but timing and coordination require per-frame motion analysis."
            },
            "balance": {
                "type": "descriptive_only",
                "observed": [
                    {"shot_number": s["shot_number"],
                     "normalized_offset": s["follow_through_balance_offset_normalized"]}
                    for s in observed_shots
                ],
                "message": "2D offsets are recorded, not graded as good or bad balance."
            },
            "guide_hand": {
                "type": "not_assessable",
                "observed": None,
                "message": "Current metrics do not measure guide-hand contact or pressure."
            },
        }

        comparisons = [{
            "principle_id": principle["id"],
            "coaching_topic": principle["name"],
            "intended_form": principle["intended_form"],
            "source_url": self.reference["source"]["url"],
            "assessment": readings[principle["id"]],
        } for principle in self.reference["principles"]]

        return {
            "method": "descriptive_measured_vs_qualitative_coaching_reference",
            "shot_count": len(observed_shots),
            "observed_shots": observed_shots,
            "release_angle_variation": release_variation,
            "comparisons": comparisons,
            "coaching_feedback": (
                "Review set, release, and follow-through frames side by side. "
                "Use measured differences as discussion points, not as proof of incorrect form."
                if len(observed_shots) >= 2
                else "More shots are needed for a meaningful shot-to-shot comparison."
            ),
            "limitations": self.reference["limitations"],
            "official_endorsement": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/analyzed/phase4_form_comparison.json"))
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    result = FormComparisonAnalyzer().analyze(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Shots compared: {result['shot_count']}")
    for label, detail in result["release_angle_variation"].items():
        print(f"Release {label}: {detail['range_degrees']} deg range"
              if detail else f"Release {label}: insufficient data")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
