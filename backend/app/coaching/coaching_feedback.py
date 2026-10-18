"""Bullseye Phase 4.3: conservative source-linked coaching review.

CLI:
  python app/coaching/coaching_feedback.py \\
      data/analyzed/phase4_form_comparison.json \\
      --output data/analyzed/phase4_coaching_feedback.json

Accepts the descriptive JSON from FormComparisonAnalyzer. This is NOT a
SeeMikeDunn-form correctness classifier. No computer vision is rerun.
"""
import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlparse

FIELDS = (
    ("elbow_angle", "Elbow"),
    ("hip_angle", "Hip"),
    ("knee_angle", "Knee"),
)
TRUSTED_SOURCE_HOSTS = {"seemikedunn.com", "www.seemikedunn.com",
                        "hustletraining.com", "www.hustletraining.com"}
SUPPORTED_RULE_IDS = ("preparation", "connection", "balance", "guide_hand")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return round(val, 3) if math.isfinite(val) else None


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CoachingFeedbackEngine:
    """Never treats angle differences as proof that a technique is wrong."""

    def __init__(self, rules_path=None):
        path = (Path(rules_path) if rules_path is not None else
                Path(__file__).with_name("coaching_rules.json"))
        self.library = _read_json(path)
        self._validate_library()

    def _validate_library(self):
        if self.library.get("official_endorsement") is not False:
            raise ValueError("Coaching rules must explicitly deny endorsement")
        sources = self.library.get("sources")
        rules = self.library.get("rules")
        if not isinstance(sources, list) or not isinstance(rules, list):
            raise ValueError("Coaching rules must contain sources and rules lists")
        if not sources:
            raise ValueError("Coaching rules must contain verified source metadata")

        ids = set()
        for source in sources:
            source_id = source.get("id")
            url = source.get("url", "")
            parsed = urlparse(url)
            if (not source_id or source_id in ids or
                    parsed.scheme != "https" or
                    parsed.hostname not in TRUSTED_SOURCE_HOSTS or
                    not source.get("title") or not source.get("scope")):
                raise ValueError("Invalid, duplicate, or unapproved source")
            ids.add(source_id)

        rules_by_id = {}
        for rule in rules:
            rule_id = rule.get("id")
            if rule_id in rules_by_id or rule_id not in SUPPORTED_RULE_IDS:
                raise ValueError("Duplicate or unknown coaching rule")
            if (not rule.get("source_ids") or
                    not set(rule["source_ids"]).issubset(ids) or
                    not rule.get("review_prompt") or
                    not rule.get("required_to_judge") or
                    not rule.get("source_backed_principle")):
                raise ValueError(f"Rule {rule_id} is missing source/evidence fields")
            rules_by_id[rule_id] = rule
        if set(rules_by_id) != set(SUPPORTED_RULE_IDS):
            raise ValueError("One or more required rules are missing")

    def analyze(self, comparison):
        if not isinstance(comparison, dict):
            raise ValueError("Expected comparison JSON object")
        shots = comparison.get("observed_shots")
        variations = comparison.get("release_angle_variation")
        if not isinstance(shots, list) or not isinstance(variations, dict):
            raise ValueError("Expected FormComparisonAnalyzer output")
        if (comparison.get("shot_count") != len(shots) or
                comparison.get("official_endorsement") is not False):
            raise ValueError("Inconsistent shot count or missing non-endorsement")

        variations_report = []
        for key, label in FIELDS:
            values = []
            for shot in shots:
                if not isinstance(shot, dict):
                    raise ValueError("Invalid observed shot entry")
                angle_dict = (shot.get("angles_degrees") or {}).get("release") or {}
                angle = _number(angle_dict.get(key))
                if angle is not None:
                    values.append({
                        "shot_number": shot.get("shot_number"),
                        "degrees": angle,
                        "release_frame": shot.get("release_frame")
                    })

            # Recalculate from observed shots instead of trusting an
            # unrelated or possibly stale release_angle_variation field.
            angle_range = (
                round(max(x["degrees"] for x in values) -
                      min(x["degrees"] for x in values), 3)
                if len(values) >= 2 else None
            )
            variations_report.append({
                "metric": key,
                "label": label,
                "range_degrees": angle_range,
                "observations": values,
                "meaning": (
                    "Measured shot-to-shot difference; not deviation from an ideal."
                    if angle_range is not None else
                    "Insufficient measured shots for a range."
                ),
            })

        measurable = [
            item for item in variations_report
            if item["range_degrees"] is not None
        ]
        # A purely descriptive review order; there is no threshold
        # or claim that the biggest difference is necessarily a defect.
        review = (max(measurable, key=lambda item: item["range_degrees"])
                  if measurable else None)
        source_by_id = {source["id"]: source for source in self.library["sources"]}
        assessments = []

        for rule in self.library["rules"]:
            rule_id = rule["id"]
            evidence = []
            status = "insufficient_evidence"

            if rule_id == "preparation":
                evidence = [
                    {"shot_number": s.get("shot_number"),
                     "set_angles_degrees": (s.get("angles_degrees") or {}).get("set"),
                     "torso_lean_degrees": _number(s.get("set_torso_lean_degrees"))}
                    for s in shots
                ]
                if any(x["torso_lean_degrees"] is not None or
                       any(_number(v) is not None for v in
                           (x["set_angles_degrees"] or {}).values())
                       for x in evidence):
                    status = "descriptive_only"

            elif rule_id == "connection":
                evidence = variations_report
                if measurable:
                    status = "descriptive_only"

            elif rule_id == "balance":
                evidence = [
                    {"shot_number": s.get("shot_number"),
                     "normalized_offset": _number(
                         s.get("follow_through_balance_offset_normalized"))}
                    for s in shots
                ]
                if any(x["normalized_offset"] is not None for x in evidence):
                    status = "descriptive_only"

            elif rule_id == "guide_hand":
                status = "not_assessable"

            assessments.append({
                "principle_id": rule_id,
                "title": rule["title"],
                "assessment_status": status,
                "coaching_topic": rule["source_backed_principle"],
                "evidence": evidence,
                "evidence_still_needed": rule["required_to_judge"],
                "suggested_next_review": rule["review_prompt"],
                "references": [source_by_id[sid] for sid in rule["source_ids"]],
                "form_correction_authorized": False,
            })

        return {
            "schema_version": "1.0",
            "shot_count": len(shots),
            "coach_reference": self.library["coach"],
            "official_endorsement": False,
            "intended_form_match": "not_assessed",
            "measurement_validation": "unverified_raw_pose_and_frame_quality",
            "release_angle_observations": variations_report,
            "first_review_suggestion": (
                f"Compare release frames for {review['label'].lower()} angle: "
                f"{review['range_degrees']:.3f} degree observed range. "
                "This does not imply incorrect technique."
                if review else "Collect at least two measurable releases."
            ),
            "assessments": assessments,
            "recommended_form_corrections": [],
            "notes": [
                "Angles and offsets come from existing 2D pose estimates; original frame accuracy has not been independently checked.",
                "Interpolated ball positions must not be treated as observed ball speed or contact.",
                "Descriptions are sourced coaching topics, not validated individual diagnoses or universal target angles.",
                "Two shots are insufficient to infer a player's long-term consistency.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_json", type=Path)
    parser.add_argument(
        "--rules", type=Path,
        default=Path(__file__).with_name("coaching_rules.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/analyzed/phase4_coaching_feedback.json"))
    args = parser.parse_args()

    report = CoachingFeedbackEngine(args.rules).analyze(
        _read_json(args.comparison_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Shots reviewed: {report['shot_count']}")
    print(f"Form match: {report['intended_form_match']}")
    print(f"First review: {report['first_review_suggestion']}")
    for entry in report["assessments"]:
        print(f"{entry['title']}: {entry['assessment_status']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
