"""Evidence-only assessment of a temporal shooting report against a source catalog.

This does NOT grade shooting technique or imply any coach's endorsement.
Run from Bullseye/backend:
    python app/coaching/technique_validator.py data/analyzed/phase4_temporal_form.json \
        --output data/analyzed/phase5_technique_review_video3.json
"""

import argparse
import json
import math
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).with_name("shooting_coaching_references.json")
SEGMENT_NAME = "set_to_release"
QUALITY_OK = "sufficient_for_descriptive_timing"
SERIES_MAP = {
    "hip_angle_series": ("observed_angle_series", "hip_angle"),
    "knee_angle_series": ("observed_angle_series", "knee_angle"),
}
# Only these catalog signal names are represented faithfully in Phase 4's
# temporal report. Full wrist positions, raw joint coordinates, ball positions,
# and guide-hand contact CANNOT be reconstructed from its summary.
SUPPORTED_SIGNALS = frozenset({
    "hip_angle_series", "knee_angle_series",
    "set_frame", "release_frame", "timing_quality", "pose_frames",
})
CONTEXT_DEPENDENT_PHASES = frozenset({
    "training", "practice", "moving_shot", "catch_and_shoot", "free_throw"
})
SOURCE_LEVELS_REVIEWABLE = frozenset({
    "published_primary_interview", "official_outline_topic_only"
})


def _is_finite_number(number):
    return (
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and math.isfinite(number)
    )


def _unique_strings(items, label):
    if not isinstance(items, list):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(i, str) or not i for i in items):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} contains duplicate entries")
    return items


def _validate_catalog(catalog):
    if not isinstance(catalog, dict):
        raise ValueError("Reference catalog must be an object")
    if catalog.get("official_endorsement") is not False:
        raise ValueError("Catalog must explicitly declare official_endorsement=false")
    policy = catalog.get("use_policy")
    if not isinstance(policy, dict) or policy.get("automatic_correctness_grading") is not False:
        raise ValueError("Catalog must explicitly prohibit automatic correctness grading")
    sources = catalog.get("sources")
    principles = catalog.get("principles")
    if not isinstance(sources, list) or not isinstance(principles, list):
        raise ValueError("Catalog needs sources and principles arrays")

    source_map = {}
    for source in sources:
        if not isinstance(source, dict) or not source.get("id") or not source.get("url"):
            raise ValueError("Every source needs id and url")
        if source["id"] in source_map:
            raise ValueError(f"Duplicate source: {source['id']}")
        source_map[source["id"]] = source

    ids = set()
    for principle in principles:
        if not isinstance(principle, dict) or not principle.get("id"):
            raise ValueError("Every principle requires an id")
        if principle["id"] in ids:
            raise ValueError(f"Duplicate principle: {principle['id']}")
        ids.add(principle["id"])
        if principle.get("can_auto_grade_correctness") is not False:
            raise ValueError(f"Unsafe auto-grading flag: {principle['id']}")
        _unique_strings(principle.get("bullseye_candidate_signals"), "bullseye_candidate_signals")
        citation = principle.get("citation")
        if not isinstance(citation, dict) or citation.get("source_id") not in source_map:
            raise ValueError(f"Missing cited source for {principle['id']}")
        if not isinstance(citation.get("evidence_level"), str):
            raise ValueError(f"Missing evidence level for {principle['id']}")
    return source_map


def _angle_available(segment, name):
    summary = segment.get("observed_angle_series", {}).get(name, {})
    return (
        isinstance(summary, dict)
        and isinstance(summary.get("observed_frames"), int)
        and summary["observed_frames"] >= 2
        and isinstance(summary.get("adjacent_observed_pairs"), int)
        and summary["adjacent_observed_pairs"] >= 1
    )


def _signal_available(signal, shot, segment):
    if signal == "set_frame":
        return type(shot.get("start_frame")) is int and shot["start_frame"] >= 1
    if signal == "release_frame":
        return type(shot.get("release_frame")) is int and shot["release_frame"] >= 1
    if signal == "timing_quality":
        return segment.get("quality") == QUALITY_OK
    if signal == "pose_frames":
        m = segment.get("quality_metrics", {})
        return (
            isinstance(m, dict)
            and isinstance(m.get("complete_pose_frames"), int)
            and m["complete_pose_frames"] > 0
        )
    if signal in SERIES_MAP:
        group, key = SERIES_MAP[signal]
        if group != "observed_angle_series":
            return False
        return _angle_available(segment, key)
    return False


def _observed_motion(shot):
    """Provide raw descriptive metrics; never produce a 'correct form' result."""
    findings = {}
    segments = shot.get("segments", {})
    for segment_key in ("set_to_release", "release_to_shot_end"):
        segment = segments.get(segment_key, {})
        if not isinstance(segment, dict):
            continue
        available = {}
        angles = segment.get("observed_angle_series", {})
        for angle in ("elbow_angle", "hip_angle", "knee_angle"):
            series = angles.get(angle, {}) if isinstance(angles, dict) else {}
            if not isinstance(series, dict):
                continue
            first = series.get("first_observed")
            last = series.get("last_observed")
            available[angle] = {
                "observed_frames": series.get("observed_frames"),
                "first_observed": first,
                "last_observed": last,
                "observed_endpoint_delta_degrees": series.get("observed_endpoint_delta"),
            }
        wrist = segment.get("wrist_vertical_position_series")
        findings[segment_key] = {
            "timing_quality": segment.get("quality", "unknown"),
            "quality_metrics": segment.get("quality_metrics", {}),
            "angle_observations": available,
            "wrist_height_relative_to_shoulder_px": (
                {
                    "observed_frames": wrist.get("observed_frames"),
                    "observed_endpoint_delta_px": wrist.get("observed_endpoint_delta"),
                } if isinstance(wrist, dict) else None
            ),
            "interpretation": "2D descriptive measurements, not a coaching grade",
        }
    return findings


class TechniqueEvidenceValidator:
    def __init__(self, catalog_path=None, catalog=None):
        if catalog is not None and catalog_path is not None:
            raise ValueError("Pass either catalog or catalog_path")
        if catalog is None:
            path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG
            catalog = json.loads(path.read_text(encoding="utf-8"))
        self.catalog = catalog
        self.sources = _validate_catalog(catalog)

    def analyze(self, temporal_report):
        if not isinstance(temporal_report, dict):
            raise ValueError("Temporal report must be a JSON object")
        shots = temporal_report.get("shots")
        if not isinstance(shots, list):
            raise ValueError("Temporal report must contain shots array")
        if temporal_report.get("shot_count") != len(shots):
            raise ValueError("Temporal shot_count doesn't match shots array")

        results = []
        for idx, shot in enumerate(shots, 1):
            if not isinstance(shot, dict):
                raise ValueError("Every shot must be an object")
            segments = shot.get("segments")
            if not isinstance(segments, dict) or not isinstance(
                segments.get(SEGMENT_NAME), dict
            ):
                raise ValueError("Missing temporal set_to_release segment")
            segment = segments[SEGMENT_NAME]
            timing_quality = segment.get("quality", "unknown")

            assessments = []
            for principle in self.catalog["principles"]:
                citation = principle["citation"]
                source = self.sources[citation["source_id"]]
                requested = principle["bullseye_candidate_signals"]
                missing = [
                    signal for signal in requested
                    if not _signal_available(signal, shot, segment)
                ]
                # Context (practice type, free throw, catch & shoot, etc.)
                # isn't encoded in temporal reports and cannot be inferred.
                context_unknown = principle.get("shot_phase") in CONTEXT_DEPENDENT_PHASES
                source_reviewed = citation["evidence_level"] in SOURCE_LEVELS_REVIEWABLE
                if context_unknown:
                    status = "shot_context_not_established"
                elif not requested or missing:
                    status = "missing_measurements"
                elif timing_quality != QUALITY_OK:
                    status = "insufficient_timing_evidence"
                elif not source_reviewed:
                    status = "source_verification_required"
                else:
                    status = "descriptive_review_possible"

                assessments.append({
                    "principle_id": principle["id"],
                    "title": principle.get("title"),
                    "status": status,
                    "intended_form_match": "not_assessed",
                    "source": {
                        "id": source["id"],
                        "title": source.get("title"),
                        "url": citation.get("url") or source["url"],
                        "evidence_level": citation["evidence_level"],
                    },
                    "required_signals": requested,
                    "missing_signals": missing,
                    "source_verified_for_review": source_reviewed,
                    "observed_timing_quality": timing_quality,
                    "review_question": (
                        principle.get("review_question")
                        if status == "descriptive_review_possible"
                        else None
                    ),
                    "explanation": {
                        "descriptive_review_possible":
                            "Source-backed topic with available descriptive pose measurements; no correctness judgment.",
                        "source_verification_required":
                            "Measurements are available, but this video claim requires direct source verification.",
                        "insufficient_timing_evidence":
                            "Observed motion is too short or incomplete under current engineering quality checks.",
                        "missing_measurements":
                            "Required measurements are unavailable; no values have been inferred.",
                        "shot_context_not_established":
                            "The clip's practice or shot context is not established by temporal pose data.",
                    }[status],
                })

            results.append({
                "shot_number": shot.get("shot_number", idx),
                "arm": shot.get("arm"),
                "measurements": _observed_motion(shot),
                "assessments": assessments,
                "intended_form_match": "not_assessed",
            })

        return {
            "schema_version": "1.0",
            "method": "source_linked_temporal_evidence_review_only",
            "shot_count": len(results),
            "shots": results,
            "reference_count": len(self.catalog["principles"]),
            "official_endorsement": False,
            "intended_form_match": "not_assessed",
            "limitations": [
                "No automatic good/bad shooting form verdict.",
                "YouTube masterclass timestamp notes are provisional pending direct video review.",
                "No individual YouTube Shorts have been verified in this catalog.",
                "2D pose quality does not establish biomechanical accuracy.",
                "Ball path, hand contact, release physics, and foot stability are not inferred from pose summaries.",
                "Shot end is tracking end, not necessarily true follow-through completion.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Source-linked descriptive shooting evidence review")
    parser.add_argument("temporal_json", type=Path, help="Phase 4 temporal JSON report")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    report = TechniqueEvidenceValidator(catalog_path=args.catalog).analyze(temporal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Shots reviewed: {report['shot_count']}")
    print(f"Principles reviewed: {report['reference_count']}")
    for shot in report["shots"]:
        tally = {}
        for assessment in shot["assessments"]:
            status = assessment["status"]
            tally[status] = tally.get(status, 0) + 1
        print(f"Shot {shot['shot_number']}: {tally}")
    print("Form verdict:", report["intended_form_match"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
