"""Phase 5.7: non-prescriptive, source-linked shooting review prompts.

Run from Bullseye/backend:
    python -m app.coaching.coaching_review_generator \\
        --report data/analyzed/phase5_reference_review_video3.json \\
        --output data/analyzed/phase5_coaching_review_video3.json

Uses ONLY the vetted source registry and catalog review questions.
No LLM, perfect-form targets, diagnoses, or claims about video timestamps.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.coaching.coaching_reference_registry import CoachingReferenceRegistry

HERE = Path(__file__).resolve().parent

ALLOWED_STATUSES = frozenset({
    "source_verification_required",
    "descriptive_review_possible",
    "insufficient_timing_evidence",
    "missing_measurements",
    "shot_context_not_established",
})

BLOCK_REASON = {
    "insufficient_timing_evidence": (
        "Tracking coverage or sequence duration is insufficient for a timing review."
    ),
    "missing_measurements": (
        "Required features are not measured reliably in this clip."
    ),
    "shot_context_not_established": (
        "The recording does not establish the required shot or drill context."
    ),
}

SOURCE_REASON = {
    "topic_only_support": (
        "The public curriculum confirms this topic, but does not verify a "
        "specific instruction or technique criterion."
    ),
    "provisional_only": (
        "The specific teaching claim is supported only by provisional or "
        "unverified material. No instruction is generated."
    ),
    "published_interview_reference": (
        "The published interview is associated with this topic; review manually."
    ),
}


class CoachingReviewGenerator:
    """Build review questions, never player-form judgments or corrective drills."""

    def __init__(self, registry: CoachingReferenceRegistry | None = None):
        self.registry = registry or CoachingReferenceRegistry()
        self.catalog = self.registry.principles
        self.canonical = self.registry.build_report()
        self.refs = {
            p["principle_id"]: p for p in self.canonical["references"]
        }

    def _validate(self, report: dict) -> None:
        if not isinstance(report, dict):
            raise ValueError("Registry report must be a JSON object")
        if report.get("official_endorsement") is not False:
            raise ValueError("Registry report must prohibit endorsement")
        if report.get("intended_form_match") != "not_assessed":
            raise ValueError("Graded registry reports are not accepted")
        if report.get("reference_count") != len(self.refs):
            raise ValueError("Reference count mismatch")
        # This also prevents replacing source URLs or escalating support labels.
        if report.get("references") != self.canonical["references"]:
            raise ValueError("Registry references differ from validated local sources")
        shots = report.get("shots")
        if not isinstance(shots, list) or type(report.get("shot_count")) is not int:
            raise ValueError("Invalid shots or shot_count")
        if report["shot_count"] != len(shots):
            raise ValueError("Shot count mismatch")
        seen_numbers: set[int] = set()
        for shot in shots:
            if not isinstance(shot, dict):
                raise ValueError("Shot must be an object")
            number = shot.get("shot_number")
            if type(number) is not int or number < 1 or number in seen_numbers:
                raise ValueError("Shot numbers must be unique positive integers")
            seen_numbers.add(number)
            if shot.get("intended_form_match") != "not_assessed":
                raise ValueError("A shot cannot contain a form grade")
            assessments = shot.get("assessments")
            if not isinstance(assessments, list) or len(assessments) != len(self.refs):
                raise ValueError("Each shot must cover all registered references")
            seen_refs: set[str] = set()
            for item in assessments:
                if not isinstance(item, dict):
                    raise ValueError("Assessment must be an object")
                pid = item.get("registry_reference_id")
                if pid not in self.refs or pid in seen_refs:
                    raise ValueError("Unknown or duplicate reference in assessments")
                seen_refs.add(pid)
                if item.get("title") != self.refs[pid]["title"]:
                    raise ValueError("Conflicting reference title")
                if item.get("registry_source_support_status") != (
                    self.refs[pid]["source_support_status"]
                ):
                    raise ValueError("Source-support label does not match registry")
                if item.get("supplemental_evidence_ids") != [
                    e["evidence_id"] for e in self.refs[pid]["supplemental_evidence"]
                ]:
                    raise ValueError("Supplemental evidence identifiers mismatch")
                if item.get("status") not in ALLOWED_STATUSES:
                    raise ValueError("Unrecognized assessment status")
                if item.get("intended_form_match") != "not_assessed":
                    raise ValueError("Unsafe assessment grade")

    def _topic(self, item: dict) -> dict:
        pid = item["registry_reference_id"]
        ref = self.refs[pid]
        status = item["status"]
        source_support = ref["source_support_status"]
        result = {
            "principle_id": pid,
            "title": ref["title"],
            "measurement_status": status,
            "source_support_status": source_support,
            "review_status": "not_ready",
            "review_question": None,
            "published_context": [],
            "source_urls": [],
            "reason": None,
            "original_video_timestamp_verified": ref[
                "original_video_timestamp_verified"
            ],
            "intended_form_match": "not_assessed",
        }

        # Blocked measurement/context always takes precedence over source support.
        if status in BLOCK_REASON:
            result["reason"] = BLOCK_REASON[status]
            return result

        if source_support == "limited_published_text_support":
            evidence = [
                e for e in ref["supplemental_evidence"]
                if e["allowed_use"] == "descriptive_context"
            ]
            if evidence:
                result["review_status"] = "manual_review_candidate"
                result["review_question"] = self.catalog[pid]["review_question"]
                result["published_context"] = [
                    {
                        "bounded_claim": e["claim_paraphrase"],
                        "scope_limit": e["scope_note"],
                        "source_id": e["source_id"],
                        "source_url": e["url"],
                        "locator": e["locator"],
                    }
                    for e in evidence
                ]
                result["source_urls"] = sorted(set(e["url"] for e in evidence))
                result["reason"] = (
                    "Observation supports posing a review question. Published "
                    "text offers limited context, not an automatic correction. "
                    "Any original YouTube timestamp remains unverified."
                )
                return result

        result["reason"] = SOURCE_REASON.get(
            source_support, "Source evidence is insufficient for detailed coaching."
        )
        return result

    def generate(self, report: dict) -> dict:
        self._validate(report)
        shots = []
        for shot in report["shots"]:
            topics = [self._topic(item) for item in shot["assessments"]]
            candidates = [
                item for item in topics
                if item["review_status"] == "manual_review_candidate"
            ]
            shots.append({
                "shot_number": shot["shot_number"],
                "review_candidates": candidates,
                "review_candidate_count": len(candidates),
                "review_topics": topics,
                "measurement_status_counts": dict(Counter(
                    t["measurement_status"] for t in topics
                )),
                "review_status_counts": dict(Counter(
                    t["review_status"] for t in topics
                )),
                "intended_form_match": "not_assessed",
            })

        return {
            "schema_version": "1.0",
            "method": "published_context_human_review_questions",
            "shot_count": len(shots),
            "reference_count": len(self.refs),
            "shots": shots,
            "intended_form_match": "not_assessed",
            "official_endorsement": False,
            "corrections_generated": 0,
            "limitations": [
                "These are human review questions, not validated form judgments.",
                "Published interview context does not verify masterclass timestamps.",
                "Topic-only course outlines do not establish technical pass/fail rules.",
                "No real video measurements are calculated by this module.",
                "Detector predictions and 2D pose estimates may be wrong.",
                "No drills or technique corrections are prescribed automatically.",
                "Not affiliated with or endorsed by the referenced coach.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-linked shooting review questions, without grading."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    result = CoachingReviewGenerator().generate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Shots reviewed: {result['shot_count']}")
    for shot in result["shots"]:
        print(
            f"Shot {shot['shot_number']}: "
            f"{shot['review_candidate_count']} manual review questions; "
            f"{shot['review_status_counts'].get('not_ready', 0)} topics not ready"
        )
    print("Intended-form verdict: not_assessed")
    print("Corrections generated: 0")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
