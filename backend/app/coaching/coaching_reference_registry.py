"""Phase 5.6: provenance-aware coaching reference registry.

Read the existing 22-topic reference library and a separate curated evidence
overlay. Preserve original citation verification levels and all upstream
technique assessment statuses; NEVER generate form corrections.

From backend/:
    python -m app.coaching.coaching_reference_registry \
        --report data/analyzed/phase5_ball_integrated_video3.json \
        --output data/analyzed/phase5_source_registry_video3.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "shooting_coaching_references.json"
DEFAULT_EVIDENCE = HERE / "coaching_source_evidence.json"

# Only these exact *source kinds* and catalog metadata qualify for supplemental
# published-text support. Neither video-summary notes nor mirrors can pass.
APPROVED_SOURCES = {
    "dunn_public_interview": ("attributed_interview", "published_transcript_read"),
    "ground_up_outline": ("official_course_outline", "official_outline_only"),
}

PRIMARY_LEVELS = {
    "published_primary_interview": "published_attributed_interview",
    "official_outline_topic_only": "official_topic_only",
}
STATUS_VALUES = {
    "source_verification_required",
    "descriptive_review_possible",
    "insufficient_timing_evidence",
    "missing_measurements",
    "shot_context_not_established",
}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _distinct_strings(value, label):
    if not isinstance(value, list) or not value or any(
        not isinstance(s, str) or not s.strip() for s in value
    ):
        raise ValueError(f"{label} must be a nonempty list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique strings")
    return value


def _normalize_title(title):
    return re.sub(r"[^a-z0-9]+", "", str(title).casefold())


class CoachingReferenceRegistry:
    """Curated source support, not a technique grade or a video transcription."""

    def __init__(self, catalog=None, evidence=None, catalog_path=None, evidence_path=None):
        if catalog is not None and catalog_path is not None:
            raise ValueError("Pass catalog or catalog_path, not both")
        if evidence is not None and evidence_path is not None:
            raise ValueError("Pass evidence or evidence_path, not both")

        self.catalog = catalog if catalog is not None else json.loads(
            Path(catalog_path or DEFAULT_CATALOG).read_text(encoding="utf-8")
        )
        self.evidence = evidence if evidence is not None else json.loads(
            Path(evidence_path or DEFAULT_EVIDENCE).read_text(encoding="utf-8")
        )
        self._validate()

    def _validate(self):
        catalog = _object(self.catalog, "Catalog")
        if catalog.get("official_endorsement") is not False:
            raise ValueError("Catalog must prohibit claims of endorsement")
        policy = _object(catalog.get("use_policy"), "Catalog use_policy")
        if policy.get("automatic_correctness_grading") is not False:
            raise ValueError("Catalog must prohibit automatic grading")
        source_list, principles = catalog.get("sources"), catalog.get("principles")
        if not isinstance(source_list, list) or not isinstance(principles, list):
            raise ValueError("Catalog needs sources and principles lists")

        self.sources = {}
        self.principles = {}
        title_index = {}
        for source in source_list:
            source = _object(source, "Source")
            sid = _nonempty(source.get("id"), "Source id")
            if sid in self.sources:
                raise ValueError(f"Duplicate source id: {sid}")
            _nonempty(source.get("url"), f"Source {sid} url")
            self.sources[sid] = source
        for principle in principles:
            principle = _object(principle, "Principle")
            pid = _nonempty(principle.get("id"), "Principle id")
            if pid in self.principles:
                raise ValueError(f"Duplicate principle id: {pid}")
            if principle.get("can_auto_grade_correctness") is not False:
                raise ValueError(f"Unsafe form grading flag: {pid}")
            if principle.get("safe_output") != "descriptive_or_not_assessed":
                raise ValueError(f"Unsafe output mode: {pid}")
            citation = _object(principle.get("citation"), f"Citation for {pid}")
            sid = citation.get("source_id")
            if sid not in self.sources:
                raise ValueError(f"Unknown cited source for {pid}: {sid}")
            _nonempty(citation.get("evidence_level"), f"Citation evidence level for {pid}")
            _nonempty(principle.get("title"), f"Title for {pid}")
            self.principles[pid] = principle
            title_index.setdefault(_normalize_title(principle["title"]), []).append(pid)
        # A renamed title never silently selects an arbitrary principle.
        self.titles = title_index

        ev = _object(self.evidence, "Evidence overlay")
        if ev.get("schema_version") != "1.0":
            raise ValueError("Unsupported evidence schema_version")
        ep = _object(ev.get("policy"), "Evidence policy")
        for key in (
            "automatic_correctness_grading",
            "replace_existing_citations",
            "promote_provisional_video_timestamps",
            "assert_official_endorsement",
        ):
            if ep.get(key) is not False:
                raise ValueError(f"Unsafe evidence policy: {key}")
        records = ev.get("evidence_records")
        if not isinstance(records, list):
            raise ValueError("evidence_records must be a list")

        self.support = {pid: [] for pid in self.principles}
        seen = set()
        for record in records:
            record = _object(record, "Evidence record")
            eid = _nonempty(record.get("id"), "Evidence id")
            if eid in seen:
                raise ValueError(f"Duplicate evidence id: {eid}")
            seen.add(eid)
            sid = _nonempty(record.get("source_id"), f"Source id for {eid}")
            if sid not in APPROVED_SOURCES or sid not in self.sources:
                raise ValueError(f"Unapproved source for reviewed evidence: {sid}")
            expected_kind, verification = APPROVED_SOURCES[sid]
            if record.get("source_kind") != expected_kind:
                raise ValueError(f"Wrong source_kind for {eid}")
            if self.sources[sid].get("verification") != verification:
                raise ValueError(f"Source verification metadata differs for {eid}")
            if record.get("source_url") != self.sources[sid]["url"]:
                raise ValueError(f"Source URL mismatch for {eid}")
            if record.get("reviewed_against_published_text") is not True:
                raise ValueError(f"Unreviewed source claim: {eid}")
            if record.get("can_auto_grade_correctness") is not False:
                raise ValueError(f"Unsafe grading in evidence claim: {eid}")
            expected_use = (
                "topic_only" if sid == "ground_up_outline" else "descriptive_context"
            )
            if record.get("allowed_use") != expected_use:
                raise ValueError(f"Unsupported allowed_use for {eid}")
            for key in ("locator", "claim_paraphrase", "scope_note"):
                _nonempty(record.get(key), f"{key} for {eid}")
            refs = _distinct_strings(record.get("reference_ids"), f"reference_ids for {eid}")
            for pid in refs:
                if pid not in self.principles:
                    raise ValueError(f"Unknown linked reference id: {pid}")
                self.support[pid].append(record)

    def _reference(self, pid):
        principle = self.principles[pid]
        citation = principle["citation"]
        primary_level = citation["evidence_level"]
        original = PRIMARY_LEVELS.get(primary_level, "unverified_specific_claim")
        support = self.support[pid]
        specific = any(r["allowed_use"] == "descriptive_context" for r in support)
        topic_only = bool(support) and not specific
        if specific:
            source_state = "limited_published_text_support"
        elif original == "published_attributed_interview":
            source_state = "published_interview_reference"
        elif topic_only or original == "official_topic_only":
            source_state = "topic_only_support"
        else:
            source_state = "provisional_only"

        return {
            "principle_id": pid,
            "title": principle["title"],
            "original_citation": dict(citation),
            "original_citation_status": original,
            # A published interview on a related topic NEVER verifies an
            # unreviewed section of a YouTube masterclass.
            "original_video_timestamp_verified": False
                if citation["source_id"] == "youtube_masterclass" else None,
            "source_support_status": source_state,
            "supplemental_evidence": [
                {
                    "evidence_id": r["id"],
                    "source_id": r["source_id"],
                    "url": r["source_url"],
                    "locator": r["locator"],
                    "claim_paraphrase": r["claim_paraphrase"],
                    "scope_note": r["scope_note"],
                    "allowed_use": r["allowed_use"],
                }
                for r in support
            ],
            "can_auto_grade_correctness": False,
            "intended_form_match": "not_assessed",
        }

    def _resolve_assessment(self, item):
        # Use the genuine stable id field exported by the Phase 5 validator.
        # `item["id"]` is a legacy/optional field; only use it when valid.
        if "principle_id" in item:
            pid = item["principle_id"]
            if pid not in self.principles:
                raise ValueError(f"Unknown principle_id in assessment: {pid}")
            matched_by = "principle_id"
        elif item.get("id") in self.principles:
            pid = item["id"]
            matched_by = "id"
        else:
            title = _nonempty(item.get("title"), "Assessment title")
            matches = self.titles.get(_normalize_title(title), [])
            if len(matches) != 1:
                raise ValueError(f"Title is missing or ambiguous: {title!r}")
            pid, matched_by = matches[0], "unique_normalized_title"

        if item.get("title") and _normalize_title(item["title"]) != _normalize_title(
            self.principles[pid]["title"]
        ):
            raise ValueError(f"Assessment id/title conflict for {pid}")
        return pid, matched_by

    def build_report(self, integrated_report=None):
        refs = [self._reference(pid) for pid in self.principles]
        ref_map = {r["principle_id"]: r for r in refs}
        result_shots = []
        if integrated_report is not None:
            rep = _object(integrated_report, "Integrated report")
            shots = rep.get("shots")
            if not isinstance(shots, list) or type(rep.get("shot_count")) is not int or (
                rep["shot_count"] != len(shots)
            ):
                raise ValueError("Integrated shot_count must match shots")
            if rep.get("intended_form_match") != "not_assessed":
                raise ValueError("Cannot import a graded integrated report")
            if rep.get("reference_count") != len(refs):
                raise ValueError("Integrated reference_count doesn't match catalog")
            shot_numbers = set()
            for shot in shots:
                shot = _object(shot, "Integrated shot")
                number = shot.get("shot_number")
                if type(number) is not int or number < 1 or number in shot_numbers:
                    raise ValueError("Integrated shot_number must be unique and positive")
                shot_numbers.add(number)
                assessments = shot.get("assessments")
                if not isinstance(assessments, list) or len(assessments) != len(refs):
                    raise ValueError("Integrated assessments must cover every reference")
                enriched = []
                seen_refs = set()
                for item in assessments:
                    item = _object(item, "Assessment")
                    pid, strategy = self._resolve_assessment(item)
                    if pid in seen_refs:
                        raise ValueError(f"Duplicate assessment reference: {pid}")
                    seen_refs.add(pid)
                    if item.get("status") not in STATUS_VALUES:
                        raise ValueError(f"Unknown assessment status: {item.get('status')}")
                    if item.get("intended_form_match") != "not_assessed":
                        raise ValueError(f"Unsafe assessment form verdict: {pid}")
                    # Preserve the original assessment and all its safety gates.
                    enriched.append({
                        **item,
                        "registry_reference_id": pid,
                        "registry_match_strategy": strategy,
                        "registry_source_support_status": ref_map[pid]["source_support_status"],
                        "supplemental_evidence_ids": [
                            r["evidence_id"] for r in ref_map[pid]["supplemental_evidence"]
                        ],
                        "registry_note": (
                            "Original assessment status is unchanged. Supplemental "
                            "published-source claims do not verify YouTube timestamps, "
                            "repair missing measurements, or allow form grading."
                        ),
                    })
                result_shots.append({
                    "shot_number": number,
                    "assessments": enriched,
                    "intended_form_match": "not_assessed",
                })

        counts = dict(Counter(r["source_support_status"] for r in refs))
        return {
            "schema_version": "1.0",
            "method": "bounded_published_source_reference_audit",
            "reference_count": len(refs),
            "shot_count": len(result_shots),
            "references": refs,
            "source_support_counts": counts,
            "shots": result_shots,
            "official_endorsement": False,
            "intended_form_match": "not_assessed",
            "limitations": [
                "This registry reviews published statements and topic outlines; it is not a form grader.",
                "A corroborating interview claim does not verify a video timestamp or paid course lesson.",
                "The published interview is attributed text, not a fresh examination of video/audio.",
                "No individual YouTube Shorts are verified by this registry.",
                "Video measurements, shot context, and landmark quality still need independent checks.",
                "Original technique assessment statuses are preserved without promotion.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Conservative coaching source registry")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path, help="Optional existing integrated technique JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = (
        json.loads(args.report.read_text(encoding="utf-8"))
        if args.report else None
    )
    registry = CoachingReferenceRegistry(
        catalog_path=args.catalog, evidence_path=args.evidence
    )
    result = registry.build_report(analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"References: {result['reference_count']}")
    print(f"Published-source claims: {len(registry.evidence['evidence_records'])}")
    print(f"Source support: {result['source_support_counts']}")
    print(f"Shots reviewed: {result['shot_count']}")
    for shot in result["shots"]:
        status = dict(Counter(a["status"] for a in shot["assessments"]))
        print(f"Shot {shot['shot_number']}: {status}")
    print(f"Form verdict: {result['intended_form_match']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
