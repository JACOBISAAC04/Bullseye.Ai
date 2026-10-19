"""Phase 6.3: add explainable evidence limits to existing player-feedback JSON.

This stage annotates display-ready Phase 6.1/6.2 feedback. It never measures new
motion, computes probabilities, verifies source content, or grades shooting form.

Run from backend:
    python -m app.coaching.player_evidence_explanations \
        --input data/analyzed/phase6_player_feedback_video3.json \
        --output data/analyzed/phase6_evidence_feedback_video3.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


DISPLAY_TEXT = {
    "posture_setup": (
        "Hip-angle observation (posture context)",
        "How does your estimated hip-angle movement appear in the replay? "
        "This angle alone does not establish torso posture.",
    ),
    "hip_loading": (
        "Hip and knee movement",
        "How do your estimated hip and knee angles change through the preparation?",
    ),
    "ground_balance": (
        "Hip-center movement in the image",
        "Can you observe the hip-center movement in the replay? "
        "Pixel displacement alone cannot establish physical balance.",
    ),
    "ball_setup": (
        "Ball and wrist positions",
        "How do the detected ball and wrist positions change together? "
        "This does not prove hand contact or ball grip.",
    ),
    "ball_path": (
        "Observed pre-release ball movement",
        "What movement can you see between the measured ball positions? "
        "Endpoints do not establish the full ball path.",
    ),
    "individualized_feedback": (
        "Joint-angle observations",
        "How do the reported set and release angle snapshots differ? "
        "Their individual source-frame IDs were not supplied.",
    ),
}

TIMING = {
    "sufficient_for_descriptive_timing",
    "insufficient_for_reliable_timing",
}

GLOBAL_LIMITS = (
    "These are descriptive 2D observations, not shooting-form grades.",
    "Camera perspective, pose estimates, and ball detections can affect measurements.",
    "A source link provides coaching context; it does not independently verify a "
    "video timestamp or a player-specific correction.",
)


def _nonneg_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value, name: str, *, minimum=0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def _validate_replay(replay: object, *, label: str, fps: float, origin: int,
                     container: dict | None = None) -> dict:
    if not isinstance(replay, dict):
        raise ValueError(f"{label} replay must be an object")
    start = replay.get("start_frame")
    end = replay.get("end_frame")
    if type(start) is not int or type(end) is not int or start < origin or end < start:
        raise ValueError(f"{label} invalid replay frames")
    a = _finite(replay.get("start_seconds"), f"{label} start_seconds")
    b = _finite(replay.get("end_seconds"), f"{label} end_seconds")
    if b < a:
        raise ValueError(f"{label} invalid replay seconds")
    # Prior module rounds video times to milliseconds.
    if abs(a - ((start - origin) / fps)) > 0.002 or \
       abs(b - ((end - origin) / fps)) > 0.002:
        raise ValueError(f"{label} timestamp does not match FPS and frame origin")
    if container is not None and not (
        container["start_frame"] <= start <= end <= container["end_frame"]
    ):
        raise ValueError(f"{label} replay lies outside the shot")
    return replay


def _validate_urls(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("source_urls must be a list")
    for link in value:
        if not isinstance(link, str):
            raise ValueError("source_urls must contain strings")
        parts = urlsplit(link)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or any(char in link for char in "\r\n\t")
        ):
            raise ValueError("Source URLs must be public HTTPS links")


def _unique_text(values: object, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        if value not in result:
            result.append(value)
    return result


def _shot_explanation(quality: str, count: int) -> dict:
    if count:
        return {
            "status": "descriptive_reviews_available",
            "headline": f"{count} measured review topic{'s' if count != 1 else ''} available",
            "reason": "These cards describe measurements that passed the existing timing and source gates.",
            "next_step": "Use each card's replay range to inspect the observed movement.",
        }
    if quality == "insufficient_for_reliable_timing":
        return {
            "status": "insufficient_timing_evidence",
            "headline": "Not enough reliable timing evidence for detailed feedback",
            "reason": (
                "The set-to-release timing segment did not meet the earlier "
                "descriptive-review gate. This does not mean the shot was incorrect."
            ),
            "next_step": (
                "Review the available shot replay. A clearer view with more visible "
                "preparation frames may make future review possible."
            ),
        }
    return {
        "status": "no_eligible_measured_reviews",
        "headline": "No eligible measurement-backed review cards",
        "reason": (
            "The upstream review produced no displayable measurement-backed topics. "
            "This alone does not identify which particular measurement is missing."
        ),
        "next_step": "Review the shot replay; do not infer a technique fault from an empty report.",
    }


def _card_explanation(card: dict, fps: float, origin: int, shot_replay: dict) -> dict:
    if not isinstance(card, dict):
        raise ValueError("Feedback card must be an object")
    if card.get("evidence_status") != "observed_2d_descriptive":
        raise ValueError("Unsupported card evidence status")
    _positive_int(card.get("measurement_count"), "measurement_count")
    _validate_urls(card.get("source_urls"))
    limitations = _unique_text(card.get("limitations"), "card limitations")
    if not limitations:
        raise ValueError("Measured card must state its limitations")
    title = card.get("title")
    observation = card.get("observation_text")
    question = card.get("review_question")
    pid = card.get("principle_id")
    if any(not isinstance(v, str) or not v.strip()
           for v in (title, observation, question, pid)):
        raise ValueError("Card missing title, principle ID, observation, or review question")
    replay = _validate_replay(
        card.get("replay"), label=f"card {pid}", fps=fps,
        origin=origin, container=shot_replay,
    )
    display_title, display_question = DISPLAY_TEXT.get(pid, (title, question))
    return {
        "display_title": display_title,
        "player_review_question": display_question,
        "evidence_explanation": {
            "status": "measured_descriptive_only",
            "label": "Observed in the video (2D)",
            "what_this_supports": (
                f"{card['measurement_count']} descriptive measurement "
                f"{'statement' if card['measurement_count'] == 1 else 'statements'} "
                "linked to the listed replay frames."
            ),
            "frame_range": {
                "start_frame": replay["start_frame"],
                "end_frame": replay["end_frame"],
            },
            "limitations": limitations,
            "source_context": (
                "Source links provide published coaching context, not an endorsement "
                "of this shot or independent verification of specific video timestamps."
                if card["source_urls"] else
                "No source URL accompanies this card; do not treat it as sourced advice."
            ),
            "confidence": {
                "kind": "not_scored",
                "label": "No calibrated confidence score available",
                "explanation": (
                    "Sample counts, detector confidence, or the presence of a source "
                    "do not establish biomechanical accuracy or form correctness."
                ),
            },
            "can_grade_form": False,
            "can_recommend_correction": False,
        },
    }


class PlayerEvidenceExplainer:
    """Enrich an existing player-feedback document without replacing old fields."""

    def enrich(self, report: dict) -> dict:
        if not isinstance(report, dict):
            raise ValueError("Player report must be a JSON object")
        if (
            report.get("intended_form_match") != "not_assessed"
            or type(report.get("corrections_generated")) is not int
            or report["corrections_generated"] != 0
            or report.get("official_endorsement") is not False
        ):
            raise ValueError("Unsafe form verdict, correction count, or endorsement")
        fps = _finite(report.get("fps"), "fps", minimum=0.001)
        origin = report.get("frame_origin")
        if type(origin) is not int or origin not in (0, 1):
            raise ValueError("Frame origin must be zero or one")
        shots = report.get("shots")
        if not isinstance(shots, list):
            raise ValueError("Missing shots array")
        if _nonneg_int(report.get("shot_count"), "shot_count") != len(shots):
            raise ValueError("Report shot count mismatch")
        # A report that already contains explanation fields must be generated anew;
        # don't silently overwrite unrelated data from an unknown source.
        if "evidence_explanations_version" in report:
            raise ValueError("Input appears to have already been enriched")

        enriched = deepcopy(report)
        seen_shots = set()
        for shot in enriched["shots"]:
            if not isinstance(shot, dict):
                raise ValueError("Shot must be an object")
            number = _positive_int(shot.get("shot_number"), "shot_number")
            if number in seen_shots:
                raise ValueError("Duplicate shot number")
            seen_shots.add(number)
            if shot.get("intended_form_match") != "not_assessed":
                raise ValueError("Shot contains unsafe verdict")
            quality = shot.get("timing_quality")
            if quality not in TIMING:
                raise ValueError("Unrecognized timing quality")
            replay = _validate_replay(
                shot.get("replay"), label=f"shot {number}", fps=fps, origin=origin
            )
            cards = shot.get("feedback_cards")
            if not isinstance(cards, list):
                raise ValueError("Shot feedback_cards must be a list")
            if _nonneg_int(shot.get("feedback_count"), "feedback_count") != len(cards):
                raise ValueError("Shot feedback_count mismatch")
            if cards and quality != "sufficient_for_descriptive_timing":
                raise ValueError("Feedback cards cannot bypass the timing gate")

            seen_ids = set()
            for card in cards:
                pid = card.get("principle_id") if isinstance(card, dict) else None
                if pid in seen_ids:
                    raise ValueError("Duplicate feedback topic")
                seen_ids.add(pid)
                extra = _card_explanation(card, fps, origin, replay)
                for key in extra:
                    if key in card:
                        raise ValueError(f"Card already contains {key}")
                card.update(extra)
            if "evidence_explanation" in shot:
                raise ValueError("Shot already contains an evidence_explanation")
            shot["evidence_explanation"] = _shot_explanation(quality, len(cards))
        if seen_shots != set(range(1, len(shots) + 1)):
            raise ValueError("Shot numbering is not contiguous")
        enriched["evidence_explanations_version"] = "1.0"
        enriched["evidence_confidence_scoring"] = "not_available"
        enriched["evidence_explanation_limits"] = list(GLOBAL_LIMITS)
        return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Saved Phase 6 player_feedback JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = PlayerEvidenceExplainer().enrich(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print("Shots:", result["shot_count"])
    for shot in result["shots"]:
        print(
            f"Shot {shot['shot_number']}: {shot['feedback_count']} evidence-annotated cards; "
            f"{shot['evidence_explanation']['status']}"
        )
    print("Form verdict:", result["intended_form_match"])
    print("Corrections generated:", result["corrections_generated"])
    print("Confidence scoring:", result["evidence_confidence_scoring"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
