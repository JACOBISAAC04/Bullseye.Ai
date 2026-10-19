"""Allow only grounded, preapproved sentences in Bullseye voice scripts.

The model may select presentation style and order of *existing* observed
statements; it cannot introduce mechanics, diagnoses, outcomes or numbers.
"""
from __future__ import annotations

from collections.abc import Sequence

STYLES = {
    "focused": (
        "Shot {shot}. Here is what the video measurements show.",
        "Replay this shot and look at how the observed movements relate.",
    ),
    "friendly": (
        "Let's review shot {shot} together.",
        "As you replay it, notice how these observed movements fit together.",
    ),
    "direct": (
        "Shot {shot}. Let's focus on the measured movements.",
        "Look at these movements again in the shot replay.",
    ),
}
SAFETY_LINE = "These camera observations do not confirm a shooting mistake."
MAX_WORDS = 85


def validate_plan(plan: object, allowed_ids: Sequence[str]) -> tuple[str, list[str]]:
    """Reject unknown/duplicate facts and all free-text claims from an LLM."""
    if not isinstance(plan, dict) or set(plan) != {"style", "evidence_ids"}:
        raise ValueError("Gemini plan must have only style and evidence_ids")
    style = plan["style"]
    if not isinstance(style, str) or style not in STYLES:
        raise ValueError("Unapproved narration style")
    ids = plan["evidence_ids"]
    if not isinstance(ids, list) or not all(type(item) is str for item in ids):
        raise ValueError("evidence_ids must be strings")
    if len(ids) != len(allowed_ids) or len(ids) != len(set(ids)):
        raise ValueError("Every approved evidence item must appear exactly once")
    if set(ids) != set(allowed_ids):
        raise ValueError("Unknown or omitted evidence IDs")
    return style, ids


def compose_script(shot_number: int, facts: Sequence[tuple[str, str]], plan: object) -> str:
    """Build a short English script without including any model-authored fact."""
    if type(shot_number) is not int or shot_number < 1:
        raise ValueError("Invalid shot number")
    if not facts or len(facts) > 2:
        raise ValueError("Expected one or two approved facts")
    ids = [item[0] for item in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence IDs")
    for identifier, sentence in facts:
        if (not isinstance(identifier, str) or not identifier or
                not isinstance(sentence, str) or not sentence.strip() or
                "\x00" in sentence):
            raise ValueError("Invalid approved fact")
    style, ordered = validate_plan(plan, ids)
    by_id = dict(facts)
    opening, review = STYLES[style]
    script = " ".join(
        [opening.format(shot=shot_number)]
        + [by_id[item] for item in ordered]
        + [review, SAFETY_LINE]
    )
    if len(script.split()) > MAX_WORDS:
        raise ValueError("Script exceeded safety length")
    return script
