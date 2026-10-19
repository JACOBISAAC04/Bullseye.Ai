"""Short, measurement-grounded offline English voice summaries for Bullseye.

This is an additive Phase 6 module. It reuses the established Phase 6
EnglishVoiceCoaching validation and Windows SAPI synthesis. Does not alter
video.py, shot tracking, source citations, or player-feedback reports.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

from app.coaching.english_voice import EnglishVoiceCoaching, synthesize_windows_wav

_ANGLE = re.compile(
    r"\b(hip|knee) angle changed from\s+(\d+(?:\.\d+)?)\s*°"
    r".{0,110}?\bto\s+(\d+(?:\.\d+)?)\s*°",
    re.IGNORECASE | re.DOTALL,
)
_BALL_PATH = re.compile(
    r"accepted ball-detector endpoint moved\s+"
    r"(\d+(?:\.\d+)?)\s+px\s+(left|right)\s+and\s+"
    r"(\d+(?:\.\d+)?)\s+px\s+(up|down)\s+in the image",
    re.IGNORECASE,
)
_WRIST = re.compile(
    r"detected ball center was\s+(\d+(?:\.\d+)?)\s+image\s+px"
    r"\s+from the wrist.*?\band\s+(\d+(?:\.\d+)?)\s+image\s+px",
    re.IGNORECASE | re.DOTALL,
)
_PRIORITY = ("hip_loading", "ball_path", "ball_setup")
_MAX_SPOKEN_TOPICS = 2


def _finite_angle(value: str) -> float | None:
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 180 else None


def _finite_px(value: str) -> float | None:
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 100_000 else None


def _angle_sentence(text: str) -> str | None:
    values: dict[str, float] = {}
    for joint, beginning, ending in _ANGLE.findall(text):
        a, b = _finite_angle(beginning), _finite_angle(ending)
        if a is None or b is None:
            continue
        values[joint.lower()] = round(b - a, 1)
    if not values:
        return None
    phrases = []
    for joint in ("hip", "knee"):
        if joint not in values:
            continue
        change = values[joint]
        if abs(change) < 0.5:
            phrases.append(f"your estimated {joint} angle was nearly unchanged")
        else:
            direction = "increased" if change > 0 else "decreased"
            phrases.append(
                f"your estimated {joint} angle {direction} by about "
                f"{abs(change):.0f} degrees"
            )
    return "Between two observed moments, " + ", and ".join(phrases) + "."


def _ball_sentence(text: str) -> str | None:
    match = _BALL_PATH.search(text)
    if match is None:
        return None
    horizontal, horizontal_dir, vertical, vertical_dir = match.groups()
    dx, dy = _finite_px(horizontal), _finite_px(vertical)
    if dx is None or dy is None:
        return None
    # This is an image-plane endpoint difference, not a reconstructed flight arc.
    return (
        f"In the camera image, the detected ball moved about "
        f"{dy:.0f} pixels {vertical_dir.lower()} between observed points."
    )


def _wrist_sentence(text: str) -> str | None:
    match = _WRIST.search(text)
    if match is None:
        return None
    first, last = (_finite_px(x) for x in match.groups())
    if first is None or last is None:
        return None
    return (
        "The detected ball-to-wrist distance changed from about "
        f"{first:.0f} to {last:.0f} pixels in the image."
    )


def _pick_observations(cards: list[dict]) -> list[tuple[str, str]]:
    by_id = {c["principle_id"]: c for c in cards}
    builders = {
        "hip_loading": _angle_sentence,
        "ball_path": _ball_sentence,
        "ball_setup": _wrist_sentence,
    }
    choices: list[tuple[str, str]] = []
    for topic in _PRIORITY:
        card = by_id.get(topic)
        if card is None:
            continue
        statement = builders[topic](card["observation_text"])
        if statement:
            choices.append((topic, statement))
        if len(choices) >= _MAX_SPOKEN_TOPICS:
            break
    return choices


class ConciseEnglishVoice:
    """Build ≤75-word narration while preserving the verified voice manifest."""

    def build(self, report: dict) -> dict:
        # Reuses the existing strict report validation, including no-grading,
        # timing gates, source URLs, replay checks, and no invented corrections.
        original = EnglishVoiceCoaching().build(report)
        result = copy.deepcopy(original)
        result["schema_version"] = "1.1"
        result["narration_style"] = "concise_measured"
        for entry, shot in zip(result["shots"], report["shots"]):
            number = entry["shot_number"]
            choices = _pick_observations(shot["feedback_cards"])
            if not shot["feedback_cards"]:
                script = (
                    f"Shot {number}. There is not enough reliable preparation "
                    "evidence for detailed coaching. That does not mean your "
                    "shooting form was wrong. A clearer, longer recording may help."
                )
            elif not choices:
                script = (
                    f"Shot {number}. Bullseye recorded shooting movement to review, "
                    "but there is not enough suitable numerical detail for a short "
                    "spoken summary. Replay the shot and check the on-screen "
                    "observations. No shooting mistake has been confirmed."
                )
            else:
                parts = [f"Shot {number}."]
                parts.extend(statement for _, statement in choices)
                parts.append(
                    "Replay the shot to review how these movements relate. "
                    "Camera measurements alone do not prove a shooting mistake."
                )
                script = " ".join(parts)
            if len(script.split()) > 75:
                raise ValueError("Concise narration exceeded 75-word limit")
            entry["narration_text"] = script
            entry["spoken_topics"] = [topic for topic, _ in choices]
            entry["spoken_word_count"] = len(script.split())
            entry["audio_filename"] = f"shot_{number:03d}_english_concise.wav"
            entry["audio_status"] = "script_only"
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--script-only", action="store_true")
    parser.add_argument("--rate", type=int, default=0)
    args = parser.parse_args()
    if not -10 <= args.rate <= 10:
        parser.error("--rate must be between -10 and 10")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = ConciseEnglishVoice().build(report)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "voice_manifest.json"
    if manifest_path.exists():
        parser.error("voice_manifest.json already exists; choose a fresh output folder")
    if not args.script_only:
        for shot in manifest["shots"]:
            target = directory / shot["audio_filename"]
            synthesize_windows_wav(shot["narration_text"], target, rate=args.rate)
            shot["audio_status"] = "generated"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    for shot in manifest["shots"]:
        print(
            f"Shot {shot['shot_number']}: {shot['spoken_word_count']} words, "
            f"{len(shot['spoken_topics'])} spoken measurement topics, "
            f"{shot['audio_status']}"
        )
    print("Made/missed: not assessed; corrections: 0; form: not_assessed")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
