"""Bullseye Phase 6: offline, English-only spoken coaching summaries.

Consumes Phase 6 player-feedback JSON (plain or evidence-enriched).
Narrates only already-qualified descriptive observations and review questions.
It does not detect made/missed shots, calculate ideal trajectories, or infer mistakes.

Windows speech synthesis uses a separate, static PowerShell System.Speech script.
No API key, network connection, or third-party Python package is required.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ALLOWED_TOPICS = {
    "posture_setup",
    "hip_loading",
    "ground_balance",
    "ball_setup",
    "ball_path",
    "individualized_feedback",
}
ALLOWED_TIMING = {"sufficient_for_descriptive_timing", "insufficient_for_reliable_timing"}
DISCLAIMER = (
    "This review describes two dimensional camera measurements only. "
    "It does not assign a shooting form grade or identify proven mistakes."
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _time(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if any(ch in value for ch in "\x00"):
        raise ValueError(f"{name} contains invalid characters")
    return value.strip()


def _english_readable(text: str) -> str:
    """Make existing descriptive text less awkward for Windows speech voices."""
    text = text.replace("°", " degrees")
    text = re.sub(r"\bpx\b", "pixels", text)
    text = re.sub(r"\b2D\b", "two dimensional", text, flags=re.I)
    text = re.sub(r"\bFPS\b", "frames per second", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_report(report: dict) -> None:
    if not isinstance(report, dict):
        raise ValueError("Report must be an object")
    if (report.get("intended_form_match") != "not_assessed"
            or report.get("official_endorsement") is not False
            or type(report.get("corrections_generated")) is not int
            or report["corrections_generated"] != 0):
        raise ValueError("Unsafe form judgment, correction count, or endorsement")
    shots = report.get("shots")
    if not isinstance(shots, list):
        raise ValueError("Missing shots list")
    if _integer(report.get("shot_count"), "shot_count") != len(shots):
        raise ValueError("Shot count mismatch")
    fps = report.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("A valid FPS is required")
    origin = report.get("frame_origin")
    if type(origin) is not int or origin not in (0, 1):
        raise ValueError("Frame origin must be 0 or 1")
    seen = set()
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("Shot must be an object")
        number = _integer(shot.get("shot_number"), "shot_number", minimum=1)
        if number in seen:
            raise ValueError("Duplicate shot numbers")
        seen.add(number)
        if shot.get("intended_form_match") != "not_assessed":
            raise ValueError("Unsafe shot verdict")
        if shot.get("timing_quality") not in ALLOWED_TIMING:
            raise ValueError("Unsupported timing quality")
        cards = shot.get("feedback_cards")
        if not isinstance(cards, list):
            raise ValueError("Missing feedback cards")
        if _integer(shot.get("feedback_count"), "feedback_count") != len(cards):
            raise ValueError("Feedback card count mismatch")
        if cards and shot["timing_quality"] != "sufficient_for_descriptive_timing":
            raise ValueError("Cannot narrate cards without reliable descriptive timing")
        _check_replay(shot.get("replay"), fps, origin, None)
        seen_topics = set()
        for card in cards:
            if not isinstance(card, dict):
                raise ValueError("Feedback card must be an object")
            topic = card.get("principle_id")
            if topic not in ALLOWED_TOPICS or topic in seen_topics:
                raise ValueError("Unknown or duplicate review topic")
            seen_topics.add(topic)
            if card.get("evidence_status") != "observed_2d_descriptive":
                raise ValueError("Unsupported card evidence")
            _integer(card.get("measurement_count"), "measurement_count", minimum=1)
            for label in ("observation_text", "review_question", "title"):
                _text(card.get(label), label)
            limits = card.get("limitations")
            if not isinstance(limits, list) or not limits or any(
                not isinstance(v, str) or not v.strip() for v in limits
            ):
                raise ValueError("Missing measurement limitations")
            urls = card.get("source_urls")
            if not isinstance(urls, list) or not urls:
                raise ValueError("Missing published supporting source")
            for url in urls:
                if not isinstance(url, str):
                    raise ValueError("Invalid supporting source")
                p = urlsplit(url)
                if (p.scheme != "https" or not p.hostname
                        or p.username or p.password or
                        any(x in url for x in "\r\n\t")):
                    raise ValueError("Invalid supporting source URL")
            _check_replay(card.get("replay"), fps, origin, shot["replay"])
            if "evidence_explanation" in card:
                evidence = card["evidence_explanation"]
                if (not isinstance(evidence, dict)
                        or evidence.get("status") != "measured_descriptive_only"
                        or evidence.get("can_grade_form") is not False
                        or evidence.get("can_recommend_correction") is not False
                        or not isinstance(evidence.get("confidence"), dict)
                        or evidence["confidence"].get("kind") != "not_scored"):
                    raise ValueError("Unsafe or malformed evidence explanation")
    if seen != set(range(1, len(shots) + 1)):
        raise ValueError("Shot numbering is not contiguous")


def _check_replay(replay: object, fps: float, origin: int, parent: object) -> None:
    if not isinstance(replay, dict):
        raise ValueError("Missing replay")
    start = _integer(replay.get("start_frame"), "replay start", minimum=origin)
    end = _integer(replay.get("end_frame"), "replay end", minimum=start)
    a = _time(replay.get("start_seconds"), "replay start seconds")
    b = _time(replay.get("end_seconds"), "replay end seconds")
    if b < a:
        raise ValueError("Replay seconds reversed")
    if abs(a - (start-origin)/fps) > .002 or abs(b - (end-origin)/fps) > .002:
        raise ValueError("Replay timestamps disagree with FPS")
    if parent is not None and not (
        parent["start_frame"] <= start <= end <= parent["end_frame"]
    ):
        raise ValueError("Card replay outside shot")


class EnglishVoiceCoaching:
    """Build speech scripts and source-linked metadata without speech synthesis."""

    def build(self, report: dict) -> dict:
        _validate_report(report)
        output = {
            "schema_version": "1.0",
            "language": "en",
            "speech_engine": "windows_system_speech_offline",
            "intended_form_match": "not_assessed",
            "corrections_generated": 0,
            "made_shots": "not_assessed",
            "missed_shots": "not_assessed",
            "limitations": [
                "Voice describes observed measurements, not validated shooting mistakes.",
                "No ideal hand or body placement or intended ball flight is computed.",
                "Shot make/miss outcomes cannot be inferred from this report.",
            ],
            "shots": [],
        }
        for shot in report["shots"]:
            no = shot["shot_number"]
            cards = shot["feedback_cards"]
            sentences = [f"Shot {no}.", DISCLAIMER]
            sources = []
            if not cards:
                sentences.append(
                    "There is not enough supported evidence for detailed coaching "
                    "on this shot. This does not mean your shooting form was wrong."
                )
                if shot["timing_quality"] == "insufficient_for_reliable_timing":
                    sentences.append(
                        "The shot preparation timing was too limited for a reliable review. "
                        "A clearer video showing more of the preparation may help."
                    )
            else:
                sentences.append(
                    f"{len(cards)} measured shooting movement topics are available."
                )
                for card in cards:
                    # Use the clarified display text only when supplied by the
                    # validated Phase 6.3 evidence enrichment.
                    title = card.get("display_title", card["title"])
                    question = card.get(
                        "player_review_question", card["review_question"]
                    )
                    sentences.append(_english_readable(_text(title, "title")) + ".")
                    sentences.append(_english_readable(card["observation_text"]))
                    sentences.append(
                        "Review question. " + _english_readable(_text(question, "question"))
                    )
                    for url in card["source_urls"]:
                        if url not in sources:
                            sources.append(url)
            sentences.append(
                "Measurements and source references are available in the on screen "
                "report. No shooting form correctness score was calculated."
            )
            output["shots"].append({
                "shot_number": no,
                "feedback_count": len(cards),
                "replay": dict(shot["replay"]),
                "source_urls": sources,
                "audio_filename": f"shot_{no:03d}_english.wav",
                "narration_text": "\n\n".join(sentences),
                "audio_status": "script_only",
            })
        return output


def synthesize_windows_wav(text: str, destination: Path, rate: int = -1) -> None:
    """Generate a local WAV with the Windows installed English SAPI voice."""
    if type(rate) is not int or not (-10 <= rate <= 10):
        raise ValueError("Rate must be an integer from -10 to 10")
    _text(text, "narration")
    script = Path(__file__).with_name("english_voice_sapi.ps1")
    if not script.is_file():
        raise FileNotFoundError(f"Missing Windows voice helper: {script}")
    if destination.suffix.lower() != ".wav":
        raise ValueError("Output voice file must have .wav extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_txt = destination.with_suffix(".narration.txt")
    if input_txt.exists() or destination.exists():
        raise FileExistsError("Refusing to overwrite existing narration or WAV")
    input_txt.write_text(text, encoding="utf-8")
    try:
        cmd = [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", str(script.resolve()),
            "-InputTextFile", str(input_txt.resolve()),
            "-OutputWav", str(destination.resolve()),
            "-Rate", str(rate),
        ]
        subprocess.run(cmd, check=True, timeout=240, capture_output=True, text=True)
        if not destination.is_file() or destination.stat().st_size < 44:
            raise RuntimeError("Windows speech engine returned no usable WAV")
    except subprocess.CalledProcessError as exc:
        destination.unlink(missing_ok=True)
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Windows English voice generation failed. " + detail
        ) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        input_txt.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--script-only", action="store_true",
                        help="Write narration JSON only; no Windows voice required")
    parser.add_argument("--rate", type=int, default=-1,
                        help="Windows speech rate from -10 to 10; default -1")
    args = parser.parse_args()
    if not -10 <= args.rate <= 10:
        parser.error("--rate must be from -10 to 10")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = EnglishVoiceCoaching().build(report)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "voice_manifest.json"
    if manifest.exists():
        parser.error("voice_manifest.json already exists; use a new output directory")
    if not args.script_only:
        for shot in output["shots"]:
            target = directory / shot["audio_filename"]
            synthesize_windows_wav(shot["narration_text"], target, rate=args.rate)
            shot["audio_status"] = "generated"
    manifest.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding="utf-8")
    print("Language: English (offline Windows speech)" if not args.script_only
          else "Language: English (script preview only)")
    for shot in output["shots"]:
        print(f"Shot {shot['shot_number']}: {shot['feedback_count']} described topics; "
              f"{shot['audio_status']} ({shot['audio_filename']})")
    print("Form verdict: not_assessed; corrections: 0; made/missed: not assessed")
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
