"""Bullseye v4: speak detailed, stage-specific coaching checkpoints.

Drop-in replacement for create_focus_voice in app.api.coach_voice_browser.
Produces the same manifest/WAV filenames in a NEW per-job folder so older
voice generations are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from app.coaching.coaching_explainer_v4 import ShotSpecificCoachingExplainer
from app.coaching.gemini_tts import GeminiTTSClient, DEFAULT_TTS_MODEL
from app.coaching.english_voice import synthesize_windows_wav

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def create_focus_voice(
    report: dict, output_dir: Path, *,
    tts: str = "gemini", voice: str = "Kore",
    tts_model: str = DEFAULT_TTS_MODEL,
    rate: int = 0, max_focus: int = 3,
    script_only: bool = False,
    tts_client=None, windows_synth=None,
) -> dict:
    if tts not in ("gemini", "windows"):
        raise ValueError("tts must be gemini or windows")
    if type(rate) is not int or not -10 <= rate <= 10:
        raise ValueError("Windows speech rate must be -10..10")
    if not isinstance(voice, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", voice) is None:
        raise ValueError("Invalid voice")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Output folder already exists; use a new versioned path")

    coaching = ShotSpecificCoachingExplainer().build(report, max_focus=max_focus)
    client = tts_client
    if tts == "gemini" and client is None and not script_only:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if key:
            try:
                client = GeminiTTSClient(
                    key, model=tts_model, voice=voice, timeout=30,
                )
            except (ValueError, TypeError):
                pass
    windows_fn = windows_synth or synthesize_windows_wav
    manifest = {
        "schema_version": "4.0",
        "language": "en",
        "narration_type": "shot_specific_evidence_gated_basketball_coaching",
        "shot_count": coaching["shot_count"],
        "made_missed": "not_assessed",
        "confirmed_mistakes": 0,
        "shots": [],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + "_checkpoint")

    import hashlib

    identity = {
        "tts": tts,
        "voice": voice,
        "model": tts_model,
        "rate": rate,
        "script_only": script_only,
        "coaching": coaching,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    marker = staged / "checkpoint.json"

    if staged.exists():
        if not marker.is_file():
            raise FileExistsError("Unrecognized checkpoint directory")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise FileExistsError("Checkpoint belongs to different settings")
    else:
        staged.mkdir()
        marker.write_text(
            json.dumps({"fingerprint": fingerprint}),
            encoding="utf-8",
        )

    for shot in coaching["shots"]:
        number = shot["shot_number"]
        filename = f"shot_{number:03d}_english_focus.wav"
        text = shot["voice_script"]
        audio_file = staged / filename
        cache_file = staged / (filename + ".cache.json")
        audio_status = "script_only"

        if not script_only:
            cached = False

            if audio_file.is_file() and cache_file.is_file():
                try:
                    metadata = json.loads(cache_file.read_text(encoding="utf-8"))
                    audio = audio_file.read_bytes()
                    cached = (
                        metadata["fingerprint"] == fingerprint
                        and metadata["script"] == text
                        and metadata["sha256"] == hashlib.sha256(audio).hexdigest()
                    )
                    if cached:
                        GeminiTTSClient.validate_wav(audio)
                        audio_status = metadata["tts_status"]
                except (OSError, ValueError, KeyError, TypeError):
                    cached = False

            if not cached:
                if tts == "gemini" and shot["focus_count"] == 0:
                    windows_fn(text, audio_file, rate=rate)
                    audio_status = "windows_no_qualified_evidence"
                elif tts == "gemini":
                    if client is None:
                        raise RuntimeError("Gemini TTS unavailable; no silent Windows fallback")
                    try:
                        audio = client.synthesize(text)
                        GeminiTTSClient.validate_wav(audio)
                        audio_file.write_bytes(audio)
                        audio_status = "gemini_generated"
                    except Exception as exc:
                        raise RuntimeError(
                            f"Gemini TTS failed for shot {number}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                else:
                    windows_fn(text, audio_file, rate=rate)
                    audio_status = "windows_generated"

                if not audio_file.is_file():
                    raise ValueError("Audio was not created")

                audio = audio_file.read_bytes()
                GeminiTTSClient.validate_wav(audio)

                metadata = {
                    "fingerprint": fingerprint,
                    "script": text,
                    "sha256": hashlib.sha256(audio).hexdigest(),
                    "tts_status": audio_status,
                }

                temporary = cache_file.with_suffix(".tmp")
                temporary.write_text(json.dumps(metadata), encoding="utf-8")
                temporary.replace(cache_file)

        manifest["shots"].append({
            "shot_number": number,
            "replay": shot["replay"],
            "focus_count": shot["focus_count"],
            "voice_script": text,
            "spoken_word_count": shot["spoken_word_count"],
            "audio_filename": filename,
            "tts_status": audio_status,
            "shot_outcome": shot["shot_outcome"],
            "technique_assessment": shot["technique_assessment"],
            "coaching_steps": shot["focuses"],
            "shot_specific_observations": shot["shot_specific_observations"],
            "narration_status": shot["narration_status"],
            "source_urls": list(dict.fromkeys(
                url for focus in shot["focuses"]
                for url in focus["source_urls"]
            )),
        })

    (staged / "coaching_decisions.json").write_text(
        json.dumps(coaching, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (staged / "voice_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    staged.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tts", choices=("gemini", "windows"), default="gemini")
    parser.add_argument("--voice", default="Kore")
    parser.add_argument("--script-only", action="store_true")
    args = parser.parse_args()
    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        result = create_focus_voice(
            report, args.output_dir, tts=args.tts, voice=args.voice,
            script_only=args.script_only,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    for shot in result["shots"]:
        print(
            f"Shot {shot['shot_number']}: {shot['focus_count']} coaching steps; "
            f"{shot['spoken_word_count']} words; TTS {shot['tts_status']}"
        )
    print("Saved:", args.output_dir / "voice_manifest.json")


if __name__ == "__main__":
    main()
