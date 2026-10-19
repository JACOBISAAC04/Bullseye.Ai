"""Bullseye Gemini-assisted scripts -> existing offline English WAV synthesis.

Run from D:\\Bullseye\\backend:
    python -m app.coaching.coaching_audio_pipeline \\
      --input data/analyzed/phase6_evidence_feedback_video3.json \\
      --output-dir data/analyzed/phase6_gemini_voice_video3 \\
      --use-gemini

No key? It automatically uses safe local narration.
Add --script-only to test output without Windows TTS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from app.coaching.english_voice import synthesize_windows_wav
from app.coaching.gemini_coaching_writer import (
    DEFAULT_MODEL,
    GeminiCoachingWriter,
)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def run_pipeline(
    report: dict, directory: Path, *,
    use_gemini: bool = False,
    model: str = DEFAULT_MODEL,
    script_only: bool = False,
    rate: int = 0,
    timeout: float = 20,
    writer: object | None = None,
) -> dict:
    """Build, optionally speak, and persist non-overwriting voice manifest."""
    if type(rate) is not int or not -10 <= rate <= 10:
        raise ValueError("Speech rate must be between -10 and 10")
    if not isinstance(directory, Path):
        directory = Path(directory)
    directory = directory.resolve()
    manifest_path = directory / "voice_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "voice_manifest.json already exists; choose a new output directory"
        )
    builder = writer if writer is not None else GeminiCoachingWriter()
    manifest = builder.build(
        report, use_gemini=use_gemini, model=model, timeout=timeout
    )
    directory.mkdir(parents=True, exist_ok=True)
    if not script_only:
        for shot in manifest["shots"]:
            target = directory / shot["audio_filename"]
            if target.exists():
                raise FileExistsError(f"Audio already exists: {target}")
            synthesize_windows_wav(shot["narration_text"], target, rate=rate)
            shot["audio_status"] = "generated"
    # Write last: do not advertise an incomplete manifest if WAV synthesis fails.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--use-gemini", action="store_true",
                        help="Opt in to a Gemini request for each qualified shot")
    parser.add_argument("--script-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rate", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()

    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        manifest = run_pipeline(
            report, args.output_dir,
            use_gemini=args.use_gemini,
            model=args.model,
            script_only=args.script_only,
            rate=args.rate,
            timeout=args.timeout,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print("Language: English | Speech: offline Windows TTS")
    for shot in manifest["shots"]:
        print(
            f"Shot {shot['shot_number']}: "
            f"{shot['spoken_word_count']} words; "
            f"Gemini: {shot['gemini_status']}; "
            f"audio: {shot['audio_status']}"
        )
    print("Mistakes: not validated | Made/missed: not assessed")
    print("Manifest:", (args.output_dir / "voice_manifest.json").resolve())


if __name__ == "__main__":
    main()
