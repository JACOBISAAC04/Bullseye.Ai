"""Coach-style English audio for Bullseye's existing shot replay page.

Separate from legacy /voice: no model/video reanalysis, no overwriting old WAVs.
Uses validated practice focuses and Gemini TTS, with a Windows voice fallback.
"""
from __future__ import annotations

import json
import re
from urllib.error import HTTPError
import logging
from copy import deepcopy
from pathlib import Path
from threading import Lock
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.voice_playback import _job_folder, build_job_player_report
from app.coaching.coaching_explanation_voice_v4 import create_focus_voice

router = APIRouter(prefix="/video", tags=["Coach-Style Voice"])
logger = logging.getLogger(__name__)
VOICE_FOLDER = "coach_voice_v4"
_LOCK = Lock()


def _directory(job_id: UUID) -> Path:
    return _job_folder(job_id) / VOICE_FOLDER


def _manifest(job_id: UUID) -> dict:
    folder = _directory(job_id)
    file = folder / "voice_manifest.json"
    if not file.is_file():
        raise FileNotFoundError("Coach-style voice has not been generated")

    manifest = json.loads(file.read_text(encoding="utf-8"))
    shots = manifest.get("shots")
    if not isinstance(shots, list):
        raise ValueError("Invalid coach-voice manifest")

    output = deepcopy(manifest)
    for shot in output["shots"]:
        number = shot.get("shot_number")
        filename = shot.get("audio_filename")
        if (
            type(number) is not int or number < 1
            or filename != f"shot_{number:03d}_english_focus.wav"
            or not (folder / filename).is_file()
        ):
            raise ValueError("Incomplete or unsafe coach-voice manifest")

        # Existing HTML can display text and replay times without knowing the
        # internal practice-focus report format.
        shot["narration_text"] = shot["voice_script"]
        shot["feedback_count"] = shot["focus_count"]
        shot["gemini_status"] = "focus_cues"  # not a claim of Gemini text writing
        shot["audio_url"] = (
            f"/video/results/{job_id}/coach-voice/shot/{number}"
        )

    output["coaching_type"] = "evidence_gated_practice_focus"
    output["confirmed_mistakes"] = 0
    output["made_missed"] = "not_assessed"
    return output


def generate_coach_voice(job_id: UUID, use_gemini: bool = True) -> dict:
    """Generate in an independent per-job folder, leaving old audio intact."""
    with _LOCK:
        directory = _directory(job_id)
        if (directory / "voice_manifest.json").is_file():
            existing = _manifest(job_id)
            statuses = {shot["tts_status"] for shot in existing["shots"]}
            if use_gemini:
                allowed = {"gemini_generated", "windows_no_qualified_evidence"}
            else:
                allowed = {"windows_generated"}
            if not statuses.issubset(allowed):
                raise FileExistsError(
                    "Existing voice uses a different TTS engine; "
                    "preserved without overwriting"
                )
            return existing
        if directory.exists():
            raise FileExistsError(
                "Incomplete coach-voice folder exists; examine it before retrying"
            )
        job = _job_folder(job_id)
        if not (job / "shooting_analysis.mp4").is_file():
            raise FileNotFoundError("Analyzed video not found")
        report = build_job_player_report(job_id)
        create_focus_voice(
            report, directory,
            tts="gemini" if use_gemini else "windows",
            voice="Kore",
        )
        return _manifest(job_id)


@router.post("/results/{job_id}/coach-voice")
async def create_coach_voice(
    job_id: UUID, request: Request,
    use_gemini: bool = Query(default=True),
):
    # Gemini API calls must not be exposed as unauthenticated public endpoints.
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Local-only voice generation")
    try:
        return await run_in_threadpool(generate_coach_voice, job_id, use_gemini)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Invalid coach-voice input for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=422, detail="Shot evidence is incomplete or inconsistent"
        ) from exc
    except Exception as exc:
        cause = exc
        while cause is not None:
            if isinstance(cause, HTTPError) and cause.code == 429:
                body = cause.read(100_000).decode("utf-8", errors="replace")
                match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?(?:h|m|s)(?:[0-9]+(?:\.[0-9]+)?(?:h|m|s))*)", body)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "gemini_quota_exhausted",
                        "message": "Gemini daily voice quota exhausted.",
                        "retry_after": match.group(1) if match else None,
                    },
                ) from exc
            cause = cause.__cause__

        logger.exception("Coach voice failed for job %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Coach voice generation failed",
        ) from exc


@router.get("/results/{job_id}/coach-voice/manifest")
def coach_voice_manifest(job_id: UUID):
    try:
        return _manifest(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="Invalid coach-voice manifest") from exc


@router.get("/results/{job_id}/coach-voice/shot/{shot_number}")
def coach_voice_shot(job_id: UUID, shot_number: int):
    if shot_number < 1:
        raise HTTPException(status_code=404, detail="Invalid shot")
    try:
        manifest = _manifest(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="Invalid coach-voice manifest") from exc
    matches = [item for item in manifest["shots"] if item["shot_number"] == shot_number]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="Shot not found")
    file = _directory(job_id) / matches[0]["audio_filename"]
    return FileResponse(
        file, media_type="audio/wav", content_disposition_type="inline",
        headers={"Cache-Control": "no-store"},
    )
