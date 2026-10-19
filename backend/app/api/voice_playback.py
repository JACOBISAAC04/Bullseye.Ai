"""Phase 6: on-demand voice coaching + replay for existing Bullseye jobs.

Mount router from app.main. Does not modify the established /video/analyze flow.
POST voice creation is intentionally restricted to local loopback clients until
authentication and rate limiting are implemented for remote deployment.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from app.coaching.player_feedback import PlayerFeedbackBuilder, _video_fps
from app.coaching.player_evidence_explanations import PlayerEvidenceExplainer
from app.coaching.coaching_audio_pipeline import run_pipeline

router = APIRouter(prefix="/video", tags=["Player Voice Playback"])
logger = logging.getLogger(__name__)

ANALYZED_DIR = Path("data/analyzed")
UPLOAD_DIR = Path("data/videos")
TEMPLATE = Path(__file__).with_name("voice_player.html")
VOICE_FOLDER = "voice_playback"
# Serialize voice creation in this local prototype: no duplicate paid Gemini calls
# from simultaneous browser requests within the same Python worker.
_VOICE_CREATE_LOCK = Lock()


def _job_folder(job_id: UUID) -> Path:
    return ANALYZED_DIR / str(job_id)


def _voice_folder(job_id: UUID) -> Path:
    return _job_folder(job_id) / VOICE_FOLDER


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis file missing: {path.name}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return obj


def _uploaded_video(job_id: UUID) -> Path:
    # The original upload has a canonical UUID basename with an allowed extension.
    matches = [
        UPLOAD_DIR / f"{job_id}{suffix}"
        for suffix in (".mp4", ".mov", ".avi", ".mkv")
        if (UPLOAD_DIR / f"{job_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError("Original uploaded video not found uniquely")
    return matches[0]


def build_job_player_report(job_id: UUID) -> dict:
    """Rebuild a display report from existing files; never re-run detectors."""
    job = _job_folder(job_id)
    temporal = _read_json(job / "temporal_form.json")
    measured = _read_json(job / "measured_coaching_report.json")
    frames = temporal.get("shots")
    if not isinstance(frames, list) or temporal.get("shot_count") != len(frames):
        raise ValueError("Temporal shot count mismatch")
    analysis_shots = []
    for shot in frames:
        if not isinstance(shot, dict):
            raise ValueError("Temporal shots must be objects")
        analysis_shots.append({
            key: shot[key]
            for key in ("shot_number", "arm", "start_frame", "release_frame", "end_frame")
        })
    analysis = {
        "job_id": str(job_id),
        "video_url": f"/video/results/{job_id}/video",
        "shot_count": len(analysis_shots),
        "shots": analysis_shots,
    }
    fps = _video_fps(_uploaded_video(job_id))
    plain = PlayerFeedbackBuilder().generate(
        analysis, measured, fps=fps, frame_origin=1
    )
    return PlayerEvidenceExplainer().enrich(plain)


def _generated_voice_exists(job_id: UUID) -> bool:
    voice = _voice_folder(job_id)
    return (voice / "voice_manifest.json").is_file() and (
        voice / "player_feedback.json"
    ).is_file()


def generate_job_voice(job_id: UUID, use_gemini: bool = False) -> dict:
    """Create playback files once, without overwriting finished/partial results."""
    with _VOICE_CREATE_LOCK:
        if _generated_voice_exists(job_id):
            return _manifest_response(job_id)
        job = _job_folder(job_id)
        if not (job / "shooting_analysis.mp4").is_file():
            raise FileNotFoundError("Analyzed video not found")
        destination = _voice_folder(job_id)
        if destination.exists():
            raise FileExistsError(
                "An incomplete voice output directory already exists; "
                "inspect it before retrying"
            )
        report = build_job_player_report(job_id)
        work = Path(tempfile.mkdtemp(prefix=".voice_build_", dir=job))
        try:
            run_pipeline(
                report,
                work,
                use_gemini=use_gemini,
                script_only=False,
            )
            # Finish the JSON report before exposing the directory to GET routes.
            (work / "player_feedback.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            work.rename(destination)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return _manifest_response(job_id)


def _manifest_response(job_id: UUID) -> dict:
    manifest = deepcopy(
        _read_json(_voice_folder(job_id) / "voice_manifest.json")
    )
    shots = manifest.get("shots")
    if not isinstance(shots, list):
        raise ValueError("Voice manifest missing shots")
    for shot in shots:
        n = shot.get("shot_number")
        if type(n) is not int or n < 1 or (
            shot.get("audio_filename") != f"shot_{n:03d}_english_ai_voice.wav"
        ):
            raise ValueError("Unexpected voice audio filename")
        shot["audio_url"] = f"/video/results/{job_id}/voice/shot/{n}"
    manifest["video_stream_url"] = f"/video/results/{job_id}/stream"
    manifest["player_feedback_url"] = (
        f"/video/results/{job_id}/voice/player-feedback"
    )
    return manifest


@router.get("/results/{job_id}/player", response_class=HTMLResponse)
def voice_player_page(job_id: UUID):
    if not (_job_folder(job_id) / "shooting_analysis.mp4").is_file():
        raise HTTPException(status_code=404, detail="Analyzed video not found")
    # UUID is validated by FastAPI; replacing it cannot inject HTML or JS.
    template = TEMPLATE.read_text(encoding="utf-8")
    return HTMLResponse(template.replace("__JOB_ID__", str(job_id)))


@router.get("/results/{job_id}/stream")
def stream_analyzed_video(job_id: UUID):
    folder = _job_folder(job_id)

    browser_video = folder / "shooting_analysis_browser.mp4"
    original_video = folder / "shooting_analysis.mp4"

    path = (
        browser_video
        if browser_video.is_file()
        else original_video
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Analyzed video not found"
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        content_disposition_type="inline"
    )


@router.post("/results/{job_id}/voice")
async def create_job_voice(
    job_id: UUID,
    request: Request,
    use_gemini: bool = Query(default=False),
):
    # Avoid an unauthenticated public endpoint that spends Gemini API quota.
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(
            status_code=403,
            detail="Voice generation is local-only until authentication is added",
        )
    try:
        return await run_in_threadpool(generate_job_voice, job_id, use_gemini)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        logger.warning("Voice input validation failed for %s: %s", job_id, exc)
        raise HTTPException(
            status_code=422,
            detail="Saved evidence is incomplete or inconsistent",
        ) from exc
    except Exception as exc:
        logger.exception("Voice creation failed for job %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Voice generation failed. Check local backend logs.",
        ) from exc


@router.get("/results/{job_id}/voice/manifest")
def get_job_voice_manifest(job_id: UUID):
    try:
        return _manifest_response(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Voice not generated") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="Invalid voice manifest") from exc


@router.get("/results/{job_id}/voice/player-feedback")
def get_job_player_feedback(job_id: UUID):
    path = _voice_folder(job_id) / "player_feedback.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Player feedback not generated")
    return FileResponse(path, media_type="application/json")


@router.get("/results/{job_id}/voice/shot/{shot_number}")
def stream_voice_shot(job_id: UUID, shot_number: int):
    if shot_number < 1:
        raise HTTPException(status_code=404, detail="Invalid shot number")
    try:
        manifest = _manifest_response(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Voice not generated") from exc
    shots = [
        item for item in manifest["shots"]
        if item["shot_number"] == shot_number
    ]
    if len(shots) != 1:
        raise HTTPException(status_code=404, detail="Shot not found")
    path = _voice_folder(job_id) / shots[0]["audio_filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Shot audio not found")
    return FileResponse(
        path, media_type="audio/wav", content_disposition_type="inline"
    )
