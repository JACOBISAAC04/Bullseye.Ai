import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.shooting_phase_detector import ShootingPhaseDetector

router = APIRouter(prefix="/video", tags=["Video"])

UPLOAD_DIR = "data/videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)

analyzer = PoseAnalyzer()
phase_detector = ShootingPhaseDetector()


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    filename = f"{uuid.uuid4()}{extension}"
    video_path = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()

    with open(video_path, "wb") as video_file:
        video_file.write(contents)

    try:
        result = analyzer.analyze_video(video_path)

        phases = phase_detector.detect(
            result["frames"]
        )

        result["shooting_phases"] = phases
    except Exception as e:
        os.remove(video_path)
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": filename,
        "analysis": result
    }