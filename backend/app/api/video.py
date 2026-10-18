import time
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from app.coaching.coaching_pipeline import CoachingReportPipeline
from app.coaching.measurement_coaching_pipeline import MeasurementCoachingPipeline

from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.shooting_phase_detector import ShootingPhaseDetector
from app.analysis.ball_analyzer import BallAnalyzer
from app.analysis.hoop_detector import HoopDetector
from app.analysis.shot_tracker import (
    ShotTracker,
    remove_ball_outliers,
    interpolate_ball_gaps,
)
from app.analysis.shooting_metrics import ShootingMetrics
from app.analysis.shooting_analysis_renderer import ShootingAnalysisRenderer

from tools.detect_yolo26_local import detect_video

from app.coaching.ball_observation_store import (
    build_ball_observation_report,
    save_report,
)


router = APIRouter(prefix="/video", tags=["Video"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/videos")
ANALYZED_DIR = Path("data/analyzed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANALYZED_DIR.mkdir(parents=True, exist_ok=True)


def run_full_analysis(video_path: Path, job_id: str):

    output_dir = ANALYZED_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # PERFORMANCE PROFILING
    # -------------------------------------------

    pipeline_start = time.perf_counter()
    last_stage = pipeline_start

    timings = {}

    def report_time(stage):
        nonlocal last_stage

        now = time.perf_counter()
        elapsed = now - last_stage

        timings[stage] = round(elapsed, 2)

        print(
            f"TIMING {stage}: {elapsed:.2f}s",
            flush=True,
        )

        last_stage = now

    # -------------------------------------------
    # OUTPUT FILE PATHS
    # -------------------------------------------

    detections_path = output_dir / "detections.json"
    metrics_path = output_dir / "shooting_metrics.json"
    rendered_path = output_dir / "shooting_analysis.mp4"

    # -------------------------------------------
    # STAGE 1: YOLO26 PLAYER DETECTION
    # -------------------------------------------

    print(
        "1/7 Running local YOLO26...",
        flush=True,
    )

    detect_video(
        video_path,
        detections_path,
    )

    report_time("YOLO26")

    # -------------------------------------------
    # STAGE 2: MEDIAPIPE POSE ANALYSIS
    # -------------------------------------------

    print(
        "2/7 Analyzing player pose...",
        flush=True,
    )

    pose_analyzer = PoseAnalyzer()

    try:
        pose_result = pose_analyzer.analyze_video(
            str(video_path),
            player_detections_path=str(detections_path),
        )

    finally:
        for name in ("pose", "crop_pose"):
            detector = getattr(
                pose_analyzer,
                name,
                None,
            )

            if detector is not None:
                detector.close()

    pose_result["shooting_phases"] = (
        ShootingPhaseDetector().detect(
            pose_result["frames"]
        )
    )

    report_time("POSE_AND_PHASES")

    # -------------------------------------------
    # STAGE 3: BASKETBALL DETECTION
    # -------------------------------------------

    print(
        "3/7 Detecting basketball...",
        flush=True,
    )

    ball_result = BallAnalyzer(imgsz=640).analyze_video(
        str(video_path)
    )

    # Preserve direct detector predictions before outlier filtering.
    # remove_ball_outliers() mutates the frames it receives.
    raw_ball_frames = deepcopy(ball_result["frames"])

    # Keep existing tracking filter settings unchanged.
    ball_result["frames"] = remove_ball_outliers(
        ball_result["frames"],
        max_speed_factor=3.0,
    )

    # These are accepted detector predictions, before interpolation.
    filtered_ball_frames = deepcopy(ball_result["frames"])

    # Preserve the existing interpolated stream for shot tracking.
    ball_result["frames"] = interpolate_ball_gaps(
        ball_result["frames"],
        max_gap=35,
    )

    report_time("BALL_DETECTION")

    # -------------------------------------------
    # STAGE 4: HOOP DETECTION
    # -------------------------------------------

    print(
        "4/7 Detecting hoop...",
        flush=True,
    )

    hoop_result = HoopDetector().analyze_video(
        str(video_path),
        detect_every=5,
    )

    rim_frames = hoop_result["rim_frames"]
    scene_cuts = hoop_result["scene_cuts"]

    report_time("HOOP_DETECTION")

    # -------------------------------------------
    # STAGE 5: SHOT TRACKING
    # -------------------------------------------

    print(
        "5/7 Detecting shots...",
        flush=True,
    )

    shot_result = ShotTracker().detect(
        pose_result["frames"],
        ball_result["frames"],
        rim_frames,
        scene_cuts,
        fps=ball_result["fps"],
    )

    report_time("SHOT_TRACKING")

    # -------------------------------------------
    # PHASE 5.4: DETECTOR BALL OBSERVATION EVIDENCE
    # -------------------------------------------

    ball_observations = build_ball_observation_report(
        raw_ball_frames,
        filtered_ball_frames,
        shot_result["shots"],
        fps=ball_result["fps"],
    )

    ball_observations_path = output_dir / "ball_observations.json"
    save_report(ball_observations, ball_observations_path)

    print(
        "Ball observations saved:",
        ball_observations["accepted_observation_count"],
        "accepted detections;",
        ball_observations["rejected_outlier_count"],
        "rejected outliers",
        flush=True,
    )

    # -------------------------------------------
    # STAGE 6: SHOOTING METRICS
    # -------------------------------------------

    print(
        "6/7 Calculating metrics...",
        flush=True,
    )

    metrics_result = ShootingMetrics().analyze(
        shot_result,
        pose_result,
        ball_result,
        rim_frames=rim_frames,
        fps=ball_result["fps"],
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_result,
            f,
            indent=2,
        )

    report_time("METRICS")

    # -------------------------------------------
    # STAGE 7: VIDEO RENDERING
    # -------------------------------------------

    print(
        "7/7 Rendering analyzed video...",
        flush=True,
    )

    ShootingAnalysisRenderer().render(
        str(video_path),
        shot_result,
        metrics_result,
        str(rendered_path),
        rim_frames=rim_frames,
        pose_frames=pose_result["frames"],
    )

    report_time("RENDERING")

    # -------------------------------------------
    # VALIDATE RENDERED OUTPUT
    # -------------------------------------------

    if (
        not rendered_path.is_file()
        or rendered_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Rendered video was not created"
        )

    # -------------------------------------------
    # PHASE 4: EVIDENCE-AWARE COACHING REPORT
    # -------------------------------------------

    print(
        "Generating shooting-form comparison...",
        flush=True,
    )

    coaching_path = output_dir / "coaching_report.json"

    coaching_report = CoachingReportPipeline().analyze({
        "analysis": pose_result,
        "metrics": metrics_result,
    })

    if coaching_report["shot_count"] != len(shot_result["shots"]):
        raise RuntimeError(
            "Coaching shot count does not match tracked shots"
        )

    with coaching_path.open("w", encoding="utf-8") as file:
        json.dump(
            coaching_report,
            file,
            indent=2,
        )

    print(
        "Coaching report generated successfully.",
        flush=True,
    )

    # -------------------------------------------
    # PREPARE SHOT SUMMARY
    # -------------------------------------------

    shots = [
        {
            "shot_number": shot["shot_number"],
            "arm": shot["arm"],
            "start_frame": shot["start_frame"],
            "release_frame": shot["release_frame"],
            "end_frame": shot["end_frame"],
            "trajectory_points": len(
                shot["trajectory"]
            ),
        }
        for shot in shot_result["shots"]
    ]

    # -------------------------------------------
    # PHASE 5: MEASUREMENT-BACKED COACHING
    # -------------------------------------------
    # Reuse in-memory pose/metrics and pre-interpolation ball evidence.
    # No second YOLO, MediaPipe, or ball-detector pass is required.

    print("Generating measured coaching feedback...", flush=True)

    # Supplemental coaching must not invalidate a completed shot analysis.
    # A failure is surfaced in the response and logged; no broken URL is returned.
    measured_coaching = None

    try:
        phase5_input = {
            "analysis": pose_result,
            "metrics": metrics_result,
            "shot_count": len(shots),
            "shots": shots,
        }
        phase5_reports = MeasurementCoachingPipeline().analyze(
            phase5_input, ball_observations
        )

        # Store intermediate evidence for audits and troubleshooting.
        phase5_paths = {
            "temporal": "temporal_form.json",
            "mechanics": "mechanics_evidence.json",
            "technique": "technique_reference_review.json",
            "sources": "source_registry_review.json",
            "review": "coaching_review_questions.json",
            "measured": "measured_coaching_report.json",
        }
        for name, filename in phase5_paths.items():
            with (output_dir / filename).open("w", encoding="utf-8") as handle:
                json.dump(phase5_reports[name], handle, indent=2)

        measured_coaching = phase5_reports["measured"]
        print("Measured coaching report generated.", flush=True)

    except Exception:
        logger.exception(
            "Supplemental measured coaching failed for job %s; "
            "preserving completed video, metrics, and Phase 4 coaching",
            job_id,
        )
        (output_dir / "measured_coaching_report.json").unlink(
            missing_ok=True
        )

    report_time("MEASURED_COACHING")

    # -------------------------------------------
    # FINAL PERFORMANCE SUMMARY
    # -------------------------------------------

    total_time = (
        time.perf_counter() - pipeline_start
    )

    timings["TOTAL_PIPELINE"] = round(
        total_time,
        2,
    )

    print(
        "\n========== BULLSEYE PERFORMANCE ==========",
        flush=True,
    )

    for stage, elapsed in timings.items():
        print(
            f"TIMING {stage}: {elapsed:.2f}s"
            if stage == "TOTAL_PIPELINE"
            else f"{stage}: {elapsed:.2f}s",
            flush=True,
        )

    print(
        f"Shots detected: {len(shots)}",
        flush=True,
    )

    print(
        "==========================================\n",
        flush=True,
    )

    print(
        f"Analysis complete: {len(shots)} shots",
        flush=True,
    )

    # -------------------------------------------
    # RETURN ANALYSIS
    # -------------------------------------------

    return {
        "analysis": pose_result,
        "shot_count": len(shots),
        "shots": shots,
        "metrics": metrics_result,
        "processing_time_seconds": round(total_time, 2),
        "stage_timings": timings,
        "measured_coaching_summary": (
            {
                "status": "available",
                "intended_form_match": measured_coaching["intended_form_match"],
                "corrections_generated": measured_coaching["corrections_generated"],
                "shot_reviews": [
                    {
                        "shot_number": shot["shot_number"],
                        "review_candidate_count": shot["review_candidate_count"],
                        "timing_quality": shot["timing_quality"],
                    }
                    for shot in measured_coaching["shots"]
                ],
            }
            if measured_coaching is not None
            else {
                "status": "unavailable",
                "detail": "Measured coaching failed; see backend logs.",
            }
        ),
        "coaching_summary": {
            "intended_form_match": coaching_report["intended_form_match"],
            "first_review_suggestion": coaching_report[
                "first_review_suggestion"
            ],
            "shots": coaching_report["shots"],
        },
    }


# -------------------------------------------
# UPLOAD + ANALYZE ENDPOINT
# -------------------------------------------

@router.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format",
        )

    job_id = str(uuid4())
    filename = f"{job_id}{extension}"

    video_path = UPLOAD_DIR / filename

    try:

        # ---------------------------------------
        # SAVE UPLOADED VIDEO
        # ---------------------------------------

        with video_path.open(
            "wb"
        ) as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output_file.write(chunk)

        # ---------------------------------------
        # RUN FULL PIPELINE
        # ---------------------------------------
        print(
            f"UPLOAD CHECK: "
            f"path={video_path.resolve()} "
            f"size={video_path.stat().st_size} bytes",
            flush=True,
        )
        result = await run_in_threadpool(
            run_full_analysis,
            video_path,
            job_id,
        )

    except Exception as exc:

        logger.exception(
            "Video analysis failed"
        )

        video_path.unlink(
            missing_ok=True
        )

        shutil.rmtree(
            ANALYZED_DIR / job_id,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed. "
                "Check backend logs."
            ),
        ) from exc

    finally:
        await file.close()

    # -------------------------------------------
    # API RESPONSE
    # -------------------------------------------

    return {
        "job_id": job_id,
        "filename": filename,
        **result,
        "video_url": (
            f"/video/results/{job_id}/video"
        ),
        "metrics_url": (
            f"/video/results/{job_id}/metrics"
        ),
        "coaching_url": f"/video/results/{job_id}/coaching",
        "ball_observations_url": (
            f"/video/results/{job_id}/ball-observations"
        ),
        "measured_coaching_url": (
            f"/video/results/{job_id}/measured-coaching"
            if result["measured_coaching_summary"]["status"] == "available"
            else None
        ),
    }


# -------------------------------------------
# DOWNLOAD ANALYZED VIDEO
# -------------------------------------------

@router.get("/results/{job_id}/video")
def download_analyzed_video(
    job_id: UUID
):

    path = (
        ANALYZED_DIR
        / str(job_id)
        / "shooting_analysis.mp4"
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Analyzed video not found",
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=(
            f"{job_id}_analysis.mp4"
        ),
    )


# -------------------------------------------
# DOWNLOAD SHOOTING METRICS
# -------------------------------------------

@router.get("/results/{job_id}/metrics")
def download_metrics(
    job_id: UUID
):

    path = (
        ANALYZED_DIR
        / str(job_id)
        / "shooting_metrics.json"
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found",
        )

    return FileResponse(
        path,
        media_type="application/json",
        filename=(
            f"{job_id}_metrics.json"
        ),
    )

@router.get("/results/{job_id}/coaching")
def download_coaching_report(job_id: UUID):
    path = (
        ANALYZED_DIR
        / str(job_id)
        / "coaching_report.json"
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Coaching report not found",
        )

    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_coaching_report.json",
    )


# -------------------------------------------
# DOWNLOAD DIRECT BALL OBSERVATIONS
# -------------------------------------------

@router.get("/results/{job_id}/ball-observations")
def download_ball_observations(job_id: UUID):
    path = (
        ANALYZED_DIR
        / str(job_id)
        / "ball_observations.json"
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Ball observation report not found",
        )

    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_ball_observations.json",
    )


# -------------------------------------------
# DOWNLOAD MEASUREMENT-BACKED COACHING REPORT
# -------------------------------------------

@router.get("/results/{job_id}/measured-coaching")
def download_measured_coaching(job_id: UUID):
    path = (
        ANALYZED_DIR
        / str(job_id)
        / "measured_coaching_report.json"
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Measured coaching report not found",
        )

    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_measured_coaching.json",
    )
