import json
import sys
from pathlib import Path

from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.ball_analyzer import BallAnalyzer
from app.analysis.hoop_detector import HoopDetector
from app.analysis.shot_tracker import ShotTracker, remove_ball_outliers, interpolate_ball_gaps
from app.analysis.shooting_metrics import ShootingMetrics
from app.analysis.shooting_analysis_renderer import ShootingAnalysisRenderer


# Usage:
#   python test_shooting_analysis_video.py                      (uses newest video in data/videos)
#   python test_shooting_analysis_video.py data/videos/x.mp4    (uses that file)
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    videos = sorted(
        Path("data/videos").glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        raise FileNotFoundError("No .mp4 files in data/videos")
    video_path = str(videos[0])

print("Using video:", video_path)

pose_analyzer = PoseAnalyzer()
ball_analyzer = BallAnalyzer()
hoop_detector = HoopDetector()
shot_tracker = ShotTracker()
shooting_metrics = ShootingMetrics()
renderer = ShootingAnalysisRenderer()

print("Analyzing pose...")
player_detections_path = (
    sys.argv[2] if len(sys.argv) > 2 else None
)

pose_result = pose_analyzer.analyze_video(
    video_path,
    player_detections_path=player_detections_path,
)

print("Analyzing ball...")
ball_result = ball_analyzer.analyze_video(video_path)
print(
    f"Ball detected in {ball_result['ball_detected_frames']}"
    f"/{ball_result['total_frames']} frames"
)
detected_ball_frames = [
    i + 1
    for i, frame in enumerate(ball_result["frames"])
    if frame.get("ball_detected")
]

print("Ball detection frames:")
print(detected_ball_frames)

print("Cleaning ball trajectory (removing outliers, filling short gaps)...")
ball_result["frames"] = remove_ball_outliers(ball_result["frames"], max_speed_factor=3.0)
ball_result["frames"] = interpolate_ball_gaps(ball_result["frames"], max_gap=35)

cleaned_ball_frames = [
    i + 1
    for i, frame in enumerate(ball_result["frames"])
    if frame.get("ball_detected")
]
print(
    f"Ball detected in {len(cleaned_ball_frames)}"
    f"/{ball_result['total_frames']} frames after cleaning"
)
print("Cleaned ball detection frames:")
print(cleaned_ball_frames)

print("Detecting hoop...")
hoop_result = hoop_detector.analyze_video(video_path)

rim_frames = hoop_result["rim_frames"]
scene_cuts = hoop_result["scene_cuts"]

rim_found = sum(r is not None for r in rim_frames)
print(f"Rim found in {rim_found}/{len(rim_frames)} frames")
print(f"Scene cuts: {sum(scene_cuts)}")

if rim_found == 0:
    print("WARNING: no rim detected, so no shot can complete. "
          "Check models/hoop.pt and the hoop confidence threshold.")

print("Detecting shots...")
shot_result = shot_tracker.detect(
    pose_result["frames"],
    ball_result["frames"],
    rim_frames,
    scene_cuts,
    fps=ball_result["fps"],
)

for shot in shot_result["shots"]:
    print(
        "  Shot", shot["shot_number"],
        "| arm:", shot["arm"],
        "| start:", shot["start_frame"],
        "| release:", shot["release_frame"],
        "| end:", shot["end_frame"],
        "| points:", len(shot["trajectory"]),
    )
print("Shots:", len(shot_result["shots"]))

for shot in shot_result["shots"]:
    print(
        "  Shot", shot["shot_number"],
        "| start:", shot["start_frame"],
        "| release:", shot["release_frame"],
        "| end:", shot["end_frame"],
        "| points:", len(shot["trajectory"]),
    )

print("Calculating shooting metrics...")
metrics_result = shooting_metrics.analyze(
    shot_result,
    pose_result,
    ball_result,
    rim_frames=rim_frames,
    fps=ball_result["fps"],
)

metrics_path = "data/analyzed/shooting_metrics.json"
Path("data/analyzed").mkdir(parents=True, exist_ok=True)

with open(metrics_path, "w") as file:
    json.dump(metrics_result, file, indent=2)

print("Metrics saved:", metrics_path)

output_path = "data/analyzed/shooting_analysis.mp4"

print("Rendering analyzed video...")
renderer.render(
    video_path,
    shot_result,
    metrics_result,
    output_path,
    rim_frames=rim_frames,          # draws a yellow rim box so you can SEE detection
    pose_frames=pose_result["frames"],   # draws the pose skeleton overlay
)

print("Video saved:", output_path)