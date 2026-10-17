import os

from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.ball_analyzer import BallAnalyzer
from app.analysis.shot_tracker import ShotTracker
from app.analysis.trajectory_renderer import TrajectoryRenderer


video_path = "data/videos/8dcdd2b1-e9bd-4c72-9373-95ca66da41c3.mp4"

pose_analyzer = PoseAnalyzer()
ball_analyzer = BallAnalyzer()
shot_tracker = ShotTracker()
renderer = TrajectoryRenderer()

print("Analyzing pose...")

pose_result = pose_analyzer.analyze_video(
    video_path
)

print("Analyzing ball...")

ball_result = ball_analyzer.analyze_video(
    video_path
)

print(
    "Ball detections:",
    ball_result["ball_detected_frames"]
)

print(
    "Ball detection rate:",
    f"{ball_result['ball_detection_rate'] * 100:.2f}%"
)

shot_result = shot_tracker.detect(
    pose_result["frames"],
    ball_result["frames"]
)

print(
    "Shooting arm:",
    shot_result["shooting_arm"]
)

print(
    "Completed shots:",
    len(shot_result["shots"])
)

for shot in shot_result["shots"]:

    print(
        "Shot",
        shot["shot_number"],
        "| start:",
        shot["start_frame"],
        "| release:",
        shot["release_frame"],
        "| end:",
        shot["end_frame"],
        "| points:",
        len(shot["trajectory"])
    )

output_path = (
    "data/analyzed/"
    "persistent_shot_trajectories.mp4"
)

os.makedirs(
    "data/analyzed",
    exist_ok=True
)

renderer.render(
    video_path,
    shot_result["shots"],
    output_path
)

print(
    "Output:",
    output_path
)