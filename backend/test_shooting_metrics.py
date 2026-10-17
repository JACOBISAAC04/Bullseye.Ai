from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.ball_analyzer import BallAnalyzer
from app.analysis.shot_tracker import ShotTracker
from app.analysis.shooting_metrics import ShootingMetrics
import json


video_path = "data/videos/529423d1-9929-4b4d-a0b5-562b1d4bf6fa.mp4"
pose_analyzer = PoseAnalyzer()
ball_analyzer = BallAnalyzer()
shot_tracker = ShotTracker()
shooting_metrics = ShootingMetrics()

pose_result = pose_analyzer.analyze_video(
    video_path
)

ball_result = ball_analyzer.analyze_video(
    video_path
)

shot_result = shot_tracker.detect(
    pose_result["frames"],
    ball_result["frames"]
)

metrics_result = shooting_metrics.analyze(
    shot_result,
    pose_result,
    ball_result
)

print(
    json.dumps(
        metrics_result,
        indent=2
    )
)