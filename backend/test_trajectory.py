from app.analysis.ball_analyzer import BallAnalyzer

video_path = "data/videos/8dcdd2b1-e9bd-4c72-9373-95ca66da41c3.mp4"

analyzer = BallAnalyzer()

result = analyzer.analyze_video(video_path)

print("Total frames:", result["total_frames"])
print("Ball detected frames:", result["ball_detected_frames"])

output_path = "data/analyzed/ball_trajectory.mp4"

analyzer.draw_trajectory(
    video_path,
    result,
    output_path
)

print("Output:", output_path)