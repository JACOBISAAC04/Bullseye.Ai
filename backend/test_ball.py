from app.analysis.ball_analyzer import BallAnalyzer

video_path = "data/videos/8dcdd2b1-e9bd-4c72-9373-95ca66da41c3.mp4"

analyzer = BallAnalyzer()

result = analyzer.analyze_video(video_path)

print("Total frames:", result["total_frames"])
print("Ball detected frames:", result["ball_detected_frames"])

for frame in result["frames"]:
    if frame["ball_detected"]:
        print(frame)
        break