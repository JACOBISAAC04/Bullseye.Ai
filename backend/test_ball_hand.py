import cv2
from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.ball_analyzer import BallAnalyzer


video_path = "data/videos/8dcdd2b1-e9bd-4c72-9373-95ca66da41c3.mp4"

pose_analyzer = PoseAnalyzer()
ball_analyzer = BallAnalyzer()

pose_result = pose_analyzer.analyze_video(video_path)
ball_result = ball_analyzer.analyze_video(video_path)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "data/analyzed/ball_hand_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

for i in range(len(ball_result["frames"])):

    success, frame = cap.read()

    if not success:
        break

    ball_frame = ball_result["frames"][i]
    pose_frame = pose_result["frames"][i]

    if (
        ball_frame["ball_detected"]
        and pose_frame.get("pose_detected")
    ):

        ball = ball_frame["ball"]

        ball_point = (
            int(ball["x"]),
            int(ball["y"])
        )

        wrist = pose_frame["right_wrist"]

        wrist_point = (
            int(wrist["x"]),
            int(wrist["y"])
        )

        cv2.circle(
            frame,
            ball_point,
            10,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            wrist_point,
            10,
            (0, 255, 0),
            -1
        )

        cv2.line(
            frame,
            ball_point,
            wrist_point,
            (255, 255, 255),
            2
        )

        distance = (
            (ball_point[0] - wrist_point[0]) ** 2
            + (ball_point[1] - wrist_point[1]) ** 2
        ) ** 0.5

        cv2.putText(
            frame,
            f"Ball-Hand: {distance:.1f}px",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    writer.write(frame)

cap.release()
writer.release()

print("Output:", output_path)