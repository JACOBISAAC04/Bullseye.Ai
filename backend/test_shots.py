import cv2
import os

from app.analysis.pose_analyzer import PoseAnalyzer
from app.analysis.ball_analyzer import BallAnalyzer
from app.analysis.hoop_detector import HoopDetector

def detect_shooting_arm(frames):

    left_movement = 0
    right_movement = 0

    for i in range(1, len(frames)):

        previous = frames[i - 1]
        current = frames[i]

        if not previous.get("pose_detected"):
            continue

        if not current.get("pose_detected"):
            continue

        left_a = previous["left_wrist"]
        left_b = current["left_wrist"]

        right_a = previous["right_wrist"]
        right_b = current["right_wrist"]

        left_movement += (
            (left_b["x"] - left_a["x"]) ** 2
            + (left_b["y"] - left_a["y"]) ** 2
        ) ** 0.5

        right_movement += (
            (right_b["x"] - right_a["x"]) ** 2
            + (right_b["y"] - right_a["y"]) ** 2
        ) ** 0.5

    if right_movement >= left_movement:
        return "right"

    return "left"


video_path = "data/videos/8dcdd2b1-e9bd-4c72-9373-95ca66da41c3.mp4"

pose_analyzer = PoseAnalyzer()
ball_analyzer = BallAnalyzer()
hoop_detector = HoopDetector()

pose_result = pose_analyzer.analyze_video(video_path)
ball_result = ball_analyzer.analyze_video(video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, first_frame = cap.read()

if not ret:
    raise ValueError("Could not read first frame")

hoop = hoop_detector.detect(first_frame)

print("Detected hoop:", hoop)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

shooting_arm = detect_shooting_arm(
    pose_result["frames"]
)

print("Detected shooting arm:", shooting_arm)

wrist_key = f"{shooting_arm}_wrist"

colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0)
]

output_path = "data/analyzed/multi_shot_trajectory.mp4"

os.makedirs(
    "data/analyzed",
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

trajectory = []
shots = []
current_shot = None
shot_color_index = 0

for i in range(len(ball_result["frames"])):

    success, frame = cap.read()

    if not success:
        break

    ball_data = ball_result["frames"][i]
    pose_data = pose_result["frames"][i]

    ball_detected = ball_data["ball_detected"]

    if ball_detected and pose_data.get("pose_detected"):

        ball = ball_data["ball"]

        ball_x = ball["x"]
        ball_y = ball["y"]

        wrist = pose_data[wrist_key]

        wrist_x = wrist["x"]
        wrist_y = wrist["y"]

        distance = (
            (ball_x - wrist_x) ** 2
            + (ball_y - wrist_y) ** 2
        ) ** 0.5

        if current_shot is None:

            if distance < 140:

                current_shot = {
                    "start_frame": i,
                    "color": colors[
                        shot_color_index % len(colors)
                    ],
                    "trajectory": []
                }

                trajectory = []

                shot_color_index += 1

        if current_shot is not None:

            point = (
                int(ball_x),
                int(ball_y)
            )

            trajectory.append(point)

            current_shot["trajectory"].append(point)

            if hoop is not None:

                hoop_x = hoop["x"]
                hoop_y = hoop["y"]

                hoop_distance = (
                    (ball_x - hoop_x) ** 2
                    + (ball_y - hoop_y) ** 2
                ) ** 0.5

                if (
                    len(trajectory) > 10
                    and hoop_distance < 100
                ):

                    current_shot["end_frame"] = i

                    shots.append(current_shot)

                    current_shot = None
                    trajectory = []

    if current_shot is not None:

        color = current_shot["color"]

        points = current_shot["trajectory"]

        for j in range(1, len(points)):

            cv2.line(
                frame,
                points[j - 1],
                points[j],
                color,
                5
            )

        if points:

            cv2.circle(
                frame,
                points[-1],
                8,
                color,
                -1
            )

    writer.write(frame)

cap.release()
writer.release()

print("Detected shots:", len(shots))

for index, shot in enumerate(shots, 1):

    print(
        "Shot",
        index,
        "start:",
        shot["start_frame"] + 1,
        "end:",
        shot.get("end_frame", -1) + 1
    )

print("Output:", output_path)
