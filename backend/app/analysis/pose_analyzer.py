import cv2
import mediapipe as mp
import math
import os
import json
from pathlib import Path


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Diagnostic fallback for small players in the full frame.
        self.crop_pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def calculate_angle(self, a, b, c):
        angle = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) -
            math.atan2(a[1] - b[1], a[0] - b[0])
        )

        angle = abs(angle)

        if angle > 180:
            angle = 360 - angle

        return angle


    @staticmethod
    def load_player_boxes(json_path, video_path, width, height, total_frames):
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        info = data["video"]

        if (
            info["filename"] != Path(video_path).name
            or info["width"] != width
            or info["height"] != height
            or info["frame_count"] != total_frames
        ):
            raise ValueError(
                "YOLO26 detections do not match the selected video"
            )

        boxes = {}
        last_box = None
        last_frame = None

        for entry in sorted(data["frames"], key=lambda e: e["frame"]):
            frame_number = entry["frame"]

            people = [
                detection
                for detection in entry["detections"]
                if detection["class_id"] == 0
            ]

            if people:
                best = max(
                    people,
                    key=lambda detection: detection["confidence"],
                )

                box = best["bbox_xyxy"]

                if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                    last_box = box
                    last_frame = frame_number
                    boxes[frame_number] = box

            elif (
                last_box is not None
                and frame_number - last_frame <= 2
            ):
                # Carry the last observed box across brief gaps.
                boxes[frame_number] = last_box

        return boxes


    def analyze_video(self, video_path, player_detections_path=None):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video")

        frame_count = 0
        detected_frames = 0
        frames = []

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        player_boxes = None

        if player_detections_path is not None:
            player_boxes = self.load_player_boxes(
                player_detections_path,
                video_path,
                width,
                height,
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )

            print(
                f"YOLO26 player boxes loaded: {len(player_boxes)} frames"
            )

        output_dir = "data/analyzed"
        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.splitext(
            os.path.basename(video_path)
        )[0] + "_pose.mp4"

        output_path = os.path.join(output_dir, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.pose.process(rgb_frame)


            # Phase 2: use cloud-generated YOLO26 player boxes
            # only when full-frame MediaPipe fails.
            if (
                results.pose_landmarks is None
                and player_boxes is not None
            ):
                bbox = player_boxes.get(frame_count)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox

                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Give MediaPipe room for hands and raised arms.
                    pad_x = box_width * 0.75
                    pad_y = box_height * 0.25

                    crop_x0 = max(0, int(x1 - pad_x))
                    crop_y0 = max(0, int(y1 - pad_y))
                    crop_x1 = min(width, int(x2 + pad_x))
                    crop_y1 = min(height, int(y2 + pad_y))

                    crop_width = crop_x1 - crop_x0
                    crop_height = crop_y1 - crop_y0

                    if crop_width > 0 and crop_height > 0:
                        cropped_rgb = rgb_frame[
                            crop_y0:crop_y1,
                            crop_x0:crop_x1,
                        ]

                        cropped_results = self.crop_pose.process(
                            cropped_rgb
                        )

                        if cropped_results.pose_landmarks is not None:
                            for landmark in (
                                cropped_results.pose_landmarks.landmark
                            ):
                                landmark.x = (
                                    crop_x0 + landmark.x * crop_width
                                ) / width

                                landmark.y = (
                                    crop_y0 + landmark.y * crop_height
                                ) / height

                                landmark.z *= crop_width / width

                            results = cropped_results




            # Temporary player-region fallback for video3.mp4.
            # The player is on the right side of this particular video.
            if results.pose_landmarks is None and player_boxes is None:
                x0 = int(width * 0.52)
                y0 = int(height * 0.30)

                cropped_rgb = rgb_frame[y0:height, x0:width]

                cropped_results = self.crop_pose.process(cropped_rgb)

                if cropped_results.pose_landmarks is not None:

                    # Convert landmarks from crop coordinates back to
                    # full-video normalized coordinates.
                    crop_width = width - x0
                    crop_height = height - y0

                    for landmark in cropped_results.pose_landmarks.landmark:
                        landmark.x = (
                            x0 + landmark.x * crop_width
                        ) / width

                        landmark.y = (
                            y0 + landmark.y * crop_height
                        ) / height

                    results = cropped_results



            frame_data = {
                "frame": frame_count,
                "time": frame_count / fps,
                "pose_detected": False
            }

            if results.pose_landmarks:
                detected_frames += 1

                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER
                ]

                left_elbow = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_ELBOW
                ]

                left_wrist = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_WRIST
                ]

                left_hip = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_HIP
                ]

                left_knee = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_KNEE
                ]

                left_ankle = landmarks[
                    self.mp_pose.PoseLandmark.LEFT_ANKLE
                ]

                right_shoulder = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                ]

                right_elbow = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW
                ]

                right_wrist = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_WRIST
                ]

                right_hip = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_HIP
                ]

                right_knee = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_KNEE
                ]

                right_ankle = landmarks[
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE
                ]

                left_elbow_angle = self.calculate_angle(
                    (left_shoulder.x, left_shoulder.y),
                    (left_elbow.x, left_elbow.y),
                    (left_wrist.x, left_wrist.y)
                )

                right_elbow_angle = self.calculate_angle(
                    (right_shoulder.x, right_shoulder.y),
                    (right_elbow.x, right_elbow.y),
                    (right_wrist.x, right_wrist.y)
                )

                left_knee_angle = self.calculate_angle(
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y),
                    (left_ankle.x, left_ankle.y)
                )

                right_knee_angle = self.calculate_angle(
                    (right_hip.x, right_hip.y),
                    (right_knee.x, right_knee.y),
                    (right_ankle.x, right_ankle.y)
                )

                frame_data.update({
                    "pose_detected": True,

                    "left_elbow_angle": left_elbow_angle,
                    "right_elbow_angle": right_elbow_angle,

                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,

                    "left_shoulder": {
                        "x": left_shoulder.x * width,
                        "y": left_shoulder.y * height,
                        "z": left_shoulder.z
                    },

                    "right_shoulder": {
                        "x": right_shoulder.x * width,
                        "y": right_shoulder.y * height,
                        "z": right_shoulder.z
                    },

                    "left_elbow": {
                        "x": left_elbow.x * width,
                        "y": left_elbow.y * height,
                        "z": left_elbow.z
                    },

                    "right_elbow": {
                        "x": right_elbow.x * width,
                        "y": right_elbow.y * height,
                        "z": right_elbow.z
                    },

                    "left_wrist": {
                        "x": left_wrist.x * width,
                        "y": left_wrist.y * height,
                        "z": left_wrist.z
                    },

                    "right_wrist": {
                        "x": right_wrist.x * width,
                        "y": right_wrist.y * height,
                        "z": right_wrist.z
                    },

                    "left_hip": {
                        "x": left_hip.x * width,
                        "y": left_hip.y * height,
                        "z": left_hip.z
                    },

                    "right_hip": {
                        "x": right_hip.x * width,
                        "y": right_hip.y * height,
                        "z": right_hip.z
                    },

                    "left_knee": {
                        "x": left_knee.x * width,
                        "y": left_knee.y * height,
                        "z": left_knee.z
                    },

                    "right_knee": {
                        "x": right_knee.x * width,
                        "y": right_knee.y * height,
                        "z": right_knee.z
                    },

                    "left_ankle": {
                        "x": left_ankle.x * width,
                        "y": left_ankle.y * height,
                        "z": left_ankle.z
                    },

                    "right_ankle": {
                        "x": right_ankle.x * width,
                        "y": right_ankle.y * height,
                        "z": right_ankle.z
                    }
                })

                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            frames.append(frame_data)

            writer.write(frame)

        cap.release()
        writer.release()

        if detected_frames == 0:
            raise ValueError("No pose detected in video")

        return {
            "total_frames": frame_count,
            "pose_detected_frames": detected_frames,
            "pose_detection_rate": detected_frames / frame_count,
            "fps": fps,
            "frames": frames,
            "analyzed_video": output_path
        }