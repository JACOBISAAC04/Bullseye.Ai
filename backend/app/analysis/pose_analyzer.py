import cv2
import mediapipe as mp
import math
import os


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

    def calculate_angle(self, a, b, c):
        angle = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) -
            math.atan2(a[1] - b[1], a[0] - b[0])
        )

        angle = abs(angle)

        if angle > 180:
            angle = 360 - angle

        return angle

    def analyze_video(self, video_path):
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