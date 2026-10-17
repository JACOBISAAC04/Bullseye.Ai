import cv2
from ultralytics import YOLO


class BallAnalyzer:

    def __init__(self):
        self.model = YOLO("app/models/ball_detector/ball.pt")
        self.ball_class = 0

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        frames = []

        while True:
            success, frame = cap.read()

            if not success:
                break

            results = self.model(
                frame,
                imgsz=640,
                conf=0.25,
                classes=[self.ball_class],
                max_det=1,
                verbose=False
            )

            best_ball = None

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    confidence = float(
                        box.conf[0].item()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    best_ball = {
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence
                    }

            frames.append({
                "frame": len(frames) + 1,
                "time": len(frames) / fps,
                "ball_detected": best_ball is not None,
                "ball": best_ball
            })

        cap.release()

        detected = sum(
            frame["ball_detected"]
            for frame in frames
        )

        return {
            "fps": fps,
            "total_frames": len(frames),
            "ball_detected_frames": detected,
            "ball_detection_rate": (
                detected / len(frames)
                if frames
                else 0
            ),
            "frames": frames
        }