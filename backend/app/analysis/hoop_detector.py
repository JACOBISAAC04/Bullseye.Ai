import cv2
import numpy as np
from ultralytics import YOLO


class HoopDetector:

    def __init__(self, weights="app/models/hoop_detector/best.pt", conf=0.25):
        self.model = YOLO(weights)
        self.conf = conf

    def analyze_video(self, video_path, smooth_radius=7):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video")

        raw, cuts = [], []
        prev_hist = None

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            hist = cv2.calcHist(
                [hsv],
                [0, 1],
                None,
                [32, 32],
                [0, 180, 0, 256]
            )

            cv2.normalize(hist, hist)

            cuts.append(
                prev_hist is not None
                and cv2.compareHist(
                    prev_hist,
                    hist,
                    cv2.HISTCMP_CORREL
                ) < 0.5
            )

            prev_hist = hist

            raw.append(self._detect(frame))

        cap.release()

        return self._smooth(raw, cuts, smooth_radius)

    def _detect(self, frame):
        result = self.model(
            frame,
            imgsz=960,
            conf=self.conf,
            verbose=False
        )[0]

        best = None

        for box in result.boxes:
            confidence = float(box.conf[0])

            if best is None or confidence > best["conf"]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                best = {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": confidence
                }

        return best

    def _smooth(self, raw, cuts, r):
        seg, s = [], 0

        for cut in cuts:
            s += int(cut)
            seg.append(s)

        out = []

        for i in range(len(raw)):

            window = [
                raw[j]
                for j in range(
                    max(0, i - r),
                    min(len(raw), i + r + 1)
                )
                if raw[j] is not None
                and seg[j] == seg[i]
            ]

            if not window:
                out.append(None)
                continue

            out.append({
                key: float(
                    np.median([item[key] for item in window])
                )
                for key in ("x", "y", "w", "h")
            })

        return {
            "rim_frames": out,
            "scene_cuts": cuts
        }