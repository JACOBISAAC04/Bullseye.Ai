
import time

import cv2
import numpy as np
from ultralytics import YOLO


class HoopDetector:

    def __init__(
        self,
        weights="app/models/hoop_detector/best.pt",
        conf=0.25,
    ):
        self.model = YOLO(weights)
        self.conf = conf

    def analyze_video(
        self,
        video_path,
        smooth_radius=7,
        detect_every=3,
    ):
        """
        Detect the hoop periodically instead of on every frame.

        Scene cuts are still checked on EVERY frame.

        detect_every=1:
            Original full-frame detection behavior.

        detect_every=3:
            Run YOLO every third frame, plus at scene cuts.
            Reuse the latest detection between inference frames.
        """

        if detect_every < 1:
            raise ValueError(
                "detect_every must be at least 1"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Could not open video")

        raw = []
        cuts = []

        prev_hist = None
        last_detection = None

        inference_calls = 0
        start_time = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                frame_index = len(raw)

                # ----------------------------------
                # SCENE CUT DETECTION
                # Always analyze every frame.
                # ----------------------------------

                hsv = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2HSV,
                )

                hist = cv2.calcHist(
                    [hsv],
                    [0, 1],
                    None,
                    [32, 32],
                    [0, 180, 0, 256],
                )

                cv2.normalize(hist, hist)

                is_cut = (
                    prev_hist is not None
                    and cv2.compareHist(
                        prev_hist,
                        hist,
                        cv2.HISTCMP_CORREL,
                    ) < 0.5
                )

                cuts.append(is_cut)
                prev_hist = hist

                # ----------------------------------
                # OPTIMIZED HOOP DETECTION
                # ----------------------------------

                should_detect = (
                    frame_index % detect_every == 0
                    or is_cut
                )

                if should_detect:
                    # Refresh bounding box using YOLO.
                    # At scene cuts, this also prevents
                    # reuse from the previous scene.
                    last_detection = self._detect(frame)

                    inference_calls += 1

                # For skipped frames, reuse the latest
                # detection within the current scene.
                raw.append(last_detection)

        finally:
            cap.release()

        # ----------------------------------
        # ORIGINAL MEDIAN SMOOTHING
        # ----------------------------------

        result = self._smooth(
            raw,
            cuts,
            smooth_radius,
        )

        elapsed = time.perf_counter() - start_time

        print(
            "\n========== HOOP DETECTOR ==========",
            flush=True,
        )

        print(
            f"Total frames: {len(raw)}",
            flush=True,
        )

        print(
            f"YOLO inference calls: {inference_calls}",
            flush=True,
        )

        print(
            f"Detection interval: {detect_every}",
            flush=True,
        )

        print(
            f"Scene cuts: {sum(cuts)}",
            flush=True,
        )

        print(
            f"Hoop processing time: {elapsed:.2f}s",
            flush=True,
        )

        print(
            "===================================\n",
            flush=True,
        )

        return result

    def _detect(self, frame):
        # Keep the original model input size
        # to minimize accuracy changes.
        result = self.model(
            frame,
            imgsz=960,
            conf=self.conf,
            verbose=False,
        )[0]

        best = None

        for box in result.boxes:
            confidence = float(box.conf[0])

            if (
                best is None
                or confidence > best["conf"]
            ):
                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                best = {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": confidence,
                }

        return best

    def _smooth(self, raw, cuts, r):
        # Preserve original scene-aware
        # median smoothing behavior.

        seg = []
        s = 0

        for cut in cuts:
            s += int(cut)
            seg.append(s)

        out = []

        for i in range(len(raw)):
            window = [
                raw[j]
                for j in range(
                    max(0, i - r),
                    min(len(raw), i + r + 1),
                )
                if (
                    raw[j] is not None
                    and seg[j] == seg[i]
                )
            ]

            if not window:
                out.append(None)
                continue

            out.append({
                key: float(
                    np.median([
                        item[key]
                        for item in window
                    ])
                )
                for key in ("x", "y", "w", "h")
            })

        return {
            "rim_frames": out,
            "scene_cuts": cuts,
        }
