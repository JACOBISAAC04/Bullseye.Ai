
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO("yolo26n.pt")

    frames = []
    person_frames = 0
    start_time = time.perf_counter()

    try:
        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            result = model.predict(
                frame,
                device="cpu",
                imgsz=640,
                conf=0.25,
                classes=[0],
                verbose=False,
            )[0]

            detections = []

            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": "person",
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ],
                })

            if detections:
                person_frames += 1

            frames.append({
                "frame": frame_number,
                "detections": detections,
            })

            if frame_number % 50 == 0:
                print(
                    f"Processed {frame_number}/{frame_count} frames",
                    flush=True,
                )

    finally:
        cap.release()

    elapsed = time.perf_counter() - start_time

    data = {
        "video": {
            "filename": video_path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_number,
        },
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print(f"Processed frames: {frame_number}")
    print(f"Person detections: {person_frames}/{frame_number}")
    print(f"YOLO26 processing time: {elapsed:.2f} seconds")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video")
    parser.add_argument("--output")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path("data/analyzed")
        / f"yolo26_local_{video_path.stem}"
        / "detections.json"
    )

    detect_video(video_path, output_path)
