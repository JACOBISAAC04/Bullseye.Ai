import cv2
from pathlib import Path

input_dir = Path("data/videos")
output_dir = Path("data/ball_frames")
output_dir.mkdir(parents=True, exist_ok=True)

for video_path in input_dir.glob("*.mp4"):
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        continue

    indices = set(
        int(i * (total - 1) / 39)
        for i in range(40)
    )

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index in indices:
            name = f"{video_path.stem}_{frame_index:05d}.jpg"
            cv2.imwrite(str(output_dir / name), frame)

        frame_index += 1

    cap.release()

print("Frame extraction complete.")