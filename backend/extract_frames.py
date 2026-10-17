import cv2, os, sys

video, out_dir, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(video)
name = os.path.splitext(os.path.basename(video))[0][:12]
i = saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % step == 0:
        cv2.imwrite(f"{out_dir}/{name}_{i:05d}.jpg", frame)
        saved += 1
    i += 1
print("saved", saved)