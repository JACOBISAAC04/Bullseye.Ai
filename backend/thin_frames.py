import os, shutil

src = "hoop_frames"
dst = "hoop_frames_sample"
target = 200  # aim for this many

files = sorted(os.listdir(src))
step = max(1, len(files) // target)

os.makedirs(dst, exist_ok=True)
kept = 0
for i, f in enumerate(files):
    if i % step == 0:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        kept += 1

print(f"kept {kept} of {len(files)}")