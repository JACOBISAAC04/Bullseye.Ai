# Bullseye Phase 2 — ₹0 YOLO26 batch workflow

Baseline: `main` at `68875e6`; current Phase 2 development checkpoint:
`phase-2-yolo26` at `7d24a85`.

This workflow **does not provide a permanent cloud connection**.
YOLO26 runs during an on-demand **free-tier Google Colab session** (GPU
when available). Your Windows PC runs the existing Bullseye
MediaPipe/ball/hoop/shot/metrics/rendering pipeline. Your existing ball
detector remains authoritative; YOLO26 provides **player candidate boxes**.
Free Colab resources are limited and availability is not guaranteed.
**Do not enable billing or purchase compute units.**

## First-time setup

Work on `phase-2-yolo26`. Copy the content of the `backend/` folder in this
handoff over your existing `D:\Bullseye\backend\` folder (merge folders).
Do **not** replace `pose_analyzer.py` or `shot_tracker.py`; the already
pushed Phase 2 code is kept intact. The added files are:

- `colab/Bullseye_YOLO26_AnyVideo_Free.ipynb`
- `tools/import_yolo26.py`
- `README_PHASE2.md`
- corrected `backend/.gitignore` (replaces the old ignore file)

## Any new video — three actions

1. Copy the original `.mp4` into `backend/data/videos/`.
2. Open `colab/Bullseye_YOLO26_AnyVideo_Free.ipynb` in
   https://colab.research.google.com/ . Run all. If a **free** T4 is
   available, use it; otherwise CPU also works (slower). Upload your
   original video and download the video-specific ZIP.
3. With the project's `.venv` activated, from
   `D:\Bullseye\backend`, run:

   ```powershell
   python tools/import_yolo26.py `
     data/videos/video9.mp4 `
     "C:\Users\Lenovo\Downloads\Bullseye_YOLO26_video9_results.zip" `
     --run
   ```

`--run` invokes the existing `test_shooting_analysis_video.py` with the
validated JSON, using the **same** dynamic-crop integration. It writes
the results to `data/analyzed/` exactly as the current test does.
The importer validates original-video filename, resolution, FPS, number
of frames and full per-frame JSON sequence. New exports also include a
SHA-256 of the original video, making it harder to accidentally combine
detections with the wrong video.

The same ZIP can be reimported; you don't need to rerun Colab for each
local analysis. The `data/analyzed/yolo26_<video-stem>/` directory
contains the imported JSON and diagnostic sample images.

**Legacy v1 detection ZIPs:** The earlier Colab reports for `video3`
and `video8` did not include SHA-256. They are accepted after metadata
validation, with a visible warning. New exports use schema v2 and SHA-256.

### Regression validation (existing exports)

```powershell
python tools/import_yolo26.py `
  data/videos/video3.mp4 `
  "C:\Users\Lenovo\Downloads\Bullseye_YOLO26_fullvideo_results.zip" `
  --run --expected-shots 2

python tools/import_yolo26.py `
  data/videos/video8.mp4 `
  "C:\Users\Lenovo\Downloads\Bullseye_YOLO26_fullvideo_results (1).zip" `
  --run --expected-shots 1
```

If these ZIPs have been moved, replace the paths as needed.
Your `test_shooting_analysis_video.py` overwrites
`data/analyzed/shooting_analysis.mp4` and `shooting_metrics.json` on every
run, so save a copy of outputs you want to keep before running another
test. A zero-shot result for a *new* video is not automatically a failure;
review the rendered video and detector logs.

### If Colab is unavailable

Existing Bullseye still runs without a YOLO26 JSON:

```powershell
python test_shooting_analysis_video.py data/videos/video8.mp4
```

**Known limitation:** Phase 1's fixed right-side crop remains in the
no-JSON fallback, so it is not reliable for arbitrary player positions.
The YOLO26-assisted path uses per-frame player boxes. The highest-
confidence candidate can still be a false positive; visually verify
new videos. The batch workflow is **not a live API**.

## Safe repository cleanup and merge

From `D:\Bullseye\backend` (branch `phase-2-yolo26`), inspect:

```powershell
git branch --show-current
git status --short
git ls-files data/analyzed
```

Existing `shooting_analysis.mp4` and `shooting_metrics.json` were
tracked **before** the ignore rules were fixed. Remove *only their
tracking*, preserving local copies:

```powershell
git rm --cached -- data/analyzed/shooting_analysis.mp4 data/analyzed/shooting_metrics.json
```

The two staged `D` entries mean "remove from Git's index", not from
your disk. If Git reports that a file is not tracked, omit that path.

Stage only the Phase 2 workflow files:

```powershell
git add .gitignore README_PHASE2.md tools/import_yolo26.py colab/Bullseye_YOLO26_AnyVideo_Free.ipynb
git diff --cached --check
git diff --cached --stat
git status --short
```

The two `D` entries for generated files are expected. Don't use
`git add .` or `git add -A` here. Review the diff carefully: only the
new tools, README, updated `.gitignore`, and removal of tracked
generated outputs should be staged.

After the **full workflow regression** and review pass, commit and push:

```powershell
git commit -m "feat: finalize free YOLO26 batch workflow and clean generated files"
git push origin phase-2-yolo26
```

### Final integration into `main`

Review the Phase 2 pull request at
https://github.com/JACOBISAAC04/Bullseye.Ai/pull/new/phase-2-yolo26

Merge the pull request on GitHub **only after the review and both
regressions pass**, then update your local `main`:

```powershell
git switch main
git pull --ff-only origin main
git status
```

Alternatively, if you prefer to merge locally and `main` has not moved:

```powershell
git switch main
git merge --ff-only phase-2-yolo26
git push origin main
```

Do **not** perform both merging approaches. If `--ff-only` refuses, stop
rather than force the merge; inspect the branch history. Existing
untracked ignored local data remains on disk.

After merging, verify:

```powershell
git log -1 --oneline
git branch -vv
git status --short
```

The two successful past runs proved shot detection counts on the
available videos, but you should **visually review the rendered output**
as the final acceptance check. This handoff does not automatically
commit or merge anything into your GitHub account.
