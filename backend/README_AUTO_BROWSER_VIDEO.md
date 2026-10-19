# Bullseye Phase 6 — Automatic browser-compatible H.264 video

## Goal

Make every **new** analyzed basketball video browser-playable automatically.

- Keep `shooting_analysis.mp4` unchanged for existing downloads.
- Write `shooting_analysis_browser.mp4` (H.264 + yuv420p + `faststart`) into the same job folder.
- Preserve FPS, frame count (within a 1-frame container rounding tolerance), and dimensions.
- Build using the locally installed `imageio-ffmpeg` binary; no paid services.
- Use a temporary file and publish the new MP4 only after validating it.
- If transcoding fails, log the error, return the regular analysis, and report `browser_video_status: "unavailable"`.
- The existing `/video/results/{job_id}/stream` endpoint must **prefer** `shooting_analysis_browser.mp4`. Your current manually fixed route already does this.

This stage does not modify YOLO, MediaPipe, ball tracking, metrics, coaching, Gemini, or TTS.

## Files and target locations

| Download | Install |
| --- | --- |
| `browser_video.py` | `backend/app/analysis/browser_video.py` |
| `test_browser_video.py` | `backend/tests/test_browser_video.py` |
| `video_h264_integration.patch` | `backend/video_h264_integration.patch` (temporary integration aid; do not commit unless desired) |

## Installation (PowerShell; VS Code)

Make sure you are on `phase-6-player-feedback`.

```powershell
cd D:\Bullseye
git branch --show-current
git status --short
```

Install only the new module and its tests:

```powershell
cd D:\Bullseye\backend
Copy-Item "$HOME\Downloads\browser_video.py" "app\analysis\" -Force
Copy-Item "$HOME\Downloads\test_browser_video.py" "tests\" -Force
Copy-Item "$HOME\Downloads\video_h264_integration.patch" "." -Force
```

Apply the minimal, reviewable patch against your existing `video.py`:

```powershell
cd D:\Bullseye
git apply --check backend\video_h264_integration.patch
```

**If the check reports a conflict, do not apply it.** Inspect `backend/app/api/video.py` in VS Code and share the conflict with your coding assistant. Do not replace `video.py` with an older file.

If `git apply --check` exits successfully:

```powershell
git apply backend\video_h264_integration.patch
code backend\app\api\video.py
```

The patch changes only:
1. An import of `ensure_browser_video`.
2. A nonblocking `try/except` block immediately after validating the rendered original video.
3. An API result field, `browser_video_status` (`available`/`unavailable`).

Keep your manually corrected `app/api/voice_playback.py` (it already prefers the browser MP4 in `/stream`). For reference, the stream endpoint should select:

```python
job = _job_folder(job_id)
browser = job / "shooting_analysis_browser.mp4"
original = job / "shooting_analysis.mp4"
path = browser if browser.is_file() else original
```

No change to the HTML player is required when it uses `/stream`.

## Tests

```powershell
cd D:\Bullseye\backend

python -m py_compile app\analysis\browser_video.py app\api\video.py

python -m unittest discover -s tests -p "test_browser_video.py" -v

python -m unittest discover -s tests -p "test_*.py"
```

The new tests use a small **local synthetic video**. They do not send video anywhere.

## Check an existing job without reprocessing

Use the Video 8 job ID from your saved response:

```powershell
cd D:\Bullseye\backend

$r = Get-Content "data/analyzed/phase5_measured_api_video8.json" -Raw | ConvertFrom-Json

python -m app.analysis.browser_video `
    --input "data/analyzed/$($r.job_id)/shooting_analysis.mp4"
```

This creates a browser copy if it does not exist, or reports `already_available` if a verified H.264 copy exists. No shooting models are rerun.

## Verify automatic conversion for NEW uploads

Restart the backend after installing the patch:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a **second** terminal:

```powershell
cd D:\Bullseye\backend

curl.exe -sS `
    -o "data/analyzed/phase6_h264_api_video8.json" `
    -w "HTTP %{http_code} | Total %{time_total}s`n" `
    -F "file=@data/videos/video8.mp4" `
    http://127.0.0.1:8000/video/analyze

$r = Get-Content "data/analyzed/phase6_h264_api_video8.json" -Raw | ConvertFrom-Json

$r.job_id
$r.shot_count
$r.browser_video_status
$r.shots | Format-Table

Get-Item "data/analyzed/$($r.job_id)/shooting_analysis_browser.mp4"
```

Expected when FFmpeg succeeds:
- HTTP 200
- `shot_count: 1`, frames 103/107/138 and 36 trajectory points
- `browser_video_status: available`
- Both the original and browser MP4 files exist.

Open the player:

```powershell
Start-Process "http://127.0.0.1:8000/video/results/$($r.job_id)/player"
```

For any conversion failure:
- HTTP video-analysis response should still succeed (assuming analysis itself succeeded).
- `browser_video_status: unavailable`; details are in the local backend log.
- The original `shooting_analysis.mp4` and reports are preserved.
- The embedded browser player may still be unable to play the original MP4 until conversion succeeds.

## Git hygiene

Stage only intended project files and verify `.env` remains ignored. **Never run `git add .` here.**

```powershell
cd D:\Bullseye
git diff --check
git status --short
git check-ignore -v backend/.env
```

Don't stage `backend/app/models/ball_detector/ball.onnx`, local `.env`, generated videos, `.pip-temp`, or backup files.

The integration patch is optional source material, not required at runtime after being applied.
