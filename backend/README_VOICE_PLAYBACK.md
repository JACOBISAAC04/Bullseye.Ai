# Bullseye — Phase 6 Voice-Enabled Shot Playback

## What this adds

An on-demand web page for an *existing analyzed job*:

- `GET /video/results/{job_id}/player` — HTML5 analyzed-video player with per-shot Replay and Listen buttons.
- `GET /video/results/{job_id}/stream` — plays existing `shooting_analysis.mp4` inline, with browser seeking.
- `POST /video/results/{job_id}/voice?use_gemini=true` — creates the evidence-backed Phase 6 report and shot WAVs using the existing Gemini/Windows speech pipeline.
- `GET /video/results/{job_id}/voice/manifest`
- `GET /video/results/{job_id}/voice/player-feedback`
- `GET /video/results/{job_id}/voice/shot/{shot_number}` — serves an existing WAV inline.

Nothing in `app/api/video.py` changes. The voice step is optional, invoked only when the local user clicks Generate. Existing YOLO/MediaPipe/ball detection is not re-run.

## Install individual files

Copy these into the project from Downloads:

```powershell
cd D:\Bullseye\backend

Copy-Item "$HOME\Downloads\voice_playback.py" "app\api\" -Force
Copy-Item "$HOME\Downloads\voice_player.html" "app\api\" -Force
Copy-Item "$HOME\Downloads\test_voice_playback.py" "tests\" -Force

# IMPORTANT: Backup local app/main.py before replacing it.
Copy-Item "app\main.py" "app\main.py.phase6backup" -Force
Copy-Item "$HOME\Downloads\main.py" "app\main.py" -Force

python -m py_compile app\api\voice_playback.py app\main.py
python -m unittest discover -s tests -p "test_voice_playback.py" -v
python -m unittest discover -s tests -p "test_*.py"
```

Expected for new tests: `Ran 15 tests` and `OK`.

Review `app/main.py` in VS Code:

```powershell
code app\main.py
```

Do not stage `app/main.py.phase6backup`. It is a local safety copy and must be removed or kept untracked.

## Reuse your existing Video 3 job

This is the previous successful Video 3 analysis job ID:

`9a9333b9-2c79-4a75-9a5c-e8a70b5d83a4`

Check whether the *existing* artifacts still exist:

```powershell
cd D:\Bullseye\backend

$job = "9a9333b9-2c79-4a75-9a5c-e8a70b5d83a4"
Get-Item "data\analyzed\$job\shooting_analysis.mp4"
Get-Item "data\analyzed\$job\temporal_form.json"
Get-Item "data\analyzed\$job\measured_coaching_report.json"
```

The original upload `data\videos\<job_id>.mp4` (or .mov/.avi/.mkv) must also still be available, because it supplies the actual FPS. If artifacts are missing, upload/analyze a video first.

Launch FastAPI in a PowerShell window if it isn't running:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a *second* PowerShell window:

```powershell
$job = "9a9333b9-2c79-4a75-9a5c-e8a70b5d83a4"
Start-Process "http://127.0.0.1:8000/video/results/$job/player"
```

Click **Generate Gemini-assisted voice** once. This creates one WAV per shot, plus `voice_manifest.json` and `player_feedback.json`, under:

`data/analyzed/<job_id>/voice_playback/`

Select **Replay shot** and **Listen to coach** independently. The video is not slowed down to match narration duration.

You can also trigger generation from PowerShell:

```powershell
curl.exe -fS -X POST "http://127.0.0.1:8000/video/results/$job/voice?use_gemini=true"
```

Gemini is optional. For local deterministic narration, use `use_gemini=false`. A failed Gemini response safely uses the local approved-sentence fallback.

## Design and safety notes

- This local prototype deliberately restricts voice-creation POSTs to loopback (`127.0.0.1`, `::1`). Before public hosting, add user authentication, access controls, request limits, and quota protection. GET endpoints similarly require authentication for any production deployment.
- Gemini requests might consume quota; the first successful voice generation is cached per job and is not overwritten by subsequent requests. `use_gemini=true` does not regenerate an already-created local voice folder.
- The Gemini key remains in `backend/.env`, ignored by Git; never put the key in frontend HTML/JS.
- Only selected measurements, not uploaded raw videos, go to Gemini. The offline speech engine creates WAV audio.
- No shooting-form corrections or make/miss results are fabricated. Existing evidence-quality gates remain in effect.
- The original analyzed MP4 must be browser-playable (H.264/AAC is typical); if the renderer used a codec your browser does not support, the file may download or show a black player despite returning HTTP 200. That is a separate encoding task.
- A new voice request will not overwrite completed or partially-created voice directories. If generation fails, inspect the backend logs, verify `.env`, then retry.
- The frontend page is intentionally small and self-contained: it requires no JavaScript framework or separate frontend server.
