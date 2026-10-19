# Bullseye — Phase 6.1 / 6.2: Player-friendly feedback and video replay metadata

**Scope:** a local, offline, additive report generator. It reads the existing
Phase 5.9 `POST /video/analyze` response and the downloaded measured-coaching JSON.
It does **not** change the FastAPI routes, shot tracker, object detectors,
pose analyzer, rendering, reference registry, or source grading.

## Delivered files

| Download file | Install to |
|---|---|
| `player_feedback.py` | `backend/app/coaching/player_feedback.py` |
| `test_player_feedback.py` | `backend/tests/test_player_feedback.py` |
| `README_PLAYER_FEEDBACK.md` | `backend/README_PLAYER_FEEDBACK.md` |

Use the `phase-6-player-feedback` branch based on **merged** Phase 5 code.
If your Phase 5 pull request has not been merged yet, merge it before creating
the new branch. Keep your local ONNX model, `.pip-temp`, and temporary backup
files untracked.

## Install (PowerShell)

```powershell
cd D:\Bullseye

# Do these three commands only once Phase 5 is merged to main:
git switch main
git pull --ff-only origin main
git switch -c phase-6-player-feedback

cd D:\Bullseye\backend

Copy-Item "$HOME\Downloads\player_feedback.py" "app\coaching\" -Force
Copy-Item "$HOME\Downloads\test_player_feedback.py" "tests\" -Force
Copy-Item "$HOME\Downloads\README_PLAYER_FEEDBACK.md" "." -Force
```

If you already created the `phase-6-player-feedback` branch, skip the branch
creation commands and just install the files.

## Run tests

```powershell
python -m unittest discover -s tests -p "test_player_feedback.py" -v
```

Expected on the delivered version: `Ran 36 tests` then `OK`.

## Run Video 3 without reprocessing video

Use the saved files from your successfully validated Phase 5.9 API run:

```powershell
python -m app.coaching.player_feedback `
    --analysis data/analyzed/phase5_measured_api_video3.json `
    --measured data/analyzed/phase5_measured_download_video3.json `
    --video data/videos/video3.mp4 `
    --output data/analyzed/phase6_player_feedback_video3.json
```

For Video 8:

```powershell
python -m app.coaching.player_feedback `
    --analysis data/analyzed/phase5_measured_api_video8.json `
    --measured data/analyzed/phase5_measured_download_video8.json `
    --video data/videos/video8.mp4 `
    --output data/analyzed/phase6_player_feedback_video8.json
```

`--video` only reads the original video's frame rate via OpenCV; it does **not**
run ball detection, MediaPipe, YOLO, or rendering. Using the original
video for FPS is important because Bullseye videos may not be 30 FPS.

Alternative if the original file is unavailable: replace `--video ...` with
`--fps <ACTUAL_FPS>` after checking the file's real frame rate. There is no
silent 30-FPS fallback.

Bullseye's ShotTracker and BallAnalyzer use **one-based frame numbers**:
frame 1 begins at 0 seconds. The default `--frame-origin 1` uses
`(frame - 1)/fps`. Specify `--frame-origin 0` only for a different
zero-based upstream dataset. Output includes both frame IDs and seek seconds.

## Inspect player-readable feedback

```powershell
$r = Get-Content "data/analyzed/phase6_player_feedback_video3.json" -Raw |
    ConvertFrom-Json

Write-Host "Shots:" $r.shot_count
Write-Host "FPS:" $r.fps
Write-Host "Form verdict:" $r.intended_form_match

foreach ($shot in $r.shots) {
    Write-Host "`nSHOT $($shot.shot_number)"
    Write-Host "Replay:" $shot.replay.start_seconds "-" $shot.replay.end_seconds "seconds"
    Write-Host "Review cards:" $shot.feedback_count

    foreach ($card in $shot.feedback_cards) {
        Write-Host "`nTOPIC:" $card.title
        Write-Host "OBSERVED:" $card.observation_text
        Write-Host "REVIEW QUESTION:" $card.review_question
        Write-Host "REPLAY:" $card.replay.start_seconds "-" $card.replay.end_seconds "seconds"
        Write-Host "SOURCE:" ($card.source_urls -join ", ")
    }
}
```

## Expected checks

| Video | Shot | Expected displayed review cards |
|---|---:|---:|
| Video 3 | 1 | 0 (timing evidence insufficient) |
| Video 3 | 2 | 6 (posture, hip loading, balance, ball setup, ball path, player-specific review) |
| Video 8 | 1 | 0 (timing evidence insufficient) |

These expectations derive from the earlier Phase 5 results. **You must verify
the output on your local saved reports.**

## Output contract for future frontend

- `shots[].headline` / `summary`: understandable shot status.
- `shots[].replay`: shot start, release, end frames and video seek seconds.
- `shots[].feedback_cards[]`: topic, measured descriptive sentence,
  coaching review question, replay interval, original published-source
  URLs, and limitations.
- `video_url`: reused exact existing relative `/video/results/<job_id>/video`
  link from the API response; never a manufactured external URL.
- `fps`, `frame_origin`: traceable timestamp conversion inputs.
- `intended_form_match: "not_assessed"`; `corrections_generated: 0`;
  `official_endorsement: false`.

A future frontend can set `video.currentTime` to each card's
`replay.start_seconds`, but this phase does not implement browser playback
controls or a frontend UI. The measured JSON and source links remain the
auditable inputs.

## Evidence safeguards

1. Input must have matching shot numbers, valid start/release/end frames,
   a safe ungraded measured report, and proper descriptive timing quality.
2. Missing or unrecognized measurement types are **not** displayed as
   measured evidence. The builder never fabricates missing ball frames.
3. Pixel movement is phrased as image-plane displacement; it cannot prove
   real-world balance, ball contact, hand grip, or ideal shooting mechanics.
4. Joint angles are 2D estimates; they are never classified as optimal.
5. Phase-angle observations without exact frame identifiers are called
   `set sample` / `release sample`; their replay spans the preparation
   interval without assigning invented sample frame IDs.
6. Approved source URLs are carried through from Phase 5 reports; the
   builder does not assert the original YouTube timestamps are verified.
7. This phase uses **no paid API** and needs no extra Python dependencies
   beyond those the Bullseye backend already uses. `--fps` mode itself
   uses Python's standard library.

## Next checkpoint

Run the 36 tests and the two report-generation commands. Inspect the six
Shot 2 cards, especially their replay timestamps, before we add API wiring
or a frontend dashboard. Keep new Phase 6 code on its own branch.
