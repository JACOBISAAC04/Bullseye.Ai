# Bullseye Phase 5.5 — Direct Ball + Pose Evidence Adapter

This is an additive, offline JSON integration. It does **not** modify `video.py`,
rerun YOLO26 or MediaPipe, or change the shot tracker. Filenames are neutral.

## Install (PowerShell, from `D:\Bullseye\backend`)

```powershell
Copy-Item "$HOME\Downloads\ball_evidence_adapter.py" "app\coaching\ball_evidence_adapter.py"
Copy-Item "$HOME\Downloads\test_ball_evidence_adapter.py" "tests\test_ball_evidence_adapter.py"
```

Dependencies: already-installed `app.coaching.mechanics_signals` and
`app.coaching.technique_signal_integration` modules.

## Test

```powershell
python -m unittest discover -s tests -p "test_ball_evidence_adapter.py" -v
```

Expected: `Ran 23 tests ... OK`.

## Enrich Video 3 mechanics from saved reports (no video reprocessing)

```powershell
python -m app.coaching.ball_evidence_adapter `
  data/analyzed/phase5_api_video3.json `
  data/analyzed/phase5_ball_observations_video3.json `
  --output data/analyzed/phase5_mechanics_with_ball_video3.json
```

For Video 8:

```powershell
python -m app.coaching.ball_evidence_adapter `
  data/analyzed/phase5_api_video8.json `
  data/analyzed/phase5_ball_observations_video8.json `
  --output data/analyzed/phase5_mechanics_with_ball_video8.json
```

You should find `ball_positions` available for observed pre-release sequences.
For Video 3, `ball_position` **must stay unavailable** at frames 108 and 396
because neither had an accepted direct ball detection. For Video 8 it may be
available at frame 107, as the existing ball-observation report shows.

The enriched report also contains per-shot `ball_body_evidence`, with direct
ball observations matched to right/left wrist, shoulder, and hip **on identical
frame numbers**. It includes raw 2D pixel distances and y offsets, not
contact classifications or ideal-form targets. Missing pose/ball frames are
never filled.

## Feed the enriched mechanics to the 22-topic technique validator

```powershell
python -m app.coaching.technique_signal_integration `
  data/analyzed/phase4_temporal_form.json `
  data/analyzed/phase5_mechanics_with_ball_video3.json `
  --output data/analyzed/phase5_ball_integrated_video3.json

python -m app.coaching.technique_signal_integration `
  data/analyzed/phase4_temporal_video8.json `
  data/analyzed/phase5_mechanics_with_ball_video8.json `
  --output data/analyzed/phase5_ball_integrated_video8.json
```

Inspect:

```powershell
@'
import json
from pathlib import Path
for tag in ("video3", "video8"):
    mech = json.loads(Path(f"data/analyzed/phase5_mechanics_with_ball_{tag}.json").read_text())
    integrated = json.loads(Path(f"data/analyzed/phase5_ball_integrated_{tag}.json").read_text())
    print("\n", tag, "form:", integrated["intended_form_match"])
    for shot in mech["shots"]:
        signals = shot["signals"]
        evidence = shot["ball_body_evidence"]
        print("shot", shot["shot_number"],
              "ball series", signals["ball_positions"]["available"],
              "exact release", signals["ball_position"]["available"],
              "same-frame matches", evidence["same_frame_pose_ball_matches"])
    for shot in integrated["shots"]:
        print("shot", shot["shot_number"], shot["status_counts"])
'@ | python -
```

## Accuracy and provenance

- A confidence-scored, outlier-filtered ball detection is *candidate evidence*,
  not verified truth or proof the hand touches the ball.
- Accepted observations must be direct BallAnalyzer predictions with source
  metadata and must agree with per-shot and global evidence listings.
- Frame counts, shot identifiers, shooting arms, boundaries, acceptance counts,
  and rejected frames must agree between reports.
- The adapter reconstructs an internal `ball_analysis.frames` input using **only**
  accepted direct observations. Missing/rejected/interpolated entries stay null.
- Both the mechanics and 22-topic validators continue to output
  `intended_form_match: not_assessed`.
- Timing quality still comes from the separate Phase 4 temporal analyzer.
- The saved reports must come from the **same video run**; the consistency
  checks cannot cryptographically prove the video origin.
- Because the adapter reads saved reports, it cannot recover original detections
  not recorded by those reports.
