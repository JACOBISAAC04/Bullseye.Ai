# Bullseye — Phase 5.8: Measurement-backed coaching review

This is an **additive, offline** module. It does not modify `video.py`, the basketball detector, shot tracker, renderer, or the coaching-reference files. No reprocessing of videos is required.

## Files

- `app/coaching/measurement_coaching_feedback.py`
- `tests/test_measurement_coaching_feedback.py`

## Install (PowerShell, from `D:\Bullseye\backend`)

```powershell
Copy-Item "$HOME\Downloads\measurement_coaching_feedback.py" "app\coaching\"
Copy-Item "$HOME\Downloads\test_measurement_coaching_feedback.py" "tests\"
python -m unittest discover -s tests -p "test_measurement_coaching_feedback.py" -v
```

Expected: `Ran 21 tests` and `OK`.

## Video 3

```powershell
python -m app.coaching.measurement_coaching_feedback `
    --temporal data/analyzed/phase4_temporal_form.json `
    --mechanics data/analyzed/phase5_mechanics_with_ball_video3.json `
    --review data/analyzed/phase5_coaching_review_video3.json `
    --output data/analyzed/phase5_measured_review_video3.json
```

Video 3 has two shots; only Shot 2 previously had six eligible human review prompts. **No additional correction or ideal-form verdict should appear.**

## Video 8

```powershell
python -m app.coaching.measurement_coaching_feedback `
    --temporal data/analyzed/phase4_temporal_video8.json `
    --mechanics data/analyzed/phase5_mechanics_with_ball_video8.json `
    --review data/analyzed/phase5_coaching_review_video8.json `
    --output data/analyzed/phase5_measured_review_video8.json
```

Video 8 previously had zero eligible review prompts, so this module will not invent new ones.

## Inspect results

```powershell
$r = Get-Content "data/analyzed/phase5_measured_review_video3.json" -Raw |
    ConvertFrom-Json

foreach ($shot in $r.shots) {
    Write-Host "`nSHOT $($shot.shot_number)"
    foreach ($topic in $shot.review_candidates) {
        Write-Host $topic.title "->" $topic.status
        $topic.observations | ConvertTo-Json -Depth 8
    }
}
```

## Interpretation, limitations

- Only candidate questions already approved by Phase 5.7 are enriched; this module does not search for more coaching claims.
- The set-to-release temporal segment must pass its **existing engineering-quality gate**.
- 2D joint-angle changes use observed frames only, not missing frames.
- Hip-center image displacement uses observed points, not a calibrated balance or lean measurement.
- Ball path uses previously validated **accepted direct BallAnalyzer observations** only. Missing release observations stay missing.
- Same-frame ball-to-wrist distances use detector ball centers and pose-estimated wrist coordinates, measured in pixels. They **cannot prove** ball contact or hand position relative to the underside of the ball.
- Source links remain associated with each review question but do not establish that the player's form is right or wrong.
- Input files are expected to be produced by the previously validated Bullseye modules. This module does not cryptographically authenticate input reports.
- `intended_form_match` remains `not_assessed`; `corrections_generated` remains `0`.
- The test suite uses synthetic fixtures. You must still run and inspect both real-video reports.

## Next checkpoint

If both results generate correctly, inspect the actual numeric observations for Video 3 Shot 2. Only after that should we consider adding measurement-backed review data to the FastAPI response.
