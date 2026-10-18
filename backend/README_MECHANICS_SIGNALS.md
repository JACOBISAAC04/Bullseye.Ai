# Bullseye — Phase 5.2: Observed mechanics signals

The extractor reads **existing saved FastAPI `/video/analyze` JSON**.
It does not rerun YOLO26, MediaPipe, rendering, or change the API.

## Files

- `app/coaching/mechanics_signals.py`
- `tests/test_mechanics_signals.py`

## Install on Windows

From `D:\Bullseye\backend`, after downloading both files:

```powershell
Copy-Item "$HOME\Downloads\mechanics_signals.py" "app\coaching\"
Copy-Item "$HOME\Downloads\test_mechanics_signals.py" "tests\"
```

## Test

```powershell
python -m unittest discover -s tests -p "test_mechanics_signals.py" -v
```

Expected: `Ran 20 tests` and `OK`.

## Run on saved video analyses (no video reprocessing)

```powershell
python app/coaching/mechanics_signals.py `
  data/analyzed/hoop5_video3.json `
  --output data/analyzed/phase5_mechanics_video3.json

python app/coaching/mechanics_signals.py `
  data/analyzed/optimized_video8_test.json `
  --output data/analyzed/phase5_mechanics_video8.json
```

## Inspect observed signals

```powershell
$r = Get-Content "data/analyzed/phase5_mechanics_video3.json" -Raw |
    ConvertFrom-Json

foreach ($shot in $r.shots) {
    Write-Host "Shot $($shot.shot_number):" `
        ($shot.available_signals -join ", ")
}
Write-Host "Raw ball frames available:" $r.ball_detection_source_present
Write-Host "Form verdict:" $r.intended_form_match
```

## Data provenance and limits

- Pose points are used only when the particular frame has `pose_detected: true` and that landmark's image x/y are finite.
- Series availability requires at least two observed frames with one consecutive observed pair. **This is not evidence of reliable shooting-sequence timing.** Continue to use Phase 4 temporal quality gates.
- `wrist_position` is the shooting-side wrist at the *exact tracked release frame*. No substitute frame or interpolation.
- `hip_center_series` is the average of both visible 2D hip positions; not a 3D center of mass.
- `ankle_positions` requires both ankles; does **not** establish ground contact, balance, or footwork correctness.
- `shot_metrics` only includes already-measured phase angles from `metrics.shots`.
- Ball observations are used **only when** the optional `ball_analysis.frames` structure exists and the detection has a positive confidence and is not flagged interpolated.
- The standard Phase 4/Phase 3 saved JSON generally does **not** include raw ball detections. In that case ball signals remain unavailable.
- `shots[].trajectory` is intentionally ignored because its points may be interpolated and lack detection provenance.
- Pose visibility/accuracy, release timing, and coaching-rule correctness have not been validated by these tests.
- No automatic form scores, pass/fail classifications, or SeeMikeDunn endorsements are produced.

## Integration checkpoint

This is a standalone, read-only signal extractor. The existing `technique_validator.py`
has **not** been changed: its current `missing_measurements` counts will stay the
same until we separately test a provenance-aware adapter that consumes both the
mechanics report and the Phase 4 temporal report. Do not treat the new signals
as coaching judgments.

Review your real outputs before integrating into the validator or FastAPI.
