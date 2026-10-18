# Bullseye — Phase 5.3 Technique Signal Integration

A read-only, source-linked integration layer. No ZIP, new model weights, web calls,
API changes, video reruns, numerical ideal-angle targets, or automatic form judgments.

## Files to install

- `technique_signal_integration.py` -> `backend/app/coaching/technique_signal_integration.py`
- `test_technique_signal_integration.py` -> `backend/tests/test_technique_signal_integration.py`

Requires already installed:
- `app/coaching/technique_validator.py`
- `app/coaching/shooting_coaching_references.json`
- Phase 4 temporal and Phase 5 mechanics reports (from the same video).

## Run from `D:\Bullseye\backend`

```powershell
python -m unittest discover -s tests -p "test_technique_signal_integration.py" -v
```

Video 3:

```powershell
python app/coaching/technique_signal_integration.py `
    data/analyzed/phase4_temporal_form.json `
    data/analyzed/phase5_mechanics_video3.json `
    --output data/analyzed/phase5_integrated_video3.json
```

Video 8:

```powershell
python app/coaching/technique_signal_integration.py `
    data/analyzed/phase4_temporal_video8.json `
    data/analyzed/phase5_mechanics_video8.json `
    --output data/analyzed/phase5_integrated_video8.json
```

Inspect statuses:

```powershell
$r = Get-Content "data/analyzed/phase5_integrated_video3.json" -Raw | ConvertFrom-Json
foreach ($shot in $r.shots) {
    Write-Host "Shot $($shot.shot_number)"
    $shot.status_counts | Format-List
    Write-Host "Available mechanics:" ($shot.available_mechanics_signals -join ", ")
}
```

## What is checked

1. Shot numbers, shooting arm, and start/release/end frames must match across reports.
2. Existing source verification and timing logic are preserved from `TechniqueEvidenceValidator`.
3. Mechanics series must have valid 2D values, sequential real frame numbers, at least
   two observations, and at least one adjacent pair. Counts are independently checked.
4. Single release samples must match the exact tracked release frame.
5. `ball_positions` or `ball_position` need reported `ball_analysis.frames` provenance,
   with valid positive confidence. Trajectory points do not count.
6. Clip context (catch-and-shoot, drill, etc.) is not inferred.
7. Every result preserves `intended_form_match: not_assessed`.

## Interpretation

- `descriptive_review_possible`: observed measurements, temporal quality, and a
  published source permit a question or review prompt — not correctness grading.
- `source_verification_required`: descriptive measurements exist, but source details
  from the video or secondary summary have not been directly verified.
- `insufficient_timing_evidence`: pose gaps or short temporal windows block timing review.
- `missing_measurements`: required observations still unavailable (including ball data,
  hand/ball contact, grip).
- `shot_context_not_established`: needs independently established shot/drill context.

A source-linked topic is not an approved mechanical pass/fail criterion. Pose coordinates
are 2D and perspective-dependent; model confidence isn't independently calibrated.
A matched shot ID and frame window do not establish that reports came from the same
original video file; use both outputs from one video, and avoid mixing jobs.
Do not use these reports to prescribe technique correction without external validation.

No FastAPI endpoint changes are required or made.
