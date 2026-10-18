# Bullseye — Technique Evidence Validation (Phase 5.1)

This package is deliberately **three separate files**, not a ZIP. Filenames are neutral. Coaching source credits remain in the source catalog.

## Files

- `technique_validator.py` → `backend/app/coaching/technique_validator.py`
- `test_technique_validator.py` → `backend/tests/test_technique_validator.py`
- `README_TECHNIQUE_VALIDATION.md` → optional, place in `backend/`

**Prerequisite:** `backend/app/coaching/shooting_coaching_references.json` must already exist. It is NOT replaced.

## What it does

- Reads an existing Phase 4 temporal report: `phase4_temporal_form.json` or `phase4_temporal_video8.json`.
- Joins each of the 22 topics to the catalog's citation and measurement requirements.
- Distinguishes `missing_measurements`, `shot_context_not_established`, `insufficient_timing_evidence`, `source_verification_required`, and `descriptive_review_possible`.
- Exposes observed 2D elbow/hip/knee angles and wrist height **relative to shoulder** with tracking-quality context.
- Always emits `intended_form_match: not_assessed`. It cannot determine technical correctness or training progress.

The validator DOES NOT infer ball positions from pose landmarks, treat all 100%-covered clips as long enough, or pretend that provisional masterclass timestamps are verified. The Shorts queue is still empty pending inspection of individual clips.

This is not a full technique-grading module. In particular, shot-phase categories such as free-throw or catch-and-shoot require separate evidence that a clip depicts that shot context.

## Install (PowerShell)

From `D:\Bullseye\backend`:

```powershell
Copy-Item "$HOME\Downloads\technique_validator.py" "app\coaching\technique_validator.py"
Copy-Item "$HOME\Downloads\test_technique_validator.py" "tests\test_technique_validator.py"
Copy-Item "$HOME\Downloads\README_TECHNIQUE_VALIDATION.md" "."
```

## Test

```powershell
python -m unittest discover -s tests -p "test_technique_validator.py" -v
```

Expected: **15 tests, OK**.

## Validate saved videos (no reprocessing)

```powershell
python app/coaching/technique_validator.py `
    data/analyzed/phase4_temporal_form.json `
    --output data/analyzed/phase5_technique_review_video3.json

python app/coaching/technique_validator.py `
    data/analyzed/phase4_temporal_video8.json `
    --output data/analyzed/phase5_technique_review_video8.json
```

## Inspect output

```powershell
$r = Get-Content "data/analyzed/phase5_technique_review_video3.json" -Raw | ConvertFrom-Json
$r.shot_count
$r.reference_count
$r.intended_form_match
foreach ($shot in $r.shots) {
    Write-Host "SHOT:" $shot.shot_number
    $shot.assessments |
        Group-Object status |
        Select-Object Name, Count |
        Format-Table
}
```

A topic tagged `source_verification_required` is NOT a coaching correction. A topic tagged `descriptive_review_possible` is NOT a correct-form verdict either.

## Next development work

1. Verify specific YouTube videos and Shorts directly, with exact individual URLs and timestamps; update evidence levels only after verification.
2. Develop dedicated measurements for ball-to-wrist path and both hands' interaction with the ball.
3. Collect representative clips and independently evaluate pose accuracy and sensitivity to view/occlusion.
4. Only then consider validated, tightly scoped feedback rules.

Do not integrate into the FastAPI endpoint until the saved-video reports have been inspected.
