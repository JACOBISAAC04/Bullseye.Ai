# Bullseye Phase 5.6 — Coaching Reference Registry

This is an **additive, read-only** source audit for your existing 22-topic
`shooting_coaching_references.json`. It does not modify YOLO26, ball detection,
shot tracking, FastAPI, the original reference library, or any scoring logic.

## Files and destinations

Copy these three individual downloads (not a ZIP):

| Download | Place in `D:\Bullseye\backend` |
|---|---|
| `coaching_reference_registry.py` | `app\coaching\coaching_reference_registry.py` |
| `coaching_source_evidence.json` | `app\coaching\coaching_source_evidence.json` |
| `test_coaching_reference_registry.py` | `tests\test_coaching_reference_registry.py` |

All filenames are neutral; source attribution is kept **inside** the JSON.

## PowerShell installation

```powershell
cd D:\Bullseye\backend

Copy-Item "$HOME\Downloads\coaching_reference_registry.py" "app\coaching\"
Copy-Item "$HOME\Downloads\coaching_source_evidence.json" "app\coaching\"
Copy-Item "$HOME\Downloads\test_coaching_reference_registry.py" "tests\"

python -m unittest discover -s tests -p "test_coaching_reference_registry.py" -v
```

Expected: `Ran 28 tests`, `OK`.

## Build the standalone 22-topic registry

```powershell
python -m app.coaching.coaching_reference_registry `
    --output data/analyzed/phase5_reference_registry.json
```

## Audit the existing Video 3 and Video 8 integrated reports

```powershell
python -m app.coaching.coaching_reference_registry `
    --report data/analyzed/phase5_ball_integrated_video3.json `
    --output data/analyzed/phase5_reference_review_video3.json

python -m app.coaching.coaching_reference_registry `
    --report data/analyzed/phase5_ball_integrated_video8.json `
    --output data/analyzed/phase5_reference_review_video8.json
```

No detector execution is needed. The source report preserves the **exact**
incoming per-topic statuses, `missing_signals`, and `not_assessed` verdicts.

## Inspect a reference and its source bounds

```powershell
@'
import json
from pathlib import Path

r = json.loads(Path("data/analyzed/phase5_reference_review_video3.json").read_text(encoding="utf-8"))

print("References:", r["reference_count"])
print("Source-support breakdown:", r["source_support_counts"])
print("Shots:", r["shot_count"])
print("Intended form:", r["intended_form_match"])

for reference in r["references"]:
    if reference["principle_id"] in ("posture_setup", "sequence", "ball_path",
                                     "avoid_forced_release_height"):
        print("\n", reference["title"])
        print("Original source level:", reference["original_citation"]["evidence_level"])
        print("Additional support:", reference["source_support_status"])
        for claim in reference["supplemental_evidence"]:
            print(" -", claim["claim_paraphrase"])
            print("   Scope:", claim["scope_note"])

for shot in r["shots"]:
    print("\nShot", shot["shot_number"])
    for assessment in shot["assessments"]:
        if assessment["status"] == "source_verification_required":
            print(assessment["title"], "->", assessment["registry_source_support_status"])
'@ | python -
```

## Evidence policy

The overlay contains 11 narrowly scoped source statements reviewed from two
public pages:

* Public interview attributed to shooting coach Mike Dunn:
  https://www.hustletraining.com/blog/how-to-shoot-with-perfect-form-from-professional-shooting-coach-mike-dunn
* Official course *outline* (not its paid video lessons):
  https://seemikedunn.com/product/ground-up-3-0-video-course/

Some published-interview paraphrases can provide **alternative, contextual**
source support for a topic. They **do not independently confirm** an original
masterclass timestamp claim. A course outline verifies only that a topic is
taught, not the instruction or numeric criterion behind it.

The masterclass timestamp summaries, unreviewed individual Shorts, and mirrored
social-media posts remain provisional. This registry never generates an
automatic shooting correction, an ideal angle, or an endorsement claim.

Important: a source registry does not validate computer-vision tracking quality,
camera geometry, shot context, or coaching effectiveness. Source verification
requires checking the actual published material and retaining its scope; a
transcript is an attributed interview, not independent proof of the physical
claim. New entries need human content review, not just passing schema tests.

## Next separate milestone

Once the real Video 3 and Video 8 audits pass, we can implement a narrowly
sourced *review prompt generator* for cases with sufficiently trustworthy
observations, while keeping original assessment decisions and the FastAPI
pipeline unchanged until regression testing.
