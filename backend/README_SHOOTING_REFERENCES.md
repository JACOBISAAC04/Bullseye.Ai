# Bullseye — Extended Shooting Coaching References

This additive pack was researched on 2026-10-18. It **does not overwrite** Phase 4 files or enable form pass/fail grading.

## Contents
- `app/coaching/shooting_coaching_references.json` — 22 topic entries, linked sources, suggested observables, evidence limitations, and Shorts-review queue
- `tests/test_shooting_coaching_references.py` — six conservative source-integrity checks

## What was actually verified
- A SeeMikeDunn public long-form YouTube masterclass: https://www.youtube.com/watch?v=jPHvxFiNq80
- Timestamps/segments for this masterclass **were obtained from a third-party AI-generated summary**, not a personally reviewed transcript. All these topic records are marked `provisional_third_party_video_summary`; verify them at their clickable timestamps before enabling as coaching criteria.
- A public Mike Dunn interview transcript: https://www.hustletraining.com/blog/how-to-shoot-with-perfect-form-from-professional-shooting-coach-mike-dunn
- Public course outline: https://seemikedunn.com/product/ground-up-3-0-video-course/
- Attributed coach posts are included for supplementary context and explicitly marked as **not YouTube**.
- SeeMikeDunn Shorts tab: https://www.youtube.com/@SeeMikeDunn/shorts

**Individual YouTube Shorts could not be directly browsed/reliably retrieved**, so this pack does not claim to have watched or extracted specific Shorts. This intentionally avoids unsupported quotations or made-up video links. `shorts_review_queue.items` is empty and must stay so until actual linked Shorts are reviewed.

## Install the separate files (no ZIP)

Download each file separately and place them in your existing repository:

- `shooting_coaching_references.json` → `D:\Bullseye\backend\app\coaching\shooting_coaching_references.json`
- `test_shooting_coaching_references.py` → `D:\Bullseye\backend\tests\test_shooting_coaching_references.py`
- `README_SHOOTING_REFERENCES.md` → `D:\Bullseye\backend\README_SHOOTING_REFERENCES.md`

Or, after downloading all three into your Windows Downloads folder:

```powershell
cd D:\Bullseye\backend
Copy-Item "$HOME\Downloads\shooting_coaching_references.json" "app\coaching\shooting_coaching_references.json"
Copy-Item "$HOME\Downloads\test_shooting_coaching_references.py" "tests\test_shooting_coaching_references.py"
Copy-Item "$HOME\Downloads\README_SHOOTING_REFERENCES.md" "."
python -m unittest discover -s tests -p "test_shooting_coaching_references.py" -v
```

Expected: `Ran 6 tests` and `OK`.

## Inspect the library

```powershell
@'
import json
from pathlib import Path
p = Path("app/coaching/shooting_coaching_references.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("Topics:", len(d["principles"]))
for v in d["principles"]:
    c = v["citation"]
    print(v["id"], "|", c["evidence_level"], "|", c["url"])
print("Shorts reviewed:", len(d["shorts_review_queue"]["items"]))
'@ | python -
```

## How Bullseye should use this library

1. **Learning/review prompts only.** No production code reads this file yet. The existing `coaching_rules.json`, `reference_principles.json`, FastAPI route, and `CoachingReportPipeline` are untouched.
2. Cross-check each masterclass timestamp against the actual uploaded lesson before upgrading a source's evidence level.
3. To add a Short, supply its original Shorts URL and review its actual words/actions. Add one record with timestamp, attribution, observations, and reviewed-by provenance. Do not infer advice from its title.
4. Only propose computed technique features after confirming required 2D/3D ball and pose data is observable and validating algorithms with coach-reviewed video.
5. Keep `intended_form_match` equal to `not_assessed` until evidence and a defensible rubric exist.

## Sample provisional item

`sequence` points to the long-form masterclass around 09:15. It suggests investigating the **relative timing** of ball movement and knee/hip extension; it does **not** score that timing or say any shooter is wrong.

## Important nuance

Dunn's interview includes a close-to-the-hip free-throw setup as an example. That does **not** establish a universal ball-starting position for every player or shot. The research library keeps this context explicit.

This research pack has no affiliation with or endorsement from Mike Dunn/SeeMikeDunn.
