# Bullseye — Phase 6.3: Player Evidence Explanations

This is an **additive** post-processing step for Phase 6.1/6.2 `player_feedback.py`.
It does not re-run detection, introduce model scoring, compute physical balance, grade form,
verify YouTube timestamps, or create corrections.

## Files

| Downloaded file | Destination under `D:\Bullseye\backend` |
|---|---|
| `player_evidence_explanations.py` | `app\coaching\` |
| `test_player_evidence_explanations.py` | `tests\` |
| `README_PLAYER_EVIDENCE_EXPLANATIONS.md` | `.` |

Keep `player_feedback.py` installed. Do not modify `app/api/video.py` yet.

## Install

PowerShell in `D:\Bullseye\backend`:

```powershell
Copy-Item "$HOME\Downloads\player_evidence_explanations.py" "app\coaching\" -Force
Copy-Item "$HOME\Downloads\test_player_evidence_explanations.py" "tests\" -Force
Copy-Item "$HOME\Downloads\README_PLAYER_EVIDENCE_EXPLANATIONS.md" "." -Force

python -m unittest discover -s tests -p "test_player_evidence_explanations.py" -v
python -m unittest discover -s tests -p "test_player_feedback.py" -v
```

Expected for the new test file: **Ran 40 tests / OK**. Existing module has **36 tests**.

## Generate reports without running YOLO or MediaPipe

Both input files are your existing Phase 6.1/6.2 JSON reports.

```powershell
python -m app.coaching.player_evidence_explanations `
  --input data/analyzed/phase6_player_feedback_video3.json `
  --output data/analyzed/phase6_evidence_feedback_video3.json

python -m app.coaching.player_evidence_explanations `
  --input data/analyzed/phase6_player_feedback_video8.json `
  --output data/analyzed/phase6_evidence_feedback_video8.json
```

**Do not use the same path for `--input` and `--output`.** The program rejects this
to protect the original report.

## Expected regression behavior

- Video 3: two shots; Shot 1 zero cards and an **insufficient timing evidence** explanation.
- Video 3: Shot 2 six cards, each enriched with evidence limits, human-friendly title,
  review question, observation frame range, and *no calibrated confidence score*.
- Video 8: one shot, zero cards, with the insufficient-timing explanation.
- In all cases: `intended_form_match == "not_assessed"`,
  `corrections_generated == 0`, `official_endorsement == false`.
- Original shot replay, card replay, observation text, source URLs, and legacy
  JSON fields are preserved.

## Read the enriched results

```powershell
@'
import json
from pathlib import Path
for video in ("video3", "video8"):
    path = Path(f"data/analyzed/phase6_evidence_feedback_{video}.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{video.upper()} — confidence: {report['evidence_confidence_scoring']}")
    for shot in report["shots"]:
        ex = shot["evidence_explanation"]
        print(f"Shot {shot['shot_number']}: {shot['feedback_count']} cards; {ex['status']}")
        print("Why:", ex["reason"])
        for card in shot["feedback_cards"]:
            evidence = card["evidence_explanation"]
            print("\n", card["display_title"])
            print("Measured:", card["observation_text"])
            print("Question:", card["player_review_question"])
            print("Replay:", card["replay"]["start_seconds"],
                  "to", card["replay"]["end_seconds"], "s")
            print("Evidence:", evidence["label"])
            print("Confidence:", evidence["confidence"]["label"])
            print("Limitations:", "; ".join(evidence["limitations"]))
'@ | python -
```

## Enriched fields

Top-level additions:
- `evidence_explanations_version: "1.0"`
- `evidence_confidence_scoring: "not_available"`
- `evidence_explanation_limits`

Per-shot addition:
- `evidence_explanation` with `status`, `headline`, `reason`, `next_step`.

Per-card additions:
- `display_title`: clarifies potentially misleading topic names (e.g. "Joint-angle observations").
- `player_review_question`: cautious review prompt.
- `evidence_explanation`: describes scope, frame range, real limitations,
  source context, and the deliberate absence of a validated confidence score.

These fields are **not scientific uncertainty bounds**. Terms like `observed_2d_descriptive`
mean only that measured 2D observations were included after existing evidence gates.
A published coach interview provides context; it does not prove a player's form is good or bad.

## Next integration checkpoint

After you verify the enriched outputs on both real videos, integrate the new layer
into the FastAPI job pipeline and expose a report endpoint for the future frontend.
Before then, **do not overwrite the existing API and do not commit this phase**.
