# Bullseye Phase 5.7 — Source-linked Coaching Review Questions

This is an **additive, read-only analysis stage**. It reads the Phase 5.6
`phase5_reference_review_*.json` report and produces review questions for a
human coach or player. It **does not** grade technique, prescribe corrections,
infer missing measurements, change timestamps, or reprocess videos.

## Files to install

Copy the three separately downloaded files (no ZIP) into your backend:

- `app/coaching/coaching_review_generator.py`
- `tests/test_coaching_review_generator.py`
- `README_COACHING_REVIEW_GENERATOR.md` (backend root)

Existing dependencies must be present:

- `app/coaching/coaching_reference_registry.py`
- `app/coaching/coaching_source_evidence.json`
- `app/coaching/shooting_coaching_references.json`

## Run from PowerShell

```powershell
cd D:\Bullseye\backend

python -m unittest discover -s tests -p "test_coaching_review_generator.py" -v

python -m app.coaching.coaching_review_generator `
    --report data/analyzed/phase5_reference_review_video3.json `
    --output data/analyzed/phase5_coaching_review_video3.json

python -m app.coaching.coaching_review_generator `
    --report data/analyzed/phase5_reference_review_video8.json `
    --output data/analyzed/phase5_coaching_review_video8.json
```

Expected output includes shots reviewed, manual review question counts, and
`Intended-form verdict: not_assessed`. A prior synthetic fixture generated
four questions, but actual question counts depend on the source support and
measurement-status combinations in your report.

### Inspect the review questions

```powershell
$r = Get-Content "data/analyzed/phase5_coaching_review_video3.json" -Raw | ConvertFrom-Json
foreach ($s in $r.shots) {
    Write-Host "`nSHOT $($s.shot_number): $($s.review_candidate_count) review candidates"
    foreach ($q in $s.review_candidates) {
        Write-Host "  - $($q.title)"
        Write-Host "    Question: $($q.review_question)"
        Write-Host "    Measurement gate: $($q.measurement_status)"
        Write-Host "    Source(s): $($q.source_urls -join ', ')"
    }
}
```

## Safety and evidence policy

1. The module reloads the **local validated Phase 5.6 registry** and checks
   that input source metadata matches it exactly; altered source claims are
   rejected.
2. An assessment is a candidate for **human review** only if it has adequate
   upstream observations (`source_verification_required` or
   `descriptive_review_possible`) *and* limited published-interview context.
   The original upstream assessment status is preserved; this does **not**
   mark the original masterclass timestamp verified.
3. Missing measurements, insufficient timing evidence, and missing shot
   context always block a review question.
4. Course-topic-only support is never promoted into a specific instruction.
   Provisional Shorts, video-summary timestamps, and mirrored social posts
   cannot supply detailed coaching claims.
5. Review questions are taken from the existing local 22-topic catalog, and
   bounded contextual notes come from the vetted evidence overlay. No
   numerical "ideal" angles, individualized form diagnoses, or drills are
   generated.
6. The tool does not compute new motion evidence. Detector predictions can
   still be incorrect even when all required signals are present.
7. The system is independent and not endorsed by the referenced coach.

Do not integrate this stage into FastAPI until the Video 3 and Video 8
reports have been inspected and the regression tests remain green.
