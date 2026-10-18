# Bullseye — Phase 5.4: Ball observation provenance

This package adds the ball-observation store **without replacing or changing** YOLO detection, outlier filtering, interpolation, ShotTracker, metrics, or rendering.

## Install on Windows

Work in `D:\\Bullseye\\backend` on `phase-5-technique-validation`.

```powershell
Copy-Item "$HOME\Downloads\ball_observation_store.py" "app\coaching\ball_observation_store.py"
Copy-Item "$HOME\Downloads\test_ball_observation_store.py" "tests\test_ball_observation_store.py"
Copy-Item "$HOME\Downloads\README_BALL_OBSERVATIONS.md" "."
```

## Run standalone tests

```powershell
python -m unittest discover -s tests -p "test_ball_observation_store.py" -v
```

Expect **25 tests** and `OK`. The tests use synthetic observations, not your real videos.

## Correct integration points (after standalone tests pass)

The analyzer uses 1-based frame numbering. Outlier removal **mutates its input**; interpolation adds estimated ball detections and sets `interpolated=True`, `confidence=0.0`. Therefore three independent data objects are needed:

```python
from copy import deepcopy
from app.coaching.ball_observation_store import (
    build_ball_observation_report,
    save_report,
)

ball_result = BallAnalyzer(imgsz=640).analyze_video(str(video_path))
raw_ball_frames = deepcopy(ball_result["frames"])  # BEFORE outlier filter

ball_result["frames"] = remove_ball_outliers(
    ball_result["frames"], max_speed_factor=3.0,
)
filtered_ball_frames = deepcopy(ball_result["frames"])  # AFTER outliers, BEFORE interpolation

ball_result["frames"] = interpolate_ball_gaps(
    ball_result["frames"], max_gap=35,
)
# Existing ShotTracker, metrics, and rendering continue UNMODIFIED.

# Only after ShotTracker produces shot_result["shots"]:
ball_observations = build_ball_observation_report(
    raw_ball_frames,
    filtered_ball_frames,
    shot_result["shots"],
    fps=ball_result["fps"],
)
save_report(ball_observations, output_dir / "ball_observations.json")
```

**Do not** call the store using the interpolated `ball_result["frames"]`. It intentionally raises an error.

The store does not mutate either snapshot and does not filter or reclassify frames. It simply records which direct detector outputs survived the already-existing filter.

## Report structure

- `detector_observation_count`: frames detected *before* outlier removal
- `accepted_observation_count`: surviving direct detector observations
- `rejected_outlier_count` and `rejected_outlier_frames`: original detection frames removed by the existing filter
- `ball_positions`: frame, time, pixel coordinates, box, confidence, and explicit provenance for every accepted direct detection
- `shots[].ball_positions`: accepted observations inside each shot window
- `shots[].release_ball_position`: direct detection at the **exact** release frame or `null`
- `shots[].ball_positions_signal_available`: at least two directly observed frames on/before release; a **candidate signal only**
- `shots[].ball_position_signal_available`: exact release-frame direct detection; a **candidate signal only**
- `intended_form_match`: always `not_assessed`

Existing saved `hoop5_video3.json` and `optimized_video8_test.json` do not contain raw/pre-interpolation detector snapshots. They **cannot** be used to reconstruct this report retroactively. A new API run (Video 8 first) is required **after** wiring in the snapshots, but only after the standalone tests pass.

## Accuracy limits

A retained detector prediction can still be false; confidence is not calibrated probability of correct tracking. The existing speed-based outlier filter may reject a real fast-moving ball or retain a plausible false positive. These records do not establish ball possession, hand contact, release velocity, or whether shooting form is correct. Missing or interpolated positions are excluded from direct-observation counts.

The existing coaching references and technique validators are intentionally unchanged in this step. A later integration will need to explicitly read and validate this report; its presence does not automatically make ball-related coaching topics assessable.
