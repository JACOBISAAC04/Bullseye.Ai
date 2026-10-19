"""Bullseye Phase 6.1/6.2 — evidence-limited, player-facing shot descriptions.

Consumes the *existing* Phase 5 measured report and video/analyze API JSON.
No detection, LLM, coaching rule, ideal-angle threshold, or correction runs here.
Replay seconds are derived from the caller's actual FPS and frame origin.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


SUPPORTED_SIGNALS = {
    "hip_angle", "knee_angle", "elbow_angle",
    "position_displacement", "same_frame_ball_to_wrist",
    "ball_path_endpoints", "phase_angles",
}
ALLOWED_CANDIDATE_STATUSES = {
    "measured_descriptive_review", "no_supported_measurement",
}
ANGLE_NAMES = {
    "hip_angle": "hip angle",
    "knee_angle": "knee angle",
    "elbow_angle": "elbow angle",
}
LIMITS = {
    "angles": "2D estimated angles depend on camera view and pose quality; no ideal-angle judgment.",
    "position": "Image-plane pixels do not measure physical balance, lean, or distance.",
    "ball": "Detector centers are not continuous, calibrated ball trajectories or proof of contact.",
    "timing": "These are observed endpoints; the data do not establish the full movement sequence.",
}


def _number(value, label, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return float(value)


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _frame(value, label, *, origin):
    if type(value) is not int or value < origin:
        raise ValueError(f"{label} must be an integer >= {origin}")
    return value


def _index_shots(data, label, *, require_boundaries=False):
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    shots = data.get("shots")
    if not isinstance(shots, list) or type(data.get("shot_count")) is not int:
        raise ValueError(f"{label} requires shots array and integer shot_count")
    if len(shots) != data["shot_count"]:
        raise ValueError(f"{label} shot_count mismatch")
    index = {}
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError(f"{label} shots must be objects")
        number = _positive_int(shot.get("shot_number"), f"{label} shot_number")
        if number in index:
            raise ValueError(f"{label} duplicate shot number {number}")
        if require_boundaries:
            for key in ("start_frame", "release_frame", "end_frame"):
                if type(shot.get(key)) is not int:
                    raise ValueError(f"{label} shot {number}: missing {key}")
        index[number] = shot
    if set(index) != set(range(1, len(shots) + 1)):
        raise ValueError(f"{label} shot numbering is inconsistent")
    return index


def _seconds(frame, fps, frame_origin):
    return round((frame - frame_origin) / fps, 3)


def _replay(start, end, fps, frame_origin):
    if start > end:
        raise ValueError("Invalid replay frame range")
    return {
        "start_frame": start,
        "end_frame": end,
        "start_seconds": _seconds(start, fps, frame_origin),
        "end_seconds": _seconds(end, fps, frame_origin),
    }


def _video_url(analysis):
    job_id = analysis.get("job_id")
    url = analysis.get("video_url")
    if job_id is None and url is None:
        return None
    try:
        canonical_id = str(UUID(job_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid analysis job_id") from None
    expected = f"/video/results/{canonical_id}/video"
    if url != expected:
        raise ValueError("Unexpected video_url for analysis job_id")
    return expected


def _source_urls(urls):
    if not isinstance(urls, list):
        raise ValueError("source_urls must be a list")
    out = []
    for u in urls:
        if not isinstance(u, str):
            raise ValueError("Invalid source URL")
        parsed = urlsplit(u)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(c in u for c in "\r\n\t")
        ):
            raise ValueError("Only valid public HTTPS source links are allowed")
        if u not in out:
            out.append(u)
    return out


def _direction(value, negative, positive):
    if abs(value) < 0.05:
        return "less than 0.1 px"
    return f"{abs(value):.1f} px {negative if value < 0 else positive}"


def _same(value, expected, label, tolerance=0.03):
    if abs(value - expected) > tolerance:
        raise ValueError(f"{label}: inconsistent endpoint difference")


def _inside(frame, shot_start, release, label, frame_origin):
    f = _frame(frame, label, origin=frame_origin)
    if not shot_start <= f <= release:
        raise ValueError(f"{label} outside shot preparation frames")
    return f


def _observation_view(obs, shot_start, release, fps, frame_origin):
    """Return (safe sentence, frame range, limitations), or None for unknown types.

    Every numeric statement is reconstructed from known, validated fields,
    never copied from a free-form source interpretation.
    """
    if not isinstance(obs, dict):
        raise ValueError("Observation must be an object")
    signal = obs.get("signal")
    if signal not in SUPPORTED_SIGNALS:
        return None
    if signal in ANGLE_NAMES:
        if obs.get("unit") != "degrees_2d":
            raise ValueError("Angle has unexpected units")
        a = _inside(obs.get("start_frame"), shot_start, release, "angle start", frame_origin)
        b = _inside(obs.get("end_frame"), shot_start, release, "angle end", frame_origin)
        if b < a:
            raise ValueError("Reversed angle frames")
        start = _number(obs.get("start_value"), "angle start", minimum=0, maximum=180)
        end = _number(obs.get("end_value"), "angle end", minimum=0, maximum=180)
        diff = _number(obs.get("observed_endpoint_difference"), "angle difference")
        _same(diff, end - start, "Angle")
        samples = _positive_int(obs.get("observed_samples"), "angle samples")
        if samples > (b - a + 1):
            raise ValueError("Angle sample count exceeds available frames")
        sentence = (
            f"Estimated {ANGLE_NAMES[signal]} changed from {start:.1f}° "
            f"(frame {a}) to {end:.1f}° (frame {b}); "
            f"an endpoint difference of {diff:+.1f}° across {samples} observed samples."
        )
        return sentence, _replay(a, b, fps, frame_origin), [
            LIMITS["angles"], LIMITS["timing"]
        ]

    if signal == "position_displacement":
        if obs.get("joint") != "hip_center" or obs.get("unit") != "uncalibrated_2d_pixels":
            raise ValueError("Unsupported position displacement or units")
        a = _inside(obs.get("start_frame"), shot_start, release, "position start", frame_origin)
        b = _inside(obs.get("end_frame"), shot_start, release, "position end", frame_origin)
        if b < a:
            raise ValueError("Reversed position frames")
        dx = _number(obs.get("delta_x_pixels"), "delta x")
        dy = _number(obs.get("delta_y_pixels"), "delta y")
        samples = _positive_int(obs.get("observed_samples"), "position samples")
        if samples > b - a + 1:
            raise ValueError("Position sample count exceeds available frames")
        sentence = (
            f"Hip-center landmark moved {_direction(dx, 'left', 'right')} and "
            f"{_direction(dy, 'up', 'down')} in the image "
            f"between frames {a} and {b} ({samples} observed samples)."
        )
        return sentence, _replay(a, b, fps, frame_origin), [LIMITS["position"]]

    if signal == "same_frame_ball_to_wrist":
        if obs.get("unit") != "uncalibrated_2d_pixels":
            raise ValueError("Ball-wrist units must be image pixels")
        a = _inside(obs.get("first_frame"), shot_start, release, "ball-wrist first", frame_origin)
        b = _inside(obs.get("last_frame"), shot_start, release, "ball-wrist last", frame_origin)
        if b < a:
            raise ValueError("Reversed ball-wrist frames")
        d1 = _number(obs.get("first_distance_pixels"), "first wrist distance", minimum=0)
        d2 = _number(obs.get("last_distance_pixels"), "last wrist distance", minimum=0)
        count = _positive_int(obs.get("same_frame_matches"), "matching ball-wrist frames")
        if count > b - a + 1:
            raise ValueError("Too many ball-wrist matches")
        sentence = (
            f"Detected ball center was {d1:.1f} image px from the wrist at frame {a} "
            f"and {d2:.1f} image px at frame {b} "
            f"({count} frames with both observations)."
        )
        return sentence, _replay(a, b, fps, frame_origin), [LIMITS["ball"]]

    if signal == "ball_path_endpoints":
        if obs.get("unit") != "uncalibrated_2d_pixels":
            raise ValueError("Ball path units must be image pixels")
        a = _inside(obs.get("start_frame"), shot_start, release, "ball first", frame_origin)
        b = _inside(obs.get("end_frame"), shot_start, release, "ball last", frame_origin)
        if b < a:
            raise ValueError("Reversed ball-path frames")
        first = obs.get("start_position")
        last = obs.get("end_position")
        if not isinstance(first, dict) or not isinstance(last, dict):
            raise ValueError("Missing ball endpoint")
        x1, y1 = _number(first.get("x"), "ball x1"), _number(first.get("y"), "ball y1")
        x2, y2 = _number(last.get("x"), "ball x2"), _number(last.get("y"), "ball y2")
        dx = _number(obs.get("delta_x_pixels"), "ball dx")
        dy = _number(obs.get("delta_y_pixels"), "ball dy")
        _same(dx, x2-x1, "Ball x", tolerance=0.05)
        _same(dy, y2-y1, "Ball y", tolerance=0.05)
        _number(obs.get("start_detector_confidence"), "start detector confidence", minimum=0, maximum=1)
        _number(obs.get("end_detector_confidence"), "end detector confidence", minimum=0, maximum=1)
        count = _positive_int(obs.get("observed_samples"), "ball observations")
        if count > b-a+1:
            raise ValueError("Ball observations exceed interval")
        sentence = (
            f"Accepted ball-detector endpoint moved {_direction(dx, 'left', 'right')} "
            f"and {_direction(dy, 'up', 'down')} in the image "
            f"from frame {a} to {b} ({count} observed detector frames)."
        )
        return sentence, _replay(a, b, fps, frame_origin), [LIMITS["ball"], LIMITS["timing"]]

    if signal == "phase_angles":
        if obs.get("unit") != "degrees_2d":
            raise ValueError("Phase angle units must be degrees_2d")
        items = obs.get("observations")
        if not isinstance(items, list) or len(items) != 2:
            raise ValueError("Expected set and release snapshots")
        if [x.get("phase") if isinstance(x, dict) else None for x in items] != ["set", "release"]:
            raise ValueError("Phase angles must be set then release")
        descriptions = []
        for item in items:
            angles = item.get("angles_degrees_2d")
            if not isinstance(angles, dict) or not angles:
                raise ValueError("Missing phase angles")
            if any(k not in ANGLE_NAMES for k in angles):
                raise ValueError("Unknown phase angle")
            parts = [
                f"{ANGLE_NAMES[k]} {(_number(v, k, minimum=0, maximum=180)):.1f}°"
                for k, v in sorted(angles.items())
            ]
            descriptions.append(f"{item['phase']} sample: {', '.join(parts)}")
        sentence = (
            "Estimated joint angles — " + "; ".join(descriptions) +
            ". The source did not provide individual frame IDs for these phase snapshots."
        )
        # The report has phase labels, not exact per-sample frames. Replay whole preparation.
        return sentence, _replay(shot_start, release, fps, frame_origin), [
            LIMITS["angles"], "Phase-snapshot frame IDs were not provided."
        ]
    return None


class PlayerFeedbackBuilder:
    """Generate display-ready, non-prescriptive feedback from trusted report shapes."""

    def generate(self, analysis: dict, measured: dict, *, fps: float, frame_origin: int = 1) -> dict:
        fps = _number(fps, "fps", minimum=0.001, maximum=1000)
        if type(frame_origin) is not int or frame_origin not in (0, 1):
            raise ValueError("frame_origin must be 0 or 1")
        shot_index = _index_shots(analysis, "analysis", require_boundaries=True)
        measured_index = _index_shots(measured, "measured")
        if set(shot_index) != set(measured_index):
            raise ValueError("Analysis and measured report shot numbers differ")
        if (
            measured.get("intended_form_match") != "not_assessed"
            or measured.get("corrections_generated") != 0
            or measured.get("official_endorsement") is not False
        ):
            raise ValueError("Unsafe measured coaching verdict or endorsement")
        if not isinstance(analysis.get("shots"), list):
            raise ValueError("Analysis missing shots")
        url = _video_url(analysis)

        shots = []
        for number in sorted(shot_index):
            src, m = shot_index[number], measured_index[number]
            start = _frame(src["start_frame"], "shot start", origin=frame_origin)
            release = _frame(src["release_frame"], "shot release", origin=frame_origin)
            end = _frame(src["end_frame"], "shot end", origin=frame_origin)
            if not start <= release <= end:
                raise ValueError("Invalid shot boundaries")
            if m.get("intended_form_match") != "not_assessed":
                raise ValueError("Unsafe shot-level technique verdict")
            candidates = m.get("review_candidates")
            if not isinstance(candidates, list):
                raise ValueError("Missing measured candidates")
            if type(m.get("review_candidate_count")) is not int or len(candidates) != m["review_candidate_count"]:
                raise ValueError("Measured review candidate count mismatch")
            quality = m.get("timing_quality")
            if not isinstance(quality, str) or not quality:
                raise ValueError("Missing timing quality")
            cards = []
            seen = set()
            for item in candidates:
                if not isinstance(item, dict):
                    raise ValueError("Review candidate must be an object")
                pid = item.get("principle_id")
                if not isinstance(pid, str) or not pid or pid in seen:
                    raise ValueError("Duplicate or invalid principle ID")
                seen.add(pid)
                if (
                    item.get("intended_form_match") != "not_assessed"
                    or item.get("source_support_status") != "limited_published_text_support"
                    or item.get("status") not in ALLOWED_CANDIDATE_STATUSES
                ):
                    raise ValueError("Candidate violates coaching safety gate")
                if item["status"] != "measured_descriptive_review":
                    continue
                if quality != "sufficient_for_descriptive_timing":
                    raise ValueError("Measured review cannot bypass timing-quality gate")
                observations = item.get("observations")
                if not isinstance(observations, list) or not observations:
                    raise ValueError("Measured candidate lacks observations")
                segments, notes, spans = [], [], []
                for observation in observations:
                    view = _observation_view(observation, start, release, fps, frame_origin)
                    if view is None:
                        continue  # Unknown signal cannot be displayed as measured evidence.
                    text, window, caveats = view
                    segments.append(text)
                    spans.append(window)
                    for note in caveats:
                        if note not in notes:
                            notes.append(note)
                if not segments:
                    continue
                title, question = item.get("title"), item.get("review_question")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError("Measured candidate missing title")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("Measured candidate missing review question")
                lo = min(s["start_frame"] for s in spans)
                hi = max(s["end_frame"] for s in spans)
                cards.append({
                    "principle_id": pid,
                    "title": title,
                    "observation_text": " ".join(segments),
                    "review_question": question,
                    "replay": _replay(lo, hi, fps, frame_origin),
                    "source_urls": _source_urls(item.get("source_urls", [])),
                    "limitations": notes,
                    "evidence_status": "observed_2d_descriptive",
                    "measurement_count": len(segments),
                })
            shot_replay = _replay(start, end, fps, frame_origin)
            shot_replay.update({
                "release_frame": release,
                "release_seconds": _seconds(release, fps, frame_origin),
                "video_url": url,
            })
            shots.append({
                "shot_number": number,
                "arm": src.get("arm") if src.get("arm") in ("left", "right") else None,
                "replay": shot_replay,
                "timing_quality": quality,
                "headline": (
                    f"{len(cards)} measured shooting-motion topics ready to review."
                    if cards else
                    "Not enough supported observations for a detailed shooting review."
                ),
                "summary": (
                    "These are camera-view measurements and coach review questions, not form corrections."
                    if cards else
                    "No technique judgment was made. A clearer or longer preparation view may allow a future review."
                ),
                "feedback_cards": cards,
                "feedback_count": len(cards),
                "intended_form_match": "not_assessed",
            })
        return {
            "schema_version": "1.0",
            "method": "frame_linked_player_facing_descriptive_feedback",
            "shot_count": len(shots),
            "fps": fps,
            "frame_origin": frame_origin,
            "video_url": url,
            "shots": shots,
            "intended_form_match": "not_assessed",
            "corrections_generated": 0,
            "official_endorsement": False,
            "limitations": [
                "This report describes measured evidence; it does not grade shooting form.",
                "Pixels and 2D angles are camera-dependent, not calibrated biomechanics.",
                "Only existing published-text-supported topics are displayed.",
                "Timestamp calculations assume the supplied FPS and frame origin are correct.",
                "Missing release ball observations are not reconstructed.",
            ],
        }


def _video_fps(path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required for --video; use --fps instead") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        return _number(cap.get(cv2.CAP_PROP_FPS), "video fps", minimum=0.001, maximum=1000)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Saved POST /video/analyze response")
    parser.add_argument("--measured", type=Path, required=True, help="Phase 5 measured-coaching report")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Read FPS from the original video (OpenCV)")
    source.add_argument("--fps", type=float, help="Explicit FPS of the original video")
    parser.add_argument("--frame-origin", type=int, choices=[0, 1], default=1,
                        help="Frame indexing; Bullseye's ShotTracker uses 1-based frames")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fps = _video_fps(args.video) if args.video is not None else args.fps
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    measured = json.loads(args.measured.read_text(encoding="utf-8"))
    report = PlayerFeedbackBuilder().generate(
        analysis, measured, fps=fps, frame_origin=args.frame_origin
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Shots: {report['shot_count']} | FPS: {report['fps']}")
    for shot in report["shots"]:
        print(
            f"Shot {shot['shot_number']}: {shot['feedback_count']} measured cards; "
            f"replay {shot['replay']['start_seconds']:.3f}–"
            f"{shot['replay']['end_seconds']:.3f}s"
        )
    print(f"Form verdict: {report['intended_form_match']}")
    print(f"Corrections generated: {report['corrections_generated']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
