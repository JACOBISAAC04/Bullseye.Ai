"""Evidence-only 2D basketball trajectory estimation.

No calibrated physical launch angle, made/missed outcome, or
biomechanical correctness is inferred.
"""

import math


def _finite(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
    )


def _observed_points(frames, release_frame, end_frame):
    if not isinstance(frames, list):
        raise ValueError("Ball frames must be a list")

    points = []
    seen = set()

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid ball frame")

        number = frame.get("frame")
        if type(number) is not int or number < 1:
            raise ValueError("Invalid ball frame number")
        if number in seen:
            raise ValueError("Duplicate ball frame number")
        seen.add(number)

        if not release_frame <= number <= end_frame:
            continue

        if frame.get("ball_detected") is not True:
            continue
        if frame.get("interpolated") is True:
            continue

        ball = frame.get("ball")
        if not isinstance(ball, dict):
            continue
        if ball.get("interpolated") is True:
            continue

        x = ball.get("x")
        y = ball.get("y")
        confidence = ball.get("confidence")

        if not all(_finite(v) for v in (x, y, confidence)):
            continue
        if not 0 < confidence <= 1:
            continue

        points.append({
            "frame": number,
            "x": float(x),
            "y": float(y),
            "confidence": float(confidence),
        })

    return sorted(points, key=lambda p: p["frame"])


def _solve_3x3(matrix, vector):
    a = [
        [float(*row[:1]), *map(float, row[1:]), float(value)]
        for row, value in zip(matrix, vector)
    ]

    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-10:
            return None

        a[col], a[pivot] = a[pivot], a[col]

        scale = a[col][col]
        for j in range(col, 4):
            a[col][j] /= scale

        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 4):
                a[row][j] -= factor * a[col][j]

    return [a[i][3] for i in range(3)]


def _quadratic_fit(points):
    # Fit vertical pixel position against time/frame offset.
    origin = points[0]["frame"]
    times = [p["frame"] - origin for p in points]
    ys = [p["y"] for p in points]

    n = len(times)
    s1 = sum(times)
    s2 = sum(t * t for t in times)
    s3 = sum(t ** 3 for t in times)
    s4 = sum(t ** 4 for t in times)

    b0 = sum(ys)
    b1 = sum(t * y for t, y in zip(times, ys))
    b2 = sum(t * t * y for t, y in zip(times, ys))

    coeff = _solve_3x3(
        [
            [n, s1, s2],
            [s1, s2, s3],
            [s2, s3, s4],
        ],
        [b0, b1, b2],
    )

    if coeff is None:
        return None

    c, b, a = coeff
    if not all(math.isfinite(v) for v in (a, b, c)):
        return None

    residuals = [
        y - (a * t * t + b * t + c)
        for t, y in zip(times, ys)
    ]
    rmse = math.sqrt(sum(r * r for r in residuals) / n)

    return {
        "a": round(a, 6),
        "b": round(b, 6),
        "c": round(c, 6),
        "rmse_pixels": round(rmse, 4),
        "origin_frame": origin,
    }


def analyze_observed_arc(
    ball_frames,
    *,
    release_frame,
    end_frame,
    max_release_gap=2,
):
    if (
        type(release_frame) is not int
        or type(end_frame) is not int
        or release_frame < 1
        or end_frame < release_frame
    ):
        raise ValueError("Invalid release/end frames")

    if type(max_release_gap) is not int or max_release_gap < 1:
        raise ValueError("Invalid release gap")

    points = _observed_points(
        ball_frames, release_frame, end_frame
    )

    result = {
        "status": "insufficient_evidence",
        "coordinate_system": "2d_image_pixels",
        "physical_angle_calibrated": False,
        "release_angle_2d_degrees": None,
        "trajectory_fit": None,
        "observed_points": points,
        "reason": None,
    }

    if len(points) < 3:
        result["reason"] = "At least three direct ball observations required"
        return result

    # A release-direction estimate needs nearby consecutive
    # observations; a long gap cannot establish initial motion.
    first = points[0]
    second = points[1]

    if (
        first["frame"] != release_frame
        or second["frame"] - first["frame"] > max_release_gap
    ):
        result["reason"] = "No sufficiently close release observations"
        return result

    dx = second["x"] - first["x"]
    dy = first["y"] - second["y"]  # Image Y increases downward.

    if math.hypot(dx, dy) < 1e-8:
        result["reason"] = "No measurable initial ball movement"
        return result

    angle = math.degrees(math.atan2(dy, abs(dx)))

    # This is a descriptive 2D direction, not a physical angle.
    result["release_angle_2d_degrees"] = round(angle, 3)

    fit = _quadratic_fit(points)
    if fit is None:
        result["reason"] = "Trajectory fit is underdetermined"
        return result

    result["trajectory_fit"] = fit
    result["status"] = "observed_descriptive_only"
    result["reason"] = (
        "Image-space direction and quadratic fit only; "
        "perspective, camera motion, and false detections may affect results"
    )
    return result