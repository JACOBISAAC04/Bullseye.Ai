import math


def remove_ball_outliers(frames, max_speed_factor=3.0):
    """Reject detections that would require the ball to move faster than a
    plausible max speed from the last ACCEPTED detection, scaled by elapsed
    frames."""
    last_valid = None  # (frame_index, x, y, d)

    for i, f in enumerate(frames):
        if not f.get("ball_detected"):
            continue

        ball = f["ball"]
        x, y = ball["x"], ball["y"]
        d = max(ball["x2"] - ball["x1"], 1.0)

        if last_valid is not None:
            i0, x0, y0, d0 = last_valid
            elapsed = max(i - i0, 1)
            dist = math.hypot(x - x0, y - y0)
            max_allowed = max_speed_factor * d0 * elapsed

            if dist > max_allowed:
                f["ball_detected"] = False
                f["ball"] = None
                continue

        last_valid = (i, x, y, d)

    return frames


def interpolate_ball_gaps(frames, max_gap=35):
    """Fill short runs of missing ball detections via linear interpolation
    between the last known and next known position."""
    n = len(frames)
    i = 0
    while i < n:
        if frames[i]["ball_detected"]:
            i += 1
            continue

        gap_start = i
        j = i
        while j < n and not frames[j]["ball_detected"]:
            j += 1

        gap_len = j - gap_start
        has_before = gap_start > 0 and frames[gap_start - 1]["ball_detected"]
        has_after = j < n and frames[j]["ball_detected"]

        if has_before and has_after and gap_len <= max_gap:
            b0 = frames[gap_start - 1]["ball"]
            b1 = frames[j]["ball"]
            w = b0["x2"] - b0["x1"]
            h = b0["y2"] - b0["y1"]
            for k in range(gap_start, j):
                t = (k - gap_start + 1) / (gap_len + 1)
                x = b0["x"] + (b1["x"] - b0["x"]) * t
                y = b0["y"] + (b1["y"] - b0["y"]) * t
                frames[k]["ball"] = {
                    "x": x, "y": y,
                    "x1": x - w / 2, "y1": y - h / 2,
                    "x2": x + w / 2, "y2": y + h / 2,
                    "confidence": 0.0,
                }
                frames[k]["ball_detected"] = True
                frames[k]["interpolated"] = True

        i = j

    return frames


class ShotTracker:

    COLORS = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    HAND_NEAR = 2.5
    RELEASE_DIST = 1.5
    LOST_POSSESSION = 4.0
    UP_STEP = 0.15
    UP_TOTAL = 0.30
    FALL_LIMIT = 4.0

    RIM_X_TOLERANCE = 1.0
    RIM_Y_TOLERANCE = 1.5
    HEADING_RATIO = 0.7

    MIN_POSSESSION_FRAMES = 3
    MIN_FLIGHT_FRAMES = 3
    FLIGHT_TIMEOUT_SECONDS = 3.0

    # A FALL RESET (ball went up then came back down without heading into
    # the rim window) only counts as a real shot attempt if the flight
    # lasted at least this many frames past release. This is stricter than
    # MIN_FLIGHT_FRAMES on purpose: MIN_FLIGHT_FRAMES just gates whether we
    # bother checking rim-heading at all, but a rebound tap or a dribble
    # pop-up can easily clear that low bar. Requiring a longer arc here
    # keeps short, non-shot upward bounces from being recorded as shots.
    MIN_SHOT_FLIGHT_FRAMES = 10

    def _closest_hand(self, pose_frame, x, y):
        """Return (hand_dist, arm_label) using whichever wrist -- left or
        right -- is currently closer to the ball. This makes the tracker
        correct regardless of which hand the player is shooting with on any
        given possession, instead of assuming one arm for the whole video."""
        if not pose_frame.get("pose_detected"):
            return None, None

        best_dist = None
        best_arm = None

        for arm in ("left", "right"):
            wrist = pose_frame.get(f"{arm}_wrist")
            if wrist is None:
                continue
            dist = self._distance(x, y, wrist["x"], wrist["y"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_arm = arm

        return best_dist, best_arm

    def detect(self, pose_frames, ball_frames, rim_frames, scene_cuts, fps=30):

        if fps is None or fps <= 0:
            fps = 30

        timeout_frames = int(self.FLIGHT_TIMEOUT_SECONDS * fps)

        shots = []
        state = "WAITING"
        current = None
        close_points = []
        previous_ball = None
        last_rim = None
        release_index = None
        apex_y = None

        waiting_count = 0
        possession_count = 0
        flight_count = 0
        release_count = 0
        near_rim_count = 0
        heading_pass_count = 0
        flight_timeout_count = 0
        flight_fall_count = 0
        flight_fall_kept_count = 0
        flight_fall_discarded_count = 0
        flight_abort_count = 0

        total = min(len(ball_frames), len(pose_frames))

        for i in range(total):

            if i < len(scene_cuts) and scene_cuts[i]:
                state = "WAITING"
                current = None
                close_points = []
                previous_ball = None
                last_rim = None

            rim = rim_frames[i] if i < len(rim_frames) else None

            if rim is not None:
                last_rim = rim
            else:
                rim = last_rim

            if state == "FLIGHT" and (i - release_index) > timeout_frames:
                flight_timeout_count += 1
                state = "WAITING"
                current = None
                close_points = []
                previous_ball = None

            ball_frame = ball_frames[i]

            if not ball_frame.get("ball_detected"):
                continue

            ball = ball_frame["ball"]
            x, y = ball["x"], ball["y"]

            d = max(ball["x2"] - ball["x1"], 1.0)

            hand_dist, closest_arm = self._closest_hand(pose_frames[i], x, y)

            point = {
                "frame": i + 1,
                "x": x,
                "y": y
            }

            if state == "WAITING":

                waiting_count += 1

                if hand_dist is not None and hand_dist < self.HAND_NEAR * d:
                    close_points.append(point)
                else:
                    close_points = []

                if len(close_points) >= self.MIN_POSSESSION_FRAMES:

                    current = {
                        "shot_number": len(shots) + 1,
                        "start_frame": close_points[0]["frame"],
                        "release_frame": None,
                        "end_frame": None,
                        "arm": None,
                        "color": self.COLORS[len(shots) % len(self.COLORS)],
                        "trajectory": list(close_points),
                    }

                    state = "POSSESSION"

            elif state == "POSSESSION":

                possession_count += 1

                current["trajectory"].append(point)

                if self._detect_release(
                    ball_frames,
                    i,
                    d,
                    hand_dist,
                    previous_ball
                ):

                    release_count += 1

                    current["release_frame"] = i + 1
                    current["arm"] = closest_arm
                    release_index = i
                    apex_y = y
                    state = "FLIGHT"

                    print(
                        f"RELEASE at frame {i + 1}, arm={closest_arm}, "
                        f"ball=({round(x, 1)}, {round(y, 1)}), d={round(d, 1)}"
                    )

                elif (
                    hand_dist is not None
                    and hand_dist > self.LOST_POSSESSION * d
                ):

                    state = "WAITING"
                    current = None
                    close_points = []

            elif state == "FLIGHT":

                flight_count += 1

                current["trajectory"].append(point)

                flight_frames = i - release_index

                if (
                    hand_dist is not None
                    and hand_dist < self.RELEASE_DIST * d
                    and flight_frames >= self.MIN_FLIGHT_FRAMES
                ):
                    flight_abort_count += 1

                    print(
                        f"FLIGHT ABORTED at frame {i + 1}: "
                        f"ball returned to hand (hand_dist={round(hand_dist, 1)}, "
                        f"threshold={round(self.RELEASE_DIST * d, 1)})"
                    )

                    state = "WAITING"
                    current = None
                    close_points = []
                    continue

                apex_y = min(apex_y, y)

                near_rim = self._near_rim(
                    x,
                    y,
                    rim
                )

                if near_rim:
                    near_rim_count += 1

                heading_to_rim = False

                if (
                    near_rim
                    and flight_frames >= self.MIN_FLIGHT_FRAMES
                ):

                    heading_to_rim = self._heading_to_rim(
                        current,
                        rim
                    )

                    if heading_to_rim:
                        heading_pass_count += 1

                if (
                    near_rim
                    and flight_frames >= self.MIN_FLIGHT_FRAMES
                    and heading_to_rim
                ):

                    current["end_frame"] = i + 1

                    shots.append(current)

                    state = "WAITING"
                    current = None
                    close_points = []

                elif (
                    not near_rim
                    and y > apex_y + self.FALL_LIMIT * d
                ):

                    flight_fall_count += 1

                    print(
                        f"FALL RESET at frame {i + 1}: "
                        f"y={round(y, 1)} apex_y={round(apex_y, 1)} "
                        f"threshold={round(apex_y + self.FALL_LIMIT * d, 1)} "
                        f"frames_since_release={i - release_index}"
                    )

                    if flight_frames >= self.MIN_SHOT_FLIGHT_FRAMES:

                        flight_fall_kept_count += 1

                        current["end_frame"] = i + 1
                        shots.append(current)

                        print(
                            f"  -> kept as Shot {current['shot_number']} "
                            f"(flight_frames={flight_frames})"
                        )

                    else:

                        flight_fall_discarded_count += 1

                        print(
                            f"  -> discarded as noise "
                            f"(flight_frames={flight_frames} < "
                            f"{self.MIN_SHOT_FLIGHT_FRAMES})"
                        )

                    state = "WAITING"
                    current = None
                    close_points = []

            previous_ball = (x, y)

        print("WAITING frames:", waiting_count)
        print("POSSESSION frames:", possession_count)
        print("FLIGHT frames:", flight_count)
        print("RELEASES:", release_count)
        print("NEAR RIM:", near_rim_count)
        print("HEADING TO RIM:", heading_pass_count)
        print("FLIGHT FALL RESETS:", flight_fall_count)
        print("  kept as shots:", flight_fall_kept_count)
        print("  discarded as noise:", flight_fall_discarded_count)
        print("FLIGHT TIMEOUT RESETS:", flight_timeout_count)
        print("FLIGHT ABORTS (returned to hand):", flight_abort_count)

        return {
            "shots": shots,
        }

    def _detect_release(
        self,
        ball_frames,
        index,
        d,
        hand_dist,
        previous_ball
    ):

        if index < 2:
            return False

        if hand_dist is None or previous_ball is None:
            return False

        if hand_dist < self.RELEASE_DIST * d:
            return False

        current = ball_frames[index]

        if not current.get("ball_detected"):
            return False

        y = current["ball"]["y"]

        if (previous_ball[1] - y) < self.UP_STEP * d:
            return False

        old = ball_frames[index - 2]

        if not old.get("ball_detected"):
            return False

        return (
            old["ball"]["y"] - y
        ) > self.UP_TOTAL * d

    def _near_rim(self, x, y, rim):

        if rim is None:
            return False

        return (
            abs(x - rim["x"])
            <= rim["w"] * self.RIM_X_TOLERANCE
            and
            abs(y - rim["y"])
            <= rim["h"] * self.RIM_Y_TOLERANCE
        )

    def _heading_to_rim(self, shot, rim):

        if rim is None:
            return False

        release_frame = shot["release_frame"]

        flight = [
            p
            for p in shot["trajectory"]
            if p["frame"] >= release_frame
        ]

        if len(flight) < 3:
            return False

        d0 = self._distance(
            flight[0]["x"],
            flight[0]["y"],
            rim["x"],
            rim["y"]
        )

        d1 = self._distance(
            flight[-1]["x"],
            flight[-1]["y"],
            rim["x"],
            rim["y"]
        )

        return d1 < self.HEADING_RATIO * d0

    @staticmethod
    def _distance(x1, y1, x2, y2):

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )