import math


class ShootingMetrics:

    # Standard basketball rim inner diameter, used to convert pixel
    # distances to real-world meters. If you know your camera's actual
    # pixels-per-meter some other way (e.g. a calibrated court marking),
    # swap out _compute_pixels_per_meter for that instead.
    RIM_DIAMETER_METERS = 0.4572  # 18 inches

    def analyze(
        self,
        shot_result,
        pose_result,
        ball_result,
        rim_frames=None,
        fps=30
    ):

        if fps is None or fps <= 0:
            fps = 30

        shots = shot_result["shots"]

        pose_frames = pose_result["frames"]
        ball_frames = ball_result["frames"]

        pixels_per_meter = self._compute_pixels_per_meter(rim_frames)

        results = []

        for shot in shots:

            # Each shot now carries its own arm (set at release time by
            # ShotTracker, since the player can shoot with either hand).
            # Fall back to "right" only in the unexpected case it's missing.
            arm = shot.get("arm") or "right"

            metrics = self._analyze_shot(
                shot,
                arm,
                pose_frames,
                ball_frames,
                fps,
                pixels_per_meter
            )

            results.append(metrics)

        return {
            "shot_count": len(results),
            "pixels_per_meter": pixels_per_meter,
            "shots": results
        }

    def _compute_pixels_per_meter(self, rim_frames):
        """Average the rim's detected pixel width across every frame it was
        found in, then scale against the real rim diameter. Returns None if
        no rim was ever detected, so callers can fall back gracefully
        instead of dividing by zero / garbage."""

        if not rim_frames:
            return None

        widths = [
            r["w"]
            for r in rim_frames
            if r is not None and r.get("w")
        ]

        if not widths:
            return None

        avg_width = sum(widths) / len(widths)

        if avg_width <= 0:
            return None

        return avg_width / self.RIM_DIAMETER_METERS

    def _analyze_shot(
        self,
        shot,
        arm,
        pose_frames,
        ball_frames,
        fps,
        pixels_per_meter
    ):

        start_frame = shot["start_frame"]
        release_frame = shot["release_frame"]
        end_frame = shot["end_frame"]

        set_metrics = self._calculate_set_metrics(
            start_frame,
            pose_frames,
            ball_frames,
            arm
        )

        release_ball, release_ball_frame, release_ball_distance = self._find_release_ball(
            release_frame,
            ball_frames,
            pose_frames,
            f"{arm}_wrist"
        )

        release_metrics = self._calculate_release_metrics(
            release_frame,
            release_ball_frame,
            release_ball,
            release_ball_distance,
            pose_frames,
            ball_frames,
            arm,
            fps,
            pixels_per_meter
        )

        follow_through = self._calculate_follow_through(
            end_frame,
            pose_frames,
            arm
        )

        trajectory_metrics = self._calculate_trajectory(
            shot["trajectory"],
            fps,
            pixels_per_meter
        )

        return {
            "shot_number": shot["shot_number"],
            "arm": arm,
            "start_frame": start_frame,
            "release_frame": release_frame,
            "end_frame": end_frame,
            "set": set_metrics,
            "release": release_metrics,
            "follow_through": follow_through,
            "trajectory": trajectory_metrics
        }

    def _side_angles(self, pose, side):
        """Elbow, hip, and knee angles for one side of the body (whichever
        arm is shooting). hip_angle is shoulder-hip-knee (torso/thigh
        hinge); knee_angle is hip-knee-ankle. Both now track the shooting
        side instead of being hardcoded to one leg/arm."""

        shoulder = pose.get(f"{side}_shoulder")
        elbow = pose.get(f"{side}_elbow")
        wrist = pose.get(f"{side}_wrist")
        hip = pose.get(f"{side}_hip")
        knee = pose.get(f"{side}_knee")
        ankle = pose.get(f"{side}_ankle")

        elbow_angle = None
        if shoulder is not None and elbow is not None and wrist is not None:
            elbow_angle = self._angle(shoulder, elbow, wrist)

        hip_angle = None
        if shoulder is not None and hip is not None and knee is not None:
            hip_angle = self._angle(shoulder, hip, knee)

        knee_angle = None
        if hip is not None and knee is not None and ankle is not None:
            knee_angle = self._angle(hip, knee, ankle)

        return elbow_angle, hip_angle, knee_angle

    def _calculate_set_metrics(
        self,
        frame_number,
        pose_frames,
        ball_frames,
        arm
    ):

        index = frame_number - 1

        pose = pose_frames[index]

        wrist = pose.get(f"{arm}_wrist")

        if wrist is None:
            return {
                "frame": frame_number
            }

        elbow_angle, hip_angle, knee_angle = self._side_angles(pose, arm)

        ball_position = self._get_ball_position(
            frame_number,
            ball_frames
        )

        body_alignment = self._body_alignment(
            pose
        )

        result = {
            "frame": frame_number,
            "elbow_angle": elbow_angle,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "wrist_position": wrist
        }

        if ball_position is not None:
            result["ball_position"] = ball_position

        result["body_alignment"] = body_alignment

        return result

    def _calculate_release_metrics(
        self,
        release_frame,
        ball_frame,
        ball,
        ball_distance,
        pose_frames,
        ball_frames,
        arm,
        fps,
        pixels_per_meter
    ):

        index = release_frame - 1

        pose = pose_frames[index]

        wrist = pose.get(f"{arm}_wrist")
        elbow = pose.get(f"{arm}_elbow")
        shoulder = pose.get(f"{arm}_shoulder")

        elbow_angle, hip_angle, knee_angle = self._side_angles(pose, arm)

        arm_extension_ratio = None

        if (
            wrist is not None
            and elbow is not None
            and shoulder is not None
        ):

            upper_arm = self._distance_points(shoulder, elbow)
            forearm = self._distance_points(elbow, wrist)
            shoulder_to_wrist = self._distance_points(shoulder, wrist)

            total_arm = upper_arm + forearm

            if total_arm > 0:
                arm_extension_ratio = shoulder_to_wrist / total_arm

        ball_position = None

        if ball is not None:

            ball_position = {
                "x": ball["x"],
                "y": ball["y"]
            }

        release_height = None
        release_height_relative_to_hip = None

        if ball is not None:

            release_height = ball["y"]

            hip = pose.get(f"{arm}_hip")

            if hip is not None:

                release_height_relative_to_hip = (
                    hip["y"] - ball["y"]
                )

        velocity = self._calculate_ball_velocity(
            ball_frame,
            ball_frames,
            fps,
            pixels_per_meter
        )

        return {
            "frame": release_frame,
            "ball_frame": ball_frame,
            "ball_to_wrist_distance_pixels": ball_distance,
            "elbow_angle": elbow_angle,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "wrist_position": wrist,
            "arm_extension_ratio": arm_extension_ratio,
            "ball_position": ball_position,
            "release_height_pixels_from_top": release_height,
            "release_height_relative_to_hip": release_height_relative_to_hip,
            "ball_velocity_pixels_per_second": velocity["pixels_per_second"],
            "ball_velocity_meters_per_second": velocity["meters_per_second"]
        }

    def _calculate_ball_velocity(
        self,
        ball_frame,
        ball_frames,
        fps,
        pixels_per_meter
    ):

        empty = {"pixels_per_second": None, "meters_per_second": None}

        if ball_frame is None:
            return empty

        selected_index = ball_frame - 1

        if (
            selected_index < 0
            or selected_index >= len(ball_frames)
        ):
            return empty

        selected = ball_frames[selected_index]

        if not selected.get("ball_detected"):
            return empty

        selected_ball = selected["ball"]

        previous = None
        previous_frame = None

        for offset in range(1, 11):

            index = selected_index - offset

            if index < 0:
                break

            candidate = ball_frames[index]

            if candidate.get("ball_detected"):

                previous = candidate["ball"]
                previous_frame = index + 1
                break

        px_per_sec = None

        if previous is not None:

            frame_difference = ball_frame - previous_frame

            if frame_difference > 0:

                distance = self._distance(
                    previous["x"],
                    previous["y"],
                    selected_ball["x"],
                    selected_ball["y"]
                )

                px_per_sec = (distance / frame_difference) * fps

        if px_per_sec is None:

            following = None
            following_frame = None

            for offset in range(1, 11):

                index = selected_index + offset

                if index >= len(ball_frames):
                    break

                candidate = ball_frames[index]

                if candidate.get("ball_detected"):

                    following = candidate["ball"]
                    following_frame = index + 1
                    break

            if following is not None:

                frame_difference = following_frame - ball_frame

                if frame_difference > 0:

                    distance = self._distance(
                        selected_ball["x"],
                        selected_ball["y"],
                        following["x"],
                        following["y"]
                    )

                    px_per_sec = (distance / frame_difference) * fps

        if px_per_sec is None:
            return empty

        m_per_sec = None

        if pixels_per_meter:
            m_per_sec = px_per_sec / pixels_per_meter

        return {
            "pixels_per_second": px_per_sec,
            "meters_per_second": m_per_sec
        }

    def _calculate_follow_through(
        self,
        frame_number,
        pose_frames,
        arm
    ):

        index = frame_number - 1

        if index < 0 or index >= len(pose_frames):
            return {
                "frame": frame_number
            }

        pose = pose_frames[index]

        wrist = pose.get(f"{arm}_wrist")
        elbow = pose.get(f"{arm}_elbow")
        shoulder = pose.get(f"{arm}_shoulder")

        elbow_angle, hip_angle, knee_angle = self._side_angles(pose, arm)

        arm_extension_ratio = None

        if (
            shoulder is not None
            and elbow is not None
            and wrist is not None
        ):

            upper_arm = self._distance_points(shoulder, elbow)
            forearm = self._distance_points(elbow, wrist)
            shoulder_to_wrist = self._distance_points(shoulder, wrist)

            total_arm = upper_arm + forearm

            if total_arm > 0:
                arm_extension_ratio = shoulder_to_wrist / total_arm

        balance = self._calculate_balance(
            pose
        )

        return {
            "frame": frame_number,
            "elbow_angle": elbow_angle,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "wrist_position": wrist,
            "arm_extension_ratio": arm_extension_ratio,
            "balance": balance
        }

    def _calculate_trajectory(
        self,
        trajectory,
        fps,
        pixels_per_meter
    ):

        if len(trajectory) < 2:
            return {
                "point_count": len(trajectory)
            }

        px_velocities = []

        for i in range(1, len(trajectory)):

            a = trajectory[i - 1]
            b = trajectory[i]

            distance = self._distance(
                a["x"],
                a["y"],
                b["x"],
                b["y"]
            )

            frame_difference = b["frame"] - a["frame"]

            if frame_difference > 0:

                velocity = (distance / frame_difference) * fps

                px_velocities.append(velocity)

        if not px_velocities:
            return {
                "point_count": len(trajectory)
            }

        def to_m(v):
            return v / pixels_per_meter if pixels_per_meter else None

        avg_px = sum(px_velocities) / len(px_velocities)
        peak_px = max(px_velocities)
        initial_px = px_velocities[0]

        return {
            "point_count": len(trajectory),
            "average_velocity_pixels_per_second": avg_px,
            "average_velocity_meters_per_second": to_m(avg_px),
            "peak_velocity_pixels_per_second": peak_px,
            "peak_velocity_meters_per_second": to_m(peak_px),
            "initial_velocity_pixels_per_second": initial_px,
            "initial_velocity_meters_per_second": to_m(initial_px)
        }

    def _find_release_ball(
        self,
        release_frame,
        ball_frames,
        pose_frames,
        wrist_key
    ):

        release_index = release_frame - 1

        candidates = []

        search_radius = 10

        for offset in range(
            -search_radius,
            search_radius + 1
        ):

            index = release_index + offset

            if index < 0 or index >= len(ball_frames):
                continue

            ball_frame = ball_frames[index]
            pose_frame = pose_frames[index]

            if not ball_frame.get("ball_detected"):
                continue

            if not pose_frame.get("pose_detected"):
                continue

            ball = ball_frame["ball"]

            wrist = pose_frame.get(wrist_key)

            if wrist is None:
                continue

            distance = self._distance(
                ball["x"],
                ball["y"],
                wrist["x"],
                wrist["y"]
            )

            temporal_distance = abs(offset)

            score = distance + temporal_distance * 3

            candidates.append(
                {
                    "frame": index + 1,
                    "ball": ball,
                    "distance": distance,
                    "score": score
                }
            )

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda x: x["score"])

        selected = candidates[0]

        return (
            selected["ball"],
            selected["frame"],
            selected["distance"]
        )

    def _get_ball_position(
        self,
        frame_number,
        ball_frames
    ):

        index = frame_number - 1

        if index < 0 or index >= len(ball_frames):
            return None

        frame = ball_frames[index]

        if not frame.get("ball_detected"):
            return None

        ball = frame["ball"]

        return {
            "x": ball["x"],
            "y": ball["y"]
        }

    def _body_alignment(
        self,
        pose
    ):

        left_shoulder = pose.get("left_shoulder")
        right_shoulder = pose.get("right_shoulder")
        left_hip = pose.get("left_hip")
        right_hip = pose.get("right_hip")

        if (
            left_shoulder is None
            or right_shoulder is None
            or left_hip is None
            or right_hip is None
        ):
            return {}

        shoulder_center = {
            "x": (left_shoulder["x"] + right_shoulder["x"]) / 2,
            "y": (left_shoulder["y"] + right_shoulder["y"]) / 2
        }

        hip_center = {
            "x": (left_hip["x"] + right_hip["x"]) / 2,
            "y": (left_hip["y"] + right_hip["y"]) / 2
        }

        dx = shoulder_center["x"] - hip_center["x"]
        dy = shoulder_center["y"] - hip_center["y"]

        torso_lean_angle = math.degrees(
            math.atan2(abs(dx), abs(dy))
        )

        return {
            "torso_lean_angle": torso_lean_angle,
            "shoulder_center": shoulder_center,
            "hip_center": hip_center
        }

    def _calculate_balance(
        self,
        pose
    ):

        left_ankle = pose.get("left_ankle")
        right_ankle = pose.get("right_ankle")
        left_hip = pose.get("left_hip")
        right_hip = pose.get("right_hip")

        if (
            left_ankle is None
            or right_ankle is None
            or left_hip is None
            or right_hip is None
        ):
            return {}

        ankle_center_x = (left_ankle["x"] + right_ankle["x"]) / 2
        hip_center_x = (left_hip["x"] + right_hip["x"]) / 2

        horizontal_offset = abs(hip_center_x - ankle_center_x)

        ankle_distance = self._distance(
            left_ankle["x"],
            left_ankle["y"],
            right_ankle["x"],
            right_ankle["y"]
        )

        normalized_offset = 0

        if ankle_distance > 0:
            normalized_offset = horizontal_offset / ankle_distance

        return {
            "horizontal_offset_pixels": horizontal_offset,
            "normalized_offset": normalized_offset
        }

    def _angle(
        self,
        a,
        b,
        c
    ):

        ba = (a["x"] - b["x"], a["y"] - b["y"])
        bc = (c["x"] - b["x"], c["y"] - b["y"])

        magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0

        cosine = (
            (ba[0] * bc[0] + ba[1] * bc[1])
            / (magnitude_ba * magnitude_bc)
        )

        cosine = max(-1, min(1, cosine))

        return math.degrees(math.acos(cosine))

    def _distance_points(
        self,
        a,
        b
    ):

        return self._distance(
            a["x"],
            a["y"],
            b["x"],
            b["y"]
        )

    def _distance(
        self,
        x1,
        y1,
        x2,
        y2
    ):

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )