import math
import os

import cv2


WHITE = (255, 255, 255)
SKELETON_COLOR = (255, 0, 220)      # bright magenta -- won't be confused
                                     # with any shot trajectory color
JOINT_COLOR = (255, 255, 255)
ANGLE_LABEL_COLOR = (0, 255, 255)   # cyan, high contrast against magenta
SUMMARY_BOX_BORDER = (0, 255, 255)  # cyan border for persistent shot cards

# (a, b) keypoint pairs to draw as skeleton lines. Covers both arms/legs
# regardless of which side is shooting, so the full body is always visible.
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class ShootingAnalysisRenderer:

    # How long release and follow-through stats stay on screen after the
    # instant they're captured, instead of vanishing after a single frame.
    HOLD_SECONDS = 1.5

    def render(
        self,
        video_path,
        shot_result,
        metrics_result,
        output_path,
        rim_frames=None,
        pose_frames=None,
    ):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,                      # real fps of the input, never assumed
            (width, height),
        )

        shots = shot_result["shots"]
        metric_shots = metrics_result.get("shots", [])

        hold_frames = int(round(self.HOLD_SECONDS * fps))

        pixels_per_meter = metrics_result.get("pixels_per_meter")

        # Built once, up front -- not per frame -- since metrics for every
        # shot are already fully computed before rendering starts.
        shot_summaries = [
            self._build_shot_summary(
                shot,
                metric_shots[i] if i < len(metric_shots) else {},
                pixels_per_meter
            )
            for i, shot in enumerate(shots)
        ]

        # everything scales with video height so 432p and 937p look alike
        scale = min(max(height / 720.0, 0.6), 1.6)
        line_w = max(2, int(round(3 * scale)))
        dot_r = max(4, int(round(6 * scale)))

        frame_index = 1

        while True:
            success, frame = cap.read()
            if not success:
                break

            # ---- active phase (computed early so the skeleton draw below
            # knows which arm is shooting, to label its angles)
            shot_idx, phase = self._active_phase(shots, frame_index, hold_frames)
            active_arm = None
            if shot_idx is not None:
                active_arm = shots[shot_idx].get("arm")

            # ---- skeleton overlay
            if pose_frames and frame_index - 1 < len(pose_frames):
                pose_frame = pose_frames[frame_index - 1]
                if pose_frame.get("pose_detected"):
                    self._draw_skeleton(frame, pose_frame, scale, active_arm)

            # ---- optional rim debug box
            if rim_frames and frame_index - 1 < len(rim_frames):
                rim = rim_frames[frame_index - 1]
                if rim is not None:
                    cv2.rectangle(
                        frame,
                        (int(rim["x"] - rim["w"] / 2), int(rim["y"] - rim["h"] / 2)),
                        (int(rim["x"] + rim["w"] / 2), int(rim["y"] + rim["h"] / 2)),
                        (0, 255, 255),
                        max(1, line_w - 1),
                    )

            # ---- persistent trajectories (one colour per shot)
            for shot in shots:
                start = shot["start_frame"]
                if start is None or frame_index < start:
                    continue

                color = shot.get("color", (0, 255, 0))
                points = [p for p in shot["trajectory"] if p["frame"] <= frame_index]

                for a, b in zip(points, points[1:]):
                    start = (int(a["x"]), int(a["y"]))
                    end = (int(b["x"]), int(b["y"]))

                    estimated = (
                        a.get("interpolated", False)
                        or b.get("interpolated", False)
                    )

                    if estimated:
                        self._draw_dashed_line(
                            frame,
                            start,
                            end,
                            color,
                            line_w,
                        )
                    else:
                        cv2.line(
                            frame,
                            start,
                            end,
                            color,
                            line_w,
                            cv2.LINE_AA,
                        )

                end = shot["end_frame"]
                if points and (end is None or frame_index <= end):
                    cv2.circle(
                        frame,
                        (int(points[-1]["x"]), int(points[-1]["y"])),
                        dot_r,
                        color,
                        -1,
                    )

            lines = ["BULLSEYE"]

            if shot_idx is not None:
                shot = shots[shot_idx]
                arm = shot.get("arm", "?")
                lines.append(f"Shot {shot['shot_number']} ({arm}) | {phase}")

                if shot_idx < len(metric_shots):
                    m = metric_shots[shot_idx]
                    data = m.get("shooting_metrics", m)
                    lines += self._metric_lines(data, phase)

            self._draw_panel(frame, lines, scale, height)

            completed_summaries = [
                shot_summaries[i]
                for i, shot in enumerate(shots)
                if shot.get("end_frame") is not None
                and frame_index >= shot["end_frame"]
            ]

            if completed_summaries:
                self._draw_summary_boxes(frame, completed_summaries, scale, width)

            writer.write(frame)
            frame_index += 1

        cap.release()
        writer.release()

        return output_path

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_dashed_line(
        frame,
        start,
        end,
        color,
        thickness,
        dash_length=9,
        gap_length=6,
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        distance = math.hypot(dx, dy)

        if distance == 0:
            return

        ux = dx / distance
        uy = dy / distance

        position = 0.0

        while position < distance:
            segment_end = min(
                position + dash_length,
                distance,
            )

            p1 = (
                int(round(start[0] + ux * position)),
                int(round(start[1] + uy * position)),
            )

            p2 = (
                int(round(start[0] + ux * segment_end)),
                int(round(start[1] + uy * segment_end)),
            )

            cv2.line(
                frame,
                p1,
                p2,
                color,
                thickness,
                cv2.LINE_AA,
            )

            position += dash_length + gap_length
    @staticmethod
    def _draw_skeleton(frame, pose_frame, scale, active_arm=None):

        line_w = max(1, int(round(2 * scale)))
        joint_r = max(2, int(round(3 * scale)))

        for a_key, b_key in SKELETON_CONNECTIONS:
            a = pose_frame.get(a_key)
            b = pose_frame.get(b_key)

            if a is None or b is None:
                continue

            cv2.line(
                frame,
                (int(a["x"]), int(a["y"])),
                (int(b["x"]), int(b["y"])),
                SKELETON_COLOR,
                line_w,
                cv2.LINE_AA,
            )

        joint_keys = {
            key
            for pair in SKELETON_CONNECTIONS
            for key in pair
        }

        for key in joint_keys:
            point = pose_frame.get(key)
            if point is None:
                continue

            cv2.circle(
                frame,
                (int(point["x"]), int(point["y"])),
                joint_r,
                JOINT_COLOR,
                -1,
                cv2.LINE_AA,
            )

        if active_arm:
            ShootingAnalysisRenderer._draw_angle_labels(
                frame, pose_frame, active_arm, scale
            )

    @staticmethod
    def _draw_angle_labels(frame, pose_frame, arm, scale):
        """Draws the live elbow/hip/knee interior angle as text right next
        to each joint, using the identical shoulder-elbow-wrist /
        shoulder-hip-knee / hip-knee-ankle definitions used in
        shooting_metrics.py, so the on-screen numbers always match the
        skeleton the viewer is looking at."""

        shoulder = pose_frame.get(f"{arm}_shoulder")
        elbow = pose_frame.get(f"{arm}_elbow")
        wrist = pose_frame.get(f"{arm}_wrist")
        hip = pose_frame.get(f"{arm}_hip")
        knee = pose_frame.get(f"{arm}_knee")
        ankle = pose_frame.get(f"{arm}_ankle")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 * scale
        thickness = max(1, int(round(1 * scale)))
        offset = int(14 * scale)

        def label(point, angle):
            if point is None or angle is None:
                return
            text = f"{angle:.0f}"
            pos = (int(point["x"]) + offset, int(point["y"]) - offset)
            cv2.putText(
                frame, text, pos, font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, text, pos, font, font_scale,
                ANGLE_LABEL_COLOR, thickness, cv2.LINE_AA,
            )

        if shoulder is not None and elbow is not None and wrist is not None:
            label(elbow, ShootingAnalysisRenderer._angle(shoulder, elbow, wrist))

        if shoulder is not None and hip is not None and knee is not None:
            label(hip, ShootingAnalysisRenderer._angle(shoulder, hip, knee))

        if hip is not None and knee is not None and ankle is not None:
            label(knee, ShootingAnalysisRenderer._angle(hip, knee, ankle))

    @staticmethod
    def _angle(a, b, c):
        """Interior angle at vertex b, in degrees. Identical formula to
        ShootingMetrics._angle -- kept local here so the renderer doesn't
        need to import/instantiate ShootingMetrics just for this."""

        ba = (a["x"] - b["x"], a["y"] - b["y"])
        bc = (c["x"] - b["x"], c["y"] - b["y"])

        mag_ba = (ba[0] ** 2 + ba[1] ** 2) ** 0.5
        mag_bc = (bc[0] ** 2 + bc[1] ** 2) ** 0.5

        if mag_ba == 0 or mag_bc == 0:
            return None

        cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
        cosine = max(-1, min(1, cosine))

        return math.degrees(math.acos(cosine))

    @staticmethod
    def _active_phase(shots, frame_index, hold_frames=0):
        """Returns (shot_index, phase) for the given frame. RELEASE and
        FOLLOW-THROUGH are single-instant events (the exact release frame,
        the exact frame the flight ends) but their stats are useful to read,
        so each gets held on screen for `hold_frames` instead of appearing
        for one frame and vanishing. The follow-through hold also extends
        past the shot's own end_frame into the gap before the next shot,
        capped so it never overlaps that next shot's own display."""

        for idx, shot in enumerate(shots):
            s = shot.get("start_frame")
            r = shot.get("release_frame")
            e = shot.get("end_frame")

            if s is None:
                continue

            next_start = (
                shots[idx + 1].get("start_frame")
                if idx + 1 < len(shots)
                else None
            )

            release_end = None
            if r is not None:
                release_end = r + hold_frames
                if e is not None:
                    release_end = min(release_end, e)

            tail_end = None
            if e is not None:
                tail_end = e + hold_frames
                if next_start is not None:
                    tail_end = min(tail_end, next_start - 1)

            if r is not None and s <= frame_index < r:
                return idx, "SET"

            if release_end is not None and r <= frame_index <= release_end:
                return idx, "RELEASE"

            if (
                e is not None
                and release_end is not None
                and release_end < frame_index < e
            ):
                return idx, "FLIGHT"

            if e is not None and tail_end is not None and e <= frame_index <= tail_end:
                return idx, "FOLLOW-THROUGH"

        return None, None

    @staticmethod
    def _metric_lines(data, phase):
        def fmt(label, value, spec):
            return None if value is None else f"{label}: {value:{spec}}"

        out = []

        if phase == "SET":
            d = data.get("set", {})
            out += [fmt("Elbow", d.get("elbow_angle"), ".1f"),
                    fmt("Hip", d.get("hip_angle"), ".1f"),
                    fmt("Knee", d.get("knee_angle"), ".1f")]

        elif phase == "RELEASE":
            d = data.get("release", {})
            out += [fmt("Elbow", d.get("elbow_angle"), ".1f"),
                    fmt("Hip", d.get("hip_angle"), ".1f"),
                    fmt("Knee", d.get("knee_angle"), ".1f"),
                    fmt("Extension", d.get("arm_extension_ratio"), ".3f"),
                    fmt("Velocity", d.get("ball_velocity_meters_per_second"), ".2f")]
            if out and out[-1] is not None:
                out[-1] += " m/s"

        elif phase == "FLIGHT":
            d = data.get("trajectory", {})
            avg = fmt("Avg velocity", d.get("average_velocity_meters_per_second"), ".2f")
            peak = fmt("Peak velocity", d.get("peak_velocity_meters_per_second"), ".2f")
            out += [
                (avg + " m/s") if avg else None,
                (peak + " m/s") if peak else None,
            ]

        elif phase == "FOLLOW-THROUGH":
            d = data.get("follow_through", {})
            out += [fmt("Elbow", d.get("elbow_angle"), ".1f"),
                    fmt("Hip", d.get("hip_angle"), ".1f"),
                    fmt("Knee", d.get("knee_angle"), ".1f"),
                    fmt("Extension", d.get("arm_extension_ratio"), ".3f")]

        return [line for line in out if line]

    @staticmethod
    def _build_shot_summary(shot, metrics, pixels_per_meter=None):
        """Cache release metrics for an easy-to-read two-column shot card.

        The raw metric values and underlying shot tracking are untouched.
        Missing measurements remain absent rather than being invented.
        """
        release = metrics.get("release", {}) if metrics else {}
        return {
            "title": f"Shot {shot['shot_number']}  /  {str(shot.get('arm', '?')).title()}",
            "elbow": release.get("elbow_angle"),
            "hip": release.get("hip_angle"),
            "knee": release.get("knee_angle"),
            "speed": release.get("ball_velocity_meters_per_second"),
            "color": shot.get("color", (0, 255, 0)),
        }

    @staticmethod
    def _put_fitted_text(
        frame, text, x, baseline, available_width, font_size,
        color, min_font_size=0.32,
    ):
        """Draw light, anti-aliased text without overflowing its column."""
        if not text or available_width < 5:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        desired_width = cv2.getTextSize(text, font, font_size, 1)[0][0]
        if desired_width > available_width:
            font_size = max(
                min_font_size,
                font_size * available_width / desired_width,
            )
        # Stop rather than paint text into a neighboring column.
        if cv2.getTextSize(text, font, font_size, 1)[0][0] > available_width:
            return
        cv2.putText(
            frame, text, (int(x), int(baseline)), font,
            font_size, color, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_summary_boxes(frame, summaries, scale, width):
        """Persistent, roomy two-column release cards in the upper-right.

        432p cards are ~226px wide. They display angles in two short rows,
        with speed on a separate row only when that metric is available.
        """
        ui = max(1.0, scale)
        h = frame.shape[0]
        margin = max(10, int(round(12 * ui)))
        gap = max(7, int(round(8 * ui)))
        card_width = min(int(round(226 * ui)), width - 2 * margin)
        if card_width < 120:
            return

        title_fs = 0.48 * ui
        body_fs = 0.46 * ui

        y = max(10, int(round(12 * ui)))
        x1 = width - margin
        x0 = x1 - card_width

        for summary in summaries:
            has_speed = summary.get("speed") is not None
            card_height = int(round((111 if has_speed else 82) * ui))
            y1 = y + card_height

            # Avoid covering the lower-left panel / extending below frame.
            if y1 > h - margin:
                break

            roi = frame[y:y1, x0:x1]
            dark = roi.copy()
            dark[:] = (16, 19, 22)
            cv2.addWeighted(dark, 0.89, roi, 0.11, 0, roi)

            # Subtle gray edge, plus a short accent keyed to the shot path.
            cv2.rectangle(
                frame, (x0, y), (x1 - 1, y1 - 1),
                (91, 99, 105), 1, cv2.LINE_AA,
            )
            raw_color = summary.get("color", (0, 255, 0))
            if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
                accent = tuple(int(max(0, min(255, channel))) for channel in raw_color[:3])
            else:
                accent = (0, 190, 120)

            cv2.line(
                frame,
                (x0 + int(round(11 * ui)), y + int(round(8 * ui))),
                (x0 + int(round(11 * ui)), y + int(round(23 * ui))),
                accent, max(2, int(round(2 * ui))), cv2.LINE_AA,
            )

            pad = int(round(23 * ui))
            right_pad = int(round(12 * ui))
            col_gap = int(round(8 * ui))
            body_left = x0 + int(round(13 * ui))
            body_width = card_width - int(round(26 * ui))
            col_width = (body_width - col_gap) // 2
            col_right = body_left + col_width + col_gap

            ShootingAnalysisRenderer._put_fitted_text(
                frame,
                summary["title"],
                x0 + pad,
                y + int(round(21 * ui)),
                card_width - pad - right_pad,
                title_fs,
                (238, 240, 242),
            )

            divider_y = y + int(round(29 * ui))
            cv2.line(
                frame, (body_left, divider_y),
                (x1 - int(round(13 * ui)), divider_y),
                (65, 72, 77), 1, cv2.LINE_AA,
            )

            def angle_text(label, value):
                return f"{label} {value:.0f}" if value is not None else None

            first_row = y + int(round(49 * ui))
            second_row = y + int(round(72 * ui))
            body_color = (225, 230, 233)

            for label, value, xx, baseline in (
                ("Elbow", summary.get("elbow"), body_left, first_row),
                ("Hip", summary.get("hip"), col_right, first_row),
                ("Knee", summary.get("knee"), body_left, second_row),
            ):
                text = angle_text(label, value)
                if text is not None:
                    ShootingAnalysisRenderer._put_fitted_text(
                        frame, text, xx, baseline, col_width, body_fs,
                        body_color,
                    )

            if has_speed:
                ShootingAnalysisRenderer._put_fitted_text(
                    frame,
                    f"Speed {summary['speed']:.2f} m/s",
                    body_left,
                    y + int(round(96 * ui)),
                    body_width,
                    body_fs,
                    body_color,
                )

            y = y1 + gap

    @staticmethod
    def _draw_panel(frame, lines, scale, height):
        """More legible lower-left info panel with restrained text weight."""
        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        ui = max(1.0, scale)
        title_size = 0.49 * ui
        text_size = 0.46 * ui
        line_h = int(round(22 * ui))
        pad_x = int(round(12 * ui))
        pad_y = int(round(11 * ui))
        margin = int(round(12 * ui))
        width = frame.shape[1]

        max_w = min(int(round(350 * ui)), width - 2 * margin)
        available_text_w = max_w - 2 * pad_x

        sizes = []
        widest = 0
        for i, text in enumerate(lines):
            font_size = title_size if i == 0 else text_size
            tw = cv2.getTextSize(text, font, font_size, 1)[0][0]
            if tw > available_text_w:
                font_size *= available_text_w / tw
            sizes.append(font_size)
            widest = max(
                widest,
                cv2.getTextSize(text, font, font_size, 1)[0][0],
            )

        panel_w = min(max_w, widest + 2 * pad_x)
        panel_h = line_h * len(lines) + 2 * pad_y

        x0 = margin
        x1 = min(width - 1, x0 + panel_w)
        y1 = height - margin
        y0 = max(0, y1 - panel_h)
        if y0 >= y1 or x0 >= x1:
            return

        roi = frame[y0:y1, x0:x1]
        dark = roi.copy()
        dark[:] = (16, 19, 22)
        cv2.addWeighted(dark, 0.88, roi, 0.12, 0, roi)
        cv2.rectangle(
            frame, (x0, y0), (x1, y1),
            (88, 94, 99), 1, cv2.LINE_AA,
        )

        for i, (text, font_size) in enumerate(zip(lines, sizes)):
            baseline = y0 + pad_y + (i + 1) * line_h - int(round(5 * ui))
            color = (235, 238, 242) if i == 0 else (226, 231, 234)
            cv2.putText(
                frame, text, (x0 + pad_x, baseline),
                font, font_size, color, 1, cv2.LINE_AA,
            )
