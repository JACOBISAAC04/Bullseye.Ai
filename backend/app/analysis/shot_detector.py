import math


class ShotDetector:

    def detect_shots(self, pose_frames, ball_frames):
        if len(pose_frames) != len(ball_frames):
            raise ValueError("Pose and ball frame counts do not match")

        shooting_arm = self._detect_shooting_arm(pose_frames)

        wrist_key = f"{shooting_arm}_wrist"

        shots = []
        current_shot = None
        possession_start = None
        release_frame = None

        for i in range(len(ball_frames)):
            ball_frame = ball_frames[i]
            pose_frame = pose_frames[i]

            if not ball_frame["ball_detected"]:
                continue

            ball = ball_frame["ball"]

            if not pose_frame.get("pose_detected"):
                continue

            wrist = pose_frame[wrist_key]

            distance = self._distance(
                ball["x"],
                ball["y"],
                wrist["x"] * self._frame_width(ball_frames),
                wrist["y"] * self._frame_height(ball_frames)
            )

            near_hand = distance < 150

            if near_hand and possession_start is None:
                possession_start = i

            if (
                possession_start is not None
                and i > possession_start + 3
            ):
                if self._ball_moving_up(ball_frames, i):
                    release_frame = i

                    current_shot = {
                        "start_frame": possession_start + 1,
                        "release_frame": release_frame + 1,
                        "trajectory_frames": []
                    }

                    shots.append(current_shot)

                    possession_start = None

        self._assign_trajectory_frames(
            shots,
            ball_frames
        )

        return {
            "shooting_arm": shooting_arm,
            "shots": shots
        }

    def _detect_shooting_arm(self, frames):
        left = self._wrist_movement(
            frames,
            "left_wrist"
        )

        right = self._wrist_movement(
            frames,
            "right_wrist"
        )

        return "right" if right >= left else "left"

    def _wrist_movement(self, frames, key):
        total = 0

        for i in range(1, len(frames)):
            if (
                not frames[i].get("pose_detected")
                or not frames[i - 1].get("pose_detected")
            ):
                continue

            a = frames[i - 1][key]
            b = frames[i][key]

            total += math.sqrt(
                (b["x"] - a["x"]) ** 2 +
                (b["y"] - a["y"]) ** 2
            )

        return total

    def _ball_moving_up(self, frames, index):
        if index < 3:
            return False

        previous = frames[index - 3]

        if not previous["ball_detected"]:
            return False

        current_y = frames[index]["ball"]["y"]
        previous_y = previous["ball"]["y"]

        return current_y < previous_y - 10

    def _distance(self, x1, y1, x2, y2):
        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    def _frame_width(self, frames):
        return 1

    def _frame_height(self, frames):
        return 1

    def _assign_trajectory_frames(self, shots, ball_frames):
        for shot in shots:
            start = shot["start_frame"] - 1

            end = min(
                len(ball_frames),
                start + 120
            )

            for i in range(start, end):
                frame = ball_frames[i]

                if frame["ball_detected"]:
                    shot["trajectory_frames"].append(
                        i + 1
                    )