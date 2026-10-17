import math


class ShootingPhaseDetector:

    def detect(self, frames):
        valid_frames = [
            frame for frame in frames
            if frame.get("pose_detected")
        ]

        if len(valid_frames) < 20:
            raise ValueError("Not enough pose data")

        shooting_arm = self._detect_shooting_arm(valid_frames)

        wrist_key = f"{shooting_arm}_wrist"
        elbow_key = f"{shooting_arm}_elbow_angle"
        knee_key = f"{shooting_arm}_knee_angle"

        release_index = self._find_release_index(
            valid_frames,
            wrist_key
        )

        window = max(8, len(valid_frames) // 20)

        set_index = max(
            0,
            release_index - window
        )

        follow_index = min(
            len(valid_frames) - 1,
            release_index + window
        )

        set_frame = valid_frames[set_index]
        release_frame = valid_frames[release_index]
        follow_frame = valid_frames[follow_index]

        return {
            "shooting_arm": shooting_arm,
            "set": self._build_phase_data(
                set_frame,
                elbow_key,
                knee_key,
                wrist_key
            ),
            "release": self._build_phase_data(
                release_frame,
                elbow_key,
                knee_key,
                wrist_key
            ),
            "follow_through": self._build_phase_data(
                follow_frame,
                elbow_key,
                knee_key,
                wrist_key
            )
        }

    def _detect_shooting_arm(self, frames):
        left_movement = self._wrist_movement(
            frames,
            "left_wrist"
        )

        right_movement = self._wrist_movement(
            frames,
            "right_wrist"
        )

        if right_movement >= left_movement:
            return "right"

        return "left"

    def _wrist_movement(self, frames, wrist_key):
        total = 0.0

        for i in range(1, len(frames)):
            previous = frames[i - 1][wrist_key]
            current = frames[i][wrist_key]

            dx = current["x"] - previous["x"]
            dy = current["y"] - previous["y"]

            total += math.sqrt(
                dx * dx + dy * dy
            )

        return total

    def _find_release_index(self, frames, wrist_key):
        best_index = 0
        best_score = float("-inf")

        for i in range(2, len(frames) - 2):
            previous = frames[i - 2][wrist_key]
            current = frames[i][wrist_key]
            next_position = frames[i + 2][wrist_key]

            upward_before = previous["y"] - current["y"]
            upward_after = current["y"] - next_position["y"]

            movement = abs(
                current["y"] - previous["y"]
            )

            score = (
                upward_before * 2
                + upward_after
                + movement
            )

            if score > best_score:
                best_score = score
                best_index = i

        return best_index

    def _build_phase_data(
        self,
        frame,
        elbow_key,
        knee_key,
        wrist_key
    ):
        return {
            "frame": frame["frame"],
            "time": frame["time"],
            "elbow_angle": frame[elbow_key],
            "knee_angle": frame[knee_key],
            "wrist": frame[wrist_key]
        }