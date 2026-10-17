import cv2


class TrajectoryRenderer:

    def render(
        self,
        video_path,
        shots,
        output_path
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise ValueError(
                "Could not open video"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (width, height)
        )

        frame_index = 1

        while True:

            success, frame = cap.read()

            if not success:
                break

            for shot in shots:

                start_frame = shot[
                    "start_frame"
                ]

                if start_frame is None:
                    continue

                if frame_index < start_frame:
                    continue

                color = shot["color"]

                visible_points = []

                for point in shot[
                    "trajectory"
                ]:

                    if point["frame"] <= frame_index:

                        visible_points.append(
                            point
                        )

                for i in range(
                    1,
                    len(visible_points)
                ):

                    previous = (
                        visible_points[i - 1]
                    )

                    current = (
                        visible_points[i]
                    )

                    cv2.line(
                        frame,
                        (
                            int(previous["x"]),
                            int(previous["y"])
                        ),
                        (
                            int(current["x"]),
                            int(current["y"])
                        ),
                        color,
                        5
                    )

                if visible_points:

                    latest = (
                        visible_points[-1]
                    )

                    cv2.circle(
                        frame,
                        (
                            int(latest["x"]),
                            int(latest["y"])
                        ),
                        8,
                        color,
                        -1
                    )

            writer.write(frame)

            frame_index += 1

        cap.release()
        writer.release()

        return output_path