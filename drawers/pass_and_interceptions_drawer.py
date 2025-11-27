import cv2
import numpy as np

class PassInterceptionDrawer:
    """
    Visual overlay for Pass & Interception statistics.
    Draws a semi-transparent box with cumulative stats on the video frames.

    Event codes expected in input lists:
      -1 or 0 = no event
       1 = Team 1 event
       2 = Team 2 event
    """

    def __init__(self, overlay_alpha: float = 0.8):
        self.overlay_alpha = float(overlay_alpha)

    def draw(self, video_frames, passes, interceptions):
        """
        Draw overlay on each frame and return a new list of annotated frames.

        Args:
            video_frames (list[np.ndarray]): The raw video frames.
            passes (list[int]): Per-frame pass events (1=Team1, 2=Team2).
            interceptions (list[int]): Per-frame interception events.

        Returns:
            list[np.ndarray]: frames with the pass/interception stats overlay.
        """
        if not video_frames:
            return []

        num_frames = len(video_frames)

        # Defensive: normalize inputs
        passes = passes if passes is not None else []
        interceptions = interceptions if interceptions is not None else []

        if len(passes) < num_frames:
            passes = list(passes) + [-1] * (num_frames - len(passes))
        if len(interceptions) < num_frames:
            interceptions = list(interceptions) + [-1] * (num_frames - len(interceptions))

        output_frames = []

        # Cumulative stats
        t1_passes = t2_passes = 0
        t1_intercepts = t2_intercepts = 0

        for i in range(num_frames):
            frame = video_frames[i]

            # Handle None / corrupted frame
            if frame is None or not isinstance(frame, np.ndarray):
                if output_frames:
                    h, w = output_frames[-1].shape[:2]
                else:
                    h, w = 720, 1280
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = frame.copy()

            # Update counts
            p_val = passes[i]
            i_val = interceptions[i]

            if p_val == 1:
                t1_passes += 1
            elif p_val == 2:
                t2_passes += 1

            if i_val == 1:
                t1_intercepts += 1
            elif i_val == 2:
                t2_intercepts += 1

            # Draw overlay
            frame = self._draw_overlay(frame, t1_passes, t2_passes, t1_intercepts, t2_intercepts)
            output_frames.append(frame)

        return output_frames

    def _draw_overlay(self, frame, p1, p2, i1, i2):
        h, w = frame.shape[:2]

        # Adaptive font based on resolution
        base_scale = max(0.6, min(w, h) / 1000.0 * 0.8)
        font_thickness = max(2, int(round(base_scale * 2)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Box position (bottom-left)
        rect_x1 = int(w * 0.05)
        rect_y1 = int(h * 0.78)
        rect_x2 = int(w * 0.45)
        rect_y2 = int(h * 0.95)

        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        box_h = rect_y2 - rect_y1
        text_x = rect_x1 + 20
        text_y1 = rect_y1 + int(box_h * 0.4)
        text_y2 = rect_y1 + int(box_h * 0.8)

        color = (0, 0, 0)

        cv2.putText(
            frame,
            f"Team 1 -- Passes: {p1}   Interceptions: {i1}",
            (text_x, text_y1),
            font, base_scale, color, font_thickness, cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Team 2 -- Passes: {p2}   Interceptions: {i2}",
            (text_x, text_y2),
            font, base_scale, color, font_thickness, cv2.LINE_AA
        )

        return frame