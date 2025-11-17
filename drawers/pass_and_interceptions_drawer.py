import cv2
import numpy as np

class PassInterceptionDrawer:
    """
    Draw pass and interception statistics as a semi-transparent overlay on each
    frame. This drawer preserves the number of frames (one output per input frame)
    and is robust to None / corrupted frames and mismatched pass/interception list lengths.
    """

    def __init__(self, overlay_alpha: float = 0.8):
        self.overlay_alpha = float(overlay_alpha)

    def get_stats(self, passes, interceptions):
        """
        Count total passes and interceptions for each team.

        Args:
            passes (list[int]): per-frame pass code (0=no pass, 1=team1, 2=team2)
            interceptions (list[int]): per-frame interception code (0=no,1=team1,2=team2)

        Returns:
            tuple: (team1_passes, team2_passes, team1_interceptions, team2_interceptions)
        """
        # Defensive: if None provided, treat as empty
        if passes is None:
            passes = []
        if interceptions is None:
            interceptions = []

        # Count occurrences - robust if lists are different lengths
        team1_passes = sum(1 for x in passes if x == 1)
        team2_passes = sum(1 for x in passes if x == 2)
        team1_interceptions = sum(1 for x in interceptions if x == 1)
        team2_interceptions = sum(1 for x in interceptions if x == 2)

        return team1_passes, team2_passes, team1_interceptions, team2_interceptions

    def draw(self, video_frames, passes, interceptions):
        """
        Draw overlay on each frame and return a new list with the same length as input frames.

        Args:
            video_frames (list[numpy.ndarray]): list of frames (may contain None).
            passes (list[int]): global list for the whole video (len may be <= frames)
            interceptions (list[int]): global list for the whole video (len may be <= frames)

        Returns:
            list[numpy.ndarray]: annotated frames (same length as video_frames)
        """
        num_frames = len(video_frames) if video_frames is not None else 0

        # Defensive padding of event lists so slicing later is safe
        def pad_list(lst, length):
            if lst is None:
                return [0] * length
            if len(lst) >= length:
                return lst
            return lst + [0] * (length - len(lst))

        passes = pad_list(list(passes) if passes is not None else [], num_frames)
        interceptions = pad_list(list(interceptions) if interceptions is not None else [], num_frames)

        output_video_frames = []
        for frame_num in range(num_frames):
            frame = video_frames[frame_num]

            # Safety: if frame is None or invalid, create a black frame fallback
            if frame is None or not isinstance(frame, (np.ndarray,)):
                # create a reasonable-sized fallback if we can't infer size
                fallback_h, fallback_w = (720, 1280)
                if isinstance(frame, np.ndarray):
                    fallback_h, fallback_w = frame.shape[:2]
                fallback = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                frame_drawn = self.draw_frame(fallback, frame_num, passes, interceptions)
                output_video_frames.append(frame_drawn)
                continue

            # main draw
            frame_drawn = self.draw_frame(frame.copy(), frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)

        return output_video_frames

    def draw_frame(self, frame, frame_num, passes, interceptions):
        """
        Draw overlay with cumulative stats up to frame_num (inclusive).
        Returns the same frame object annotated.
        """
        # Frame shape
        h, w = frame.shape[:2]

        # Compute adaptive font & sizes based on resolution
        base_scale = max(0.5, min(w, h) / 1000 * 0.7)  # scales with resolution
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(round(base_scale * 2)))

        # Overlay rectangle coordinates (relative)
        rect_x1 = int(w * 0.05)
        rect_y1 = int(h * 0.70)
        rect_x2 = int(w * 0.45)
        rect_y2 = int(h * 0.90)

        # Text positions
        text_x = rect_x1 + int(w * 0.02)
        text_y1 = rect_y1 + int((rect_y2 - rect_y1) * 0.35)
        text_y2 = rect_y1 + int((rect_y2 - rect_y1) * 0.75)

        # Build overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)

        # Blend overlay onto frame
        alpha = max(0.0, min(1.0, self.overlay_alpha))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Prepare cumulative lists up to current frame
        passes_till_frame = passes[:frame_num + 1]
        interceptions_till_frame = interceptions[:frame_num + 1]

        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, interceptions_till_frame
        )

        # Compose text lines
        line1 = f"Team  — Passes: {team1_passes}  Interceptions: {team1_interceptions}"
        """line2 = f"Team 2 — Passes: {team2_passes}  Interceptions: {team2_interceptions}"""

        # Put text (with shadow for readability)
        def put_text_with_shadow(img, text, org, font, scale, color=(0, 0, 0), thickness=1):
            # shadow
            cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
            # main
            cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

        put_text_with_shadow(frame, line1, (text_x, text_y1), font, base_scale, (0, 0, 0), thickness)
        """put_text_with_shadow(frame, line2, (text_x, text_y2), font, base_scale, (0, 0, 0), thickness)"""

        return frame