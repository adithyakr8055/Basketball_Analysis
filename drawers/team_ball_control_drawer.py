# drawers/team_ball_control_drawer.py
import cv2
import numpy as np

class TeamBallControlDrawer:
    """
    Robust drawer for team ball control.

    Changes made:
    - get_team_ball_control tolerates None frames and mismatched lengths.
    - draw returns a list of frames with the same length as input video_frames.
    - frame 0 is not skipped; original frame is returned with overlay (so lengths match).
    - division by zero guarded when computing percentages.
    """

    def __init__(self):
        pass

    def get_team_ball_control(self, player_assignment, ball_aquisition, target_len=None):
        """
        Calculate which team has ball control for each frame.

        Args:
            player_assignment (list): per-frame dict mapping player_id -> team_id (or None)
            ball_aquisition (list): per-frame player_id in possession or -1/None
            target_len (int|None): desired output length. If provided, result is padded/truncated.

        Returns:
            numpy.ndarray: array length == target_len (or max input length) with values:
                1 (team1), 2 (team2), -1 (no control)
        """
        if player_assignment is None:
            player_assignment = []
        if ball_aquisition is None:
            ball_aquisition = []

        n = max(len(player_assignment), len(ball_aquisition), 0)
        if target_len is not None:
            n = max(n, int(target_len))

        team_ball_control = []
        for i in range(n):
            pa_frame = player_assignment[i] if i < len(player_assignment) else None
            ba_frame = ball_aquisition[i] if i < len(ball_aquisition) else None

            # normalize missing values
            if ba_frame is None or ba_frame == -1:
                team_ball_control.append(-1)
                continue

            if not pa_frame or not isinstance(pa_frame, dict):
                # no assignment information -> unknown
                team_ball_control.append(-1)
                continue

            # If ba_frame (player id) not in assignments -> unknown
            if ba_frame not in pa_frame:
                team_ball_control.append(-1)
                continue

            team_id = pa_frame.get(ba_frame, None)
            if team_id == 1:
                team_ball_control.append(1)
            elif team_id == 2:
                team_ball_control.append(2)
            else:
                team_ball_control.append(-1)

        return np.array(team_ball_control, dtype=int)

    def draw(self, video_frames, player_assignment, ball_aquisition):
        """
        Draw team ball control statistics on each frame and return a list same length as video_frames.

        Args:
            video_frames (list): list of frames (numpy arrays)
            player_assignment (list): per-frame dicts mapping player_id->team_id
            ball_aquisition (list): per-frame player_id who has ball (-1/None if none)

        Returns:
            list: frames with overlay (same length as input)
        """
        if video_frames is None:
            return []

        N = len(video_frames)
        # compute team control for length at least N (pads/truncates internally)
        team_ball_control = self.get_team_ball_control(player_assignment, ball_aquisition, target_len=N)

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            # ensure we always append a frame (don't skip)
            frame_copy = frame.copy()

            # draw overlay using team_ball_control up to this frame
            try:
                frame_drawn = self.draw_frame(frame_copy, frame_num, team_ball_control)
            except Exception as e:
                # if anything goes wrong for this frame, log (print) and append original copy
                print(f"[DBG] TeamBallControlDrawer.draw: exception on frame {frame_num}: {e}")
                frame_drawn = frame_copy

            output_video_frames.append(frame_drawn)

        return output_video_frames

    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent overlay of team ball control percentages on a single frame.

        - Handles edge cases where there is no data yet (division by zero).
        - Uses available team_ball_control values up to current frame.
        """
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.90)

        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Safely slice team_ball_control up to frame_num (inclusive)
        upto = frame_num + 1
        if upto > len(team_ball_control):
            upto = len(team_ball_control)

        if upto == 0:
            # no data yet — show zeros
            team_1_pct = 0.0
            team_2_pct = 0.0
        else:
            team_ball_control_till_frame = team_ball_control[:upto]
            total_count = team_ball_control_till_frame.shape[0]
            if total_count == 0:
                team_1_pct = 0.0
                team_2_pct = 0.0
            else:
                team_1_num_frames = int((team_ball_control_till_frame == 1).sum())
                team_2_num_frames = int((team_ball_control_till_frame == 2).sum())

                # percentages (guard division)
                team_1_pct = (team_1_num_frames / total_count) * 100.0 if total_count else 0.0
                team_2_pct = (team_2_num_frames / total_count) * 100.0 if total_count else 0.0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct:.2f}%", (text_x, text_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct:.2f}%", (text_x, text_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        return frame