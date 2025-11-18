# drawers/team_ball_control_drawer.py
import cv2
import numpy as np


class TeamBallControlDrawer:
    """
    Draw running ball-control percentages for:
        White Team (team 1)
        Blue  Team (team 2)

    We IGNORE ball_aquisition from the detector and instead:
      - get ball position from ball_tracks
      - find the nearest player to the ball in each frame
      - use that player's team (from player_assignment) as the team in control
      - if we can't find anyone, keep last team's possession
    """

    def __init__(self, max_dist_px: float = 200.0):
        # if the ball is further than this from all players, treat as "unknown" for that frame
        self.max_dist_px = float(max_dist_px)

    # ---------- helpers ---------- #

    @staticmethod
    def _center_from_bbox(bbox):
        if not bbox or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        except Exception:
            return None

    def _get_ball_center(self, ball_tracks_frame):
        """
        ball_tracks_frame is assumed to be dict[track_id -> { 'bbox': [...] , ... }]
        We'll just take the first one.
        """
        if not ball_tracks_frame or not isinstance(ball_tracks_frame, dict):
            return None
        for _, data in ball_tracks_frame.items():
            bbox = data.get("bbox") if isinstance(data, dict) else None
            c = self._center_from_bbox(bbox)
            if c is not None:
                return c
        return None

    def _infer_team_per_frame(self, player_tracks, player_assignment, ball_tracks):
        """
        For each frame, infer controlling team using nearest-player-to-ball rule.
        Returns np.array of length N with values 1,2,-1.
        """
        N = max(len(player_tracks or []), len(player_assignment or []), len(ball_tracks or []))
        team_per_frame = []
        last_team = -1

        for i in range(N):
            pt_frame = player_tracks[i] if i < len(player_tracks) else None
            pa_frame = player_assignment[i] if i < len(player_assignment) else None
            bt_frame = ball_tracks[i] if i < len(ball_tracks) else None

            ball_c = self._get_ball_center(bt_frame)
            controlling_team = last_team

            if ball_c is not None and pt_frame and isinstance(pt_frame, dict) and isinstance(pa_frame, dict):
                bx, by = ball_c
                best_pid = None
                best_dist2 = None

                for pid, pdata in pt_frame.items():
                    bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
                    pc = self._center_from_bbox(bbox)
                    if pc is None:
                        continue
                    px, py = pc
                    dx = px - bx
                    dy = py - by
                    d2 = dx * dx + dy * dy
                    if best_dist2 is None or d2 < best_dist2:
                        best_dist2 = d2
                        best_pid = pid

                if best_pid is not None and best_dist2 is not None:
                    dist = best_dist2 ** 0.5
                    if dist <= self.max_dist_px:
                        team_id = pa_frame.get(best_pid, None)
                        if team_id in (1, 2):
                            controlling_team = team_id

            if controlling_team in (1, 2):
                last_team = controlling_team
            else:
                controlling_team = -1

            team_per_frame.append(controlling_team)

        return np.array(team_per_frame, dtype=int)

    

    def draw(self, video_frames, player_tracks, player_assignment, ball_tracks):
        """
        Draw overlay on each frame; output list has same length as video_frames.

        Args:
            video_frames: list[np.ndarray]
            player_tracks: list[dict]
            player_assignment: list[dict player_id -> team_id]
            ball_tracks: list[dict]
        """
        if video_frames is None:
            return []

        N = len(video_frames)
        team_ball_control = self._infer_team_per_frame(player_tracks, player_assignment, ball_tracks)

        # pad / trim to N just in case
        if len(team_ball_control) < N:
            pad = np.full((N - len(team_ball_control),), -1, dtype=int)
            team_ball_control = np.concatenate([team_ball_control, pad])
        else:
            team_ball_control = team_ball_control[:N]

        out_frames = []
        for fi, frame in enumerate(video_frames):
            frame_copy = frame.copy()
            try:
                frame_drawn = self._draw_frame(frame_copy, fi, team_ball_control)
            except Exception as e:
                print(f"[DBG] TeamBallControlDrawer.draw: exception on frame {fi}: {e}")
                frame_drawn = frame_copy
            out_frames.append(frame_drawn)

        return out_frames

    def _draw_frame(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        h, w = overlay.shape[:2]
        rect_x1 = int(w * 0.60)
        rect_y1 = int(h * 0.75)
        rect_x2 = int(w * 0.99)
        rect_y2 = int(h * 0.92)

        text_x = int(w * 0.63)
        text_y1 = int(h * 0.80)
        text_y2 = int(h * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        upto = min(frame_num + 1, len(team_ball_control))
        valid = team_ball_control[:upto]
        valid = valid[valid != -1]
        if len(valid) == 0:
            t1_pct = t2_pct = 0.0
        else:
            t1_frames = int((valid == 1).sum())
            t2_frames = int((valid == 2).sum())
            total = len(valid)
            t1_pct = t1_frames * 100.0 / total
            t2_pct = t2_frames * 100.0 / total

        cv2.putText(
            frame,
            f"White Team Ball Control: {t1_pct:.2f}%",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Blue Team Ball Control:  {t2_pct:.2f}%",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        return frame
