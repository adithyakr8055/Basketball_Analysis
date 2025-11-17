# drawers/pass_and_interception_drawer.py
import cv2
import numpy as np


class PassInterceptionDrawer:
    """
    Compute passes & interceptions from *inferred* ball possession:

        - For each frame, we assign the ball to the nearest player.
        - When the owner changes between frames and both owners have teams:

            same team  -> pass for that team
            different  -> interception for new owner's team
    """

    def __init__(self, overlay_alpha: float = 0.8, max_dist_px: float = 200.0):
        self.overlay_alpha = float(overlay_alpha)
        self.max_dist_px = float(max_dist_px)

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
        if not ball_tracks_frame or not isinstance(ball_tracks_frame, dict):
            return None
        for _, data in ball_tracks_frame.items():
            bbox = data.get("bbox") if isinstance(data, dict) else None
            c = self._center_from_bbox(bbox)
            if c is not None:
                return c
        return None

    def _infer_owner_per_frame(self, player_tracks, player_assignment, ball_tracks):
        """
        Returns list of (owner_pid or None, team_id or -1) per frame.
        """
        N = max(len(player_tracks or []), len(player_assignment or []), len(ball_tracks or []))
        owners = []
        last_owner = None
        last_team = -1

        for i in range(N):
            pt_frame = player_tracks[i] if i < len(player_tracks) else None
            pa_frame = player_assignment[i] if i < len(player_assignment) else None
            bt_frame = ball_tracks[i] if i < len(ball_tracks) else None

            ball_c = self._get_ball_center(bt_frame)
            owner_pid = last_owner
            owner_team = last_team

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
                            owner_pid = best_pid
                            owner_team = team_id

            if owner_pid is None or owner_team not in (1, 2):
                owners.append((None, -1))
            else:
                last_owner = owner_pid
                last_team = owner_team
                owners.append((owner_pid, owner_team))

        return owners

    def draw(self, video_frames, player_tracks, player_assignment, ball_tracks):
        if video_frames is None:
            return []

        N = len(video_frames)
        owners = self._infer_owner_per_frame(player_tracks, player_assignment, ball_tracks)

        if len(owners) < N:
            owners += [(None, -1)] * (N - len(owners))
        else:
            owners = owners[:N]

        passes_t1 = passes_t2 = 0
        inter_t1 = inter_t2 = 0

        out_frames = []

        for i, frame in enumerate(video_frames):
            frame_copy = frame.copy()

            if i > 0:
                prev_owner, prev_team = owners[i - 1]
                owner, team = owners[i]

                if (
                    prev_owner is not None and owner is not None and
                    prev_owner != owner and prev_team in (1, 2) and team in (1, 2)
                ):
                    if prev_team == team:
                        # pass
                        if team == 1:
                            passes_t1 += 1
                        else:
                            passes_t2 += 1
                    else:
                        # interception by new team
                        if team == 1:
                            inter_t1 += 1
                        else:
                            inter_t2 += 1

            try:
                frame_drawn = self._draw_overlay(
                    frame_copy, passes_t1, passes_t2, inter_t1, inter_t2
                )
            except Exception as e:
                print(f"[DBG] PassInterceptionDrawer.draw: exception on frame {i}: {e}")
                frame_drawn = frame_copy

            out_frames.append(frame_drawn)

        return out_frames

    def _draw_overlay(self, frame, passes_t1, passes_t2, inter_t1, inter_t2):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        rect_x1 = int(w * 0.01)
        rect_y1 = int(h * 0.75)
        rect_x2 = int(w * 0.45)
        rect_y2 = int(h * 0.92)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = max(0.0, min(1.0, self.overlay_alpha))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        font_scale = 0.7
        thickness = 2
        line_h = int(h * 0.05)

        text_x = rect_x1 + int(w * 0.02)
        text_y1 = rect_y1 + line_h
        text_y2 = rect_y1 + 2 * line_h

        text1 = f"White Team  Passes: {passes_t1}  Interceptions: {inter_t1}"
        text2 = f"Blue Team   Passes: {passes_t2}  Interceptions: {inter_t2}"

        cv2.putText(frame, text1, (text_x, text_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, text2, (text_x, text_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return frame
