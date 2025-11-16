# drawers/tactical_view_drawer.py
import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0], start_x=20, start_y=40):
        self.start_x = start_x
        self.start_y = start_y
        # ensure colors are integer tuples
        self.team_1_color = tuple(int(c) for c in team_1_color)
        self.team_2_color = tuple(int(c) for c in team_2_color)

    def _safe_read_and_resize_court(self, court_image_path: str, width: int, height: int):
        """Read court image, resize to width x height. If read fails, return a blank image sized to width x height."""
        court_img = None
        if court_image_path:
            court_img = cv2.imread(court_image_path)
            if court_img is None:
                print(f"[DBG] TacticalViewDrawer: failed to read court image at '{court_image_path}'")
        if court_img is None:
            # fallback: blank court sized width x height
            court_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            try:
                court_img = cv2.resize(court_img, (width, height))
            except Exception:
                court_img = cv2.resize(court_img, (width, height), interpolation=cv2.INTER_AREA)
        return court_img

    def _get_per_frame_keypoints(self, tactical_court_keypoints, num_frames):
        """
        - If tactical_court_keypoints is None -> return [None]*num_frames
        - If it's a list with length == num_frames -> return as-is
        - Else treat as single-frame keypoints repeated for all frames
        """
        if tactical_court_keypoints is None:
            return [None] * num_frames
        try:
            if isinstance(tactical_court_keypoints, (list, tuple)) and len(tactical_court_keypoints) == num_frames:
                return list(tactical_court_keypoints)
        except Exception:
            pass
        return [tactical_court_keypoints] * num_frames

    def _clamp_region(self, frame_shape, x1, y1, x2, y2):
        h, w = frame_shape[:2]
        x1c = max(0, min(w, int(x1)))
        y1c = max(0, min(h, int(y1)))
        x2c = max(0, min(w, int(x2)))
        y2c = max(0, min(h, int(y2)))
        if x2c <= x1c or y2c <= y1c:
            return None
        return x1c, y1c, x2c, y2c

    def draw(self,
             video_frames: List[np.ndarray],
             court_image_path: str,
             width: int,
             height: int,
             tactical_court_keypoints: Optional[Any],
             tactical_player_positions: Optional[List[Dict[int, Any]]] = None,
             player_assignment: Optional[List[Dict[int, int]]] = None,
             ball_acquisition: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Draw tactical view overlay and player positions on each frame.
        Defensive: will not raise if optional data is missing or malformed.
        """
        if video_frames is None or len(video_frames) == 0:
            print("[DBG] TacticalViewDrawer: no frames provided")
            return []

        num_frames = len(video_frames)
        court_image = self._safe_read_and_resize_court(court_image_path, width, height)
        per_frame_keypoints = self._get_per_frame_keypoints(tactical_court_keypoints, num_frames)

        tactical_player_positions = tactical_player_positions or [None] * num_frames
        player_assignment = player_assignment or [None] * num_frames
        ball_acquisition = ball_acquisition or [-1] * num_frames

        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            if frame is None:
                fallback_h, fallback_w = video_frames[0].shape[:2] if isinstance(video_frames[0], np.ndarray) else (720, 1280)
                frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                print(f"[DBG] TacticalViewDrawer: frame {frame_idx} was None, replaced with black frame")

            out_frame = frame.copy()
            fh, fw = out_frame.shape[:2]

            # Overlay area
            y1 = self.start_y
            y2 = self.start_y + height
            x1 = self.start_x
            x2 = self.start_x + width
            region = self._clamp_region(out_frame.shape, x1, y1, x2, y2)
            if region is not None:
                rx1, ry1, rx2, ry2 = region
                overlay = out_frame[ry1:ry2, rx1:rx2].copy()
                target_h = ry2 - ry1
                target_w = rx2 - rx1
                if (target_w, target_h) != (court_image.shape[1], court_image.shape[0]):
                    court_for_frame = cv2.resize(court_image, (target_w, target_h))
                else:
                    court_for_frame = court_image
                alpha = 0.6
                cv2.addWeighted(court_for_frame, alpha, overlay, 1 - alpha, 0, out_frame[ry1:ry2, rx1:rx2])
            else:
                # overlay out-of-bounds -> ignore overlay
                print(f"[DBG] TacticalViewDrawer: overlay region out of bounds for frame {frame_idx} ({fw}x{fh})")

            # Draw court keypoints if present
            keypoints = per_frame_keypoints[frame_idx] if per_frame_keypoints is not None else None
            if keypoints:
                try:
                    if hasattr(keypoints, "cpu"):
                        kp_arr = keypoints.cpu().numpy()
                    else:
                        kp_arr = np.array(keypoints)
                    # flatten if shape (1, N, 2)
                    if kp_arr.ndim == 3 and kp_arr.shape[0] == 1:
                        kp_arr = kp_arr[0]
                    for kp_idx, kp in enumerate(kp_arr):
                        try:
                            x_raw, y_raw = int(float(kp[0])), int(float(kp[1]))
                        except Exception:
                            continue
                        x = x_raw + self.start_x
                        y = y_raw + self.start_y
                        if 0 <= x < fw and 0 <= y < fh:
                            cv2.circle(out_frame, (x, y), 4, (0, 0, 255), -1)
                            # small index label (optional); remove if cluttered
                            cv2.putText(out_frame, str(kp_idx), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                except Exception as e:
                    print(f"[DBG] TacticalViewDrawer: failed to draw keypoints for frame {frame_idx}: {e}")

            # Draw tactical player positions (and ball as ID 0)
            frame_positions = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else None
            frame_assignments = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            player_with_ball = ball_acquisition[frame_idx] if frame_idx < len(ball_acquisition) else -1

            if frame_positions:
                try:
                    for player_id, position in frame_positions.items():
                        try:
                            px = int(float(position[0])) + self.start_x
                            py = int(float(position[1])) + self.start_y
                        except Exception:
                            continue

                        # special-case: ball mapped as id 0
                        if player_id == 0:
                            # draw ball as small orange circle with thin border (Orange in BGR is (0, 140, 255))
                            if 0 <= px < fw and 0 <= py < fh:
                                cv2.circle(out_frame, (px, py), 6, (0, 140, 255), -1)  # orange filled
                                cv2.circle(out_frame, (px, py), 8, (0, 0, 0), 1)  # border
                            continue

                        # Default team id is 1
                        team_id = frame_assignments.get(player_id, 1) if isinstance(frame_assignments, dict) else 1
                        color = self.team_1_color if team_id == 1 else self.team_2_color
                        
                        if not (0 <= px < fw and 0 <= py < fh):
                            continue
                            
                        radius = 8
                        cv2.circle(out_frame, (px, py), radius, color, -1)
                        # highlight player with ball
                        if player_with_ball == player_id:
                            cv2.circle(out_frame, (px, py), radius + 3, (0, 0, 255), 2)
                            
                except Exception as e:
                    print(f"[DBG] TacticalViewDrawer: failed to draw players/ball for frame {frame_idx}: {e}")

            output_video_frames.append(out_frame)

        return output_video_frames