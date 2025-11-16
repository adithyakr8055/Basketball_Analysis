# drawers/speed_and_distance_drawer.py
import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class SpeedAndDistanceDrawer:
    """
    Robust drawer for speed and distance per player.

    Usage:
      drawer = SpeedAndDistanceDrawer()
      output_frames = drawer.draw(
          video_frames,
          player_tracks_list,
          player_distances_per_frame,    # per-frame delta distances in meters
          player_speed_per_frame,        # per-frame speeds (km/h)
          speed_in_m_per_s=False         # This flag is now effectively ignored or assumed False
      )
    """

    def __init__(self):
        pass

    def _pad_or_default(self, lst, length, default):
        """Ensure list-like `lst` has length `length` by padding with `default` or creating defaults."""
        if lst is None:
            return [default] * length
        try:
            lst = list(lst)
        except Exception:
            return [default] * length
        if len(lst) >= length:
            return lst
        return lst + [default] * (length - len(lst))

    def draw(self,
             video_frames: List[np.ndarray],
             player_tracks_list: List[Dict[int, Dict[str, Any]]],
             player_distances_per_frame: List[Dict[int, float]],
             player_speed_per_frame: List[Dict[int, float]],
             speed_in_m_per_s: bool = True) -> List[np.ndarray]:
        """
        Draw speed & cumulative distance near each player's bbox foot.

        - video_frames: list of frames (len = N)
        - player_tracks_list: list (len = N) where each item is dict player_id -> {'bbox': (x1,y1,x2,y2)}
        - player_distances_per_frame: list (len = N) where each item is dict player_id -> distance_delta_in_m
        - player_speed_per_frame: list (len = N) where each item is dict player_id -> speed (km/h is assumed)
        - speed_in_m_per_s: If True, it implies the raw speed is in m/s, but based on the fix, 
          we now assume the speed is already in km/h. This flag will be effectively ignored for the conversion.
        """
        num_frames = len(video_frames) if video_frames is not None else 0
        if num_frames == 0:
            print("[DBG] SpeedAndDistanceDrawer: no frames received, returning empty list")
            return []

        # pad inputs so indexing won't IndexError
        player_tracks_list = self._pad_or_default(player_tracks_list, num_frames, {})
        player_distances_per_frame = self._pad_or_default(player_distances_per_frame, num_frames, {})
        player_speed_per_frame = self._pad_or_default(player_speed_per_frame, num_frames, {})

        output_video_frames = []
        total_distances: Dict[int, float] = {}  # cumulative per-player (meters)

        for frame_idx in range(num_frames):
            frame = video_frames[frame_idx]
            # Defensive frame handling
            if frame is None or not isinstance(frame, (np.ndarray,)):
                # create fallback black frame using a reasonable default size (try to infer from other frames)
                fallback_h, fallback_w = 720, 1280
                for f in video_frames:
                    if isinstance(f, np.ndarray):
                        fallback_h, fallback_w = f.shape[:2]
                        break
                frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                print(f"[DBG] SpeedAndDistanceDrawer: frame {frame_idx} was None/invalid — using fallback {fallback_w}x{fallback_h}")

            out = frame.copy()

            per_frame_tracks = player_tracks_list[frame_idx] or {}
            per_frame_distance = player_distances_per_frame[frame_idx] or {}
            per_frame_speed = player_speed_per_frame[frame_idx] or {}

            # accumulate distances (per-frame distances are expected to be delta meters)
            for pid, dist in per_frame_distance.items():
                try:
                    if pid not in total_distances:
                        total_distances[pid] = 0.0
                    total_distances[pid] += float(dist)
                except Exception:
                    print(f"[DBG] SpeedAndDistanceDrawer: invalid distance for player {pid} on frame {frame_idx}: {dist}")

            # Draw each player's info available in per_frame_tracks
            for player_id, player_info in per_frame_tracks.items():
                try:
                    bbox = player_info.get('bbox') if isinstance(player_info, dict) else None
                except Exception:
                    bbox = None

                if not bbox or not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                    continue

                try:
                    x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
                except Exception:
                    print(f"[DBG] SpeedAndDistanceDrawer: invalid bbox for player {player_id} on frame {frame_idx}: {bbox}")
                    continue

                # position to place text: below the bbox bottom-center
                pos_x = int((x1 + x2) / 2)
                pos_y = int(y2) + 30

                # get totals and speed (may be missing)
                total_distance = total_distances.get(player_id, None)
                raw_speed = per_frame_speed.get(player_id, None)

                # --- START MODIFIED LOGIC ---
                # convert speed for display
                speed_display = None
                if raw_speed is not None:
                    try:
                        s = float(raw_speed)  # trust it's km/h already
                        # The conversion logic (s = s * 3.6) is removed.
                        speed_display = f"{s:.1f} km/h" # Format to one decimal place
                    except Exception:
                        speed_display = None
                # --- END MODIFIED LOGIC ---
                
                distance_display = None
                if total_distance is not None:
                    try:
                        distance_display = f"{total_distance:.2f} m"
                    except Exception:
                        distance_display = None

                # Draw small background rectangle then text stacked
                try:
                    texts = [t for t in [speed_display, distance_display] if t is not None]
                    if texts:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_thickness = 1
                        max_w = 0
                        total_h = 0
                        line_heights = []
                        for t in texts:
                            (w, h), _ = cv2.getTextSize(t, font, font_scale, font_thickness)
                            line_heights.append(h)
                            if w > max_w:
                                max_w = w
                            total_h += h + 6
                        rect_w = int(max_w + 12)
                        rect_h = int(total_h + 6)

                        rect_x1 = pos_x - rect_w // 2
                        rect_y1 = pos_y - rect_h
                        rect_x2 = rect_x1 + rect_w
                        rect_y2 = rect_y1 + rect_h

                        # clamp coords inside frame
                        rect_x1 = max(0, rect_x1)
                        rect_y1 = max(0, rect_y1)
                        rect_x2 = min(out.shape[1] - 1, rect_x2)
                        rect_y2 = min(out.shape[0] - 1, rect_y2)

                        overlay = out.copy()
                        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
                        alpha = 0.7
                        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

                        current_y = rect_y1 + 14
                        for t in texts:
                            cv2.putText(out, t, (rect_x1 + 6, current_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                            current_y += 14 + 4
                except Exception as e:
                    print(f"[DBG] SpeedAndDistanceDrawer: failed to draw overlay for player {player_id} on frame {frame_idx}: {e}")

            output_video_frames.append(out)

        return output_video_frames