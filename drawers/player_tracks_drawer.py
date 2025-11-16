# drawers/player_tracks_drawer.py
"""
PlayerTracksDrawer - robust & defensive version

Draws players (ellipses), a possession triangle, and jersey numbers on frames while ensuring
the output list length equals the input frame list length and never raises
on missing/None inputs.
"""

import numpy as np

# Import from utils - allow both package and top-level import paths for flexibility
try:
    from utils import draw_ellipse, draw_traingle
except Exception:
    try:
        # when used as a package
        from .utils import draw_ellipse, draw_traingle
    except Exception:
        # fallback stubs (won't draw but won't crash)
        def draw_ellipse(frame, bbox, color, track_id=None):
            return frame
        def draw_traingle(frame, bbox, color=(0,255,0)):
            return frame

# Import cv2 defensively for drawing text
try:
    import cv2
except ImportError:
    cv2 = None

class PlayerTracksDrawer:
    """
    Draw player tracks, ball-possession marker, and jersey numbers on video frames.

    This implementation is defensive: it tolerates missing frames, missing
    bbox entries, and mismatched lengths in tracks/player_assignment/ball_aquisition/ocr_results.
    """

    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0], default_player_team_id=1):
        self.default_player_team_id = default_player_team_id
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def _pad_list(self, lst, length, fill):
        """
        Ensure list-like `lst` has length `length` by padding with `fill`.
        If lst is None, create a list of `fill`s.
        """
        if lst is None:
            return [fill] * length
        # It's possible lst is not a list (e.g., generator) — coerce to list
        try:
            lst = list(lst)
        except Exception:
            return [fill] * length
        if len(lst) >= length:
            return lst
        return lst + [fill] * (length - len(lst))

    def _norm_track_id(self, tid):
        """
        Normalize track id so comparisons are stable (int if possible, else original).
        """
        try:
            return int(tid)
        except Exception:
            return tid

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition, ocr_results=None, player_number_map=None):
        """
        Process frames and draw player ellipses, possession triangle, and jersey numbers.

        Args:
            video_frames (list[np.ndarray]): list of frames (may contain None entries).
            tracks (list[dict]): per-frame dict mapping player_id -> player_info (expects 'bbox' key).
            player_assignment (list[dict]): per-frame dict mapping player_id -> team_id.
            ball_aquisition (list): per-frame player_id who has the ball or -1/None.
            ocr_results (list[dict], optional): per-frame OCR dict mapping player_id -> (number, confidence).
            player_number_map (dict, optional): stable map of player_id -> jersey_number.

        Returns:
            list[np.ndarray]: annotated frames (same length as input list).
        """
        num_frames = len(video_frames) if video_frames is not None else 0

        # pad inputs so indexing is safe
        tracks = self._pad_list(tracks, num_frames, {})                      # default no players in frame
        player_assignment = self._pad_list(player_assignment, num_frames, {})# default no assignment
        ball_aquisition = self._pad_list(ball_aquisition, num_frames, -1)    # default nobody has ball
        # new: pad OCR results
        ocr_results = self._pad_list(ocr_results, num_frames, {})            # default no OCR result

        # use empty dicts as fallback for stable map
        player_number_map = player_number_map or {}

        output_video_frames = []

        # Attempt to find a sensible fallback frame size if some frames are None
        fallback_h, fallback_w = 720, 1280
        for f in (video_frames or []):
            if isinstance(f, np.ndarray):
                fallback_h, fallback_w = f.shape[:2]
                break

        for frame_num in range(num_frames):
            frame = video_frames[frame_num]

            # If frame invalid, create a black fallback frame with a reasonable default size
            if frame is None or not isinstance(frame, np.ndarray):
                frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                print(f"[DBG] PlayerTracksDrawer: frame {frame_num} was None/invalid — using fallback {fallback_w}x{fallback_h}")

            # We keep a copy so we don't mutate the original frame outside this function
            try:
                annotated = frame.copy()
            except Exception:
                annotated = frame

            player_dict = tracks[frame_num] or {}
            assignment_for_frame = player_assignment[frame_num] or {}
            player_id_has_ball = ball_aquisition[frame_num]
            ocr_results_for_frame = ocr_results[frame_num] or {}

            # Normalize ball holder id for stable comparisons
            norm_ball_holder = self._norm_track_id(player_id_has_ball)

            # iterate players safely (track id might not be int keys)
            for raw_track_id, player in list(player_dict.items()):
                try:
                    track_id = self._norm_track_id(raw_track_id)

                    # team id fallback (assignment keys may be raw types)
                    try:
                        team_id = assignment_for_frame.get(raw_track_id, assignment_for_frame.get(track_id, self.default_player_team_id))
                    except Exception:
                        team_id = self.default_player_team_id

                    color = self.team_1_color if team_id == 1 else self.team_2_color

                    # 1. Get Bounding Box
                    bbox = None
                    if isinstance(player, dict):
                        bbox = player.get("bbox") or player.get("box") or player.get("bbox_xyxy")
                    if bbox is None and isinstance(player, (list, tuple)) and len(player) >= 1:
                        candidate = player[0]
                        if isinstance(candidate, (list, tuple)):
                            bbox = candidate

                    if not bbox:
                        # skip players without bbox
                        continue

                    # Attempt to extract bbox corners for text placement
                    try:
                        # Convert to int (x1, y1, x2, y2)
                        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
                    except Exception:
                        x1, y1 = 0, 0 # Fallback for text placement if needed

                    # 2. Draw Ellipse
                    try:
                        annotated = draw_ellipse(annotated, bbox, color, track_id)
                    except Exception as e:
                        print(f"[DBG] draw_ellipse failed for frame {frame_num} player {track_id}: {e}")

                    # 3. Draw Possession Triangle
                    if norm_ball_holder is not None and norm_ball_holder != -1 and (track_id == norm_ball_holder or raw_track_id == norm_ball_holder):
                        try:
                            annotated = draw_traingle(annotated, bbox, (0, 0, 255))
                        except Exception as e:
                            print(f"[DBG] draw_traingle failed for frame {frame_num} player {track_id}: {e}")

                    # 4. Draw Jersey Number (OCR Integration)
                    if cv2 is not None:
                        player_id_key = raw_track_id # Use the original key for dictionary lookups
                        num = None

                        # a. Priority 1: Use stable aggregated map
                        if player_id_key in player_number_map:
                            num = player_number_map[player_id_key]
                        
                        # b. Priority 2: Fall back to per-frame OCR result (if not using stable number)
                        if num is None:
                            per_frame_ocr_data = ocr_results_for_frame.get(player_id_key)
                            if per_frame_ocr_data:
                                # per_frame_ocr_data is (number, confidence)
                                num = per_frame_ocr_data[0]
                        
                        if num is not None:
                            try:
                                # Place text slightly above the top-left corner of the bounding box
                                cv2.putText(annotated, str(num), (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                            except Exception as e:
                                print(f"[DBG] cv2.putText failed for jersey number on frame {frame_num} player {track_id}: {e}")

                except Exception as e:
                    # guard against malformed player entries or unexpected keys
                    print(f"[DBG] PlayerTracksDrawer: skipping player {raw_track_id} on frame {frame_num} due to error: {e}")
                    continue

            output_video_frames.append(annotated)

        return output_video_frames