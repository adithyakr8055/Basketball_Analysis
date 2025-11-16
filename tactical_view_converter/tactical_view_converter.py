# tactical_view_converter/tactical_view_converter.py
import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import get_foot_position  # measure_distance not used here

# small Homography wrapper fallback if your .homography import fails
try:
    from .homography import Homography
except Exception:
    # Minimal homography wrapper using cv2.getPerspectiveTransform / warpPerspective
    class Homography:
        def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray):
            # expects Nx2 arrays; if >4 points use findHomography
            if src_pts.shape[0] >= 4:
                H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
            else:
                H = None
            self.H = H

        def transform_points(self, pts: np.ndarray) -> np.ndarray:
            # pts shape (N,2)
            if self.H is None:
                # fallback: identity mapping
                return pts.copy()
            pts_h = np.concatenate([pts.reshape(-1, 2), np.ones((pts.shape[0], 1))], axis=1)
            transformed = (self.H @ pts_h.T).T
            transformed = transformed[:, :2] / transformed[:, 2:3]
            return transformed


class TacticalViewConverter:
    def __init__(self, court_image_path: str):
        self.court_image_path = court_image_path
        # tactical canvas size (pixels)
        self.width = 300
        self.height = 161

        # real court size used for pixel->meter scaling
        self.actual_width_in_meters = 28.0
        self.actual_height_in_meters = 15.0

        # tactical template points (index ordering must match model keypoints shape/ordering)
        self.key_points = [
            (0.0, 0.0),
            (0.0, (0.91 / self.actual_height_in_meters) * self.height),
            (0.0, (5.18 / self.actual_height_in_meters) * self.height),
            (0.0, (10.0 / self.actual_height_in_meters) * self.height),
            (0.0, (14.1 / self.actual_height_in_meters) * self.height),
            (0.0, float(self.height)),
            (float(self.width / 2), float(self.height)),
            (float(self.width / 2), 0.0),
            ( (5.79 / self.actual_width_in_meters) * self.width, (5.18 / self.actual_height_in_meters) * self.height),
            ( (5.79 / self.actual_width_in_meters) * self.width, (10.0 / self.actual_height_in_meters) * self.height),
            (float(self.width), float(self.height)),
            (float(self.width), (14.1 / self.actual_height_in_meters) * self.height),
            (float(self.width), (10.0 / self.actual_height_in_meters) * self.height),
            (float(self.width), (5.18 / self.actual_height_in_meters) * self.height),
            (float(self.width), (0.91 / self.actual_height_in_meters) * self.height),
            (float(self.width), 0.0),
            ( ((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width, (5.18 / self.actual_height_in_meters) * self.height),
            ( ((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width, (10.0 / self.actual_height_in_meters) * self.height),
        ]

    def _frame_keypoints_to_list(self, frame_kp) -> List[Tuple[float, float]]:
        if frame_kp is None:
            return []
        # if has .xy attribute (our validate wrapper sets this)
        if hasattr(frame_kp, "xy"):
            try:
                arr = np.asarray(frame_kp.xy)
                if arr.ndim == 3:  # shape (1, N, 2)
                    arr = arr[0]
                return [(float(x), float(y)) for x, y in arr.tolist()]
            except Exception:
                return []
        # numpy array
        if isinstance(frame_kp, np.ndarray):
            arr = frame_kp
            if arr.ndim == 2 and arr.shape[1] == 2:
                return [(float(x), float(y)) for x, y in arr.tolist()]
            # if nested
            try:
                flat = arr.reshape(-1, 2)
                return [(float(x), float(y)) for x, y in flat.tolist()]
            except Exception:
                return []
        # list-like
        try:
            if isinstance(frame_kp, (list, tuple)):
                # maybe nested [[x,y],...]
                candidate = frame_kp
                if len(candidate) > 0 and isinstance(candidate[0], (list, tuple)):
                    return [(float(x), float(y)) for x, y in candidate]
        except Exception:
            pass
        return []

    def validate_keypoints(self, keypoints_list: List[Any]) -> List[Optional[Any]]:
        """
        Return a list of objects with .xy or None. This ensures downstream drawers expecting .xy work.
        """
        if keypoints_list is None:
            return []

        cleaned = []
        valid_count = 0
        for frame_kp in keypoints_list:
            pts = self._frame_keypoints_to_list(frame_kp)
            if len(pts) < 4:
                cleaned.append(None)
                continue
            # pad to template length
            full = []
            for idx in range(len(self.key_points)):
                if idx < len(pts):
                    full.append((float(pts[idx][0]), float(pts[idx][1])))
                else:
                    full.append((0.0, 0.0))
            class KPObj:
                def __init__(self, arr):
                    self.xy = np.array([arr], dtype=np.float32)   # shape (1, N, 2)
                    self.xyn = np.array([arr], dtype=np.float32)
            cleaned.append(KPObj(full))
            valid_count += 1
        print(f"[DBG] validate_keypoints: total_frames={len(keypoints_list)} valid_frames={valid_count}")
        return cleaned

    def transform_players_to_tactical_view(
        self,
        keypoints_list: List[Any],
        player_tracks: List[Dict[int, Dict[str, Any]]],
        ball_tracks: Optional[List[Dict]] = None
    ) -> List[Dict[int, List[float]]]:
        """
        Map player foot positions (and ball if provided) to tactical view coordinates.
        Returns list length == number of frames, each item dict player_id->[x,y] (floats).
        """
        tactical_player_positions: List[Dict[int, List[float]]] = []

        n_frames = max(len(keypoints_list) if keypoints_list is not None else 0,
                       len(player_tracks) if player_tracks is not None else 0,
                       len(ball_tracks) if ball_tracks is not None else 0)

        for frame_idx in range(n_frames):
            tactical_positions = {}
            # get validated keypoints object for this frame
            kp_obj = keypoints_list[frame_idx] if (keypoints_list and frame_idx < len(keypoints_list)) else None
            pts = self._frame_keypoints_to_list(kp_obj) if kp_obj is not None else []

            # need at least 4 correspondences
            if len(pts) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            # choose valid indices (pts > 0)
            valid_indices = [i for i, p in enumerate(pts) if p[0] > 0 and p[1] > 0]
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            src_pts = np.array([pts[i] for i in valid_indices], dtype=np.float32)
            dst_pts = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            try:
                homography = Homography(src_pts, dst_pts)
            except Exception as e:
                tactical_player_positions.append(tactical_positions)
                continue

            # players for this frame
            players_for_frame = player_tracks[frame_idx] if (player_tracks and frame_idx < len(player_tracks)) else {}
            for pid, pdata in (players_for_frame or {}).items():
                try:
                    bbox = pdata.get("bbox")
                    if not bbox:
                        continue
                    foot = get_foot_position(bbox)
                    pts_np = np.array([foot], dtype=np.float32)
                    transformed = homography.transform_points(pts_np)
                    tx, ty = float(transformed[0][0]), float(transformed[0][1])
                    if -5 <= tx <= (self.width + 5) and -5 <= ty <= (self.height + 5):
                        tactical_positions[int(pid)] = [tx, ty]
                except Exception:
                    continue

            # ball mapping (optional) -> map ball center if available
            if ball_tracks and frame_idx < len(ball_tracks):
                ball_frame = ball_tracks[frame_idx]
                if isinstance(ball_frame, dict):
                    # attempt to find ball bbox or center
                    binfo = None
                    # if ball_frame has id-less single detection
                    # common formats: {'bbox': (...)} or {id: {'bbox': (...)}} -> handle both
                    if 'bbox' in ball_frame:
                        binfo = ball_frame
                    else:
                        # try first dict value
                        try:
                            first = next(iter(ball_frame.values()))
                            if isinstance(first, dict) and 'bbox' in first:
                                binfo = first
                        except Exception:
                            binfo = None
                    if binfo:
                        try:
                            bbox = binfo.get('bbox')
                            if bbox:
                                foot = get_foot_position(bbox)
                                pts_np = np.array([foot], dtype=np.float32)
                                transformed = homography.transform_points(pts_np)
                                tx, ty = float(transformed[0][0]), float(transformed[0][1])
                                # store with special key -1 for ball
                                if -5 <= tx <= (self.width + 5) and -5 <= ty <= (self.height + 5):
                                    tactical_positions[-1] = [tx, ty]
                        except Exception:
                            pass

            tactical_player_positions.append(tactical_positions)

        non_empty = sum(1 for d in tactical_player_positions if d)
        print(f"[DBG] transform_players_to_tactical_view: frames={len(tactical_player_positions)} non_empty_frames={non_empty}")
        return tactical_player_positions