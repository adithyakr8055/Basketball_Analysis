# shot_detector.py
from typing import List, Dict, Optional, Any, Tuple
import math

def detect_shots_heuristic(ball_tracks: List[Optional[Dict]],
                           tactical_ball_positions: Optional[List[Dict[int, Tuple[float,float]]]] = None,
                           player_positions: Optional[List[Dict[int, Tuple[float,float]]]] = None,
                           pose_landmarks: Optional[List[Optional[Dict[str, Tuple[float,float,float]]]]] = None,
                           meters_per_pixel: float = 0.03,
                           fps: float = 30.0,
                           upward_velocity_threshold_m_s: float = 2.0,
                           wrist_distance_px_threshold: float = 60.0) -> List[bool]:
    """
    Very small heuristic:
      - compute ball vertical velocity (dy/dt in meters/sec) across frames; detect significant upward motion (negative dy in image coords usually)
      - if upward motion > threshold AND there exists a player whose wrist landmark is near the ball -> mark shot True.
    Returns list of booleans (len = frames).
    Defensive: returns False for frames where info missing.
    """
    n = max(len(ball_tracks) if ball_tracks else 0, len(player_positions) if player_positions else 0)
    if n == 0:
        return []

    # extract ball center per frame (x,y) in pixels if possible
    ball_centers = [None] * n
    for i in range(n):
        bt = (ball_tracks[i] or {}) if i < len(ball_tracks) else {}
        # ball_tracks entry may be dict of detections; take first or look for key 'center'
        center = None
        if isinstance(bt, dict) and bt:
            # try common formats
            # if single detection per frame: pick the first value
            try:
                first = next(iter(bt.values()))
                if isinstance(first, dict) and "bbox" in first:
                    bx = first['bbox']
                    cx = (bx[0] + bx[2]) / 2.0
                    cy = (bx[1] + bx[3]) / 2.0
                    center = (float(cx), float(cy))
                elif isinstance(first, (list, tuple)) and len(first) >= 2:
                    center = (float(first[0]), float(first[1]))
            except Exception:
                center = None
        ball_centers[i] = center

    # compute vertical velocities (meters/s). image y increases downward -> upward motion => negative dy
    dt = 1.0 / fps if fps > 0 else 1.0/30.0
    v_y = [0.0] * n
    for i in range(1, n):
        a = ball_centers[i-1]
        b = ball_centers[i]
        if a is None or b is None:
            v_y[i] = 0.0
            continue
        dy_px = b[1] - a[1]
        dy_m = dy_px * meters_per_pixel
        v_y[i] = dy_m / dt

    shots = [False] * n
    for i in range(n):
        vy = v_y[i]
        # upward velocity means negative vy (since dy negative)
        if vy < -upward_velocity_threshold_m_s:
            # check proximity to a player's wrist (or hand). Use pose_landmarks if available
            nearby_player_found = False
            if pose_landmarks and i < len(pose_landmarks) and pose_landmarks[i]:
                # check left/right wrist names
                for wrist in ("LEFT_WRIST", "RIGHT_WRIST", "LEFT_INDEX", "RIGHT_INDEX"):
                    if wrist in pose_landmarks[i]:
                        wx, wy, _ = pose_landmarks[i][wrist]
                        ball = ball_centers[i]
                        if ball is None:
                            continue
                        dist_px = math.hypot(wx - ball[0], wy - ball[1])
                        if dist_px <= wrist_distance_px_threshold:
                            nearby_player_found = True
                            break
            # fallback use nearest tactical player (if provided)
            if not nearby_player_found and player_positions and i < len(player_positions) and player_positions[i]:
                # find nearest player center to ball
                b = ball_centers[i]
                if b is not None:
                    min_d = 1e9
                    for pid,ppos in player_positions[i].items():
                        try:
                            px,py = float(ppos[0]), float(ppos[1])
                            d = math.hypot(px - b[0], py - b[1])
                            if d < min_d:
                                min_d = d
                        except Exception:
                            continue
                    if min_d < wrist_distance_px_threshold:
                        nearby_player_found = True
            shots[i] = nearby_player_found
        else:
            shots[i] = False
    return shots


def evaluate_shots(detected: List[bool], ground_truth_frames: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Evaluate if you have ground-truth shot frames as a list of frame indices.
    Returns precision, recall, f1, support.
    """
    if ground_truth_frames is None:
        return {}
    gt_set = set(ground_truth_frames)
    det_set = set([i for i, v in enumerate(detected) if v])
    tp = len(gt_set & det_set)
    fp = len(det_set - gt_set)
    fn = len(gt_set - det_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}