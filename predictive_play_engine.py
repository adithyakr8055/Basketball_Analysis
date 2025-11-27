"""
predictive_play_engine.py

Rule-based "Predictive Play Engine" for Basketball_Analysis.

This module does NOT use ML. Instead, it builds a rich context per frame
and then applies interpretable heuristics to predict high-level next actions.

Public API (used by main.py):
    build_frame_contexts(...)
    predict_next_actions(frame_contexts)

Frame context keys (per frame):
    - frame_idx
    - ball_handler_id
    - offense_team
    - defense_team
    - handler_speed_mps
    - handler_tx, handler_ty          # tactical coordinates (or None)
    - handler_zone                    # COURT_ZONE_*
    - offense_spacing                 # average dist between attacking teammates
    - defensive_pressure              # distance to nearest defender
    - numerical_advantage             # attackers in frontcourt - defenders in frontcourt
    - is_fast_break                   # bool
    - time_since_possession_change    # frames
    - possession_id                   # naive segmentation based on holder
"""

from typing import List, Dict, Any, Tuple, Optional
import math


def _normalize(val: float, vmin: float, vmax: float, eps: float = 1e-6) -> float:
    if vmax - vmin == 0:
        return 0.5
    return (float(val) - float(vmin)) / (float(vmax - vmin) + eps)


def _compute_court_bounds(tactical_player_positions: List[Dict[int, Tuple[float, float]]]):
    xs, ys = [], []
    for fr in tactical_player_positions:
        if not isinstance(fr, dict):
            continue
        for (x, y) in fr.values():
            try:
                xs.append(float(x))
                ys.append(float(y))
            except Exception:
                continue
    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _classify_zone(x: Optional[float],
                   y: Optional[float],
                   xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float) -> str:
    if x is None or y is None:
        return "COURT_ZONE_UNKNOWN"
    nx = _normalize(x, xmin, xmax)
    ny = _normalize(y, ymin, ymax)

    in_center_x = 0.3 <= nx <= 0.7
    in_paint_y = 0.3 <= ny <= 0.7

    if in_center_x and in_paint_y:
        return "COURT_ZONE_PAINT"

    if ny > 0.75:
        if nx < 0.25:
            return "COURT_ZONE_CORNER_LEFT"
        if nx > 0.75:
            return "COURT_ZONE_CORNER_RIGHT"

    if 0.25 <= ny <= 0.75:
        if nx < 0.3:
            return "COURT_ZONE_WING_LEFT"
        if nx > 0.7:
            return "COURT_ZONE_WING_RIGHT"

    if ny < 0.25:
        return "COURT_ZONE_PERIMETER_TOP"

    return "COURT_ZONE_OTHER"


def _pairwise_avg_distance(positions: List[Tuple[float, float]]) -> Optional[float]:
    n = len(positions)
    if n < 2:
        return None
    total = 0.0
    count = 0
    for i in range(n):
        x1, y1 = positions[i]
        for j in range(i + 1, n):
            x2, y2 = positions[j]
            dx = float(x1) - float(x2)
            dy = float(y1) - float(y2)
            total += math.sqrt(dx * dx + dy * dy)
            count += 1
    return total / max(1, count)


def _nearest_defender_distance(handler_id: int,
                               frame_positions: Dict[int, Tuple[float, float]],
                               frame_assign: Dict[int, int],
                               offense_team: int) -> Optional[float]:
    if handler_id is None or handler_id == -1:
        return None
    if handler_id not in frame_positions:
        return None
    if offense_team not in (1, 2):
        return None

    hx, hy = frame_positions[handler_id]
    best = None
    for pid, (x, y) in frame_positions.items():
        team = frame_assign.get(pid, -1)
        if team in (1, 2) and team != offense_team:
            dx = float(hx) - float(x)
            dy = float(hy) - float(y)
            d = math.sqrt(dx * dx + dy * dy)
            if best is None or d < best:
                best = d
    return best


def _numerical_advantage(frame_positions: Dict[int, Tuple[float, float]],
                         frame_assign: Dict[int, int],
                         offense_team: int) -> int:
    """Same heuristic as in analytics table."""
    if not frame_positions or offense_team not in (1, 2):
        return 0

    xs = [float(x) for (x, _) in frame_positions.values()]
    if not xs:
        return 0
    median_x = sorted(xs)[len(xs) // 2]

    off_in_front = 0
    def_in_front = 0
    for pid, (x, y) in frame_positions.items():
        team = frame_assign.get(pid, -1)
        if float(x) <= median_x:
            continue
        if team == offense_team:
            off_in_front += 1
        elif team in (1, 2) and team != offense_team:
            def_in_front += 1
    return off_in_front - def_in_front


def _segment_possessions(ball_acquisition: List[int]) -> List[int]:
    """Simple segmentation by holder changes (same as analytics)."""
    n = len(ball_acquisition)
    if n == 0:
        return []
    possession_ids = [-1] * n
    current_id = 0
    last_holder = ball_acquisition[0]
    possession_ids[0] = 0 if last_holder != -1 else -1

    for i in range(1, n):
        holder = ball_acquisition[i]
        if holder == -1:
            possession_ids[i] = -1
        else:
            if last_holder == -1 or holder != last_holder:
                current_id += 1
            possession_ids[i] = current_id
        last_holder = holder

    return possession_ids


# --------------------------------------------------------------------
# Frame context builder
# --------------------------------------------------------------------
def build_frame_contexts(
    tactical_player_positions: List[Dict[int, Tuple[float, float]]],
    ball_acquisition: List[int],
    player_assignment: List[Dict[int, int]],
    ball_tracks: List[Dict[int, Dict[str, Any]]],
    court_keypoints: List[Any],
    player_speed_per_frame: List[Dict[int, float]],
    fps: float,
    meters_per_pixel: float,
    tactical_width_m: float,
    tactical_height_m: float,
) -> List[Dict[str, Any]]:
    """
    Build rich per-frame contexts.

    All lists are assumed to be aligned by frame index.
    """
    n = min(
        len(tactical_player_positions),
        len(ball_acquisition),
        len(player_assignment),
        len(player_speed_per_frame),
    )
    if n == 0:
        return []

    # Bounds for zone classification
    xmin, xmax, ymin, ymax = _compute_court_bounds(tactical_player_positions)

    possession_ids = _segment_possessions(ball_acquisition)

    contexts: List[Dict[str, Any]] = []
    last_possession_id = possession_ids[0] if possession_ids else -1
    frames_since_change = 0

    for i in range(n):
        holder = ball_acquisition[i]
        frame_assign = player_assignment[i] or {}
        frame_positions = tactical_player_positions[i] or {}
        frame_speed = player_speed_per_frame[i] or {}

        poss_id = possession_ids[i] if i < len(possession_ids) else -1
        if poss_id != last_possession_id:
            frames_since_change = 0
            last_possession_id = poss_id
        else:
            frames_since_change += 1

        offense_team = frame_assign.get(holder, -1) if holder != -1 else -1
        defense_team = 1 if offense_team == 2 else 2 if offense_team == 1 else -1

        # handler position + zone
        tx = ty = None
        if holder != -1 and holder in frame_positions:
            tx, ty = frame_positions[holder]
        zone = _classify_zone(tx, ty, xmin, xmax, ymin, ymax)

        # offense spacing
        off_positions = []
        if offense_team in (1, 2):
            for pid, (x, y) in frame_positions.items():
                if frame_assign.get(pid, -1) == offense_team:
                    off_positions.append((x, y))
        spacing = _pairwise_avg_distance(off_positions)

        # defensive pressure
        def_pressure = _nearest_defender_distance(holder, frame_positions, frame_assign, offense_team)

        # numerical advantage
        num_adv = _numerical_advantage(frame_positions, frame_assign, offense_team)

        # simple fast break heuristic: high speed + advantage
        handler_speed = frame_speed.get(holder)
        is_fast_break = False
        try:
            if handler_speed is not None and float(handler_speed) > 4.5 and num_adv >= 1:
                is_fast_break = True
        except Exception:
            is_fast_break = False

        ctx = {
            "frame_idx": i,
            "ball_handler_id": holder,
            "offense_team": offense_team,
            "defense_team": defense_team,
            "handler_speed_mps": handler_speed,
            "handler_tx": tx,
            "handler_ty": ty,
            "handler_zone": zone,
            "offense_spacing": spacing,
            "defensive_pressure": def_pressure,
            "numerical_advantage": num_adv,
            "is_fast_break": is_fast_break,
            "time_since_possession_change": frames_since_change,
            "possession_id": poss_id,
        }
        contexts.append(ctx)

    return contexts


# --------------------------------------------------------------------
# Predictive logic
# --------------------------------------------------------------------
def _risk_level(ctx: Dict[str, Any]) -> str:
    """Map context to LOW / MEDIUM / HIGH risk levels."""
    dp = ctx.get("defensive_pressure")
    spacing = ctx.get("offense_spacing")
    num_adv = ctx.get("numerical_advantage", 0)
    risk_score = 0.0

    # defensive pressure: closer defender -> higher risk
    if dp is not None:
        d = float(dp)
        # 0..8 mapped to ~1..0
        risk_score += max(0.0, min(1.0, 1.0 - d / 8.0)) * 0.5
    else:
        risk_score += 0.2

    # spacing: tighter -> higher risk
    if spacing is not None:
        s = float(spacing)
        if s <= 3.0:
            risk_score += 0.4
        elif s <= 6.0:
            risk_score += 0.25
        else:
            risk_score += 0.1
    else:
        risk_score += 0.2

    # numerical advantage
    if num_adv <= -2:
        risk_score += 0.3
    elif num_adv == -1:
        risk_score += 0.2
    elif num_adv == 0:
        risk_score += 0.15
    elif num_adv >= 1:
        risk_score += 0.05

    if risk_score >= 0.8:
        return "HIGH"
    if risk_score >= 0.45:
        return "MEDIUM"
    return "LOW"


def predict_next_actions(frame_contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Given frame contexts, return a list of string labels, one per frame.

    Possible labels (designed to look impressive on overlays & CSVs):
      - NO_POSSESSION
      - STABLE_POSSESSION
      - FAST_BREAK_PUSH
      - LIKELY_PASS
      - LIKELY_SHOT
      - LIKELY_DRIVE
      - RESET_PLAY
      - HIGH_RISK_PASS_ZONE
    """
    labels: List[str] = []

    for ctx in frame_contexts:
        handler = ctx.get("ball_handler_id", -1)
        offense_team = ctx.get("offense_team", -1)
        zone = ctx.get("handler_zone", "COURT_ZONE_UNKNOWN")
        speed = ctx.get("handler_speed_mps")
        spacing = ctx.get("offense_spacing")
        def_pressure = ctx.get("defensive_pressure")
        num_adv = ctx.get("numerical_advantage", 0)
        is_fast_break = bool(ctx.get("is_fast_break", False))
        time_since_change = int(ctx.get("time_since_possession_change", 0))

        if handler == -1 or offense_team not in (1, 2):
            labels.append("NO_POSSESSION")
            continue

        # 1) Fast-break scenario
        if is_fast_break:
            labels.append("FAST_BREAK_PUSH")
            continue

        # 2) High-risk zone: tight defense, bad spacing
        risk_level = _risk_level(ctx)
        if risk_level == "HIGH":
            labels.append("HIGH_RISK_PASS_ZONE")
            continue

        # 3) Likely drive: handler in wing/paint, moving with decent speed
        if zone in ("COURT_ZONE_WING_LEFT", "COURT_ZONE_WING_RIGHT", "COURT_ZONE_PAINT"):
            try:
                if speed is not None and float(speed) > 3.5:
                    labels.append("LIKELY_DRIVE")
                    continue
            except Exception:
                pass

        # 4) Likely shot: handler at perimeter or corner, spacing good, pressure low
        good_spacing = False
        low_pressure = False
        if spacing is not None:
            try:
                s = float(spacing)
                good_spacing = s >= 6.5
            except Exception:
                pass
        if def_pressure is not None:
            try:
                d = float(def_pressure)
                low_pressure = d >= 4.5
            except Exception:
                pass

        if zone in ("COURT_ZONE_PERIMETER_TOP",
                    "COURT_ZONE_CORNER_LEFT",
                    "COURT_ZONE_CORNER_RIGHT") and good_spacing and low_pressure:
            labels.append("LIKELY_SHOT")
            continue

        # 5) Likely pass: moderate or good spacing, moderate pressure
        if spacing is not None:
            try:
                s = float(spacing)
                if 4.0 <= s <= 8.5:
                    labels.append("LIKELY_PASS")
                    continue
            except Exception:
                pass

        # 6) Reset: early possession, handler outside paint, no clear pressure
        if time_since_change < 10 and zone not in ("COURT_ZONE_PAINT",):
            labels.append("RESET_PLAY")
            continue

        # 7) Stable possession catch-all
        labels.append("STABLE_POSSESSION")

    return labels