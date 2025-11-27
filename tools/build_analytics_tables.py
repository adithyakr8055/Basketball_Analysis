#!/usr/bin/env python3
"""
build_analytics_tables.py

Generate **professional-level analytics CSVs** from stubs + (optional) predictive actions.

Outputs (in analytics/):

1) enhanced_play_analysis.csv
   Per-frame, context-rich data:
     - frame
     - possession_id
     - possession_frame_index        # index of this frame within the current possession
     - offense_team, defense_team
     - ball_handler_id
     - ball_handler_jersey
     - ball_handler_team
     - handler_zone                  # COURT_ZONE_* (perimeter/paint/wing/corner/etc.)
     - offense_spacing_m             # average distance between attacking teammates
     - defensive_pressure_m          # distance to nearest defender (smaller = tight pressure)
     - numerical_advantage           # (#attackers in frontcourt - #defenders in frontcourt)
     - is_fast_break                 # 1 if fast-break-like context, else 0
     - event_type                    # NONE / PASS / INTERCEPTION
     - possession_change             # 1 if new holder vs previous frame (and both valid)
     - tactical_x, tactical_y        # handler location on tactical court
     - speed_mps, distance_delta_m
     - predicted_action              # from predictive_actions.csv if available
     - predicted_confidence          # from predictive CSV if present; else blank
     - context_risk_index            # 0..1 (higher = more turnover / interception risk)
     - next_5f_event                 # what happens in next 5 frames: PASS / INTERCEPTION / SHOT / NONE

2) player_interaction_network.csv
   Aggregated directed edges between players:
     from_player_id, from_jersey, from_team,
     to_player_id,   to_jersey,   to_team,
     event_type,     # PASS / INTERCEPTION
     event_count

3) player_summary.csv
   Player-level summary:
     player_id, jersey, team,
     total_possession_frames,
     passes_made,
     passes_received,
     interceptions_made,
     avg_speed_mps,
     total_distance_m,
     usage_rate,               # possession_frames / total_frames_seen
     on_court_frames           # frames where player appears in assignment

4) possession_summary.csv
   Possession-level analytics:
     possession_id,
     team,
     start_frame,
     end_frame,
     duration_frames,
     duration_seconds (approx, if fps known in stub; else None),
     total_passes,
     total_interceptions_against,
     avg_spacing_m,
     avg_defensive_pressure_m,
     avg_context_risk_index,
     num_shots_in_possession,
     terminal_event_type       # SHOT / INTERCEPTION / END_NO_EVENT

5) team_summary.csv
   Team-level aggregates:
     team_id,
     total_possessions,
     avg_possession_length_frames,
     avg_possession_length_seconds,
     total_passes,
     total_interceptions_made,
     total_interceptions_against,
     total_shots,
     avg_spacing_m,
     avg_defensive_pressure_m

6) predictive_patterns.csv
   Simple pattern mining around predicted_action vs actual events:
     predicted_action,
     count_frames,
     count_possessions,
     next_5f_passes,
     next_5f_interceptions,
     next_5f_shots,
     avg_context_risk_index

Run from repo root (or via main.py auto-call):

    python tools/build_analytics_tables.py
"""

import os
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils import read_stub  # type: ignore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _load_stub(name: str, stubs_dir: str = "stubs", default=None):
    path = os.path.join(stubs_dir, name)
    try:
        data = read_stub(True, path)
        if data is None:
            return default
        return data
    except Exception as e:
        print(f"[WARN] Failed to load stub {path}: {e}")
        return default


def _safe_len(x, fallback=0):
    try:
        return len(x)
    except Exception:
        return fallback


def _ensure_len(lst, target_len, filler):
    """Defensively resize list to target_len."""
    if not isinstance(lst, list):
        return [filler] * target_len
    if len(lst) < target_len:
        return lst + [filler] * (target_len - len(lst))
    if len(lst) > target_len:
        return lst[:target_len]
    return lst


def _compute_court_bounds(tactical_positions: List[Dict[int, Tuple[float, float]]]):
    """Get approximate min/max x,y on tactical court from all frames."""
    xs, ys = [], []
    for fr in tactical_positions:
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


def _normalize(val, vmin, vmax, eps: float = 1e-6):
    if vmin == vmax:
        return 0.5
    return (float(val) - float(vmin)) / (float(vmax - vmin) + eps)


def _classify_zone(x: Optional[float], y: Optional[float],
                   xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    """
    Map (x,y) to a coarse court zone label.
    Zones: PERIMETER_TOP, WING_LEFT, WING_RIGHT, CORNER_LEFT, CORNER_RIGHT, PAINT, UNKNOWN
    """
    if x is None or y is None:
        return "COURT_ZONE_UNKNOWN"
    try:
        nx = _normalize(x, xmin, xmax)
        ny = _normalize(y, ymin, ymax)
    except Exception:
        return "COURT_ZONE_UNKNOWN"

    # Define "paint" as center-ish horizontally, lower/mid vertically
    in_center_x = 0.3 <= nx <= 0.7
    in_paint_y = 0.3 <= ny <= 0.7

    if in_center_x and in_paint_y:
        return "COURT_ZONE_PAINT"

    # Corners: bottom-left and bottom-right
    if ny > 0.75:
        if nx < 0.25:
            return "COURT_ZONE_CORNER_LEFT"
        if nx > 0.75:
            return "COURT_ZONE_CORNER_RIGHT"

    # Wings: left/right mid area
    if 0.25 <= ny <= 0.75:
        if nx < 0.3:
            return "COURT_ZONE_WING_LEFT"
        if nx > 0.7:
            return "COURT_ZONE_WING_RIGHT"

    # Top perimeter
    if ny < 0.25:
        return "COURT_ZONE_PERIMETER_TOP"

    return "COURT_ZONE_OTHER"


def _pairwise_avg_distance(positions: List[Tuple[float, float]]) -> Optional[float]:
    """Average pairwise distance; returns None if <2 positions."""
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
    """Distance from handler to nearest opponent (defender)."""
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
    """
    Crude numerical advantage:
      (#offense players in "frontcourt") - (#defense players in "frontcourt")

    We approximate "frontcourt" as x > median x on tactical court (relative).
    """
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
    """
    Build a possession_id list the same length as ball_acquisition.
    possession_id increments whenever the possessing team changes or we go from -1→valid or valid→-1.
    """
    n = len(ball_acquisition)
    possession_ids = [-1] * n
    current_possession = -1
    current_team = -1

    def _team_of(holder: int, frame_assign: Dict[int, int]) -> int:
        return frame_assign.get(holder, -1)

    # team info is not available here yet; we’ll adjust later when building rows.
    # For now, we segment simply by holder change including -1 transitions.
    current_possession = 0
    last_holder = ball_acquisition[0] if n > 0 else -1

    for i in range(n):
        holder = ball_acquisition[i]
        if i == 0:
            possession_ids[i] = current_possession if holder != -1 else -1
            last_holder = holder
            continue

        if holder == -1:
            # no possession -> mark as -1
            possession_ids[i] = -1
        else:
            if last_holder == -1:
                # new possession begins
                current_possession += 1
                possession_ids[i] = current_possession
            else:
                if holder != last_holder:
                    # holder changed -> new possession
                    current_possession += 1
                possession_ids[i] = current_possession
        last_holder = holder

    return possession_ids


def _compute_context_risk_index(def_pressure: Optional[float],
                                spacing: Optional[float],
                                numerical_advantage: int) -> float:
    """
    Heuristic 0..1 risk index:
      - Higher when defensive_pressure is high (distance small),
      - Higher when spacing is bad (small),
      - Higher when numerical advantage is negative.
    """
    # Normalize defensive pressure: smaller distance -> higher risk
    if def_pressure is None:
        dp_score = 0.3
    else:
        # for distances 0..8 (tunable), map to [1..0]
        dp = float(def_pressure)
        dp_score = max(0.0, min(1.0, 1.0 - dp / 8.0))

    # Spacing: small spacing => higher risk
    if spacing is None:
        sp_score = 0.4
    else:
        sp = float(spacing)
        # typical spacing 3..10; below 3 is very tight, above 10 is very spread
        if sp <= 3:
            sp_score = 1.0
        elif sp >= 10:
            sp_score = 0.0
        else:
            sp_score = 1.0 - (sp - 3.0) / 7.0

    # Numerical advantage: negative => more risk
    if numerical_advantage <= -2:
        adv_score = 1.0
    elif numerical_advantage == -1:
        adv_score = 0.75
    elif numerical_advantage == 0:
        adv_score = 0.5
    elif numerical_advantage == 1:
        adv_score = 0.3
    else:
        adv_score = 0.15  # strong advantage

    # Combine with weights
    risk = 0.5 * dp_score + 0.3 * sp_score + 0.2 * adv_score
    return float(max(0.0, min(1.0, risk)))


# -------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------
def build_analytics(
    stubs_dir: str = "stubs",
    analytics_dir: str = "analytics",
    predictive_csv: str = "output_csv/predictive_actions.csv",
):
    os.makedirs(analytics_dir, exist_ok=True)

    print("[INFO] Loading core stubs...")

    # Core timelines (per frame)
    ball_acquisition: List[int] = _load_stub("ball_acquisition_stub.pkl", stubs_dir, default=[])
    player_assignment: List[Optional[Dict[int, int]]] = _load_stub("player_assignment_stub.pkl", stubs_dir, default=[])
    tactical_positions: List[Optional[Dict[int, Tuple[float, float]]]] = _load_stub("tactical_player_positions_stub.pkl", stubs_dir, default=[])
    player_distances: List[Dict[int, float]] = _load_stub("player_distances_stub.pkl", stubs_dir, default=[])
    player_speeds: List[Dict[int, float]] = _load_stub("player_speeds_stub.pkl", stubs_dir, default=[])
    passes: List[int] = _load_stub("passes_stub.pkl", stubs_dir, default=[])
    interceptions: List[int] = _load_stub("interceptions_stub.pkl", stubs_dir, default=[])
    shots: List[Any] = _load_stub("shots_stub.pkl", stubs_dir, default=[])

    # OCR results & jersey map (optional)
    ocr_results: List[Dict[int, Tuple[Optional[str], float]]] = _load_stub("jersey_ocr_results_stub.pkl", stubs_dir, default=[])
    jersey_number_map: Dict[int, str] = _load_stub("jersey_numbers_map_stub.pkl", stubs_dir, default={})

    # Approx FPS if stored somewhere (optional)
    # If you later dump fps into a stub, you can read it here; for now, we'll not guess.
    approx_fps = None  # leave None -> durations in seconds will be blank

    # Determine number of frames to process
    lengths = [
        _safe_len(ball_acquisition),
        _safe_len(player_assignment),
        _safe_len(tactical_positions),
        _safe_len(player_distances),
        _safe_len(player_speeds),
        _safe_len(passes),
        _safe_len(interceptions),
        _safe_len(shots),
    ]
    n_frames = max(lengths) if lengths else 0
    if n_frames == 0:
        print("[ERROR] No timeline data found in stubs. Have you run main.py?")
        return

    print(f"[INFO] Using n_frames = {n_frames}")

    # Normalize lists to same length (defensive)
    ball_acquisition = _ensure_len(ball_acquisition, n_frames, -1)
    player_assignment = _ensure_len(player_assignment, n_frames, {})
    tactical_positions = _ensure_len(tactical_positions, n_frames, {})
    player_distances = _ensure_len(player_distances, n_frames, {})
    player_speeds = _ensure_len(player_speeds, n_frames, {})
    passes = _ensure_len(passes, n_frames, -1)
    interceptions = _ensure_len(interceptions, n_frames, -1)
    ocr_results = _ensure_len(ocr_results, n_frames, {})
    shots = _ensure_len(shots, n_frames, False)

    # ----------------------------------------------------------------
    # Load predictive actions CSV (if exists)
    # ----------------------------------------------------------------
    predictive_by_frame: Dict[int, Dict[str, Any]] = {}

    if os.path.exists(predictive_csv):
        print(f"[INFO] Loading predictive actions from {predictive_csv}")
        with open(predictive_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_idx = int(row.get("frame", row.get("frame_idx", -1)))
                except Exception:
                    continue
                predictive_by_frame[frame_idx] = row
    else:
        print(f"[WARN] Predictive actions CSV not found at {predictive_csv}. "
              "predicted_action will be 'UNKNOWN'.")

    # ----------------------------------------------------------------
    # Precompute tactical bounds for zone classification
    # ----------------------------------------------------------------
    # Convert None entries to {} for bounds computation
    norm_tact = [fr if isinstance(fr, dict) else {} for fr in tactical_positions]
    xmin, xmax, ymin, ymax = _compute_court_bounds(norm_tact)
    print(f"[INFO] Tactical bounds approx: xmin={xmin:.2f}, xmax={xmax:.2f}, "
          f"ymin={ymin:.2f}, ymax={ymax:.2f}")

    # ----------------------------------------------------------------
    # Possession segmentation
    # ----------------------------------------------------------------
    possession_ids = _segment_possessions(ball_acquisition)

    # ----------------------------------------------------------------
    # Prepare containers
    # ----------------------------------------------------------------
    enhanced_rows: List[Dict[str, Any]] = []
    edge_counter = Counter()
    possession_stats = defaultdict(lambda: {
        "team": -1,
        "start_frame": None,
        "end_frame": None,
        "frames": [],
        "passes": 0,
        "interceptions_against": 0,
        "shots": 0,
        "spacing_vals": [],
        "pressure_vals": [],
        "risk_vals": [],
    })

    # Pattern mining for predicted_action
    predictive_pattern_counter = defaultdict(lambda: {
        "frames": 0,
        "possessions": set(),
        "next_5f_passes": 0,
        "next_5f_interceptions": 0,
        "next_5f_shots": 0,
        "risk_sum": 0.0,
    })

    # ----------------------------------------------------------------
    # Core frame-by-frame loop
    # ----------------------------------------------------------------
    for i in range(n_frames):
        holder = ball_acquisition[i]
        prev_holder = ball_acquisition[i - 1] if i > 0 else -1

        frame_assign = player_assignment[i] or {}
        frame_tactical = tactical_positions[i] or {}
        frame_speed = player_speeds[i] or {}
        frame_dist = player_distances[i] or {}

        # possession metadata
        poss_id = possession_ids[i]
        poss_index = 0
        if poss_id != -1:
            # index within possession: count previous frames with same poss_id
            poss_index = sum(1 for k in range(0, i + 1) if possession_ids[k] == poss_id)

        # offense/defense team
        offense_team = frame_assign.get(holder, -1) if holder != -1 else -1
        defense_team = 1 if offense_team == 2 else 2 if offense_team == 1 else -1

        # predicted action from CSV
        pred_row = predictive_by_frame.get(i, {})
        predicted_action = pred_row.get("predicted_action", "UNKNOWN")
        predicted_confidence = pred_row.get("confidence", pred_row.get("prob", ""))

        # jersey info for handler
        jersey = ""
        if holder != -1:
            if jersey_number_map:
                jersey = jersey_number_map.get(holder, "") or ""
            if not jersey:
                fr_ocr = ocr_results[i] or {}
                val = fr_ocr.get(holder)
                if isinstance(val, (tuple, list)) and len(val) >= 1:
                    jersey = val[0] or ""

        # tactical pos & zone
        tx, ty = None, None
        if holder != -1 and holder in frame_tactical:
            try:
                tx, ty = frame_tactical[holder]
            except Exception:
                tx, ty = None, None
        handler_zone = _classify_zone(tx, ty, xmin, xmax, ymin, ymax)

        # offense spacing
        off_positions = []
        if offense_team in (1, 2):
            for pid, (x, y) in frame_tactical.items():
                if frame_assign.get(pid, -1) == offense_team:
                    off_positions.append((x, y))
        offense_spacing = _pairwise_avg_distance(off_positions)

        # defensive pressure
        def_pressure = _nearest_defender_distance(holder, frame_tactical, frame_assign, offense_team)

        # numerical advantage
        num_adv = _numerical_advantage(frame_tactical, frame_assign, offense_team)

        # simple fast-break heuristic: high handler speed + numerical advantage in front
        speed_mps = frame_speed.get(holder)
        is_fast_break = 0
        try:
            if speed_mps is not None and float(speed_mps) > 4.5 and num_adv >= 1:
                is_fast_break = 1
        except Exception:
            is_fast_break = 0

        # distance delta
        distance_delta_m = frame_dist.get(holder)

        # context risk index (0..1)
        context_risk_index = _compute_context_risk_index(def_pressure, offense_spacing, num_adv)

        # event_type + receiver
        event_type = "NONE"
        receiver_id = -1
        receiver_jersey = ""
        receiver_team = -1
        sender_id = -1

        if passes[i] not in (-1, 0) and holder != -1 and prev_holder != -1 and holder != prev_holder:
            event_type = "PASS"
            receiver_id = holder
            sender_id = prev_holder
        elif interceptions[i] not in (-1, 0) and holder != -1 and prev_holder != -1 and holder != prev_holder:
            event_type = "INTERCEPTION"
            receiver_id = holder
            sender_id = prev_holder

        if receiver_id != -1:
            receiver_team = frame_assign.get(receiver_id, -1)
            if jersey_number_map:
                receiver_jersey = jersey_number_map.get(receiver_id, "") or ""
            if not receiver_jersey:
                fr_ocr = ocr_results[i] or {}
                val = fr_ocr.get(receiver_id)
                if isinstance(val, (tuple, list)) and len(val) >= 1:
                    receiver_jersey = val[0] or ""

            if sender_id != -1:
                edge_counter[(sender_id, receiver_id, event_type)] += 1

        possession_change = 1 if (i > 0 and holder != prev_holder and holder != -1 and prev_holder != -1) else 0

        # look ahead next 5 frames to see what happens soon
        lookahead = 5
        next_event = "NONE"
        for j in range(i + 1, min(n_frames, i + 1 + lookahead)):
            if shots[j]:
                next_event = "SHOT"
                break
            if interceptions[j] not in (-1, 0):
                next_event = "INTERCEPTION"
                break
            if passes[j] not in (-1, 0):
                next_event = "PASS"
                break

        # Register possession-level aggregates
        if poss_id != -1:
            ps = possession_stats[poss_id]
            if ps["start_frame"] is None:
                ps["start_frame"] = i
            ps["end_frame"] = i
            ps["frames"].append(i)
            if offense_team in (1, 2) and ps["team"] == -1:
                ps["team"] = offense_team
            if event_type == "PASS":
                ps["passes"] += 1
            if event_type == "INTERCEPTION":
                ps["interceptions_against"] += 1
            if shots[i]:
                ps["shots"] += 1
            if offense_spacing is not None:
                ps["spacing_vals"].append(offense_spacing)
            if def_pressure is not None:
                ps["pressure_vals"].append(def_pressure)
            ps["risk_vals"].append(context_risk_index)

        # Predictive pattern aggregates
        pa = predicted_action or "UNKNOWN"
        pattern_bucket = predictive_pattern_counter[pa]
        pattern_bucket["frames"] += 1
        if poss_id != -1:
            pattern_bucket["possessions"].add(poss_id)
        if next_event == "PASS":
            pattern_bucket["next_5f_passes"] += 1
        elif next_event == "INTERCEPTION":
            pattern_bucket["next_5f_interceptions"] += 1
        elif next_event == "SHOT":
            pattern_bucket["next_5f_shots"] += 1
        pattern_bucket["risk_sum"] += context_risk_index

        # Build per-frame row
        row = {
            "frame": i,
            "possession_id": poss_id,
            "possession_frame_index": poss_index,
            "offense_team": offense_team,
            "defense_team": defense_team,

            "ball_handler_id": holder,
            "ball_handler_jersey": jersey,
            "ball_handler_team": offense_team,
            "handler_zone": handler_zone,

            "offense_spacing_m": offense_spacing if offense_spacing is not None else "",
            "defensive_pressure_m": def_pressure if def_pressure is not None else "",
            "numerical_advantage": num_adv,
            "is_fast_break": is_fast_break,

            "event_type": event_type,
            "possession_change": possession_change,

            "tactical_x": tx if tx is not None else "",
            "tactical_y": ty if ty is not None else "",

            "speed_mps": speed_mps if speed_mps is not None else "",
            "distance_delta_m": distance_delta_m if distance_delta_m is not None else "",

            "predicted_action": predicted_action,
            "predicted_confidence": predicted_confidence,
            "context_risk_index": round(context_risk_index, 3),

            "receiver_id": receiver_id,
            "receiver_jersey": receiver_jersey,
            "receiver_team": receiver_team,

            "next_5f_event": next_event,
        }

        enhanced_rows.append(row)

    # ----------------------------------------------------------------
    # Write enhanced_play_analysis.csv
    # ----------------------------------------------------------------
    enhanced_csv_path = os.path.join(analytics_dir, "enhanced_play_analysis.csv")
    print(f"[INFO] Writing enhanced play analysis to {enhanced_csv_path}")

    fieldnames = [
        "frame",
        "possession_id",
        "possession_frame_index",
        "offense_team",
        "defense_team",
        "ball_handler_id",
        "ball_handler_jersey",
        "ball_handler_team",
        "handler_zone",
        "offense_spacing_m",
        "defensive_pressure_m",
        "numerical_advantage",
        "is_fast_break",
        "event_type",
        "possession_change",
        "tactical_x",
        "tactical_y",
        "speed_mps",
        "distance_delta_m",
        "predicted_action",
        "predicted_confidence",
        "context_risk_index",
        "receiver_id",
        "receiver_jersey",
        "receiver_team",
        "next_5f_event",
    ]

    with open(enhanced_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in enhanced_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    # Build interaction network CSV
    # ----------------------------------------------------------------
    network_csv_path = os.path.join(analytics_dir, "player_interaction_network.csv")
    print(f"[INFO] Writing player interaction network to {network_csv_path}")

    network_rows = []
    for (from_pid, to_pid, etype), count in edge_counter.items():
        if etype not in ("PASS", "INTERCEPTION"):
            continue

        from_team = -1
        to_team = -1
        from_jersey = jersey_number_map.get(from_pid, "")
        to_jersey = jersey_number_map.get(to_pid, "")

        # Infer team from any frame where they appear
        for i in range(n_frames):
            fa = player_assignment[i] or {}
            if from_team == -1 and from_pid in fa:
                from_team = fa[from_pid]
            if to_team == -1 and to_pid in fa:
                to_team = fa[to_pid]
            if from_team != -1 and to_team != -1:
                break

        network_rows.append({
            "from_player_id": from_pid,
            "from_jersey": from_jersey,
            "from_team": from_team,
            "to_player_id": to_pid,
            "to_jersey": to_jersey,
            "to_team": to_team,
            "event_type": etype,
            "event_count": count,
        })

    network_fieldnames = [
        "from_player_id",
        "from_jersey",
        "from_team",
        "to_player_id",
        "to_jersey",
        "to_team",
        "event_type",
        "event_count",
    ]

    with open(network_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=network_fieldnames)
        writer.writeheader()
        for r in network_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    # Build per-player summary CSV (UPGRADED)
    # ----------------------------------------------------------------
    summary_csv_path = os.path.join(analytics_dir, "player_summary.csv")
    print(f"[INFO] Writing player summary to {summary_csv_path}")

    all_players = set()
    for fr_assign in player_assignment:
        if isinstance(fr_assign, dict):
            all_players.update(fr_assign.keys())
    for holder in ball_acquisition:
        if holder != -1:
            all_players.add(holder)

    possession_frames = Counter()
    passes_made = Counter()
    passes_received = Counter()
    interceptions_made = Counter()
    speed_sum = Counter()
    speed_count = Counter()
    distance_sum = Counter()
    on_court_frames = Counter()

    for i in range(n_frames):
        holder = ball_acquisition[i]
        prev_holder = ball_acquisition[i - 1] if i > 0 else -1

        fr_assign = player_assignment[i] or {}
        fr_speed = player_speeds[i] or {}
        fr_dist = player_distances[i] or {}

        # on-court frames (player appears in assignment or tactical)
        for pid in fr_assign.keys():
            on_court_frames[pid] += 1

        # possession frames
        if holder != -1:
            possession_frames[holder] += 1

        # speed & distance per holder
        if holder in fr_speed:
            try:
                v = float(fr_speed[holder])
                speed_sum[holder] += v
                speed_count[holder] += 1
            except Exception:
                pass
        if holder in fr_dist:
            try:
                d = float(fr_dist[holder])
                distance_sum[holder] += d
            except Exception:
                pass

        # passes & interceptions
        if passes[i] not in (-1, 0) and holder != -1 and prev_holder != -1 and holder != prev_holder:
            passes_made[prev_holder] += 1
            passes_received[holder] += 1
        if interceptions[i] not in (-1, 0) and holder != -1 and prev_holder != -1 and holder != prev_holder:
            interceptions_made[holder] += 1

    summary_rows = []
    for pid in sorted(all_players):
        jersey = jersey_number_map.get(pid, "")
        team = -1
        for i in range(n_frames):
            fa = player_assignment[i] or {}
            if pid in fa:
                team = fa[pid]
                break

        avg_speed = 0.0
        if speed_count[pid] > 0:
            avg_speed = speed_sum[pid] / speed_count[pid]

        oc_frames = on_court_frames[pid]
        poss_frames = possession_frames[pid]
        usage_rate = 0.0
        if oc_frames > 0:
            usage_rate = poss_frames / float(oc_frames)

        row = {
            "player_id": pid,
            "jersey": jersey,
            "team": team,
            "total_possession_frames": poss_frames,
            "passes_made": passes_made[pid],
            "passes_received": passes_received[pid],
            "interceptions_made": interceptions_made[pid],
            "avg_speed_mps": round(avg_speed, 3) if avg_speed > 0 else 0.0,
            "total_distance_m": round(distance_sum[pid], 3) if distance_sum[pid] > 0 else 0.0,
            "usage_rate": round(usage_rate, 3),
            "on_court_frames": oc_frames,
        }
        summary_rows.append(row)

    summary_fieldnames = [
        "player_id",
        "jersey",
        "team",
        "total_possession_frames",
        "passes_made",
        "passes_received",
        "interceptions_made",
        "avg_speed_mps",
        "total_distance_m",
        "usage_rate",
        "on_court_frames",
    ]

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    # Possession summary CSV
    # ----------------------------------------------------------------
    possession_csv_path = os.path.join(analytics_dir, "possession_summary.csv")
    print(f"[INFO] Writing possession summary to {possession_csv_path}")

    possession_rows = []
    for poss_id, ps in possession_stats.items():
        if not ps["frames"]:
            continue
        start_frame = ps["start_frame"]
        end_frame = ps["end_frame"]
        duration_frames = (end_frame - start_frame + 1) if (start_frame is not None and end_frame is not None) else 0
        if approx_fps is not None:
            duration_seconds = duration_frames / float(approx_fps)
        else:
            duration_seconds = ""

        avg_spacing = sum(ps["spacing_vals"]) / len(ps["spacing_vals"]) if ps["spacing_vals"] else ""
        avg_pressure = sum(ps["pressure_vals"]) / len(ps["pressure_vals"]) if ps["pressure_vals"] else ""
        avg_risk = sum(ps["risk_vals"]) / len(ps["risk_vals"]) if ps["risk_vals"] else ""

        # terminal event
        terminal_event = "END_NO_EVENT"
        last_frame = ps["end_frame"]
        if last_frame is not None:
            if shots[last_frame]:
                terminal_event = "SHOT"
            elif interceptions[last_frame] not in (-1, 0):
                terminal_event = "INTERCEPTION"
            elif passes[last_frame] not in (-1, 0):
                terminal_event = "PASS"

        possession_rows.append({
            "possession_id": poss_id,
            "team": ps["team"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds,
            "total_passes": ps["passes"],
            "total_interceptions_against": ps["interceptions_against"],
            "avg_spacing_m": avg_spacing,
            "avg_defensive_pressure_m": avg_pressure,
            "avg_context_risk_index": round(avg_risk, 3) if avg_risk != "" else "",
            "num_shots_in_possession": ps["shots"],
            "terminal_event_type": terminal_event,
        })

    possession_fieldnames = [
        "possession_id",
        "team",
        "start_frame",
        "end_frame",
        "duration_frames",
        "duration_seconds",
        "total_passes",
        "total_interceptions_against",
        "avg_spacing_m",
        "avg_defensive_pressure_m",
        "avg_context_risk_index",
        "num_shots_in_possession",
        "terminal_event_type",
    ]

    with open(possession_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=possession_fieldnames)
        writer.writeheader()
        for r in possession_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    # Team summary CSV
    # ----------------------------------------------------------------
    team_csv_path = os.path.join(analytics_dir, "team_summary.csv")
    print(f"[INFO] Writing team summary to {team_csv_path}")

    team_stats = defaultdict(lambda: {
        "possessions": 0,
        "duration_frames_sum": 0,
        "passes": 0,
        "interceptions_made": 0,        # from interaction edges
        "interceptions_against": 0,
        "shots": 0,
        "spacing_sum": 0.0,
        "spacing_count": 0,
        "pressure_sum": 0.0,
        "pressure_count": 0,
    })

    # Add from possession table
    for row in possession_rows:
        team_id = row["team"]
        if team_id not in (1, 2):
            continue
        ts = team_stats[team_id]
        ts["possessions"] += 1
        ts["duration_frames_sum"] += (row["duration_frames"] or 0)
        ts["passes"] += row["total_passes"] or 0
        ts["interceptions_against"] += row["total_interceptions_against"] or 0
        ts["shots"] += row["num_shots_in_possession"] or 0
        if row["avg_spacing_m"] not in ("", None):
            ts["spacing_sum"] += float(row["avg_spacing_m"])
            ts["spacing_count"] += 1
        if row["avg_defensive_pressure_m"] not in ("", None):
            ts["pressure_sum"] += float(row["avg_defensive_pressure_m"])
            ts["pressure_count"] += 1

    # Interceptions made (from edges)
    for (from_pid, to_pid, etype), cnt in edge_counter.items():
        if etype != "INTERCEPTION":
            continue
        # to_pid is receiver (interceptor), find its team
        rec_team = -1
        for i in range(n_frames):
            fa = player_assignment[i] or {}
            if to_pid in fa:
                rec_team = fa[to_pid]
                break
        if rec_team in (1, 2):
            team_stats[rec_team]["interceptions_made"] += cnt

    team_rows = []
    for team_id, ts in sorted(team_stats.items()):
        poss = ts["possessions"]
        avg_frames = (ts["duration_frames_sum"] / poss) if poss > 0 else 0
        if approx_fps is not None:
            avg_secs = avg_frames / float(approx_fps)
        else:
            avg_secs = ""
        avg_spacing = (ts["spacing_sum"] / ts["spacing_count"]) if ts["spacing_count"] > 0 else ""
        avg_pressure = (ts["pressure_sum"] / ts["pressure_count"]) if ts["pressure_count"] > 0 else ""

        team_rows.append({
            "team_id": team_id,
            "total_possessions": poss,
            "avg_possession_length_frames": avg_frames,
            "avg_possession_length_seconds": avg_secs,
            "total_passes": ts["passes"],
            "total_interceptions_made": ts["interceptions_made"],
            "total_interceptions_against": ts["interceptions_against"],
            "total_shots": ts["shots"],
            "avg_spacing_m": avg_spacing,
            "avg_defensive_pressure_m": avg_pressure,
        })

    team_fieldnames = [
        "team_id",
        "total_possessions",
        "avg_possession_length_frames",
        "avg_possession_length_seconds",
        "total_passes",
        "total_interceptions_made",
        "total_interceptions_against",
        "total_shots",
        "avg_spacing_m",
        "avg_defensive_pressure_m",
    ]

    with open(team_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=team_fieldnames)
        writer.writeheader()
        for r in team_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    # Predictive patterns CSV
    # ----------------------------------------------------------------
    predictive_csv_path = os.path.join(analytics_dir, "predictive_patterns.csv")
    print(f"[INFO] Writing predictive patterns to {predictive_csv_path}")

    pred_rows = []
    for action, stats in predictive_pattern_counter.items():
        frames = stats["frames"]
        poss_count = len(stats["possessions"])
        if frames <= 0:
            continue
        avg_risk = stats["risk_sum"] / float(frames)
        pred_rows.append({
            "predicted_action": action,
            "count_frames": frames,
            "count_possessions": poss_count,
            "next_5f_passes": stats["next_5f_passes"],
            "next_5f_interceptions": stats["next_5f_interceptions"],
            "next_5f_shots": stats["next_5f_shots"],
            "avg_context_risk_index": round(avg_risk, 3),
        })

    pred_fieldnames = [
        "predicted_action",
        "count_frames",
        "count_possessions",
        "next_5f_passes",
        "next_5f_interceptions",
        "next_5f_shots",
        "avg_context_risk_index",
    ]

    with open(predictive_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
        writer.writeheader()
        for r in pred_rows:
            writer.writerow(r)

    # ----------------------------------------------------------------
    print("[INFO] Analytics CSV generation complete.")
    print(f"  - {enhanced_csv_path}")
    print(f"  - {network_csv_path}")
    print(f"  - {summary_csv_path}")
    print(f"  - {possession_csv_path}")
    print(f"  - {team_csv_path}")
    print(f"  - {predictive_csv_path}")


if __name__ == "__main__":
    build_analytics()