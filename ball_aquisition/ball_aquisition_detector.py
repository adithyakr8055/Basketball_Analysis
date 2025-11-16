# ball_aquisition.py
from typing import List, Dict, Tuple, Any, Optional
from utils.bbox_utils import measure_distance, get_center_of_bbox

class BallAquisitionDetector:
    """
    Detect ball acquisition (possession) by players in each frame.

    Input shapes:
      - player_tracks: list[ dict[player_id -> {'bbox': (x1,y1,x2,y2), ...}, ... ]  (len == num_frames)
      - ball_tracks:   list[ dict[ball_id -> {'bbox': (x1,y1,x2,y2), ...}, ... ]    (len == num_frames)

    Output:
      - possession_list: list[int] of length num_frames. value is player_id or -1 (no possession)
    """

    def __init__(self,
                 possession_threshold: int = 50,
                 min_frames: int = 11,
                 containment_threshold: float = 0.8):
        self.possession_threshold = possession_threshold
        self.min_frames = min_frames
        self.containment_threshold = containment_threshold

    def get_key_basketball_player_assignment_points(
        self,
        player_bbox: Tuple[int, int, int, int],
        ball_center: Tuple[float, float]
    ) -> List[Tuple[int, int]]:
        """
        Compute useful key points around player bbox to measure distances more robustly.
        """
        x1, y1, x2, y2 = player_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        bx, by = int(ball_center[0]), int(ball_center[1])

        output_points = []

        # Points aligned to ball row/column if inside
        if y1 < by < y2:
            output_points.append((x1, by))
            output_points.append((x2, by))
        if x1 < bx < x2:
            output_points.append((bx, y1))
            output_points.append((bx, y2))

        # canonical key points around bbox
        output_points += [
            (x1 + width // 2, y1),                 # top center
            (x2, y1),                              # top right
            (x1, y1),                              # top left
            (x2, y1 + height // 2),                # center right
            (x1, y1 + height // 2),                # center left
            (x1 + width // 2, y1 + height // 2),   # center
            (x2, y2),                              # bottom right
            (x1, y2),                              # bottom left
            (x1 + width // 2, y2),                 # bottom center
            (x1 + width // 2, y1 + max(1, height // 3)),  # mid-top center
        ]

        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for pt in output_points:
            if pt not in seen:
                seen.add(pt)
                uniq.append(pt)
        return uniq

    def calculate_ball_containment_ratio(
        self,
        player_bbox: Tuple[int, int, int, int],
        ball_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Fraction of the ball bbox area that intersects with player's bbox.
        Returns 0.0..1.0
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        # Normalize/guard
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
        bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return 0.0

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        ball_area = max(1, (bx2 - bx1) * (by2 - by1))  # avoid division by zero

        return float(intersection_area) / float(ball_area)

    def find_minimum_distance_to_ball(
        self,
        ball_center: Tuple[float, float],
        player_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Return minimum euclidean distance from ball center to the key points around player's bbox.
        If no key points, fall back to distance to bbox center.
        """
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        if not key_points:
            # fallback: center of player's bbox
            x1, y1, x2, y2 = player_bbox
            player_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            return measure_distance(ball_center, player_center)
        distances = [measure_distance(ball_center, pt) for pt in key_points]
        return min(distances) if distances else float('inf')

    def _choose_ball_for_frame(self, ball_frame_dict: Dict[Any, Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
        """
        From a dict of ball detections for a frame, pick one ball bbox to use.
        Simple strategy: choose the first ball; if multiple, choose the one with smallest bbox area.
        """
        if not ball_frame_dict:
            return None
        # ball_frame_dict: {ball_id: {'bbox': (...) , ...}, ...}
        candidates = []
        for b_id, info in ball_frame_dict.items():
            bbox = info.get('bbox')
            if not bbox:
                continue
            bx1, by1, bx2, by2 = bbox
            area = max(0, (bx2 - bx1) * (by2 - by1))
            candidates.append((area, bbox))
        if not candidates:
            return None
        # choose smallest area (likely the real ball)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def find_best_candidate_for_possession(
        self,
        ball_center: Tuple[float, float],
        player_tracks_frame: Dict[int, Dict[str, Any]],
        ball_bbox: Tuple[int, int, int, int]
    ) -> int:
        """
        Return best player_id candidate in this frame, or -1 if none.
        Priority:
         1) any player with containment >= containment_threshold (choose the one with smallest distance)
         2) otherwise, choose closest player whose min-distance < possession_threshold
        """
        if not player_tracks_frame:
            return -1

        high_containment = []
        regular = []

        for player_id, pinfo in player_tracks_frame.items():
            player_bbox = pinfo.get('bbox')
            if not player_bbox:
                continue
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_dist = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment >= self.containment_threshold:
                high_containment.append((player_id, min_dist, containment))
            else:
                regular.append((player_id, min_dist, containment))

        # choose among high containment: pick smallest distance (most likely touching/holding)
        if high_containment:
            best = min(high_containment, key=lambda x: x[1])  # (player_id, dist, containment)
            return int(best[0])

        # else choose closest within possession threshold
        if regular:
            best = min(regular, key=lambda x: x[1])
            if best[1] <= self.possession_threshold:
                return int(best[0])

        return -1

    def detect_ball_possession(
        self,
        player_tracks: List[Dict[int, Dict[str, Any]]],
        ball_tracks: List[Dict[int, Dict[str, Any]]]
    ) -> List[int]:
        """
        Main function. Walk frames and decide possession.

        Approach:
          - For each frame, select a ball bbox (if multiple)
          - Find best candidate player for that frame
          - Maintain consecutive counters per player; only when a player's counter reaches self.min_frames
            we mark possession. When a player becomes candidate, increment that player's counter and reset others.
          - Once possession is confirmed, we mark the current frame with that player, and also backfill previous frames
            in which the candidate was being accumulated (so output is more continuous).
        """
        num_frames = max(len(ball_tracks), len(player_tracks))
        possession = [-1] * num_frames
        consecutive_counts: Dict[int, int] = {}  # player_id -> consecutive frames as best candidate

        # Keep a ring/list of last candidate player ids to enable backfill when min_frames reached
        last_candidates: List[Optional[int]] = [None] * num_frames  # we will just use indices directly for backfill

        for f in range(num_frames):
            ball_frame = ball_tracks[f] if f < len(ball_tracks) else {}
            player_frame = player_tracks[f] if f < len(player_tracks) else {}

            chosen_ball_bbox = self._choose_ball_for_frame(ball_frame)
            if not chosen_ball_bbox:
                # no ball detected in this frame
                # reset candidate counts? here we reset consecutive_counts (no continuity when no ball)
                consecutive_counts = {}
                last_candidates[f] = None
                continue

            ball_center = get_center_of_bbox(chosen_ball_bbox)
            best_player = self.find_best_candidate_for_possession(ball_center, player_frame, chosen_ball_bbox)

            last_candidates[f] = best_player if best_player != -1 else None

            if best_player == -1:
                # no candidate in this frame, reset counters
                consecutive_counts = {}
                continue

            # increment chosen player's counter and reset others
            for pid in list(consecutive_counts.keys()):
                if pid != best_player:
                    consecutive_counts[pid] = 0
            consecutive_counts[best_player] = consecutive_counts.get(best_player, 0) + 1

            # if this player's consecutive count reached threshold, mark possession.
            if consecutive_counts[best_player] >= self.min_frames:
                # mark this frame
                possession[f] = best_player

                # backfill the previous min_frames-1 frames if they were unassigned and candidate was same
                backfill = self.min_frames - 1
                i = f - 1
                while backfill > 0 and i >= 0:
                    if possession[i] == -1 and last_candidates[i] == best_player:
                        possession[i] = best_player
                    i -= 1
                    backfill -= 1

                # (optionally) keep marking subsequent frames until candidate changes or ball lost;
                # here we'll continue loop: if next frames continue to have same best_player,
                # consecutive_counts will stay high and frames will be set when we reach min_frames again.
                # If you prefer to mark every following frame once possession started, uncomment below:
                # (but be conservative — we'll just mark when >= min_frames)
                # possession[f] = best_player

        return possession