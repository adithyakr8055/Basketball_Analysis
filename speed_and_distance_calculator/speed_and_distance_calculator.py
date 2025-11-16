# speed_and_distance_calculator.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import math

class SpeedAndDistanceCalculator:
    """
    Convert tactical pixel positions -> physical distances and speeds.

    Expects:
      - tactical_image_width_pixels, tactical_image_height_pixels
      - actual_width_in_meters, actual_height_in_meters
      - optional scale_factor to tune empirically (default 1.0)
    """

    def __init__(self,
                 tactical_image_width_pixels: int,
                 tactical_image_height_pixels: int,
                 actual_width_in_meters: float,
                 actual_height_in_meters: float,
                 scale_factor: float = 1.0,
                 smoothing_alpha: float = 0.4,
                 max_speed_m_s: float = 12.0):
        # tactical image dims (pixels)
        self.tactical_w_px = max(1, int(tactical_image_width_pixels))
        self.tactical_h_px = max(1, int(tactical_image_height_pixels))

        # real world dims (meters)
        self.actual_w_m = float(actual_width_in_meters)
        self.actual_h_m = float(actual_height_in_meters)

        self.scale_factor = float(scale_factor)
        # meters per pixel (use average of width/height mapping)
        self.m_per_px_x = (self.actual_w_m / self.tactical_w_px) * self.scale_factor
        self.m_per_px_y = (self.actual_h_m / self.tactical_h_px) * self.scale_factor
        # use isotropic scale as average
        self.m_per_px = (self.m_per_px_x + self.m_per_px_y) / 2.0

        # smoothing and outlier caps
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_speed_m_s = float(max_speed_m_s)

        # internal state for exponential smoothing per-player
        self._last_smoothed_speed = {}  # pid -> last smoothed m/s
        self._last_positions = {}      # pid -> (x_px, y_px) from previous frame

    def _euclidean_px(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def calculate_distance(self,
                           tactical_positions_per_frame: List[Optional[Dict[int, Tuple[float, float]]]]
                          ) -> List[Dict[int, float]]:
        """
        Returns per-frame per-player delta distances in meters (distance travelled since previous frame).
        tactical_positions_per_frame: list length N frames; each item is dict player_id -> (x_px, y_px)
        """
        n = len(tactical_positions_per_frame)
        out: List[Dict[int, float]] = [dict() for _ in range(n)]

        # reset internal state
        self._last_positions = {}

        for i in range(n):
            frame_positions = tactical_positions_per_frame[i] or {}
            distances = {}
            for pid, pos in frame_positions.items():
                try:
                    pid_i = int(pid)
                except Exception:
                    pid_i = pid
                # if we have last position, compute px distance
                last = self._last_positions.get(pid_i)
                try:
                    cur = (float(pos[0]), float(pos[1]))
                except Exception:
                    continue
                if last is None:
                    # no displacement (first observed frame)
                    distances[pid_i] = 0.0
                else:
                    px_d = self._euclidean_px(cur, last)
                    meters = px_d * self.m_per_px
                    # guard against NaN/inf
                    if not math.isfinite(meters) or meters < 0:
                        meters = 0.0
                    distances[pid_i] = float(meters)
                # update last position
                self._last_positions[pid_i] = cur
            out[i] = distances
        return out

    def calculate_speed(self,
                        player_distances_per_frame: List[Dict[int, float]],
                        fps: float = 30.0
                       ) -> List[Dict[int, float]]:
        """
        Convert per-frame distances (meters) -> per-frame speeds (m/s).
        Uses simple smoothing and caps unrealistic spikes.

        Returns list length N (frames): dict player_id -> speed_m_s
        """
        if fps <= 0:
            fps = 30.0
        dt = 1.0 / float(fps)
        n = len(player_distances_per_frame)
        out: List[Dict[int, float]] = [dict() for _ in range(n)]

        # initialize smoothing state
        self._last_smoothed_speed = {}

        for i in range(n):
            distances = player_distances_per_frame[i] or {}
            speeds_frame = {}
            for pid, dist_m in distances.items():
                try:
                    pid_i = int(pid)
                except Exception:
                    pid_i = pid
                try:
                    dist = float(dist_m)
                except Exception:
                    dist = 0.0
                # raw instantaneous speed (m/s)
                raw_v = dist / dt if dt > 0 else 0.0
                # cap extreme values before smoothing
                if not math.isfinite(raw_v) or raw_v < 0:
                    raw_v = 0.0
                # outlier rejection: if raw_v is absurdly large, clip to max_speed * 5 then smoothing will reduce
                if raw_v > self.max_speed_m_s * 5:
                    raw_v = self.max_speed_m_s * 5

                prev_sm = self._last_smoothed_speed.get(pid_i, raw_v)
                smoothed = (self.smoothing_alpha * raw_v) + ((1.0 - self.smoothing_alpha) * prev_sm)
                # final cap to realistic max
                if smoothed > self.max_speed_m_s:
                    smoothed = self.max_speed_m_s

                speeds_frame[pid_i] = float(smoothed)
                self._last_smoothed_speed[pid_i] = smoothed
            out[i] = speeds_frame
        return out