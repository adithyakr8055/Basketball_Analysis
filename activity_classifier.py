# activity_classifier.py
from typing import List, Dict, Optional, Tuple, Any
import math
import numpy as np

class ActivityClassifier:
    """
    Classify player as 'running' or 'standing' using tactical player positions (pixel coords).
    Heuristic:
      - compute smoothed speed (meters/sec using meters_per_pixel and fps)
      - speed > running_threshold_m_s => running, else standing
    """

    def __init__(self, meters_per_pixel: float, fps: float = 30.0, running_threshold_m_s: float = 3.0, smooth_alpha: float = 0.4):
        self.m_per_px = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else 0.03
        self.fps = float(fps) if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.running_threshold = float(running_threshold_m_s)
        self.alpha = float(smooth_alpha)
        self._last_smoothed_speed = {}  # pid -> m/s
        self._last_pos = {}            # pid -> (x_px,y_px)

    @staticmethod
    def _euclid(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def classify(self, tactical_positions: List[Optional[Dict[int, Tuple[float,float]]]]) -> List[Dict[int, str]]:
        n = len(tactical_positions)
        out = [dict() for _ in range(n)]
        self._last_smoothed_speed = {}
        self._last_pos = {}
        for i in range(n):
            frame_positions = tactical_positions[i] or {}
            frame_out = {}
            for pid, pos in frame_positions.items():
                try:
                    pid_i = int(pid)
                except Exception:
                    pid_i = pid
                try:
                    cur = (float(pos[0]), float(pos[1]))
                except Exception:
                    continue
                last = self._last_pos.get(pid_i)
                if last is None:
                    raw_v = 0.0
                else:
                    px_d = self._euclid(cur, last)
                    meters = px_d * self.m_per_px
                    raw_v = meters / self.dt if self.dt > 0 else 0.0
                    if raw_v < 0 or not math.isfinite(raw_v):
                        raw_v = 0.0
                prev = self._last_smoothed_speed.get(pid_i, raw_v)
                sm = self.alpha * raw_v + (1.0 - self.alpha) * prev
                self._last_smoothed_speed[pid_i] = sm
                self._last_pos[pid_i] = cur
                frame_out[pid_i] = "running" if sm >= self.running_threshold else "standing"
            out[i] = frame_out
        return out