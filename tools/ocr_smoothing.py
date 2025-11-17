"""
utils/ocr_smoothing.py

Temporal smoothing & weighted-voting for per-frame OCR results.

Input format expected:
  per_frame_results: list of length = n_frames, each element is dict(pid -> (num_or_None, conf_float))

Output:
  mapping: dict pid -> stable_number (string)
"""
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import math

def temporal_smooth_player_labels(per_frame_results: List[Dict[Any, Tuple[Any, float]]],
                                 min_confidence: float = 0.3,
                                 min_occurrence: int = 2,
                                 window_size: int = 11,
                                 temporal_weighting: str = "gaussian") -> Dict[Any, str]:
    """
    - window_size: odd int; sliding window size for local voting (e.g. 11 -> +/-5 frames)
    - temporal_weighting: "uniform" or "linear" or "gaussian" (applies weights inside window)
    Returns pid -> number (string)
    """
    n = len(per_frame_results)
    pid_frames = defaultdict(list)  # pid -> list of (frame_idx, num, conf)

    # collect all candidates per pid
    for fi, fr in enumerate(per_frame_results):
        if not fr:
            continue
        for pid, (num, conf) in fr.items():
            if num is None:
                continue
            try:
                c = float(conf)
            except Exception:
                c = 0.0
            pid_frames[pid].append((fi, str(num), c))

    final = {}
    half = max(1, (window_size // 2))

    def weight_fn(offset):
        # offset is absolute distance |center - frame|
        if temporal_weighting == "uniform":
            return 1.0
        if temporal_weighting == "linear":
            return max(0.0, (half - offset + 1))
        # gaussian
        sigma = max(1.0, half / 2.0)
        return math.exp(- (offset**2) / (2 * sigma * sigma))

    for pid, entries in pid_frames.items():
        if not entries:
            continue
        # build per-frame candidate dict for this pid
        frame_map = defaultdict(list)
        for fi, num, conf in entries:
            frame_map[fi].append((num, conf))
        # sliding window voting: compute score per candidate across frames
        candidate_scores = defaultdict(float)
        occurrence_counts = defaultdict(int)
        for center in range(n):
            # accumulate votes in window centered at center
            start = max(0, center - half)
            end = min(n - 1, center + half)
            for f in range(start, end + 1):
                if f not in frame_map:
                    continue
                offset = abs(center - f)
                w = weight_fn(offset)
                for num, conf in frame_map[f]:
                    if conf < min_confidence:
                        continue
                    # candidate score contribution
                    candidate_scores[num] += (w * conf)
                    occurrence_counts[num] += 1
        if not candidate_scores:
            continue
        # pick top candidate by score, require min_occurrence
        best = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[0]
        best_num, best_score = best
        if occurrence_counts.get(best_num, 0) >= min_occurrence:
            final[pid] = best_num
    return final