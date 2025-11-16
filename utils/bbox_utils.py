"""
Utility functions for bounding box calculations.

This version is hardened for safety:
- Handles None bboxes
- Handles empty lists/tuples
- Handles invalid numeric values
- Ensures functions never crash the pipeline
"""

import math


def _safe_bbox(bbox):
    """
    Ensures bbox is valid and in (x1,y1,x2,y2) format.
    Returns None if invalid.
    """
    if bbox is None:
        return None
    
    if not isinstance(bbox, (list, tuple)):
        return None

    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except Exception:
        return None

    if any(math.isnan(v) for v in [x1, y1, x2, y2]):
        return None

    return x1, y1, x2, y2


def get_center_of_bbox(bbox):
    """
    Returns (cx, cy) or (0,0) if bbox invalid.
    """
    b = _safe_bbox(bbox)
    if b is None:
        return (0, 0)

    x1, y1, x2, y2 = b
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Returns width or 0 if invalid.
    """
    b = _safe_bbox(bbox)
    if b is None:
        return 0

    return max(0, b[2] - b[0])


def measure_distance(p1, p2):
    """
    Safe Euclidean distance between p1 and p2.
    """
    try:
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    except Exception:
        return 0.0


def measure_xy_distance(p1, p2):
    """
    Safe (dx, dy).
    """
    try:
        return p1[0] - p2[0], p1[1] - p2[1]
    except Exception:
        return (0, 0)


def get_foot_position(bbox):
    """
    Returns bottom center of bbox or (0,0) if invalid.
    """
    b = _safe_bbox(bbox)
    if b is None:
        return (0, 0)

    x1, y1, x2, y2 = b
    return int((x1 + x2) / 2), int(y2)