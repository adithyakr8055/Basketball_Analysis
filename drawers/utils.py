# utils.py
"""
Drawing helpers used by the drawers:
- draw_traingle (kept misspelled to match repo imports)
- draw_ellipse

This module expects bbox coordinates in either:
- (x1, y1, x2, y2)  OR
- (x, y, w, h)

The helper `_normalize_bbox` converts either form to (x1,y1,x2,y2).
"""

import cv2
import numpy as np

# try relative import first (works when utils is a package), otherwise absolute
try:
    from .bbox_utils import (
        get_center_of_bbox,
        get_bbox_width,
        measure_distance,
        measure_xy_distance,
        get_foot_position,
    )
except Exception:
    try:
        # second try for different import paths
        from utils.bbox_utils import (
            get_center_of_bbox,
            get_bbox_width,
            measure_distance,
            measure_xy_distance,
            get_foot_position,
        )
    except Exception:
        # fallback safe stubs so module won't completely break during import;
        # real functions should exist in utils/bbox_utils.py in your repo.
        def get_center_of_bbox(bbox):
            # bbox -> (x1,y1,x2,y2) assumed
            if not bbox:
                return (0, 0)
            try:
                x1, y1, x2, y2 = bbox
                return int((x1 + x2) / 2), int((y1 + y2) / 2)
            except Exception:
                return (0, 0)

        def get_bbox_width(bbox):
            if not bbox:
                return 0
            try:
                x1, y1, x2, y2 = bbox
                return abs(int(x2 - x1))
            except Exception:
                return 0

        def get_foot_position(bbox):
            # returns bottom center
            if not bbox:
                return (0, 0)
            try:
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                return (cx, int(y2))
            except Exception:
                return (0, 0)

        def measure_distance(p1, p2):
            try:
                return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
            except Exception:
                return 0.0

        def measure_xy_distance(p1, p2):
            try:
                return p1[0] - p2[0], p1[1] - p2[1]
            except Exception:
                return 0, 0


def _normalize_bbox(bbox):
    """
    Normalize bbox into (x1, y1, x2, y2).
    Accepts:
      - (x1,y1,x2,y2)
      - (x,y,w,h)
      - list/np.array of either
    Returns tuple of ints (x1,y1,x2,y2) or None if invalid.
    """
    if bbox is None:
        return None
    try:
        b = list(bbox)
    except Exception:
        return None
    if len(b) != 4:
        return None

    # Unpack floats safely
    try:
        x0, y0, x1, y1 = map(float, b)
    except Exception:
        return None

    width_guess = x1 - x0
    height_guess = y1 - y0

    # If width_guess < 0 or height_guess < 0, assume (x,y,w,h) where w/h might be negative-ish
    if width_guess < 0 or height_guess < 0:
        x, y, w, h = int(x0), int(y0), int(abs(x1)), int(abs(y1))
        return (x, y, x + w, y + h)

    # Heuristic: if width_guess smaller than x0 (likely x,y,w,h)
    try:
        if width_guess < x0 or height_guess < y0:
            x, y, w, h = int(x0), int(y0), int(width_guess), int(height_guess)
            return (x, y, x + w, y + h)
    except Exception:
        pass

    # Otherwise assume (x1,y1) are bottom-right coords
    return (int(x0), int(y0), int(x1), int(y1))


def draw_traingle(frame, bbox, color=(0, 255, 0)):
    """
    Draws a filled triangle "pointer" above the bbox bottom center.
    Returns the frame (modified in-place and also returned).
    """
    if frame is None:
        return frame
    nb = _normalize_bbox(bbox)
    if nb is None:
        return frame

    x1, y1, x2, y2 = nb
    cx = int((x1 + x2) / 2)
    # place triangle base at the bottom (y2) and tip upward
    base_y = int(y2)
    pts = np.array(
        [[cx, base_y], [cx - 12, base_y - 24], [cx + 12, base_y - 24]],
        dtype=np.int32,
    )

    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED, lineType=cv2.LINE_AA)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return frame


def draw_ellipse(frame, bbox, color=(255, 255, 255), track_id=None):
    """
    Draws an ellipse at the foot position of bbox and optionally a filled rectangle
    with the track_id above the ellipse.

    - Works with (x1,y1,x2,y2) or (x,y,w,h)
    - Does not raise on invalid input; returns original frame.
    """
    if frame is None:
        return frame
    nb = _normalize_bbox(bbox)
    if nb is None:
        return frame

    x1, y1, x2, y2 = nb
    cx = int((x1 + x2) / 2)
    width = max(1, int(x2 - x1))
    height = max(1, int(0.35 * width))
    foot_y = int(y2)

    # ellipse center at bottom center (slightly below the bbox bottom for visibility)
    center = (cx, foot_y + 5)
    axes = (int(width / 2), height)

    # draw shadow / filled ellipse then border
    cv2.ellipse(
        frame,
        center=center,
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=cv2.FILLED,
        lineType=cv2.LINE_AA,
    )
    cv2.ellipse(
        frame,
        center=center,
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(0, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    if track_id is not None:
        # small rounded rectangle above the ellipse
        rect_w = 42
        rect_h = 20
        x1r = cx - rect_w // 2
        y1r = foot_y - rect_h - 6
        x2r = x1r + rect_w
        y2r = y1r + rect_h
        cv2.rectangle(frame, (int(x1r), int(y1r)), (int(x2r), int(y2r)), color, cv2.FILLED)
        txt_x = int(x1r + rect_w * 0.18)
        txt_y = int(y1r + rect_h * 0.72)
        cv2.putText(
            frame, str(track_id), (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA
        )

    return frame