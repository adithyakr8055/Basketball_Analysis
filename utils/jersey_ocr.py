"""
utils/jersey_ocr.py

Rewritten robust OCR utilities for jersey numbers (back-only optimized).

Features:
- Focus on BACK region cropping (configurable).
- Supports EasyOCR (preferred) and pytesseract fallback.
- Many preprocessing pipelines (CLAHE, adaptive threshold, sharpen, resize, invert, color mask).
- Deskewing of binary images.
- Multi-attempt OCR across preprocess variants and engines.
- Candidate scoring by (engine_confidence, bbox_area, digit_length) with sanity checks.
- Batch helpers and aggregation (same API as original).
- Optional debug saving of crops and preprocess variants to inspect inputs.

Usage:
  from utils import jersey_ocr
  num, conf = jersey_ocr.recognize_jersey_number_from_crop(crop)
  num, conf = jersey_ocr.recognize_jersey_number_on_frame(frame, bbox)
  per_frame = jersey_ocr.batch_recognize_over_video(frames, player_tracks)
  mapping = jersey_ocr.aggregate_player_numbers(per_frame, min_confidence=0.45, min_occurrence=2)

Note: this file intentionally does not force GPU usage for easyocr; keep `gpu=False` unless you explicitly want GPU.
"""

from typing import Tuple, Optional, Dict, List, Any
import cv2
import numpy as np
import os
from pathlib import Path
import math
import logging

# Try EasyOCR
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# Try pytesseract fallback
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

# Setup simple logger (module-level)
logger = logging.getLogger("jersey_ocr")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[jersey_ocr] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -------------------------
# Reader caching / helpers
# -------------------------
_easyocr_reader: Optional[Any] = None

def _ensure_reader(engine: str = "easyocr"):
    """
    Return initialized reader or None (for pytesseract).
    Raises runtime error if requested engine unavailable.
    """
    global _easyocr_reader
    engine = (engine or "easyocr").lower()
    if engine == "easyocr":
        if not _HAS_EASYOCR:
            raise RuntimeError("easyocr not installed. `pip install easyocr`")
        if _easyocr_reader is None:
            # Restrict to digits for speed/accuracy; GPU disabled by default
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return _easyocr_reader
    elif engine == "pytesseract":
        if not _HAS_PYTESSERACT:
            raise RuntimeError("pytesseract not installed. `pip install pytesseract` and system tesseract")
        return None
    else:
        raise ValueError("Unknown OCR engine: " + str(engine))


# -------------------------
# Preprocessing pipelines
# -------------------------
def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def _resize_keep_aspect(img: np.ndarray, min_dim: int = 64) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= min_dim and w >= min_dim:
        return img
    scale = max(min_dim / max(h, w), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def _preproc_variant(img: np.ndarray, variant: int = 0) -> np.ndarray:
    """
    variant list (defaults):
    0 - gray -> bilateral -> adaptiveThreshold (binary inverted)
    1 - CLAHE -> blur -> Otsu -> invert (good for dark shirts with light digits)
    2 - sharpen -> upscale -> Otsu
    3 - equalizeHist -> Otsu -> invert
    4 - contrast stretch -> adaptiveThreshold
    5 - color mask try (isolate light colors) -> morphological -> binary
    6 - simple resized grayscale (no heavy ops)
    """
    if img is None:
        return img
    img0 = img.copy()
    # ensure BGR or grayscale expected by operations
    if len(img0.shape) == 3:
        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    else:
        gray = img0

    if variant == 0:
        g = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 7)
        return _resize_keep_aspect(th, 64)

    if variant == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        blur = cv2.GaussianBlur(cl, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th)  # digits dark->light depending on engine
        return _resize_keep_aspect(th, 64)

    if variant == 2:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        up = _resize_keep_aspect(sharp, 128)
        _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    if variant == 3:
        equ = cv2.equalizeHist(gray)
        _, th = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th)
        return _resize_keep_aspect(th, 64)

    if variant == 4:
        # contrast stretch
        p2, p98 = np.percentile(gray, (2, 98))
        if p98 - p2 > 1:
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = gray
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th)
        return _resize_keep_aspect(th, 64)

    if variant == 5:
        # color mask attempt: isolate very light colors (white numbers)
        if len(img0.shape) == 3:
            hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
            # mask light regions: high V, low saturation
            mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            return _resize_keep_aspect(mask, 64)
        else:
            return _resize_keep_aspect(gray, 64)

    # default (6)
    return _resize_keep_aspect(gray, 64)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Deskew binary or gray images using minAreaRect heuristic."""
    if img is None or img.size == 0:
        return img
    try:
        g = _to_gray(img)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        if coords.shape[0] < 10:
            return img
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img


# -------------------------
# OCR wrappers
# -------------------------
def _ocr_with_easyocr(reader, img: np.ndarray, allow_digits_only: bool = True) -> List[Tuple[str, float]]:
    results = []
    if img is None or img.size == 0:
        return results
    try:
        # EasyOCR wants color images OR grayscale; pass as-is
        out = reader.readtext(img, detail=1)
        for bbox, text, conf in out:
            txt = (text or "").strip()
            if allow_digits_only:
                txt = "".join([c for c in txt if c.isdigit()])
                if txt == "":
                    continue
            results.append((txt, float(conf)))
    except Exception:
        # swallow exceptions to keep pipeline robust
        pass
    return results


def _ocr_with_pytesseract(img: np.ndarray, psm: int = 7, oem: int = 3, allow_digits_only: bool = True) -> List[Tuple[str, float]]:
    results = []
    if img is None or img.size == 0:
        return results
    try:
        config = f'--oem {oem} --psm {psm} -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        n = len(data.get('text', []))
        for i in range(n):
            txt = (data['text'][i] or "").strip()
            if txt == "":
                continue
            if allow_digits_only:
                txt = "".join([c for c in txt if c.isdigit()])
                if txt == "":
                    continue
            conf_raw = data.get('conf', [None] * n)[i]
            try:
                conf = float(conf_raw)
            except Exception:
                # fallback if conf cannot be parsed
                conf = 0.0
            if conf < 0:
                conf = 0.0
            results.append((txt, conf / 100.0))
    except Exception:
        pass
    return results


# -------------------------
# Candidate scoring
# -------------------------
def _score_candidate(text: str, conf: float, crop_area: float) -> float:
    """
    Compute a composite score for candidate (higher = better).
    - conf: engine-provided (0..1)
    - crop_area: pixel area of crop (bigger crops preferred)
    - text length preference: prefer 1-2 digits typical of jerseys
    Score is a weighted sum with saturations.
    """
    if text is None or text == "":
        return 0.0
    try:
        length = len(str(text))
    except Exception:
        length = 0
    length_score = 1.0 if 1 <= length <= 2 else max(0.25, 2.0 / (length + 1))
    # area influence: scale log-wise to avoid domination
    area_score = math.log(max(1.0, crop_area)) / 10.0
    conf_score = float(conf)
    # combine
    score = 0.6 * conf_score + 0.25 * length_score + 0.15 * min(area_score, 1.0)
    return float(score)


# -------------------------
# Cropping (BACK-oriented)
# -------------------------
def crop_jersey_region(frame: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       region: str = "back",
                       expand: float = 0.12) -> Optional[np.ndarray]:
    """
    Crop region likely to contain BACK jersey number.
    For back numbers we focus slightly lower than chest and include full back width.

    region choices:
      - "back" (default, optimized)
      - "center" (general)
      - "upper" (chest/front)

    expand: fraction of bbox width/height to expand to improve capture.
    """
    if frame is None or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
    except Exception:
        return None
    h, w = frame.shape[:2]
    # clamp
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if region == "upper":
        top = y1 + int(0.12 * bh)
        bottom = y1 + int(0.55 * bh)
        left = x1 + int(0.05 * bw)
        right = x2 - int(0.05 * bw)
    elif region == "center":
        top = y1 + int(0.25 * bh)
        bottom = y1 + int(0.75 * bh)
        left = x1 + int(0.05 * bw)
        right = x2 - int(0.05 * bw)
    else:  # "back"
        # For back numbers, place crop in middle-lower part of bbox.
        # Many players' back numbers sit around center-to-lower half.
        top = y1 + int(0.30 * bh)
        bottom = y1 + int(0.85 * bh)
        # include more horizontal margin to capture full number printed across back
        left = x1 - int(0.10 * bw)
        right = x2 + int(0.10 * bw)

    # apply expand padding
    pad_x = int(expand * bw) + 1
    pad_y = int(expand * bh) + 1
    lx = max(0, left - pad_x)
    rx = min(w, right + pad_x)
    ty = max(0, top - pad_y)
    by = min(h, bottom + pad_y)

    if rx <= lx or by <= ty:
        return None

    crop = frame[ty:by, lx:rx].copy()

    # ensure a reasonable size for OCR
    ch, cw = crop.shape[:2]
    if ch < 48 or cw < 48:
        try:
            crop = cv2.resize(crop, (max(64, cw * 2), max(64, ch * 2)), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
    return crop


# -------------------------
# Main recognition functions
# -------------------------
def recognize_jersey_number_from_crop(crop: np.ndarray,
                                      ocr_engine: str = "easyocr",
                                      max_attempts: int = 6,
                                      debug_save_dir: Optional[str] = None) -> Tuple[Optional[str], float]:
    """
    Recognize jersey number from a cropped image. Returns (number_str or None, confidence 0..1).
    - max_attempts: how many preprocessing variants to try (0..6)
    - debug_save_dir: optional path to save intermediate crops for inspection
    """
    if crop is None or crop.size == 0:
        return None, 0.0

    # Ensure reader if needed (easyocr) - fallback logic below
    engine_order = []
    if ocr_engine == "easyocr":
        if _HAS_EASYOCR:
            engine_order = ["easyocr", "pytesseract"] if _HAS_PYTESSERACT else ["easyocr"]
        elif _HAS_PYTESSERACT:
            engine_order = ["pytesseract"]
        else:
            raise RuntimeError("No OCR backend available. Install easyocr or pytesseract+tesseract.")
    elif ocr_engine == "pytesseract":
        if _HAS_PYTESSERACT:
            engine_order = ["pytesseract"]
        elif _HAS_EASYOCR:
            engine_order = ["easyocr"]
        else:
            raise RuntimeError("No OCR backend available.")
    else:
        # try both
        engine_order = []
        if _HAS_EASYOCR: engine_order.append("easyocr")
        if _HAS_PYTESSERACT: engine_order.append("pytesseract")
        if not engine_order:
            raise RuntimeError("No OCR backend available.")

    # Prepare debug folder
    debug_path = None
    if debug_save_dir:
        debug_path = Path(debug_save_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[str, float, float]] = []  # (text, conf, score)
    crop_area = float(max(1, crop.shape[0] * crop.shape[1]))

    # iterate preprocess variants
    max_attempts = max(1, min(max_attempts, 7))
    for variant in range(max_attempts):
        proc = _preproc_variant(crop, variant=variant)
        try:
            proc = _deskew(proc)
        except Exception:
            pass

        # optional debug save
        if debug_path is not None:
            try:
                cv2.imwrite(str(debug_path / f"variant_{variant}.png"), proc if len(proc.shape) == 2 else cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

        for engine in engine_order:
            if engine == "easyocr":
                if not _HAS_EASYOCR:
                    continue
                try:
                    reader = _ensure_reader("easyocr")
                    out = _ocr_with_easyocr(reader, proc, allow_digits_only=True)
                except Exception:
                    out = []
            else:  # pytesseract
                if not _HAS_PYTESSERACT:
                    continue
                out = _ocr_with_pytesseract(proc, psm=7, oem=3, allow_digits_only=True)

            for txt, conf in out:
                if not txt or txt.strip() == "":
                    continue
                # canonicalize digits: strip leading zeros but keep '0' if it was just zero
                txt_norm = txt.lstrip("0") or "0"
                try:
                    n = int(txt_norm)
                except Exception:
                    continue
                # only allow typical jersey numbers 0..99
                if not (0 <= n <= 99):
                    continue
                s = _score_candidate(txt_norm, conf, crop_area)
                candidates.append((str(int(n)), float(conf), float(s)))

    if not candidates:
        return None, 0.0

    # choose best candidate by composite score, tiebreaker by confidence
    candidates = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
    best_text, best_conf, best_score = candidates[0]

    # normalize final confidence: combine engine conf and score
    final_conf = float(min(1.0, best_conf * 0.85 + 0.15 * (best_score)))
    return str(best_text), float(final_conf)


def recognize_jersey_number_on_frame(frame: np.ndarray,
                                     player_bbox: Tuple[int, int, int, int],
                                     ocr_engine: str = "easyocr") -> Tuple[Optional[str], float]:
    """
    Convenience wrapper that crops the back region and runs recognition.
    """
    crop = crop_jersey_region(frame, player_bbox, region="back")
    if crop is None:
        return None, 0.0
    return recognize_jersey_number_from_crop(crop, ocr_engine=ocr_engine)


def batch_recognize_over_video(frames: List[np.ndarray],
                               player_tracks: List[Dict[int, Dict]],
                               ocr_engine: str = "easyocr",
                               sample_rate: int = 1,
                               debug_save_dir: Optional[str] = None) -> List[Dict[int, Tuple[Optional[str], float]]]:
    """
    Run jersey OCR for all frames and players.
      - sample_rate: process every Nth frame (default 1 = every frame)
      - debug_save_dir: optional folder to store crops (organized by frame/pid)
    Returns: list (len = len(frames)) of dict pid -> (num_or_None, conf)
    """
    n = len(frames)
    out = [dict() for _ in range(n)]

    debug_path = None
    if debug_save_dir:
        debug_path = Path(debug_save_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, n, max(1, sample_rate)):
        frame = frames[i]
        players = {}
        try:
            players = player_tracks[i] or {}
        except Exception:
            players = {}

        for pid, pdata in (players or {}).items():
            try:
                bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
                if not bbox:
                    out[i][pid] = (None, 0.0)
                    continue
                crop = crop_jersey_region(frame, bbox, region="back")
                if crop is None:
                    out[i][pid] = (None, 0.0)
                    continue
                # optional per-crop debug folder
                per_crop_dir = None
                if debug_path is not None:
                    per_crop_dir = debug_path / f"frame_{i}_pid_{pid}"
                    per_crop_dir.mkdir(parents=True, exist_ok=True)
                num, conf = recognize_jersey_number_from_crop(crop, ocr_engine=ocr_engine, debug_save_dir=str(per_crop_dir) if per_crop_dir else None)
                out[i][pid] = (num, conf)
            except Exception:
                out[i][pid] = (None, 0.0)
    # fill skipped frames (if sample_rate > 1) with empty dicts so length stays consistent
    return out


def aggregate_player_numbers(frame_results: List[Dict[int, Tuple[Optional[str], float]]],
                             min_confidence: float = 0.45,
                             min_occurrence: int = 2) -> Dict[int, str]:
    """
    Aggregate per-frame OCR outputs into stable player_id -> jersey_number mapping.
    Keeps numbers that have at least min_occurrence frames with conf >= min_confidence.
    """
    from collections import defaultdict, Counter
    occurrences = defaultdict(list)
    for fr in frame_results:
        for pid, (num, conf) in fr.items():
            if num is None:
                continue
            try:
                if float(conf) >= float(min_confidence):
                    occurrences[int(pid)].append(str(num))
            except Exception:
                continue

    final = {}
    for pid, nums in occurrences.items():
        if not nums:
            continue
        cnt = Counter(nums)
        number, count = cnt.most_common(1)[0]
        if count >= int(min_occurrence):
            final[int(pid)] = str(number)
    return final


# -------------------------
# small CLI-ish helper (not executed on import)
# -------------------------
if __name__ == "__main__":
    # quick smoke test if user runs module directly
    logger.setLevel(logging.DEBUG)
    import sys, pickle
    if len(sys.argv) < 3:
        logger.info("Usage: python jersey_ocr.py <video_path> <tracks_stub.pkl> [pid]")
        sys.exit(0)
    video = sys.argv[1]
    tracks = sys.argv[2]
    pid = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    # load frames
    cap = cv2.VideoCapture(video)
    frames = []
    ok, fr = cap.read()
    while ok:
        frames.append(fr.copy())
        ok, fr = cap.read()
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from {video}")

    # load tracks (list of dicts)
    with open(tracks, "rb") as f:
        tracks_list = pickle.load(f)
    logger.info(f"Loaded tracks (len={len(tracks_list)})")

    if pid is None:
        # run OCR for top players: print debug crop stats
        logger.info("Running quick OCR for first 50 frames for top players...")
        # collect top players by frequency
        counts = {}
        for i, fr_tracks in enumerate(tracks_list[:200]):
            if not fr_tracks:
                continue
            for p in fr_tracks.keys():
                counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:6]
        logger.info(f"Top players: {top}")
        pids = [p for p, _ in top]
    else:
        pids = [pid]

    # run for first 200 frames for each pid
    for p in pids:
        results = []
        for i in range(min(len(frames), len(tracks_list), 200)):
            frt = tracks_list[i] or {}
            pdata = frt.get(p)
            if not pdata:
                continue
            bbox = pdata.get("bbox")
            if not bbox:
                continue
            crop = crop_jersey_region(frames[i], bbox, region="back")
            num, conf = recognize_jersey_number_from_crop(crop, ocr_engine="easyocr", max_attempts=6)
            logger.info(f"frame={i} pid={p} -> {num} (conf={conf:.3f})")
            results.append((i, num, conf))
        logger.info(f"Found {len(results)} crops for pid {p}")
        from collections import Counter
        nums = [n for (_, n, c) in results if n is not None and c >= 0.3]
        logger.info(f"Counts: {Counter(nums)}")