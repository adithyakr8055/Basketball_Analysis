"""
utils/jersey_ocr.py

Utilities to extract and recognize jersey numbers from player bounding boxes using OCR.

Features:
- Supports EasyOCR (preferred) and pytesseract fallback.
- Multiple preprocessing attempts to improve OCR reliability.
- Returns best candidate text plus a simple confidence score.
- Batch helper to run over frames & player_tracks and return mapping frame->player_id->(number,conf).
"""

from typing import Tuple, Optional, Dict, List
import cv2
import numpy as np
import math
import os

# Try to import EasyOCR first (better for digits on complex backgrounds)
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# pytesseract fallback
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

# Small helper to ensure at least one OCR backend available
if not (_HAS_EASYOCR or _HAS_PYTESSERACT):
    # Not raising here so module can be imported; functions will raise when used.
    pass

def _ensure_reader(engine: str = "easyocr"):
    """
    Return an initialized reader for engine. engine in {"easyocr","pytesseract"}.
    For easyocr we cache the reader on module level to avoid repeated loads.
    """
    global _easyocr_reader
    if engine == "easyocr":
        if not _HAS_EASYOCR:
            raise RuntimeError("easyocr not installed; install via `pip install easyocr`")
        try:
            if " _easyocr_reader" not in globals() or _easyocr_reader is None:
                # English digits model; restrict to digits for speed/accuracy
                _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            return _easyocr_reader
        except Exception as e:
            # propagate
            raise
    elif engine == "pytesseract":
        if not _HAS_PYTESSERACT:
            raise RuntimeError("pytesseract not installed; install via `pip install pytesseract` and system tesseract")
        return None
    else:
        raise ValueError("Unknown OCR engine: " + str(engine))


def _preprocess_for_ocr(img: np.ndarray, method: int = 0) -> np.ndarray:
    """
    Preprocess cropped jersey image for OCR. `method` selects different pipelines.
    Always returns a BGR or grayscale image suitable for OCR.
    Methods:
      0: basic gray + bilateral + adaptive threshold
      1: increase contrast (CLAHE) + threshold
      2: morphological opening + resize + sharpen
      3: color isolation (try to remove background by kmeans)
    """
    if img is None or img.size == 0:
        return img

    # Work on a copy
    im = img.copy()

    # Convert to gray
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if method == 0:
        # smooth and adaptive threshold
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 7)
        return th

    if method == 1:
        # CLAHE then Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(gray)
        blur = cv2.GaussianBlur(cl, (3,3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # invert such that digits are black on white for pytesseract (if needed)
        th = cv2.bitwise_not(th)
        return th

    if method == 2:
        # sharpen & morphological
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        resized = cv2.resize(sharp, (max(64, sharp.shape[1]*2), max(64, sharp.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
        _, th = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    if method == 3:
        # attempt to isolate bright numbers vs dark jersey via simple contrast stretch
        # normalize and equalize
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        equ = cv2.equalizeHist(norm)
        _, th = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th)
        return th

    # default: return grayscale
    return gray


def _deskew(img: np.ndarray) -> np.ndarray:
    """Try to deskew using minAreaRect if text is rotated slightly."""
    if img is None or img.size == 0:
        return img
    # Only works on binary image
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


def _ocr_with_easyocr(reader, img: np.ndarray, allow_digits_only=True) -> List[Tuple[str, float]]:
    """
    Run easyocr on img and return list of (text, confidence) candidates.
    We return one or more candidates as found by EasyOCR.
    """
    results = []
    if img is None or img.size == 0:
        return results
    # EasyOCR expects color images or grayscale fine
    try:
        out = reader.readtext(img, detail=1)  # returns list [(bbox, text, conf), ...]
        for bbox, text, conf in out:
            # postprocess text: keep digits and a bit of filtering
            txt = text.strip()
            if allow_digits_only:
                txt = "".join([c for c in txt if c.isdigit()])
                if txt == "":
                    continue
            results.append((txt, float(conf)))
    except Exception:
        pass
    return results


def _ocr_with_pytesseract(img: np.ndarray, psm: int = 7, oem: int = 3, allow_digits_only=True) -> List[Tuple[str, float]]:
    """
    Run pytesseract on image. Returns list of (text, pseudo_confidence).
    psm: page segmentation mode
    oem: OCR engine mode
    Note: pytesseract's confidences are trickier; we compute simple heuristic confidence.
    """
    results = []
    if img is None or img.size == 0:
        return results
    config = f'--oem {oem} --psm {psm} -c tessedit_char_whitelist=0123456789'
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        n = len(data['text'])
        for i in range(n):
            txt = data['text'][i].strip()
            if txt == "":
                continue
            if allow_digits_only:
                txt = "".join([c for c in txt if c.isdigit()])
                if txt == "":
                    continue
            # tesseract gives confidences as int in 'conf'
            conf = float(data['conf'][i]) if str(data['conf'][i]).replace('-', '').isdigit() else 0.0
            # normalize negative or weird conf
            if conf < 0:
                conf = 0.0
            results.append((txt, conf / 100.0))
    except Exception:
        pass
    return results


def recognize_jersey_number_from_crop(crop: np.ndarray,
                                      ocr_engine: str = "easyocr",
                                      max_attempts: int = 4) -> Tuple[Optional[str], float]:
    """
    Recognize jersey number from a cropped image (player's jersey region).

    Returns:
        best_text (str or None), confidence (0..1)
    """
    if crop is None or crop.size == 0:
        return None, 0.0

    # ensure reader
    # We don't raise here if backend missing; handle gracefully
    reader = None
    if ocr_engine == "easyocr" and _HAS_EASYOCR:
        reader = _ensure_reader("easyocr")
    elif ocr_engine == "pytesseract" and _HAS_PYTESSERACT:
        reader = _ensure_reader("pytesseract")
    else:
        # try easyocr first, then pytesseract
        if _HAS_EASYOCR:
            reader = _ensure_reader("easyocr")
            ocr_engine = "easyocr"
        elif _HAS_PYTESSERACT:
            reader = _ensure_reader("pytesseract")
            ocr_engine = "pytesseract"
        else:
            raise RuntimeError("No OCR backend available. Install easyocr or pytesseract + tesseract.")

    # Attempt multiple preprocess methods and deskewing to maximize chance
    candidates: List[Tuple[str, float]] = []
    for method in range(max_attempts):
        proc = _preprocess_for_ocr(crop, method=method)
        proc = _deskew(proc)

        if ocr_engine == "easyocr":
            out = _ocr_with_easyocr(reader, proc, allow_digits_only=True)
        else:
            out = _ocr_with_pytesseract(proc, psm=7, oem=3, allow_digits_only=True)

        # Collect results
        for t, c in out:
            if t is None or t == "":
                continue
            # canonicalize: remove leading zeros unless standalone zero
            t_norm = t.lstrip("0") or "0"
            candidates.append((t_norm, float(c)))

    if not candidates:
        return None, 0.0

    # choose best candidate by confidence then by digit-length (prefer 1-2 digits, common on jerseys)
    candidates = sorted(candidates, key=lambda x: (x[1], len(str(x[0]))), reverse=True)
    best_text, best_conf = candidates[0]
    # constrain to reasonable jersey numbers (1..99)
    try:
        n = int(best_text)
        if not (0 <= n <= 99):
            return None, 0.0
    except Exception:
        return None, 0.0
    return str(int(best_text)), float(best_conf)


def crop_jersey_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], region="upper") -> Optional[np.ndarray]:
    """
    Given a full player bbox (x1,y1,x2,y2), return a crop likely to contain the jersey number.
    Strategies:
      - Use upper half or center of bbox (numbers are usually on chest/back)
      - Optionally expand horizontally for visibility
    region: "upper" or "back" or "center"
    """
    if frame is None or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
    except Exception:
        return None
    h, w = frame.shape[:2]
    # clamp
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if region == "upper":
        # upper 40%-65% of bbox (chest/upper back)
        top = y1 + int(0.12 * bh)
        bottom = y1 + int(0.55 * bh)
        left = x1 + int(0.05 * bw)
        right = x2 - int(0.05 * bw)
    elif region == "center":
        top = y1 + int(0.25 * bh)
        bottom = y1 + int(0.75 * bh)
        left = x1 + int(0.05 * bw)
        right = x2 - int(0.05 * bw)
    else:
        # "back" fallback same as upper
        top = y1 + int(0.12 * bh)
        bottom = y1 + int(0.55 * bh)
        left = x1 + int(0.05 * bw)
        right = x2 - int(0.05 * bw)

    # expand a little
    pad_x = int(0.02 * bw) + 1
    pad_y = int(0.02 * bh) + 1
    lx = max(0, left - pad_x)
    rx = min(w, right + pad_x)
    ty = max(0, top - pad_y)
    by = min(h, bottom + pad_y)

    if rx <= lx or by <= ty:
        return None

    crop = frame[ty:by, lx:rx].copy()
    # if crop is tiny, resize to a reasonable size
    if crop.shape[0] < 24 or crop.shape[1] < 24:
        try:
            crop = cv2.resize(crop, (max(48, crop.shape[1]*2), max(48, crop.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
    return crop


def recognize_jersey_number_on_frame(frame: np.ndarray,
                                     player_bbox: Tuple[int, int, int, int],
                                     ocr_engine: str = "easyocr") -> Tuple[Optional[str], float]:
    """
    Convenience wrapper: crop an estimated jersey region and run OCR.
    Returns recognized number string (or None) and confidence 0..1.
    """
    crop = crop_jersey_region(frame, player_bbox, region="upper")
    if crop is None:
        return None, 0.0
    return recognize_jersey_number_from_crop(crop, ocr_engine=ocr_engine, max_attempts=4)


def batch_recognize_over_video(frames: List[np.ndarray],
                               player_tracks: List[Dict[int, Dict]],
                               ocr_engine: str = "easyocr") -> List[Dict[int, Tuple[Optional[str], float]]]:
    """
    Run jersey OCR for all frames and players.
    Returns list of length = len(frames) where each element is dict player_id -> (number_str_or_None, conf)
    This function is defensive and will skip invalid frames/tracks.
    """
    n = len(frames)
    out = [dict() for _ in range(n)]
    for i in range(n):
        frame = frames[i]
        players = {}
        try:
            players = player_tracks[i] or {}
        except Exception:
            players = {}
        for pid, pdata in (players or {}).items():
            try:
                bbox = pdata.get("bbox")
                if not bbox:
                    out[i][pid] = (None, 0.0)
                    continue
                num, conf = recognize_jersey_number_on_frame(frame, bbox, ocr_engine=ocr_engine)
                out[i][pid] = (num, conf)
            except Exception:
                out[i][pid] = (None, 0.0)
    return out


def aggregate_player_numbers(frame_results: List[Dict[int, Tuple[Optional[str], float]]],
                             min_confidence: float = 0.45,
                             min_occurrence: int = 2) -> Dict[int, str]:
    """
    Aggregate per-frame OCR outputs into stable player_id -> jersey_number mapping.
    Policy:
      - For each player_id, collect recognized numbers with confidences.
      - Keep numbers with confidence >= min_confidence.
      - Choose number with most occurrences across frames; require min_occurrence.
    Returns mapping player_id -> number_str (only if satisfied), else not present.
    """
    from collections import defaultdict, Counter
    occurrences = defaultdict(list)
    for fr in frame_results:
        for pid, (num, conf) in fr.items():
            if num is None:
                continue
            if conf >= min_confidence:
                occurrences[int(pid)].append(num)

    final = {}
    for pid, nums in occurrences.items():
        if not nums:
            continue
        cnt = Counter(nums)
        number, count = cnt.most_common(1)[0]
        if count >= min_occurrence:
            final[pid] = number
    return final