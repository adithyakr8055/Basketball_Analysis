#!/usr/bin/env python3
import os, glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# adjust import path if needed; run from repo root
from utils import jersey_ocr

# folder where tools/debug_crops saved earlier
folder = 'tools/debug_crops'
if not os.path.exists(folder):
    print("No debug_crops folder:", folder); raise SystemExit(1)

img_paths = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
print("found", len(img_paths), "crops")

# Try both engines
engines = []
if getattr(jersey_ocr, "_HAS_EASYOCR", False):
    engines.append("easyocr")
if getattr(jersey_ocr, "_HAS_PYTESSERACT", False):
    engines.append("pytesseract")
if not engines:
    print("No OCR backends available in jersey_ocr module.")
    raise SystemExit(1)

for p in img_paths[:200]:
    import cv2
    crop = cv2.imread(p)
    print("----", os.path.basename(p))
    for e in engines:
        try:
            txt, conf = jersey_ocr.recognize_jersey_number_from_crop(crop, ocr_engine=e, max_attempts=6)
            print(" engine:", e, "->", txt, conf)
        except Exception as ex:
            print(" engine:", e, "EXC:", ex)