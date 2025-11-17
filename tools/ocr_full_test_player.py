#!/usr/bin/env python3
import sys, os

# add repo root to path so `from utils import jersey_ocr` works no matter where script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import pickle
from utils import jersey_ocr

# --- config ---
video = 'input_videos/video1.mp4'
tracks = 'stubs/player_track_stubs.pkl'
pid_to_test = 8  # change to a top player ID you want to test
# ----------------

# load frames (all)
cap = cv2.VideoCapture(video)
frames = []
ok, fr = cap.read()
while ok:
    frames.append(fr.copy())
    ok, fr = cap.read()
cap.release()
print("loaded frames:", len(frames))

# load tracks
tracks_list = pickle.load(open(tracks, 'rb'))

results = []
for i, fr in enumerate(frames):
    if i >= len(tracks_list):
        break
    pdata = tracks_list[i].get(pid_to_test) if tracks_list[i] else None
    if not pdata:
        continue
    bbox = pdata.get('bbox')
    if not bbox:
        continue
    num, conf = jersey_ocr.recognize_jersey_number_on_frame(fr, bbox, ocr_engine="easyocr")
    results.append((i, num, conf))

print("Found", len(results), "crops for pid", pid_to_test)
for r in results[:40]:
    print(r)

# Aggregate quick
from collections import Counter
nums = [n for (_, n, cf) in results if n is not None and cf >= 0.3]
print("Counts:", Counter(nums))