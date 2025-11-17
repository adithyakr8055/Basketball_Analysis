#!/usr/bin/env python3
"""
tools/run_ocr_and_export.py

Run jersey OCR over video/stubs and export CSV + example crops.

Usage:
  python tools/run_ocr_and_export.py --video input_videos/video1.mp4 --tracks stubs/player_track_stubs.pkl \
      --out_csv out/ocr_results.csv --crop_dir out/ocr_crops --engine easyocr --sample_rate 1 --propagate 2 --smoothing True
"""
import argparse, os, pickle, csv
from pathlib import Path
import cv2
import sys

# ensure repo root import
folder = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(folder, "..")))

try:
    from utils import read_stub
    from utils.jersey_ocr import batch_recognize_over_video, crop_jersey_region, recognize_jersey_number_from_crop, recognize_jersey_number_on_frame
except Exception as e:
    print("ERROR importing jersey_ocr:", e)
    batch_recognize_over_video = None
    recognize_jersey_number_on_frame = None
    crop_jersey_region = None
    recognize_jersey_number_from_crop = None

# optional smoothing
try:
    from tools.ocr_smoothing import temporal_smooth_player_labels
    _HAS_SMOOTH = True
except Exception:
    _HAS_SMOOTH = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--out_csv", default="out/ocr_results.csv")
    p.add_argument("--crop_dir", default="out/ocr_crops")
    p.add_argument("--engine", default="easyocr", choices=["easyocr","pytesseract"])
    p.add_argument("--sample_rate", type=int, default=1, help="process every Nth frame")
    p.add_argument("--propagate", type=int, default=1, help="propagate sampled OCR to +/- this many frames")
    p.add_argument("--min_size_w", type=int, default=48)
    p.add_argument("--min_size_h", type=int, default=80)
    p.add_argument("--smoothing", action="store_true", help="apply temporal smoothing and save per-player mapping")
    return p.parse_args()

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    ok, fr = cap.read()
    while ok:
        frames.append(fr.copy())
        ok, fr = cap.read()
    cap.release()
    return frames

def filter_tracks_for_ocr(frames, tracks, min_w, min_h):
    out = []
    for i, t in enumerate(tracks):
        d = {}
        for pid, pdata in (t or {}).items():
            bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
            if not bbox:
                continue
            try:
                x1,y1,x2,y2 = [int(float(v)) for v in bbox[:4]]
                w = x2-x1; h = y2-y1
            except Exception:
                continue
            if w >= min_w and h >= min_h:
                d[pid] = pdata
        out.append(d)
    return out

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    Path(args.crop_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading frames...")
    frames = load_frames(args.video)
    print(f"[INFO] Loaded {len(frames)} frames")

    print("[INFO] Loading tracks...")
    with open(args.tracks, "rb") as f:
        tracks = pickle.load(f)

    # filter small crops
    filtered_tracks = filter_tracks_for_ocr(frames, tracks, args.min_size_w, args.min_size_h)

    n = len(frames)
    sample_indices = list(range(0, n, max(1, args.sample_rate)))
    print(f"[INFO] Sampling {len(sample_indices)} frames (every {args.sample_rate})")

    # We'll use per-frame results list
    frame_results = [dict() for _ in range(n)]

    # If batch_recognize_over_video exists we can use it directly on sampled frames
    if batch_recognize_over_video:
        sampled_frames = [frames[i] for i in sample_indices]
        sampled_tracks = [filtered_tracks[i] for i in sample_indices]
        print("[INFO] Running batch_recognize_over_video on sampled frames (may use easyocr/pytesseract)...")
        sampled_results = batch_recognize_over_video(sampled_frames, sampled_tracks, ocr_engine=args.engine, sample_rate=1, debug_save_dir=None)
        for idx, frame_idx in enumerate(sample_indices):
            frame_results[frame_idx] = sampled_results[idx] or {}
    else:
        print("[INFO] Running per-crop OCR fallback...")
        for i in sample_indices:
            players = filtered_tracks[i]
            for pid, pdata in (players or {}).items():
                bbox = pdata.get("bbox")
                crop = crop_jersey_region(frames[i], bbox, region="back")
                if crop is None:
                    frame_results[i][pid] = (None, 0.0)
                    continue
                num, conf = recognize_jersey_number_from_crop(crop, ocr_engine=args.engine)
                frame_results[i][pid] = (num, conf)

    # propagate sampled results to neighbors
    for fi in sample_indices:
        for p, v in (frame_results[fi] or {}).items():
            for d in range(1, args.propagate+1):
                if fi - d >= 0 and not frame_results[fi-d]:
                    frame_results[fi-d] = frame_results[fi].copy()
                if fi + d < n and not frame_results[fi+d]:
                    frame_results[fi+d] = frame_results[fi].copy()

    # Save crops and CSV
    with open(args.out_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["frame", "pid", "number", "conf", "bbox_w", "bbox_h"])
        for i in range(n):
            players = (filtered_tracks[i] or {})
            for pid, pdata in players.items():
                bbox = pdata.get("bbox")
                if not bbox:
                    continue
                x1,y1,x2,y2 = [int(float(v)) for v in bbox[:4]]
                w = x2-x1; h = y2-y1
                num_conf = frame_results[i].get(pid, (None, 0.0))
                num, conf = num_conf
                writer.writerow([i, pid, num if num is not None else "", float(conf)])
                # save crop image for visual debugging
                crop = crop_jersey_region(frames[i], bbox, region="back")
                if crop is not None:
                    outname = os.path.join(args.crop_dir, f"frame_{i:04d}_pid_{pid}.jpg")
                    try:
                        cv2.imwrite(outname, crop)
                    except Exception:
                        pass

    print("[INFO] Saved CSV:", args.out_csv)

    # run smoothing if requested and smoothing helper available
    if args.smoothing:
        if not _HAS_SMOOTH:
            print("[WARN] smoothing requested but utils.ocr_smoothing not found. Install or place utils/ocr_smoothing.py")
        else:
            # build per-frame simple structure: list of dict pid -> (num, conf)
            per_frame = frame_results
            mapping = temporal_smooth_player_labels(per_frame, min_confidence=0.25, min_occurrence=2, window_size=11)
            out_map_csv = os.path.splitext(args.out_csv)[0] + "_player_map.csv"
            with open(out_map_csv, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["pid","number"])
                for pid, num in mapping.items():
                    w.writerow([pid, num])
            print("[INFO] Saved player mapping:", out_map_csv)

if __name__ == "__main__":
    main()