#!/usr/bin/env python3
"""
tools/ocr_crop_analyzer.py

Analyze a folder of debug crop images (variant_*.png or frame_pid_*.jpg), and produce
automatic suggestions (resize thresholds, color-mask suggestions, variant tweaks).
"""
import os, sys
from pathlib import Path
import cv2
import numpy as np
import argparse
import math

def analyze_image_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    h, w = g.shape[:2]
    mean = float(np.mean(g))
    std = float(np.std(g))
    minv = int(np.min(g)); maxv = int(np.max(g))
    area = h*w
    return {"w":w,"h":h,"mean":mean,"std":std,"min":minv,"max":maxv,"area":area}

def scan_folder(folder):
    files = sorted(Path(folder).glob("**/*"))
    stats = []
    for p in files:
        if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg"]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            st = analyze_image_stats(img)
            st["path"] = str(p)
            stats.append(st)
    return stats

def summarize_stats(stats):
    if not stats:
        print("[INFO] No images found")
        return
    areas = [s["area"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    hs = [s["h"] for s in stats]
    ws = [s["w"] for s in stats]
    print("Total crops:", len(stats))
    print("W range:", min(ws), "->", max(ws), "median:", int(np.median(ws)))
    print("H range:", min(hs), "->", max(hs), "median:", int(np.median(hs)))
    print("Area median:", int(np.median(areas)))
    print("Brightness mean:", float(np.mean(means)), "std:", float(np.std(means)))
    print("Pixel std (contrast) mean:", float(np.mean(stds)))

    # suggestions
    if np.median(ws) < 64 or np.median(hs) < 64:
        print("SUGGESTION: many crops are small. Increase min_width/min_height (or change resize thresholds).")
    if np.mean(means) < 50:
        print("SUGGESTION: crops are dark - try CLAHE/contrast stretch or color mask for bright digits.")
    if np.mean(means) > 200:
        print("SUGGESTION: crops are very bright - try threshold invert or color mask for dark digits.")
    if np.mean(stds) < 20:
        print("SUGGESTION: low contrast crops - try sharpening or increase CLAHE clip limit.")
    print("\nSample crops (first 10):")
    for s in stats[:10]:
        print(s["path"], "w,h:", s["w"], s["h"], "mean:", round(s["mean"],1), "std:", round(s["std"],1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop_folder", default="tools/debug_crops")
    args = p.parse_args()
    stats = scan_folder(args.crop_folder)
    summarize_stats(stats)

if __name__ == "__main__":
    main()