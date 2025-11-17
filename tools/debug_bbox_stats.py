#!/usr/bin/env python3
import pickle, os, numpy as np
from collections import defaultdict, Counter

st = 'stubs/player_track_stubs.pkl'
if not os.path.exists(st):
    print("stub not found:", st); raise SystemExit(1)
tracks = pickle.load(open(st,'rb'))

counts = Counter()
area_stats = defaultdict(list)
small_count = 0
total_crops = 0

for fi,fr in enumerate(tracks):
    if not fr: continue
    for pid,p in fr.items():
        bbox = p.get('bbox')
        if not bbox: continue
        x1,y1,x2,y2 = [float(v) for v in bbox[:4]]
        w = max(0, x2-x1); h = max(0, y2-y1)
        area = w*h
        counts[int(pid)] += 1
        area_stats[int(pid)].append((w,h,area,fi))
        total_crops += 1
        if w < 40 or h < 40:
            small_count += 1

print("total frames with any player:", len(tracks))
print("total bbox crops:", total_crops)
print("small crops (<40px dim):", small_count)
print("top players by frames:", counts.most_common(10))

# show sample sizes for top player
top = counts.most_common(1)[0][0]
print("\nsample sizes for player", top, "first 8 samples:")
for tpl in area_stats[top][:8]:
    print("w,h,area,frame:", tpl)