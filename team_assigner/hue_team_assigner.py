# team_assigner/hue_team_assigner.py
import numpy as np
from sklearn.cluster import KMeans
from utils.jersey_ocr import crop_jersey_region
import cv2
from collections import defaultdict

def assign_teams_by_hue(frames, player_tracks, sample_fraction=0.2):
    """
    frames: list of images
    player_tracks: list indexed by frame -> dict pid->pdata (with 'bbox')
    Returns: dict pid -> team_label (0 or 1)
    """
    per_player_hues = defaultdict(list)
    n = len(frames)
    step = max(1, int(1.0/(sample_fraction) if sample_fraction<1 else 1))
    # sample frames evenly
    indices = list(range(0,n, max(1, n//max(1,int(n*sample_fraction)))))
    for i in indices:
        frame = frames[i]
        players = player_tracks[i] if i < len(player_tracks) else {}
        for pid, pdata in (players or {}).items():
            bbox = pdata.get("bbox")
            if not bbox: continue
            crop = crop_jersey_region(frame, bbox)
            if crop is None: continue
            try:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h = hsv[:,:,0].ravel()
                h = h[h>0]
                if h.size == 0: continue
                per_player_hues[int(pid)].append(float(np.median(h)))
            except Exception:
                continue

    ids = []
    medians = []
    for pid, hs in per_player_hues.items():
        if len(hs) == 0: continue
        ids.append(int(pid))
        medians.append(float(np.median(hs)))

    if len(ids) < 2:
        # fallback: assign zeros
        return {pid: 0 for pid in ids}

    X = np.array(medians).reshape(-1,1)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    labels = {ids[i]: int(km.labels_[i]) for i in range(len(ids))}
    return labels