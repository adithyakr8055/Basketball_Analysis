# utils/debug_team_and_distance.py
import cv2, numpy as np, sys, os
from collections import defaultdict

# import functions from your repo (adjust path if needed)
from utils.jersey_ocr import crop_jersey_region
# Try to import your modules that produce player_tracks and tactical conversion
try:
    from tactical_view_converter.homography import pixel_to_court_coords  # if exists
except Exception:
    pixel_to_court_coords = None

def mean_hue_of_crop(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0].ravel()
        # remove zeros if background
        h = h[h>0]
        if len(h)==0:
            return None
        return float(np.median(h))
    except Exception:
        return None

def debug_team_hues(frames, player_tracks, sample_frames=50):
    # Collect per-player median hue across frames
    hues = {}
    for i, frame in enumerate(frames):
        if i % max(1, len(frames)//sample_frames) != 0:
            continue
        players = player_tracks[i] if i < len(player_tracks) else {}
        for pid,pdata in (players or {}).items():
            bbox = pdata.get("bbox")
            if not bbox: continue
            crop = crop_jersey_region(frame, bbox)
            if crop is None: continue
            mh = mean_hue_of_crop(crop)
            if mh is None: continue
            hues.setdefault(int(pid), []).append(mh)
    # aggregate
    medians = {pid: float(np.median(hs)) for pid,hs in hues.items() if len(hs)>0}
    print("Collected median hues for players:", medians)
    return medians

def quick_kmeans_labels(medians):
    # medians: pid -> hue value
    from sklearn.cluster import KMeans
    if len(medians) < 2:
        print("Not enough players for clustering")
        return {pid:0 for pid in medians.keys()}
    ids = list(medians.keys())
    X = np.array([medians[p] for p in ids]).reshape(-1,1)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    labels = {ids[i]: int(km.labels_[i]) for i in range(len(ids))}
    print("KMeans labels (pid -> label):", labels)
    return labels

def debug_distances_for_player(player_tracks, homography=None, tactical_px=(1000,500), court_m=(28.0,15.0)):
    # Show per-player raw pixel displacement and converted meters if homography provided.
    results = {}
    for pid in set().union(*[set(fr.keys()) for fr in player_tracks]):
        # gather centers per frame
        pts = []
        for fr in player_tracks:
            p = fr.get(pid)
            if not p: continue
            bbox = p.get("bbox")
            if not bbox: continue
            x1,y1,x2,y2 = [int(float(v)) for v in bbox[:4]]
            cx = (x1+x2)/2.0
            cy = (y1+y2)/2.0
            pts.append((cx,cy))
        if len(pts) < 2: continue
        # compute pixel displacements
        pix_dists = [np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1)]
        total_pix = float(np.sum(pix_dists))
        # convert using homography if provided
        total_m = None
        if homography is not None:
            # assume homography maps pixel -> tactical_image pixels sized tactical_px
            tactical_w, tactical_h = tactical_px
            court_w_m, court_h_m = court_m
            world_pts = cv2.perspectiveTransform(np.array([[p] for p in pts], dtype='float32'), homography).reshape(-1,2)
            world_m = [( (wp[0]/tactical_w)*court_w_m, (wp[1]/tactical_h)*court_h_m ) for wp in world_pts]
            m_dists = [np.hypot(world_m[i+1][0]-world_m[i][0], world_m[i+1][1]-world_m[i][1]) for i in range(len(world_m)-1)]
            total_m = float(np.sum(m_dists))
        results[pid] = {"n_samples":len(pts), "total_px":total_pix, "total_m":total_m}
    print("Distance debug (per player):")
    for pid,vals in results.items():
        print(pid, vals)
    return results

if __name__ == "__main__":
    # Example usage: needs frames list & player_tracks from your pipeline
    print("This is a helper - integrate into your run before/after tracking step.")
    print("Import frames (list of images) and player_tracks (list of dicts per frame) into this module and call debug functions.")