# tools/distance_debugger.py
import os, sys, argparse, pickle
import cv2, numpy as np
from collections import Counter
from math import hypot

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../"))

def load_tracks(tracks_path):
    return pickle.load(open(tracks_path,"rb"))

def prompt_click_two(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Can't read video frame")
    pts = []
    def onmouse(event,x,y,flags,param):
        nonlocal pts
        if event==cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x),int(y)))
            print("clicked", pts[-1])
    name = "click baseline endpoints (left, right) then q"
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, onmouse)
    while True:
        tmp = frame.copy()
        for p in pts:
            cv2.circle(tmp, p, 6, (0,255,0), -1)
        cv2.imshow(name, tmp)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if len(pts) < 2:
        raise RuntimeError("Need two clicks")
    p0, p1 = pts[0], pts[1]
    px = hypot(p1[0]-p0[0], p1[1]-p0[1])
    return px, p0, p1

def compute_centers(tracks):
    per_pid = {}
    counts = Counter()
    for fi, fr in enumerate(tracks):
        if not fr: continue
        for pid_raw, pdata in fr.items():
            try:
                pid = int(pid_raw)
            except Exception:
                pid = pid_raw
            bbox = pdata.get("bbox")
            if not bbox: continue
            x1,y1,x2,y2 = [float(v) for v in bbox[:4]]
            cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
            per_pid.setdefault(pid, []).append((fi, cx, cy))
            counts[pid]+=1
    top = [pid for pid,_ in counts.most_common(6)]
    return per_pid, top

def apply_homography_points(pts, H):
    # pts: Nx2 array
    pts = np.asarray(pts)
    ones = np.ones((pts.shape[0],1))
    hom = np.hstack([pts, ones])
    trans = (H @ hom.T).T
    trans = trans[:, :2] / trans[:, 2:3]
    return trans

def path_distance(pts2d):
    d = 0.0
    for i in range(1,len(pts2d)):
        dx = pts2d[i][0]-pts2d[i-1][0]
        dy = pts2d[i][1]-pts2d[i-1][1]
        d += hypot(dx,dy)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--court_length_m", type=float, default=28.0)
    args = ap.parse_args()

    if not os.path.exists(args.video) or not os.path.exists(args.tracks):
        raise RuntimeError("Missing files.")

    tracks = load_tracks(args.tracks)
    per_pid, top = compute_centers(tracks)
    print("Top players:", top)

    # check homography stub
    H = None
    if os.path.exists("stubs/homography_stub.pkl"):
        try:
            H = pickle.load(open("stubs/homography_stub.pkl","rb"))
            H = np.array(H)
            print("Loaded homography shape:", H.shape)
        except Exception as e:
            print("Failed loading homography:", e)

    # get pixel baseline
    px_len, p0, p1 = prompt_click_two(args.video)
    print("Clicked pixel baseline length:", px_len)
    m_per_px = args.court_length_m / px_len
    print("Estimated meters per pixel (naive):", m_per_px)

    for pid in top:
        samples = per_pid.get(pid, [])
        if len(samples) < 2:
            print(f"pid {pid}: not enough positions ({len(samples)})")
            continue
        pts_px = [(cx,cy) for _,cx,cy in samples]
        if H is not None:
            try:
                mapped = apply_homography_points(pts_px, H)
                # If homography maps to meters already, compute distance in mapped space
                # We will print both raw pixel-dist and homography-dist
                d_px = path_distance(pts_px)
                d_hom = path_distance(mapped)
                print(f"pid {pid}: frames={len(pts_px)} d_px={d_px:.1f} px d_hom={d_hom:.2f} (homography units)")
                # also show naive px->m
                print(f"         naive meters = {d_px * m_per_px:.2f} m")
            except Exception as e:
                print("homography mapping failed:", e)
        else:
            d_px = path_distance(pts_px)
            print(f"pid {pid}: frames={len(pts_px)} d_px={d_px:.1f} px naive_meters={d_px * m_per_px:.2f} m")

    print("If distances are huge (e.g. 100+m), homography is missing or m_per_px is wrong due to perspective.")
    print("A correct approach: compute homography mapping image -> court coordinates (meters), then compute distances in court coords.")

if __name__ == "__main__":
    main()