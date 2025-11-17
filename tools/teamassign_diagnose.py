# tools/teamassign_diagnose.py
import os, sys, pickle, argparse
from collections import Counter, defaultdict
import cv2
import numpy as np

# ensure repo utils path if needed
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../"))

# import TeamAssigner (adjust import path if needed)
try:
    from team_assigner import TeamAssigner
except Exception as e:
    # try from utils or root module path
    try:
        from utils.team_assigner import TeamAssigner
    except Exception as ee:
        print("ERROR: cannot import TeamAssigner:", e, " / ", ee)
        raise

def load_frames(video_path, n=120):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max(1, n))
    idxs = list(range(0, total, step))[:n] if total>0 else list(range(n))
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames (sampled indices).")
    return frames

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_crop_image(crop, outpath):
    try:
        cv2.imwrite(outpath, crop)
    except Exception as e:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--stub_out", default="stubs/player_team_map.pkl")
    ap.add_argument("--frames_sample", type=int, default=120)
    ap.add_argument("--show_samples", type=int, default=30)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise RuntimeError("Video not found: " + args.video)
    if not os.path.exists(args.tracks):
        raise RuntimeError("Tracks not found: " + args.tracks)

    print("Loading frames...")
    frames = load_frames(args.video, n=args.frames_sample)
    print("Loading player tracks...")
    tracks = pickle.load(open(args.tracks, "rb"))

    # create assigner in fast mode (kmeans) to get quick reproducible results
    ta_fast = TeamAssigner(fast_mode=True)
    print("Running TeamAssigner in fast_mode (kmeans heuristic)...")
    per_frame_fast = ta_fast.get_player_teams_across_frames(frames, tracks, read_from_stub=False, stub_path=None, min_votes=1)

    # now collect votes by calling get_player_color directly for debugging
    votes_debug = defaultdict(list)
    sample_crops_dir = "tools/debug_crops"
    ensure_dir(sample_crops_dir)

    max_save = args.show_samples
    saved = 0
    for fi, frame_tracks in enumerate(tracks[:len(frames)]):
        if not frame_tracks:
            continue
        frame = frames[fi]
        for pid_raw, pdata in (frame_tracks or {}).items():
            try:
                pid = int(pid_raw)
            except Exception:
                pid = pid_raw
            bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
            if not bbox:
                continue
            label = ta_fast.get_player_color(frame, bbox, player_id=pid, frame_num=fi)
            votes_debug[pid].append(label)
            # save a few crops for visual inspection
            if saved < max_save:
                crop = ta_fast._crop_safe(frame, bbox, player_id=pid, frame_num=fi)
                out = os.path.join(sample_crops_dir, f"pid{pid}_f{fi}_{label}.jpg")
                save_crop_image(crop, out)
                saved += 1

    # summarize votes
    print("\nPer-player vote summary (fast_mode kmeans):")
    summary = []
    for pid, lbls in votes_debug.items():
        cnt = Counter(lbls)
        most, mostc = cnt.most_common(1)[0]
        summary.append((pid, len(lbls), dict(cnt), most, mostc))
    summary_sorted = sorted(summary, key=lambda x:-x[1])
    print("Top players by number of label votes (pid, votes_count, counts):")
    for item in summary_sorted[:30]:
        pid, n_votes, counts, most, mostc = item
        print(f"pid={pid:>3} votes={n_votes:>3} top_label={most} ({mostc}) counts={counts}")

    # check final mapping created by fast run
    print("\nPer-frame mapping example (frame 0,1,2):")
    for i in range(min(3, len(per_frame_fast))):
        print(f"frame {i}: {per_frame_fast[i]}")

    # Save fast-mode stable map for inspection
    try:
        stable = {}
        # re-derive stable map from votes_debug
        for pid, lbls in votes_debug.items():
            cnt = Counter(lbls)
            # map textual labels to team 1 or 2 using TeamAssigner defaults
            if cnt:
                top_label = cnt.most_common(1)[0][0]
                stable[pid] = 1 if top_label == ta_fast.team_1_class_name else 2
        pickle.dump(stable, open(args.stub_out, "wb"))
        print(f"\nSaved derived stable mapping to {args.stub_out} (size {len(stable)})")
    except Exception as e:
        print("Failed to save stable map:", e)

    print("\nSaved sample crops to:", sample_crops_dir)
    print("Review those images to see whether crops contain uniform colors or background.")

if __name__ == "__main__":
    main()