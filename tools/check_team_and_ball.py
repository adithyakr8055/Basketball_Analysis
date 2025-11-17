# tools/check_team_and_ball.py
import os, pickle, argparse
from collections import Counter, defaultdict

def load(path):
    return pickle.load(open(path, "rb"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="stubs/player_track_stubs.pkl")
    ap.add_argument("--team_map", default="stubs/player_team_map.pkl")
    ap.add_argument("--team_tracks", default="stubs/player_track_stubs_with_team.pkl")
    ap.add_argument("--ball_stub", default="stubs/team_ball_control_stub.pkl")
    args = ap.parse_args()

    for p in [args.tracks, args.team_map]:
        if not os.path.exists(p):
            print("MISSING:", p)
        else:
            print("FOUND:", p)

    # examine tracks
    if os.path.exists(args.tracks):
        tracks = load(args.tracks)
        total_frames = len(tracks)
        print("Frames in tracks:", total_frames)
        per_frame_team_counts = []
        any_team_field = False
        pid_types = set()
        for i, fr in enumerate(tracks[:min(len(tracks), 120)]):
            cnt = Counter()
            if not fr:
                per_frame_team_counts.append(cnt)
                continue
            for pid, pdata in fr.items():
                pid_types.add(type(pid))
                if isinstance(pdata, dict):
                    t = pdata.get("team", pdata.get("team_id", None))
                    if t is not None:
                        any_team_field = True
                        cnt[int(t)] += 1
            per_frame_team_counts.append(cnt)
        print("Sample frame team counts (first 5):")
        for i,c in enumerate(per_frame_team_counts[:5]):
            print(f" frame {i}: {dict(c)}")
        print("Any 'team' field present in original tracks?:", any_team_field)
        print("Player ID types in tracks (sample):", pid_types)

    # check whether team-injected stub exists (prior script output)
    if os.path.exists(args.team_tracks):
        print("Injected team tracks found at:", args.team_tracks)
    else:
        print("No injected-team tracks found:", args.team_tracks)

    # check ball control stub
    if os.path.exists(args.ball_stub):
        print("Ball control stub found:", args.ball_stub)
        try:
            d = load(args.ball_stub)
            print("Ball control stub type:", type(d))
            # try to print a small sample
            if isinstance(d, list):
                print("length:", len(d), "sample (0):", d[0] if len(d)>0 else None)
            else:
                sample_keys = list(d.keys())[:10]
                print("sample keys:", sample_keys)
        except Exception as e:
            print("Failed to read ball stub:", e)
    else:
        print("Ball control stub missing:", args.ball_stub)
        print("If missing, the ball-control logic may be recomputed or failing — need to run pipeline without stubs to rebuild it.")

if __name__ == "__main__":
    main()