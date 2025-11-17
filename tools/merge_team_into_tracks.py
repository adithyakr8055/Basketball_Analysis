# tools/merge_team_into_tracks.py
import os, pickle, argparse
from collections import defaultdict

def load(path):
    return pickle.load(open(path, "rb"))

def save(obj, path):
    pickle.dump(obj, open(path, "wb"))
    print("Saved:", path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="stubs/player_track_stubs.pkl", help="player_tracks stub")
    p.add_argument("--team_map", default="stubs/player_team_map.pkl", help="stable map produced by TeamAssigner")
    p.add_argument("--out", default="stubs/player_track_stubs_with_team.pkl", help="output tracks with team injected")
    args = p.parse_args()

    if not os.path.exists(args.tracks):
        raise SystemExit(f"Tracks not found: {args.tracks}")
    if not os.path.exists(args.team_map):
        raise SystemExit(f"Team map not found: {args.team_map}")

    print("Loading tracks:", args.tracks)
    tracks = load(args.tracks)
    print("Loading team_map:", args.team_map)
    team_map = load(args.team_map)

    # Normalize keys to ints in team_map (safe)
    team_map_norm = {}
    for k,v in team_map.items():
        try:
            team_map_norm[int(k)] = int(v)
        except Exception:
            team_map_norm[k] = int(v)

    modified_count = 0
    # tracks assumed list of frames, each dict pid -> pdata
    for fi, frame in enumerate(tracks):
        if not frame:
            continue
        for pid_raw, pdata in list(frame.items()):
            # Normalize pid to int if possible
            try:
                pid = int(pid_raw)
            except Exception:
                pid = pid_raw
            team = team_map_norm.get(pid)
            if team is None:
                # no stable mapping for this pid — skip
                continue
            # ensure pdata is a dict
            if not isinstance(pdata, dict):
                continue
            # inject 'team' field (int)
            old = pdata.get("team", pdata.get("team_id"))
            if old != team:
                pdata["team"] = int(team)
                modified_count += 1

    print(f"Injected 'team' into {modified_count} player-frame entries.")
    save(tracks, args.out)
    print("DONE. Now re-run your pipeline using this updated stub (replace path or point pipeline to new stub).")

if __name__ == "__main__":
    main()