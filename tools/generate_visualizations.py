import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_visualizations(stubs_dir="stubs", analytics_dir="analytics", viz_dir="viz"):
    """
    Generates professional heatmaps & movement analysis visuals.
    """

    os.makedirs(viz_dir, exist_ok=True)

    tactical_path = os.path.join(stubs_dir, "tactical_player_positions_stub.pkl")
    jersey_path = os.path.join(stubs_dir, "jersey_numbers_map_stub.pkl")

    tactical_positions = pickle.load(open(tactical_path, "rb"))
    jersey_map = pickle.load(open(jersey_path, "rb"))

    all_points = []
    per_player = defaultdict(list)

    for fdict in tactical_positions:
        if not isinstance(fdict, dict):
            continue
        for pid, (x, y) in fdict.items():
            all_points.append((x, y))
            per_player[pid].append((x, y))

    if not all_points:
        print("[WARN] No tactical data found")
        return

    max_x = max(p[0] for p in all_points) + 1
    max_y = max(p[1] for p in all_points) + 1

    def draw_heat(points, name):
        H, _, _ = np.histogram2d(
            [p[0] for p in points],
            [p[1] for p in points],
            bins=[50, 50],
            range=[[0, max_x], [0, max_y]]
        )

        plt.figure(figsize=(6, 10))
        plt.imshow(H.T, origin="lower", cmap="hot")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, name.replace(" ", "_") + ".png"), dpi=250)
        plt.close()

    # Overall heatmap
    draw_heat(all_points, "Overall_Player_Heatmap")

    # Top 6 most active players
    top_players = sorted(per_player.items(), key=lambda x: -len(x[1]))[:6]
    for pid, pts in top_players:
        jersey = jersey_map.get(pid, "")
        name = f"Player_{pid}_#{jersey}_Heatmap" if jersey else f"Player_{pid}_Heatmap"
        draw_heat(pts, name)

    print("[DONE] Heatmaps generated.")