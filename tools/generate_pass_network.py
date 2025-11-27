import os
import csv
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def generate_pass_network(stubs_dir="stubs", analytics_dir="analytics", output_path="viz/pass_network.png"):
    """
    Professional pass + interception network graph.

    Primary source:
        analytics/player_interaction_network.csv
    Fallback:
        stubs/passes_stub.pkl + ball_acquisition_stub.pkl
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    network_csv = os.path.join(analytics_dir, "player_interaction_network.csv")

    G = nx.DiGraph()

    # =====================================
    # ✅ PREFERRED METHOD: Use analytics CSV
    # =====================================
    if os.path.exists(network_csv):
        print("[INFO] Building pass network from analytics CSV")

        with open(network_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = int(row["from_player_id"])
                v = int(row["to_player_id"])
                weight = int(row["event_count"])

                jersey_u = row.get("from_jersey", "")
                jersey_v = row.get("to_jersey", "")
                team_u = int(row.get("from_team", -1))
                team_v = int(row.get("to_team", -1))

                G.add_node(u, jersey=jersey_u, team=team_u)
                G.add_node(v, jersey=jersey_v, team=team_v)

                if G.has_edge(u, v):
                    G[u][v]["weight"] += weight
                else:
                    G.add_edge(u, v, weight=weight)

    # =====================================
    # ✅ FALLBACK METHOD: raw stubs
    # =====================================
    else:
        print("[WARN] Analytics CSV not found, falling back to raw stubs")

        passes = pickle.load(open(os.path.join(stubs_dir, "passes_stub.pkl"), "rb"))
        ball = pickle.load(open(os.path.join(stubs_dir, "ball_acquisition_stub.pkl"), "rb"))
        jersey_map = pickle.load(open(os.path.join(stubs_dir, "jersey_numbers_map_stub.pkl"), "rb"))

        last_owner = None
        for i in range(len(passes)):
            if passes[i] != -1:
                current = ball[i]
                if last_owner and current != last_owner and current != -1:
                    G.add_node(last_owner, jersey=jersey_map.get(last_owner, ""))
                    G.add_node(current, jersey=jersey_map.get(current, ""))

                    if G.has_edge(last_owner, current):
                        G[last_owner][current]['weight'] += 1
                    else:
                        G.add_edge(last_owner, current, weight=1)
                last_owner = current

    if G.number_of_nodes() == 0:
        print("[WARN] No data to build pass network")
        return

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    plt.figure(figsize=(12, 12))

    # Node style
    labels = {}
    node_colors = []

    for pid, data in G.nodes(data=True):
        jersey = data.get("jersey", "")
        labels[pid] = f"#{jersey}" if jersey else f"ID {pid}"

        team = data.get("team", -1)
        if team == 1:
            node_colors.append("#1f78b4")
        elif team == 2:
            node_colors.append("#33a02c")
        else:
            node_colors.append("#999999")

    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight="bold")

    edges = G.edges()
    widths = [G[u][v]["weight"] * 1.5 for u, v in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=widths,
        edge_color="red",
        arrows=True
    )

    plt.title("Professional Pass Interaction Network", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Pass network graph saved: {output_path}")