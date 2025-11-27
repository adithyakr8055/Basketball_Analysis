import os
import glob
import re

import pandas as pd
import numpy as np
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# PATHS (relative to repo root)
# =========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYTICS_DIR = os.path.join(ROOT_DIR, "analytics")
HEATMAP_DIR = os.path.join(ANALYTICS_DIR, "heatmaps")
VIZ_DIR = os.path.join(ROOT_DIR, "viz")
OUTPUT_CSV_DIR = os.path.join(ROOT_DIR, "output_csv")


# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Basketball Analytics System (BVA) Dashboard",
    page_icon="🏀",
    layout="wide",
)

# =========================
# GLOBAL STYLE
# =========================
st.markdown(
    """
    <style>
    /* General */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }
    .main {
        padding-top: 1rem;
    }
    /* Metric cards tweak */
    [data-testid="metric-container"] {
        background-color: #111827;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid #1f2933;
    }
    [data-testid="metric-container"] > div {
        color: #e5e7eb;
    }
    /* Make images rounded */
    img {
        border-radius: 12px !important;
    }
    /* Dataframe header */
    .stDataFrame thead tr th {
        background-color: #111827;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_image(path: str):
    if not os.path.exists(path):
        return None
    return Image.open(path)

def list_player_heatmaps():
    if not os.path.isdir(HEATMAP_DIR):
        return {}
    mapping = {}
    for p in glob.glob(os.path.join(HEATMAP_DIR, "heatmap_player_*.png")):
        fname = os.path.basename(p)
        # heatmap_player_<pid>_optionalJersey.png
        m = re.match(r"heatmap_player_(\d+)", fname)
        if m:
            pid = int(m.group(1))
            mapping[pid] = p
    return mapping

# =========================
# LOAD ALL DATA
# =========================
enhanced_play_path = os.path.join(ANALYTICS_DIR, "enhanced_play_analysis.csv")
player_summary_path = os.path.join(ANALYTICS_DIR, "player_summary.csv")
interaction_network_path = os.path.join(ANALYTICS_DIR, "player_interaction_network.csv")
predictive_actions_path = os.path.join(OUTPUT_CSV_DIR, "predictive_actions.csv")

df_enh = load_csv(enhanced_play_path)
df_summary = load_csv(player_summary_path)
df_network = load_csv(interaction_network_path)
df_predict = load_csv(predictive_actions_path)

overall_heatmap_img = load_image(os.path.join(HEATMAP_DIR, "heatmap_overall.png"))
pass_network_img = load_image(os.path.join(VIZ_DIR, "pass_network.png"))

player_heatmap_map = list_player_heatmaps()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🏀 BVA Dashboard")
st.sidebar.markdown("### Basketball Analytics System")

st.sidebar.markdown("---")

if not df_enh.empty:
    total_frames = df_enh["frame"].max() + 1
else:
    total_frames = 0

st.sidebar.markdown(f"**Loaded frames:** `{total_frames}`")
st.sidebar.markdown(f"**Analytics dir:** `analytics/`")
st.sidebar.markdown(f"**Visuals dir:** `viz/`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtering**")

# Global frame range filter
if not df_enh.empty:
    min_frame = int(df_enh["frame"].min())
    max_frame = int(df_enh["frame"].max())
    frame_range = st.sidebar.slider(
        "Frame window",
        min_value=min_frame,
        max_value=max_frame,
        value=(min_frame, max_frame),
        step=1,
    )
else:
    frame_range = (0, 0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: run `python main.py ...` again to refresh this dashboard with a new video.")

# =========================
# TITLE
# =========================
st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Basketball Analytics System (BVA)</h1>
    <p style="color:#9ca3af; font-size:0.95rem; margin-bottom:0.5rem;">
        Automated player tracking, event detection & predictive play engine – visualized.
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# KPIs (Overview)
# =========================
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

if not df_enh.empty:
    mask_global = (df_enh["frame"].between(frame_range[0], frame_range[1]))
    df_range = df_enh[mask_global]

    total_passes = (df_range["event_type"] == "PASS").sum()
    total_interceptions = (df_range["event_type"] == "INTERCEPTION").sum()
    total_possessions = (df_range["ball_handler_id"] != -1).sum()
else:
    total_passes = total_interceptions = total_possessions = 0

if not df_summary.empty:
    total_players = df_summary["player_id"].nunique()
else:
    total_players = 0

with col_kpi1:
    st.metric("Total Passes", f"{total_passes}")
with col_kpi2:
    st.metric("Total Interceptions", f"{total_interceptions}")
with col_kpi3:
    st.metric("Players Detected", f"{total_players}")
with col_kpi4:
    st.metric("Possession Frames", f"{total_possessions}")

st.markdown("---")

# =========================
# TOP-LEVEL TABS
# =========================
tab_overview, tab_players, tab_playbyplay, tab_predictive = st.tabs(
    ["🏟 Overview", "👤 Player Insights", "📋 Play-by-Play", "🧠 Predictive Engine"]
)

# =========================
# TAB 1: OVERVIEW
# =========================
with tab_overview:
    st.subheader("Game Overview")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("#### Overall Court Occupancy Heatmap")
        if overall_heatmap_img:
            st.image(overall_heatmap_img, caption="Where players spent the most time on the court")
        else:
            st.info("Overall heatmap not found. Make sure `analytics/heatmaps/heatmap_overall.png` exists.")

    with col2:
        st.markdown("#### Pass Interaction Network")
        if pass_network_img:
            st.image(pass_network_img, caption="Pass & interception relationships between players")
        else:
            st.info("Pass network image not found. Make sure `viz/pass_network.png` exists.")

    st.markdown("### Event Dynamics Over Time")

    if not df_enh.empty:
        df_plot = df_enh.copy()
        df_plot = df_plot[df_plot["frame"].between(frame_range[0], frame_range[1])]

        # Aggregate events by frame
        agg = df_plot.groupby("frame").agg(
            passes=("event_type", lambda x: (x == "PASS").sum()),
            interceptions=("event_type", lambda x: (x == "INTERCEPTION").sum()),
            possession=("ball_handler_id", lambda x: (x != -1).sum()),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg["frame"], y=agg["passes"],
            mode="lines+markers",
            name="Passes",
        ))
        fig.add_trace(go.Scatter(
            x=agg["frame"], y=agg["interceptions"],
            mode="lines+markers",
            name="Interceptions",
        ))
        fig.add_trace(go.Scatter(
            x=agg["frame"], y=agg["possession"],
            mode="lines",
            name="Active Possessions",
            opacity=0.4,
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Frame",
            yaxis_title="Count per frame",
            legend_title="Events",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("enhanced_play_analysis.csv not found or empty.")

# =========================
# TAB 2: PLAYER INSIGHTS
# =========================
with tab_players:
    st.subheader("Player-Level Insights")

    if df_summary.empty:
        st.info("player_summary.csv is empty or missing. Run main.py + analytics again.")
    else:
        # Sorting controls
        sort_metric = st.selectbox(
            "Sort players by",
            ["passes_made", "passes_received", "interceptions_made", "total_distance_m", "avg_speed_mps", "total_possession_frames"],
            index=0
        )
        ascending = st.checkbox("Sort ascending", value=False)

        df_sorted = df_summary.sort_values(by=sort_metric, ascending=ascending)

        st.markdown("#### Player Summary Table")
        st.dataframe(
            df_sorted,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Player selector
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            player_ids = df_summary["player_id"].tolist()
            id_to_label = {}

            for _, row in df_summary.iterrows():
                pid = row["player_id"]
                jersey = row.get("jersey", "")
                team = row.get("team", -1)
                label = f"Player {pid}"
                if jersey not in ("", "nan", None):
                    label += f"  (#{jersey})"
                if team in (1, 2):
                    label += f"  [Team {team}]"
                id_to_label[pid] = label

            selected_pid = st.selectbox(
                "Select player",
                options=player_ids,
                format_func=lambda x: id_to_label.get(x, f"Player {x}")
            )

        with col_sel2:
            st.markdown("##### Key Metrics")

            row = df_summary[df_summary["player_id"] == selected_pid].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Possession Frames", int(row["total_possession_frames"]))
            c2.metric("Passes Made", int(row["passes_made"]))
            c3.metric("Passes Received", int(row["passes_received"]))
            c4.metric("Interceptions", int(row["interceptions_made"]))

            c5, c6 = st.columns(2)
            c5.metric("Average Speed (m/s)", float(row["avg_speed_mps"]))
            c6.metric("Total Distance (m)", float(row["total_distance_m"]))

        st.markdown("### Spatial & Tempo Profile")

        col_heat, col_time = st.columns([1.1, 1])

        with col_heat:
            st.markdown("#### Movement Heatmap")

            if selected_pid in player_heatmap_map:
                img = load_image(player_heatmap_map[selected_pid])
                st.image(img, caption=id_to_label.get(selected_pid, f"Player {selected_pid}"), use_container_width=True)
            else:
                st.info("No dedicated heatmap found for this player (check analytics/heatmaps).")

        with col_time:
            st.markdown("#### Speed & Possession Timeline")

            if not df_enh.empty:
                df_p = df_enh[df_enh["ball_handler_id"] == selected_pid].copy()
                if not df_p.empty:
                    # Speed line
                    fig2 = go.Figure()
                    if "speed_mps" in df_p.columns:
                        fig2.add_trace(go.Scatter(
                            x=df_p["frame"],
                            y=df_p["speed_mps"].replace("", np.nan).astype(float),
                            mode="lines+markers",
                            name="Speed (m/s)",
                        ))
                    fig2.update_layout(
                        height=300,
                        xaxis_title="Frame",
                        yaxis_title="Speed (m/s)",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Possession indicator
                    df_pos = df_enh.copy()
                    df_pos["has_ball"] = np.where(df_pos["ball_handler_id"] == selected_pid, 1, 0)
                    df_pos = df_pos[df_pos["frame"].between(frame_range[0], frame_range[1])]

                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=df_pos["frame"],
                        y=df_pos["has_ball"],
                        mode="lines",
                        name="Possession (1=has ball)",
                    ))
                    fig3.update_layout(
                        height=200,
                        xaxis_title="Frame",
                        yaxis_title="Possession",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("This player never appears as primary ball handler in the current frame range.")
            else:
                st.info("enhanced_play_analysis.csv not loaded.")

# =========================
# TAB 3: PLAY-BY-PLAY
# =========================
with tab_playbyplay:
    st.subheader("Play-by-Play Analysis")

    if df_enh.empty:
        st.info("No enhanced_play_analysis.csv found.")
    else:
        df_pb = df_enh[df_enh["frame"].between(frame_range[0], frame_range[1])].copy()

        colf1, colf2, colf3 = st.columns(3)

        with colf1:
            event_filter = st.multiselect(
                "Filter by event type",
                options=["PASS", "INTERCEPTION", "NONE"],
                default=["PASS", "INTERCEPTION", "NONE"],
            )
        with colf2:
            team_filter = st.multiselect(
                "Filter by ball handler team",
                options=sorted(df_pb["ball_handler_team"].dropna().unique().tolist()),
                default=sorted(df_pb["ball_handler_team"].dropna().unique().tolist())
            )
        with colf3:
            possession_only = st.checkbox("Only rows with possession change", value=False)

        if event_filter:
            df_pb = df_pb[df_pb["event_type"].isin(event_filter)]
        if team_filter:
            df_pb = df_pb[df_pb["ball_handler_team"].isin(team_filter)]
        if possession_only:
            df_pb = df_pb[df_pb["possession_change"] == 1]

        st.markdown("#### Filtered Events Table")
        st.dataframe(
            df_pb[
                [
                    "frame",
                    "event_type",
                    "ball_handler_id",
                    "ball_handler_jersey",
                    "ball_handler_team",
                    "receiver_id",
                    "receiver_jersey",
                    "receiver_team",
                    "predicted_action",
                    "speed_mps",
                    "distance_delta_m",
                ]
            ].sort_values("frame"),
            use_container_width=True,
            hide_index=True,
        )

        # Small distribution of event types
        st.markdown("#### Event Type Distribution")
        counts = df_pb["event_type"].value_counts().reset_index()
        counts.columns = ["event_type", "count"]

        fig_evt = px.bar(
            counts,
            x="event_type",
            y="count",
            color="event_type",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig_evt, use_container_width=True)

# =========================
# TAB 4: PREDICTIVE ENGINE
# =========================
with tab_predictive:
    st.subheader("Predictive Play Engine Insights")

    if df_enh.empty:
        st.info("enhanced_play_analysis.csv missing; cannot show predictions.")
    else:
        df_pred = df_enh.copy()
        df_pred = df_pred[df_pred["frame"].between(frame_range[0], frame_range[1])]

        st.markdown("#### Predicted Action Distribution")

        counts = df_pred["predicted_action"].value_counts().reset_index()
        counts.columns = ["predicted_action", "count"]

        fig_pred = px.bar(
            counts,
            x="predicted_action",
            y="count",
            color="predicted_action",
            template="plotly_dark",
            height=360,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("#### Inspect Specific Predicted Scenario")

        unique_actions = sorted([a for a in df_pred["predicted_action"].dropna().unique().tolist() if a not in ("", "NONE")])
        if unique_actions:
            chosen_action = st.selectbox(
                "Choose predicted action to inspect",
                options=unique_actions,
            )

            df_sel = df_pred[df_pred["predicted_action"] == chosen_action].copy()

            st.markdown(f"Showing frames where model predicts: **{chosen_action}**")
            st.dataframe(
                df_sel[
                    [
                        "frame",
                        "ball_handler_id",
                        "ball_handler_jersey",
                        "ball_handler_team",
                        "receiver_id",
                        "receiver_jersey",
                        "receiver_team",
                        "event_type",
                        "speed_mps",
                        "distance_delta_m",
                    ]
                ].sort_values("frame"),
                use_container_width=True,
                hide_index=True,
            )

            # timeline of where those predictions happen
            fig_timeline = px.scatter(
                df_sel,
                x="frame",
                y="ball_handler_team",
                color="event_type",
                hover_data=["ball_handler_id", "receiver_id"],
                template="plotly_dark",
                title="Timeline of selected predicted action",
                height=350,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No non-trivial predicted actions found (all NONE or empty).")