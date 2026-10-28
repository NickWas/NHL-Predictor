import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(__file__))

from update_data import update_data

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("Settings")
days = st.sidebar.slider("Days to look back:", 1, 60, 7)

if st.sidebar.button("Run Analysis"):
    with st.spinner("Scraping Reddit and analyzing sentiment..."):
        update_data(days=days)
        st.success("✅ Analysis complete!")

st.title("🏒 NHL Sentiment Analysis Dashboard")

# -------------------------------
# Tabs for Output Views
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Sentiment", "📄 Summary", "🧍 Player Abnormalities", "📂 Raw Data"])

# Paths to output files
base = "results"
chart_path = f"{base}/team_sentiment_{days}days.png"
summary_path = f"{base}/nhl_sentiment_report_{days}days.json"
player_csv = f"{base}/player_abnormalities_{days}days.csv"
raw_csv = f"{base}/reddit_raw_data_latest.csv"

# -------------------------------
# 📊 Team Sentiment Chart
# -------------------------------
with tab1:
    st.header("Team Sentiment Chart")
    if os.path.exists(chart_path):
        st.image(Image.open(chart_path), use_container_width=True)
    else:
        st.warning("Sentiment chart not found yet. Run analysis to generate it.")

# -------------------------------
# 📄 Summary JSON Report with Metrics
# -------------------------------
with tab2:
    st.header("Sentiment Summary Report")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            report = json.load(f)

        team_summary = report.get("team_sentiment", {})
        player_summary = report.get("player_abnormalities", {})

        # Display key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="❤️ Most Loved Team",
                value=team_summary.get("most_loved", "N/A"),
                delta=f"{team_summary.get('most_loved_score', 0):.2f}"
            )
        with col2:
            st.metric(
                label="💔 Most Hated Team",
                value=team_summary.get("most_hated", "N/A"),
                delta=f"{team_summary.get('most_hated_score', 0):.2f}"
            )

        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="😐 Neutral Teams",
                value=len(team_summary.get("neutral", []))
            )
        with col4:
            st.metric(
                label="📈 Teams Analyzed",
                value=team_summary.get("total_teams", 0)
            )

        st.markdown("---")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("🧍 Player Abnormalities", player_summary.get("total", 0))
        with col6:
            st.metric("🏥 Injuries", player_summary.get("injuries", 0), help="Detected injury-related discussions")

        st.markdown("### Raw JSON Report")
        st.json(report)

    else:
        st.warning("Summary report not found. Run analysis to create it.")


# -------------------------------
# 🧍 Player Abnormalities Table
# -------------------------------
with tab3:
    st.header("Player Abnormalities")
    if os.path.exists(player_csv):
        df_abnormal = pd.read_csv(player_csv)
        st.dataframe(df_abnormal)
    else:
        st.info("No abnormalities found or file missing.")

# -------------------------------
# 📂 Raw Reddit Data (optional)
# -------------------------------
with tab4:
    st.header("Raw Reddit Posts + Comments")
    if os.path.exists(raw_csv):
        df_raw = pd.read_csv(raw_csv)
        st.dataframe(df_raw.head(200))  # show first 200 for speed
    else:
        st.info("Raw data not found. Run analysis to collect it.")
