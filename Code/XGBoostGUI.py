# XGBoostGUI.py

import streamlit as st
import pandas as pd
from XGBoost.XGBoost import train_model, preprocess_data
import requests

@st.cache_resource
def get_trained_model():
    return train_model(season=2023)

def predict_game(game_id, model):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    res = requests.get(url)

    if res.status_code != 200:
        st.error("❌ Game not found or not available.")
        return

    game_data = res.json()
    game_df, _ = preprocess_data(game_data)

    if game_df is None or game_df.empty:
        st.warning("⚠️ No usable events in this game.")
        return

    probs = model.predict_proba(game_df)
    game_df["Minute"] = 60 - game_df["timeRemaining"]
    game_df["Home Win %"] = probs[:, 1] * 100
    game_df["Away Win %"] = probs[:, 0] * 100

    st.subheader("📈 Win Probability Over Time")
    st.line_chart(game_df[["Minute", "Home Win %", "Away Win %"]].set_index("Minute"))

    st.subheader("📊 Raw Prediction Table")
    st.dataframe(game_df[["Minute", "Home Win %", "Away Win %"]].round(2), use_container_width=True)

# Streamlit UI
st.title("🏒 NHL Game Win Probability Predictor (XGBoost)")
game_id_input = st.text_input("Enter NHL Game ID", value="2023020034", help="Use format like 2023020034")

if st.button("Predict"):
    with st.spinner("⏳ Loading model and processing game..."):
        model = get_trained_model()
        predict_game(game_id_input, model)
