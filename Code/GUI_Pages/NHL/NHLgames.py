import streamlit as st
import requests
import json
from datetime import datetime, timedelta

st.markdown("<h1 style='text-align: center;'>Upcoming NHL Games</h1>", unsafe_allow_html=True)

games_url = "https://api-web.nhle.com/v1/score/now"
res = requests.get(games_url)
game_data = res.json()

games = []

for game in game_data["games"]:
    homeTeamName = game["homeTeam"]["abbrev"]
    homeTeamLogo = game["homeTeam"]["logo"]
    awayTeamName = game["awayTeam"]["abbrev"]
    awayTeamLogo = game["awayTeam"]["logo"]

    game_time = datetime.strptime(game["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ")
    game_time = game_time - timedelta(hours=4)  # Convert to local time (EST)

    games.append([homeTeamName, homeTeamLogo, awayTeamName, awayTeamLogo, game_time])

st.markdown("""
    <style>
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

for game in games:
    col1, col2, col3, col4, col5 = st.columns([2, 1, .8, 1, 2])

    with col1:
        st.markdown(f'<div class="centered"> <span style="font-size:24px; font-weight:bold;"> {game[0]}</div>', unsafe_allow_html=True)

    with col2:
        st.image(game[1], width=100)

    with col3:
        st.image("GUI_Pages/NHL/vs.png", width=50)

    with col4:
        st.image(game[3], width=100)

    with col5:
        st.markdown(f'<div class="centered"> <span style="font-size:24px; font-weight:bold;"> {game[2]}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="centered"><b>Game time:</b> {game[4].strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)

    st.markdown("---")

st.warning("Upcoming games are updated at 13:00 UTC every day.")
