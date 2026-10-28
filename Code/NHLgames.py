import streamlit as st
from Get_Games_For_GUI import get_predicted_games
from datetime import datetime

st.markdown("<h1 style='text-align: center;'>Upcoming NHL Games</h1>", unsafe_allow_html=True)

games = get_predicted_games()

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
    homeTeamName, homeTeamLogo, home_prob = game[0], game[1], game[2]
    awayTeamName, awayTeamLogo, away_prob = game[3], game[4], game[5]
    game_time = game[6]

    col1, col2, col3, col4, col5 = st.columns([2, 1, .8, 1, 2])

    with col1:
        st.markdown(
            f"<div class='centered'>"
            f"<span style='font-size:24px; font-weight:bold;'>{homeTeamName}</span><br>"
            f"<span style='font-size:20px;'>Win Probability: {home_prob*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.image(homeTeamLogo, width=100)

    with col3:
        st.image("GUI_Pages/NHL/vs.png", width=50)

    with col4:
        st.image(awayTeamLogo, width=100)

    with col5:
        st.markdown(
            f"<div class='centered'>"
            f"<span style='font-size:24px; font-weight:bold;'>{awayTeamName}</span><br>"
            f"<span style='font-size:20px;'>Win Probability: {away_prob*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"<div class='centered'><b>Game time:</b> {game_time.strftime('%Y-%m-%d %H:%M')}</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

st.warning("Upcoming games are updated at 13:00 UTC every day.")
