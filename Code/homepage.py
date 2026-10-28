
import streamlit as st

st.set_page_config(page_title="EdgeBot", page_icon="💵")

pg = st.navigation({'NHL': [st.Page('GUI_Pages/NHL/NHLgames.py', title='Convolutional Neural Network', icon= '🏒'),
                             st.Page('XGBoost/XGBoostGUI.py', title='XGBoost', icon= '🥅'),
                             st.Page('SentimentAnalysis/SentimentAnalysisGUI.py', title='Sentiment Analysis', icon= '⛸️'),]
                    })
                    

#'NBA': [st.Page('GUI_Pages/NBA/NBAgames.py', title = 'Upcoming Games', icon= '🏀'), 
#           st.Page('GUI_Pages/NBA/NBAPlayerStats.py', title = 'Player Statistics', icon= '⛹️‍♂️')],
#   'NFL': [st.Page('GUI_Pages/NFL.py', title = 'NFL', icon= '🏈')]
#'''
                


pg.run()
