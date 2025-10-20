import streamlit as st

from lib import ui
from pages import (
    dashboard_overview,
    score_prediction,
    data_analytics,
    user_profile,
    user_management,
)
from lib.db import init_database

st.set_page_config(
    page_title="Nova Score Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_theme()

# Ensure DB exists
init_database()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Dashboard Overview",
        "Score Prediction",
        "Data Analytics",
        "User Profile",
        "User Management",
    ],
    index=0,
)

if page == "Dashboard Overview":
    dashboard_overview.render()
elif page == "Score Prediction":
    score_prediction.render()
elif page == "Data Analytics":
    data_analytics.render()
elif page == "User Profile":
    user_profile.render()
else:
    user_management.render()
