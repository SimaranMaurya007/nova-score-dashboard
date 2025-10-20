import streamlit as st

def inject_theme():
    st.markdown(
        """
        <style>
        .main-header { background: linear-gradient(90deg, #2E8B57, #228B22);
            padding:1rem; border-radius:10px; color:white; text-align:center; margin-bottom:2rem; }
        .metric-card { background:white; padding:1rem; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);
            border-left:4px solid #2E8B57; margin:0.5rem 0; }
        .prediction-card { background:linear-gradient(135deg, #F0FFF0, #E6FFE6);
            padding:2rem; border-radius:15px; border:2px solid #2E8B57; text-align:center; margin:1rem 0; }
        .sidebar .sidebar-content { background: linear-gradient(180deg, #F0FFF0, #E6FFE6); }
        .stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input { background-color:white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
