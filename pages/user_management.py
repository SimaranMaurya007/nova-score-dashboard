import streamlit as st
import pandas as pd

from lib.db import get_conn

def render():
    st.header("👥 User Management")

    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM user_scores ORDER BY prediction_date DESC", conn)

    if df.empty:
        st.info("No user data available yet.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        search_worker = st.text_input("Search by Worker ID")
    with col2:
        filter_region = st.selectbox("Filter by Region", ["All"] + sorted(df["region"].dropna().unique().tolist()))
    with col3:
        filter_gender = st.selectbox("Filter by Gender", ["All"] + sorted(df["gender"].dropna().unique().tolist()))

    filtered = df.copy()
    if search_worker:
        filtered = filtered[filtered["worker_id"].astype(str).str.contains(search_worker)]
    if filter_region != "All":
        filtered = filtered[filtered["region"] == filter_region]
    if filter_gender != "All":
        filtered = filtered[filtered["gender"] == filter_gender]

    st.subheader("User Records")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="nova_scores_export.csv", mime="text/csv")
