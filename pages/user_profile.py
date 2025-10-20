import streamlit as st
import pandas as pd
import plotly.express as px  # noqa: F401

from lib.db import get_conn
from lib.models import load_all_models_csv
from lib.offers import get_offers_for_score, is_eligible_for_credit

def render():
    st.header("👤 User Profile")

    worker_id_query = st.text_input("Enter Worker ID to view profile")
    if not worker_id_query:
        st.info("Enter a Worker ID to see profile details.")
        return

    try:
        worker_id = int(worker_id_query)
    except ValueError:
        st.error("Worker ID must be an integer.")
        return

    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM user_scores WHERE worker_id = ? ORDER BY week ASC, prediction_date ASC",
            conn,
            params=(worker_id,),
        )

    both_df = load_all_models_csv()

    if both_df is not None:
        try:
            for c in ["worker_id", "week"]:
                if c in df.columns:
                    df[c] = df[c].astype(int)
                if c in both_df.columns:
                    both_df[c] = both_df[c].astype(int)
            keep = ["worker_id", "week"] + [
                c
                for c in ["lightgbm_predicted_score", "catboost_predicted_score", "lstm_predicted_score"]
                if c in both_df.columns
            ]
            sub = both_df[both_df["worker_id"] == worker_id][keep]
            if not sub.empty:
                df = df.merge(sub, on=["worker_id", "week"], how="left")
        except Exception:
            pass

    if df.empty:
        st.warning(f"No records found for Worker {worker_id}.")
        return

    latest_row = df.sort_values(["prediction_date"]).iloc[-1]
    latest_score = latest_row["nova_score"]

    st.metric("Latest Nova Score", f"{latest_score:.2f}")
    st.metric("Last Updated", str(latest_row["prediction_date"]))

    # Optional model metrics
    for label, key in [
        ("CatBoost Predicted Score", "catboost_predicted_score"),
        ("LightGBM Predicted Score", "lightgbm_predicted_score"),
        ("LSTM Predicted Score", "lstm_predicted_score"),
    ]:
        if key in df.columns and pd.notna(latest_row.get(key)):
            st.metric(label, f"{float(latest_row.get(key)):.2f}")

    # History plot
    plot_df = df.copy()

    def pick(name):
        for c in [name, name + "_x", name + "_y"]:
            if c in plot_df.columns:
                return c
        return None

    value_vars = [("Nova Score", "nova_score")]
    for label, col in [
        ("CatBoost", "catboost_predicted_score"),
        ("LightGBM", "lightgbm_predicted_score"),
        ("LSTM", "lstm_predicted_score"),
    ]:
        c = pick(col)
        if c:
            value_vars.append((label, c))

    long_parts = []
    for label, col in value_vars:
        if col in plot_df.columns:
            t = plot_df[["week", col]].copy()
            t.rename(columns={col: "score"}, inplace=True)
            t["score"] = pd.to_numeric(t["score"], errors="coerce")
            t.dropna(subset=["score"], inplace=True)
            if not t.empty:
                t["Model"] = label
                long_parts.append(t)

    if long_parts:
        long_df = pd.concat(long_parts, ignore_index=True)
        st.plotly_chart(
            px.line(
                long_df,
                x="week",
                y="score",
                color="Model",
                title=f"Worker {worker_id} - Weekly Scores by Model",
                markers=True,
                color_discrete_map={"Nova Score": "#2E8B57", "CatBoost": "#1f77b4", "LightGBM": "#90EE90", "LSTM": "#FF8C00"},
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            px.line(
                df,
                x="week",
                y="nova_score",
                title=f"Worker {worker_id} - Weekly Nova Score",
                markers=True,
                color_discrete_sequence=["#2E8B57"],
            ),
            use_container_width=True,
        )

    st.subheader("Weekly Records")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download User CSV", data=df.to_csv(index=False), file_name=f"worker_{worker_id}_history.csv", mime="text/csv")

    # Offers
    st.subheader("Offers")
    eligible, _ = is_eligible_for_credit(float(latest_score))
    if eligible:
        for o in get_offers_for_score(float(latest_score), str(latest_row.get("worker_type", "other"))):
            _offer(o)
    else:
        st.error("Not eligible due to low Nova Score")

def _offer(o: dict):
    st.markdown(
        f"""
    <div style="background: white; border-left: 4px solid #2E8B57; padding: .75rem 1rem; border-radius:8px; margin-bottom:.5rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-weight:600;">{o.get('title','Offer')}</div>
                <div style="color:#2E8B57; font-size:.9rem;">{o.get('subtitle','')}\n</div>
            </div>
            <div style="background:#E6FFE6; color:#2E8B57; padding:2px 8px; border-radius:12px; font-size:.8rem;">{o.get('tag','')}</div>
        </div>
        <div style="margin-top:6px; font-size:.95rem;">{o.get('detail','')}</div>
        <div style="margin-top:8px;"><span style="background:#2E8B57; color:white; padding:6px 10px; border-radius:6px; font-size:.85rem;">{o.get('cta','Apply Now')}</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )
