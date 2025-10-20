import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from lib.db import get_conn
from lib.models import load_all_models_csv

def render():
    st.header("Data Analytics")

    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM user_scores", conn)

    both_df = load_all_models_csv()

    if df.empty and both_df is None:
        st.info("No data available yet. Make some predictions or add a combined CSV!")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Demographics", "Feature Analysis", "Model Comparison"])

    with tab1:
        st.subheader("Score Distribution")
        if both_df is not None:
            lgb_col = "lightgbm_predicted_score" if "lightgbm_predicted_score" in both_df.columns else None
            dist_specs = [
                ("Nova Score", "nova_score"),
                ("CatBoost Predicted", "catboost_predicted_score"),
                ("LSTM Predicted", "lstm_predicted_score"),
                ("LightGBM Predicted", lgb_col),
            ]
            _hist_grid(both_df, dist_specs)

    with tab2:
        st.subheader("Demographic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            gender_stats = df.groupby("gender")["nova_score"].agg(["mean", "count"]).reset_index()
            st.plotly_chart(
                px.bar(gender_stats, x="gender", y="mean", title="Average Nova Score by Gender", color="mean", color_continuous_scale="Greens"),
                use_container_width=True,
            )
        with col2:
            region_stats = df.groupby("region")["nova_score"].agg(["mean", "count"]).reset_index()
            st.plotly_chart(
                px.bar(region_stats, x="region", y="mean", title="Average Nova Score by Region", color="mean", color_continuous_scale="Greens"),
                use_container_width=True,
            )

    with tab3:
        st.subheader("Feature Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numerical_cols].corr()
        st.plotly_chart(px.imshow(corr, title="Feature Correlation Matrix", color_continuous_scale="RdYlBu_r"), use_container_width=True)

    with tab4:
        st.subheader("Model Comparison: CatBoost vs LightGBM vs LSTM")
        if both_df is None:
            st.info("Upload `all_models_predicted_score.csv` to view model comparisons.")
        else:
            view_cols = [
                c
                for c in ["worker_id", "week", "catboost_predicted_score", "lightgbm_predicted_score", "lstm_predicted_score", "nova_score"]
                if c in both_df.columns
            ]
            st.dataframe(both_df[view_cols].sort_values(["worker_id", "week"]), use_container_width=True)

            # Distributions overlay
            parts = []
            for name, col in [
                ("CatBoost", "catboost_predicted_score"),
                ("LightGBM", "lightgbm_predicted_score"),
                ("LSTM", "lstm_predicted_score"),
            ]:
                if col in both_df.columns:
                    t = both_df[[col]].copy()
                    t.rename(columns={col: "score"}, inplace=True)
                    t["Model"] = name
                    parts.append(t)

            if parts:
                dist_df = pd.concat(parts, ignore_index=True)
                fig = px.histogram(
                    dist_df, x="score", color="Model", barmode="overlay", opacity=0.6, title="Prediction Distributions",
                    color_discrete_map={"CatBoost": "#1f77b4", "LightGBM": "#90EE90", "LSTM": "#FF8C00"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Gap histogram & scatter
            if all(c in both_df.columns for c in ["catboost_predicted_score", "lightgbm_predicted_score"]):
                tmp = both_df.copy()
                tmp["gap"] = tmp["catboost_predicted_score"] - tmp["lightgbm_predicted_score"]
                st.plotly_chart(
                    px.histogram(tmp, x="gap", nbins=40, title="Distribution of Model Gap (CatBoost - LightGBM)", color_discrete_sequence=["#2E8B57"]),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.scatter(
                        both_df,
                        x="lightgbm_predicted_score",
                        y="catboost_predicted_score",
                        title="Prediction Agreement: LightGBM vs CatBoost",
                        labels={"lightgbm_predicted_score": "LightGBM", "catboost_predicted_score": "CatBoost"},
                        color_discrete_sequence=["#2E8B57"],
                    ),
                    use_container_width=True,
                )

def _hist_grid(df, specs):
    c1, c2 = st.columns(2)
    if specs[0][1] and specs[0][1] in df.columns:
        with c1:
            st.plotly_chart(
                px.histogram(df, x=specs[0][1], nbins=40, title=f"Distribution - {specs[0][0]}", color_discrete_sequence=["#2E8B57"]),
                use_container_width=True,
            )
    if specs[1][1] and specs[1][1] in df.columns:
        with c2:
            st.plotly_chart(
                px.histogram(df, x=specs[1][1], nbins=40, title=f"Distribution - {specs[1][0]}", color_discrete_sequence=["#2E8B57"]),
                use_container_width=True,
            )

    c3, c4 = st.columns(2)
    if specs[2][1] and specs[2][1] in df.columns:
        with c3:
            st.plotly_chart(
                px.histogram(df, x=specs[2][1], nbins=40, title=f"Distribution - {specs[2][0]}", color_discrete_sequence=["#2E8B57"]),
                use_container_width=True,
            )
    if specs[3][1] and specs[3][1] in df.columns:
        with c4:
            st.plotly_chart(
                px.histogram(df, x=specs[3][1], nbins=40, title=f"Distribution - {specs[3][0]}", color_discrete_sequence=["#2E8B57"]),
                use_container_width=True,
            )
