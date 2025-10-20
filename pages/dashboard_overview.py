import streamlit as st
import pandas as pd
import plotly.express as px

# Simple fixed metrics (as in original)
CAT_R2, CAT_RMSE = 0.8245, 4.5953
LGB_R2, LGB_RMSE = 0.7654, 5.3128
LSTM_R2, LSTM_RMSE = 0.597, 7.947

CB_TOP = [
    ("tenure_weeks_norm", 26.126285),
    ("earnings_roll_mean_norm", 12.704976),
    ("earnings_norm", 10.958783),
    ("gap_days_roll_mean_norm", 4.256911),
    ("repeat_client_share_roll_mean_norm", 3.685169),
    ("customer_rating_roll_mean_norm", 3.194941),
    ("avg_job_value_roll_mean_norm", 3.092342),
    ("cancellation_rate_roll_mean_norm", 2.578988),
    ("hours_worked_roll_mean_norm", 2.508323),
    ("customer_rating_norm", 2.037360),
    ("peak_hour_jobs_share_roll_mean_norm", 1.869009),
    ("earnings_trend_norm", 1.725810),
]

LGB_TOP = [
    ("tenure_weeks", 21433),
    ("customer_rating_roll_mean", 7883),
    ("avg_job_value_roll_mean", 7297),
    ("earnings_roll_mean", 7268),
    ("cancellation_rate_roll_mean", 7228),
    ("repeat_client_share_roll_mean", 7225),
    ("peak_hour_jobs_share_roll_mean", 7093),
    ("earnings_trend", 6972),
    ("customer_rating_trend", 6599),
    ("tips_share_roll_mean", 6557),
    ("cancellation_rate_trend", 6497),
    ("customer_rating", 6441),
]

LSTM_TOP = {
    "tenure_weeks": 2.9898,
    "earnings": 1.7980,
    "avg_job_value": 1.6990,
    "gap_days": 0.9047,
    "jobs_completed": 0.5763,
    "hours_worked": 0.4936,
    "repeat_client_share": 0.3614,
    "customer_rating": 0.3182,
    "transaction_count": 0.0370,
    "tips_share": 0.0217,
    "dispute_count": 0.0023,
    "peak_hour_jobs_share": -0.0044,
    "cancellation_rate": -0.1175,
}

def render():
    st.markdown(
        '<div class="main-header"><h1>Nova Score Prediction Dashboard</h1></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric("CatBoost R²", CAT_R2)
    with col2:
        _metric("CatBoost RMSE", CAT_RMSE)
    with col3:
        _metric("LightGBM R²", LGB_R2)
    with col4:
        _metric("LightGBM RMSE", LGB_RMSE)

    lc1, lc2 = st.columns(2)
    with lc1:
        _metric("LSTM R²", LSTM_R2)
    with lc2:
        _metric("LSTM RMSE", LSTM_RMSE)

    st.subheader("Feature Importance by Model")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.caption("CatBoost Feature Importance")
        _bar(CB_TOP, "CatBoost - Top Features")
    with fc2:
        st.caption("LightGBM Feature Importance")
        _bar(LGB_TOP, "LightGBM - Top Features")
    with fc3:
        st.caption("LSTM Top Features (weight analysis)")
        lstm_df = (
            pd.DataFrame(list(LSTM_TOP.items()), columns=["Feature", "Weight"])
            .sort_values("Weight", ascending=False)
            .head(12)
        )
        fig = px.bar(
            lstm_df,
            x="Weight",
            y="Feature",
            orientation="h",
            color="Weight",
            color_continuous_scale="Greens",
            title="LSTM - Top Features",
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if "predictions_history" in st.session_state and st.session_state.predictions_history:
        st.subheader("📋 Recent Predictions")
        recent_df = pd.DataFrame(st.session_state.predictions_history[-10:])
        st.dataframe(recent_df, use_container_width=True)

def _metric(label, value):
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <h2 style="color:#2E8B57;">{value:.3f}</h2>
        <p>Model metric</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

def _bar(data, title):
    df = pd.DataFrame(data, columns=["Feature", "Importance"])
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Greens",
        title=title,
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
