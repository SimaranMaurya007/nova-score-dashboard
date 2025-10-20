import streamlit as st
import pandas as pd
import numpy as np
from plotly import express as px  # noqa: F401
from catboost import Pool

from lib.config import (
    GENDER_OPTIONS,
    REGION_OPTIONS,
    PARTNER_TIER_OPTIONS,
    WORKER_TYPE_OPTIONS,
    NUMERIC_FEATURES,
)
from lib.db import get_previous_score, get_conn
from lib.normalization import normalize_input
from lib.feature_prep import prepare_features_for_lgbm
from lib.models import load_catboost, load_lightgbm, load_all_models_csv  # noqa: F401
from lib.lstm import load_lstm_bundle, prepare_features_for_lstm_single
from lib.utils import now_str
from lib.offers import get_offers_for_score, is_eligible_for_credit

@st.cache_resource
def _models():
    return load_catboost(), load_lightgbm(), load_lstm_bundle()

def render():
    st.header("Nova Score Prediction")

    cb_model, (lgbm_model, lgbm_feats), lstm_bundle = _models()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Worker Data")
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                worker_id = st.number_input("Worker ID", min_value=0, value=0)
                week = st.number_input("Week", min_value=1, max_value=52, value=1)
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", GENDER_OPTIONS)
                region = st.selectbox("Region", REGION_OPTIONS)
                partner_tier = st.selectbox("Partner Tier", PARTNER_TIER_OPTIONS)
                worker_type = st.selectbox("Worker Type", WORKER_TYPE_OPTIONS)
            with col_b:
                earnings = st.number_input("Earnings", min_value=0.0, value=1500.0, step=100.0)
                jobs_completed = st.number_input("Jobs Completed", min_value=0, value=10)
                repeat_client_share = st.slider("Repeat Client Share", 0.0, 1.0, 0.5)
                avg_job_value = st.number_input("Avg Job Value", min_value=0.0, value=150.0, step=10.0)
                hours_worked = st.number_input("Hours Worked", min_value=0, value=20)
                dispute_count = st.number_input("Dispute Count", min_value=0, value=0)
                tips_share = st.slider("Tips Share", 0.0, 1.0, 0.2)
                customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0)

            col_c, col_d = st.columns(2)
            with col_c:
                cancellation_rate = st.slider("Cancellation Rate", 0.0, 1.0, 0.05)
                peak_hour_jobs_share = st.slider("Peak Hour Jobs Share", 0.0, 1.0, 0.3)
            with col_d:
                transaction_count = st.number_input("Transaction Count", min_value=0, value=15)
                gap_days = st.number_input("Gap Days", min_value=0, value=2)

            nova_prev_score = get_previous_score(worker_id) if worker_id > 0 else 60.0
            submitted = st.form_submit_button("Predict Nova Score", use_container_width=True)

        if submitted:
            input_data = {
                "worker_id": worker_id,
                "week": week,
                "age": age,
                "gender": gender,
                "region": region,
                "partner_tier": partner_tier,
                "worker_type": worker_type,
                "earnings": earnings,
                "jobs_completed": jobs_completed,
                "repeat_client_share": repeat_client_share,
                "avg_job_value": avg_job_value,
                "hours_worked": hours_worked,
                "dispute_count": dispute_count,
                "tips_share": tips_share,
                "customer_rating": customer_rating,
                "cancellation_rate": cancellation_rate,
                "peak_hour_jobs_share": peak_hour_jobs_share,
                "transaction_count": transaction_count,
                "gap_days": gap_days,
                "nova_prev_score": nova_prev_score,
            }

            try:
                # Normalize for CatBoost
                input_df = pd.DataFrame([input_data])
                input_df = normalize_input(input_df)
                features_for_model = [f + "_norm" for f in NUMERIC_FEATURES] + [
                    "gender",
                    "region",
                    "partner_tier",
                    "worker_type",
                ]
                cat_idx = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + 4))
                pool = Pool(input_df[features_for_model], cat_features=cat_idx)
                cb_prediction = float(cb_model.predict(pool)[0])

                # LightGBM (if available)
                lgbm_prediction = None
                if lgbm_model is not None and lgbm_feats is not None:
                    lgbm_features = prepare_features_for_lgbm(input_data, lgbm_feats)
                    lgbm_prediction = float(
                        lgbm_model.predict(
                            lgbm_features,
                            num_iteration=getattr(lgbm_model, "best_iteration", None),
                        )[0]
                    )

                # LSTM (optional)
                lstm_score = None
                if lstm_bundle is not None:
                    try:
                        X_seq_scaled = prepare_features_for_lstm_single(input_data, lstm_bundle)
                        pred_scaled = lstm_bundle["model"].predict(X_seq_scaled, verbose=0).ravel()[0]
                        lstm_score = float(
                            lstm_bundle["y_scaler"].inverse_transform([[pred_scaled]]).ravel()[0]
                        )
                    except Exception:
                        pass

                record = {
                    "timestamp": now_str(),
                    "worker_id": worker_id,
                    "predicted_score": round(cb_prediction, 2),
                    "catboost_score": round(cb_prediction, 2),
                    "lightgbm_score": round(lgbm_prediction, 2) if lgbm_prediction is not None else None,
                    "lstm_score": round(lstm_score, 2) if lstm_score is not None else None,
                    "model_gap": round(cb_prediction - lgbm_prediction, 2) if lgbm_prediction is not None else None,
                    "previous_score": nova_prev_score,
                    "score_change": round(cb_prediction - nova_prev_score, 2),
                    "worker_type": worker_type,
                }
                st.session_state.setdefault("predictions_history", []).append(record)

                # Insert if new (ignore duplicates)
                with get_conn() as conn:
                    before = conn.total_changes
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO user_scores (
                          worker_id, week, age, gender, region, partner_tier, worker_type,
                          earnings, jobs_completed, repeat_client_share, avg_job_value,
                          hours_worked, dispute_count, tips_share, customer_rating,
                          cancellation_rate, peak_hour_jobs_share, transaction_count,
                          gap_days, nova_prev_score, nova_score
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            worker_id,
                            week,
                            age,
                            gender,
                            region,
                            partner_tier,
                            worker_type,
                            earnings,
                            jobs_completed,
                            repeat_client_share,
                            avg_job_value,
                            hours_worked,
                            dispute_count,
                            tips_share,
                            customer_rating,
                            cancellation_rate,
                            peak_hour_jobs_share,
                            transaction_count,
                            gap_days,
                            nova_prev_score,
                            cb_prediction,
                        ),
                    )
                    inserted = (conn.total_changes - before) == 1
                    conn.commit()

                if inserted:
                    st.success(
                        "Weekly score update completed successfully!"
                        if nova_prev_score != 60.0
                        else "New user created with credit score 60.0! Prediction completed successfully!"
                    )
                else:
                    st.warning(f"⚠️ Data for Worker {worker_id} in Week {week} already exists. Skipping insert.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

        st.subheader("Batch Predictions (CSV)")
        uploaded = st.file_uploader(
            "Upload CSV with required columns",
            type=["csv"],
            help=(
                "Columns: worker_id, week, age, gender, region, partner_tier, worker_type, "
                "earnings, jobs_completed, repeat_client_share, avg_job_value, hours_worked, "
                "dispute_count, tips_share, customer_rating, cancellation_rate, "
                "peak_hour_jobs_share, transaction_count, gap_days, nova_prev_score"
            ),
        )

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Unable to read CSV: {e}")
                df_in = None

            if df_in is not None:
                required = [
                    "worker_id",
                    "week",
                    "age",
                    "gender",
                    "region",
                    "partner_tier",
                    "worker_type",
                    "earnings",
                    "jobs_completed",
                    "repeat_client_share",
                    "avg_job_value",
                    "hours_worked",
                    "dispute_count",
                    "tips_share",
                    "customer_rating",
                    "cancellation_rate",
                    "peak_hour_jobs_share",
                    "transaction_count",
                    "gap_days",
                    "nova_prev_score",
                ]
                missing = [c for c in required if c not in df_in.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    st.info(f"Processing {len(df_in)} rows...")
                    preds = []
                    prog = st.progress(0)
                    total = len(df_in)

                    for idx, row in df_in.iterrows():
                        try:
                            input_data = {c: row[c] for c in required}
                            single_df = pd.DataFrame([input_data])
                            from lib.normalization import normalize_input as _norm  # local alias

                            single_df = _norm(single_df)
                            feats = [f + "_norm" for f in NUMERIC_FEATURES] + [
                                "gender",
                                "region",
                                "partner_tier",
                                "worker_type",
                            ]
                            cat_idx = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + 4))
                            pool = Pool(single_df[feats], cat_features=cat_idx)
                            cb_pred = float(cb_model.predict(pool)[0])

                            lgb_pred = None
                            if lgbm_model is not None and lgbm_feats is not None:
                                lgb_feats = prepare_features_for_lgbm(input_data, lgbm_feats)
                                lgb_pred = float(
                                    lgbm_model.predict(
                                        lgb_feats, num_iteration=getattr(lgbm_model, "best_iteration", None)
                                    )[0]
                                )

                            lstm_pred = None
                            if lstm_bundle is not None:
                                try:
                                    X_seq_scaled = prepare_features_for_lstm_single(input_data, lstm_bundle)
                                    pred_scaled = lstm_bundle["model"].predict(X_seq_scaled, verbose=0).ravel()[0]
                                    lstm_pred = float(
                                        lstm_bundle["y_scaler"].inverse_transform([[pred_scaled]]).ravel()[0]
                                    )
                                except Exception:
                                    pass

                            preds.append(
                                {
                                    "catboost_predicted_score": cb_pred,
                                    "lightgbm_predicted_score": lgb_pred,
                                    "lstm_predicted_score": lstm_pred,
                                }
                            )
                        except Exception as e:
                            preds.append(
                                {
                                    "catboost_predicted_score": None,
                                    "lightgbm_predicted_score": None,
                                    "lstm_predicted_score": None,
                                    "error": str(e),
                                }
                            )

                        if (idx + 1) % 10 == 0 or (idx + 1) == total:
                            prog.progress(int((idx + 1) / total * 100))

                    prog.progress(100)

                    out_df = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
                    st.success("Batch predictions completed.")
                    st.dataframe(out_df.head(20), use_container_width=True)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{now_str().replace(':','').replace(' ','_')}.csv",
                        mime="text/csv",
                    )

    with col2:
        st.subheader("Prediction Result")
        if st.session_state.get("predictions_history"):
            latest = st.session_state["predictions_history"][-1]
            st.markdown(
                f"""
                <div class="prediction-card">
                    <h2>Nova Score</h2>
                    <h1 style="color:#2E8B57; font-size:3rem;">{latest['predicted_score']}</h1>
                    <p>Previous Score: {latest['previous_score']}</p>
                    <p>Change: {latest['score_change']:+.2f}</p>
                    <p>Predicted on: {latest['timestamp']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Model comparison
            models, scores = ["CatBoost"], [latest["catboost_score"]]
            if latest.get("lightgbm_score") is not None:
                models.append("LightGBM")
                scores.append(latest["lightgbm_score"])
            if latest.get("lstm_score") is not None:
                models.append("LSTM")
                scores.append(latest["lstm_score"])

            if len(models) > 1:
                comp_df = pd.DataFrame({"Model": models, "Score": scores})
                fig = px.bar(
                    comp_df,
                    x="Model",
                    y="Score",
                    color="Model",
                    color_discrete_sequence=["#2E8B57", "#90EE90", "#1f77b4"],
                    title="Predicted Scores by Model",
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Score interpretation
            score = latest["predicted_score"]
            if score >= 80:
                st.success("🌟 Excellent Performance!")
            elif score >= 70:
                st.info("👍 Good Performance")
            elif score >= 60:
                st.warning("⚠️ Average Performance")
            else:
                st.error("🔴 Needs Improvement")

            # Credit eligibility & offers
            worker_type_for_offer = latest.get("worker_type", "other")
            eligible, _ = is_eligible_for_credit(score)
            st.subheader("Credit Eligibility")
            if eligible:
                st.success("Eligible for credit offers")
                for o in get_offers_for_score(score, worker_type_for_offer):
                    _offer(o)
            else:
                st.error("Not eligible due to low Nova Score")
        else:
            st.info("Fill out the form to get a prediction")

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
