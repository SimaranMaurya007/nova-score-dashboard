import pandas as pd
from .config import NUMERIC_FEATURES  # noqa: F401 (kept for clarity)

RENAME_MAP = {
    "age": "age_norm",
    "earnings": "earnings_norm",
    "jobs_completed": "jobs_completed",
    "repeat_client_share": "repeat_client_share",
    "avg_job_value": "avg_job_value_norm",
    "hours_worked": "hours_worked",
    "dispute_count": "dispute_count_norm",
    "tips_share": "tips_share",
    "customer_rating": "customer_rating",
    "cancellation_rate": "cancellation_rate",
    "peak_hour_jobs_share": "peak_hour_jobs_share",
    "transaction_count": "transaction_count",
    "gap_days": "gap_days",
    "gender": "gender",
    "region": "region",
    "partner_tier": "partner_tier",
    "worker_type": "worker_type",
    "nova_prev_score": "nova_prev_score",
}

FEATURE_COLUMNS = list(RENAME_MAP.values())

def prepare_features_for_prediction(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data]).rename(columns=RENAME_MAP)
    return df[FEATURE_COLUMNS]

def prepare_features_for_lgbm(input_data: dict, lgbm_feature_names: list) -> pd.DataFrame:
    raw_cols = [
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
    raw_df = pd.DataFrame([{k: input_data.get(k) for k in raw_cols}])
    raw_dummies = pd.get_dummies(raw_df, drop_first=True)

    # Ensure same columns as training
    for col in lgbm_feature_names:
        if col not in raw_dummies.columns:
            raw_dummies[col] = 0

    return raw_dummies[lgbm_feature_names]
