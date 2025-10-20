from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "nova_scores.db"

# Files
NORM_STATS_CSV = DATA_DIR / "group_norm_stats_65_35.csv"
CB_MODEL_PATH = DATA_DIR / "nova_score_model_65_35.cbm"
LGBM_MODEL_PATH = DATA_DIR / "nova_lightgbm.pkl"
LGBM_FEATURES_PATH = DATA_DIR / "nova_lightgbm_features.pkl"
LSTM_BUNDLE_PATH = DATA_DIR / "lstm_nova_bundle.pkl"
ALL_MODELS_CSV = DATA_DIR / "all_models_predicted_score.csv"  # optional

# Categorical Options
GENDER_OPTIONS = ["M", "F", "Other"]
REGION_OPTIONS = ["North", "South", "East", "West"]
PARTNER_TIER_OPTIONS = ["silver", "gold", "platinum"]
WORKER_TYPE_OPTIONS = ["merchant", "driver", "other"]

# Numeric features that are normalized
NUMERIC_FEATURES = [
    "age",
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
]
