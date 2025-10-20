import os
import joblib
import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd

from .config import (
    CB_MODEL_PATH,
    LGBM_MODEL_PATH,
    LGBM_FEATURES_PATH,
    ALL_MODELS_CSV,
)

@st.cache_resource
def load_catboost():
    model = CatBoostRegressor()
    model.load_model(str(CB_MODEL_PATH))
    return model

@st.cache_resource
def load_lightgbm():
    try:
        model = joblib.load(LGBM_MODEL_PATH)
        features = joblib.load(LGBM_FEATURES_PATH) if os.path.exists(LGBM_FEATURES_PATH) else None
        return model, features
    except Exception as e:
        st.warning(f"LightGBM artifacts not found or failed: {e}")
        return None, None

@st.cache_data(ttl=10, show_spinner=False)
def load_all_models_csv():
    if os.path.exists(ALL_MODELS_CSV):
        try:
            df = pd.read_csv(ALL_MODELS_CSV)
            df.columns = [c.lower() for c in df.columns]
            # normalize expected names
            rename = {}
            if "light_gbm_predicted_score" in df.columns:
                rename["light_gbm_predicted_score"] = "lightgbm_predicted_score"
            df = df.rename(columns=rename)
            return df
        except Exception:
            return None
    return None
