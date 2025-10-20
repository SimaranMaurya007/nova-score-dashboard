import pickle
import numpy as np
import pandas as pd
import streamlit as st
from .config import LSTM_BUNDLE_PATH

try:
    from tensorflow.keras.models import model_from_json
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

@st.cache_resource
def load_lstm_bundle():
    if not TF_AVAILABLE or not LSTM_BUNDLE_PATH.exists():
        return None
    try:
        with open(LSTM_BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = model_from_json(bundle["model_json"])
        model.set_weights(bundle["model_weights"])
        return {
            "model": model,
            "x_scaler": bundle["x_scaler"],
            "y_scaler": bundle["y_scaler"],
            "feature_columns": bundle["feature_columns"],
            "w_feat": bundle["w_feat"],
            "seq_len": bundle["meta"]["seq_len"],
        }
    except Exception as e:
        st.warning(f"Unable to load LSTM bundle: {e}")
        return None

def prepare_features_for_lstm_single(input_data: dict, lstm_bundle: dict):
    feature_columns = lstm_bundle["feature_columns"]
    seq_len = lstm_bundle["seq_len"]

    raw_df = pd.DataFrame([input_data])

    inferred_cats = list({c.split("_")[0] for c in feature_columns if "_" in c})
    cat_cols_present = [c for c in inferred_cats if c in raw_df.columns]
    df_encoded = pd.get_dummies(raw_df, columns=cat_cols_present, drop_first=False)

    for c in feature_columns:
        if c not in df_encoded.columns:
            df_encoded[c] = 0

    X_df = df_encoded[feature_columns].astype(float)
    padding = np.zeros((seq_len - 1, X_df.shape[1]))
    X_seq = np.vstack([padding, X_df.values]).reshape(1, seq_len, X_df.shape[1])

    x_scaler = lstm_bundle["x_scaler"]
    w_feat = lstm_bundle["w_feat"]

    X_seq_flat = X_seq.reshape(-1, X_seq.shape[2])
    X_seq_scaled = x_scaler.transform(X_seq_flat).reshape(X_seq.shape)
    return X_seq_scaled * w_feat
