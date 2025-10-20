import pandas as pd
from .config import NORM_STATS_CSV, NUMERIC_FEATURES

# Load once
norm_stats = pd.read_csv(NORM_STATS_CSV, header=[0, 1], index_col=0)

def normalize_input(user_input: pd.DataFrame) -> pd.DataFrame:
    group_key = f"{user_input['gender'].iloc[0]}_{user_input['region'].iloc[0]}"
    if group_key in norm_stats.index:
        for f in NUMERIC_FEATURES:
            mean = norm_stats.loc[group_key, (f, "mean")]
            std = norm_stats.loc[group_key, (f, "std")]
            user_input[f + "_norm"] = (user_input[f] - mean) / (std + 1e-8)
    else:  # global fallback
        for f in NUMERIC_FEATURES:
            mean = norm_stats.xs(f, axis=1, level=0)["mean"].mean()
            std = norm_stats.xs(f, axis=1, level=0)["std"].mean()
            user_input[f + "_norm"] = (user_input[f] - mean) / (std + 1e-8)
    return user_input
