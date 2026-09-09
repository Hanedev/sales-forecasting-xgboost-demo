import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12]

FEATURE_COLUMNS = [
    "sales",
    "price",
    "promotion",
    "stockout_days",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "month_sin",
    "month_cos",
    "time_index",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["company", "product_id"]

    out = out.sort_values(group_cols + ["date"]).reset_index(drop=True)

    for lag in LAGS:
        out[f"lag_{lag}"] = (
            out.groupby(group_cols)["sales"].shift(lag)
        )

    out["rolling_mean_3"] = (
        out.groupby(group_cols)["sales"]
        .transform(lambda s: s.shift(1).rolling(3).mean())
    )

    out["rolling_mean_6"] = (
        out.groupby(group_cols)["sales"]
        .transform(lambda s: s.shift(1).rolling(6).mean())
    )

    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["time_index"] = out.groupby(group_cols).cumcount().astype(float)

    return out


def add_direct_targets(df: pd.DataFrame, horizons: int = 6) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["company", "product_id"]

    for horizon in range(1, horizons + 1):
        out[f"target_h{horizon}"] = (
            out.groupby(group_cols)["sales"].shift(-horizon)
        )

    return out
