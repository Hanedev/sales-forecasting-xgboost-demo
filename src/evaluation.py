from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS, add_direct_targets, add_features


def _build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )


def _metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual
    abs_error = np.abs(error)

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error ** 2)))

    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.where(
        denominator == 0,
        0.0,
        2.0 * abs_error / denominator,
    )
    smape = float(np.mean(smape_terms) * 100)

    actual_sum = np.abs(actual).sum()
    wmape = (
        float(abs_error.sum() / actual_sum * 100)
        if actual_sum != 0
        else 0.0
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "wMAPE": wmape,
    }


def _add_weighted_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["company", "product_id"]

    rolling_3 = (
        out.groupby(group_cols)["sales"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    rolling_12 = (
        out.groupby(group_cols)["sales"]
        .transform(lambda s: s.rolling(12, min_periods=1).mean())
    )

    out["baseline_m1"] = (
        0.5 * out["sales"]
        + 0.3 * rolling_3
        + 0.2 * rolling_12
    )

    return out


def temporal_holdout_backtest(
    df: pd.DataFrame,
    horizons: int = 6,
    holdout_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate direct multi-horizon XGBoost models on a temporal holdout.

    The holdout is defined by TARGET dates, not feature-row dates.
    For example, with data ending in December 2025 and a six-month
    holdout, July-December 2025 are held out as target months.
    """
    featured = add_features(df)
    prepared = add_direct_targets(featured, horizons)
    prepared = _add_weighted_baseline(prepared)

    max_date = pd.Timestamp(prepared["date"].max())
    holdout_start = max_date - pd.DateOffset(
        months=holdout_months - 1
    )

    horizon_rows = []
    m1_model_metrics = None
    m1_baseline_metrics = None

    for horizon in range(1, horizons + 1):
        target = f"target_h{horizon}"
        target_date = prepared["date"] + pd.DateOffset(
            months=horizon
        )

        complete = prepared.dropna(
            subset=FEATURE_COLUMNS + [target]
        ).copy()

        complete_target_date = (
            complete["date"]
            + pd.DateOffset(months=horizon)
        )

        train_mask = complete_target_date < holdout_start
        test_mask = (
            (complete_target_date >= holdout_start)
            & (complete_target_date <= max_date)
        )

        train = complete.loc[train_mask]
        test = complete.loc[test_mask]

        if train.empty or test.empty:
            raise ValueError(
                f"Not enough data to evaluate horizon M+{horizon}."
            )

        model = _build_model()
        model.fit(
            train[FEATURE_COLUMNS],
            train[target],
        )

        predictions = np.maximum(
            0.0,
            model.predict(test[FEATURE_COLUMNS]),
        )

        model_metrics = _metrics(
            test[target],
            predictions,
        )

        horizon_rows.append(
            {
                "Horizon": f"M+{horizon}",
                "Observations": int(len(test)),
                "MAE": model_metrics["MAE"],
                "RMSE": model_metrics["RMSE"],
                "sMAPE (%)": model_metrics["sMAPE"],
                "wMAPE (%)": model_metrics["wMAPE"],
            }
        )

        if horizon == 1:
            baseline_predictions = test["baseline_m1"].to_numpy()

            m1_model_metrics = model_metrics
            m1_baseline_metrics = _metrics(
                test[target],
                baseline_predictions,
            )

    horizon_metrics = pd.DataFrame(horizon_rows)

    m1_comparison = pd.DataFrame(
        [
            {
                "Method": "XGBoost",
                "MAE": m1_model_metrics["MAE"],
                "RMSE": m1_model_metrics["RMSE"],
                "sMAPE (%)": m1_model_metrics["sMAPE"],
                "wMAPE (%)": m1_model_metrics["wMAPE"],
            },
            {
                "Method": "Weighted baseline",
                "MAE": m1_baseline_metrics["MAE"],
                "RMSE": m1_baseline_metrics["RMSE"],
                "sMAPE (%)": m1_baseline_metrics["sMAPE"],
                "wMAPE (%)": m1_baseline_metrics["wMAPE"],
            },
        ]
    )

    return horizon_metrics, m1_comparison
