from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS, add_direct_targets, add_features


@dataclass
class DirectMultiHorizonForecaster:
    horizons: int = 6
    models: dict[int, XGBRegressor] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "DirectMultiHorizonForecaster":
        featured = add_features(df)
        training = add_direct_targets(featured, self.horizons)

        for horizon in range(1, self.horizons + 1):
            target = f"target_h{horizon}"
            train_h = training.dropna(subset=FEATURE_COLUMNS + [target])

            model = XGBRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )

            model.fit(train_h[FEATURE_COLUMNS], train_h[target])
            self.models[horizon] = model

        return self

    def predict_latest(
        self,
        df: pd.DataFrame,
        company: str,
        product_id: str,
    ) -> pd.DataFrame:
        featured = add_features(df)

        subset = featured[
            (featured["company"] == company)
            & (featured["product_id"] == product_id)
        ].dropna(subset=FEATURE_COLUMNS)

        if subset.empty:
            raise ValueError("Not enough history to build forecast features.")

        latest = subset.sort_values("date").iloc[-1]
        X = latest[FEATURE_COLUMNS].to_frame().T.astype(float)
        last_date = pd.Timestamp(latest["date"])

        rows = []

        for horizon, model in sorted(self.models.items()):
            prediction = max(0.0, float(model.predict(X)[0]))

            rows.append(
                {
                    "horizon": f"M+{horizon}",
                    "forecast_date": last_date + pd.DateOffset(months=horizon),
                    "xgboost_forecast": prediction,
                }
            )

        return pd.DataFrame(rows)


def weighted_baseline(history: pd.Series) -> float:
    values = history.dropna().astype(float)

    if len(values) < 12:
        return float(values.tail(min(3, len(values))).mean())

    last_1 = values.iloc[-1]
    last_3 = values.tail(3).mean()
    last_12 = values.tail(12).mean()

    return float(
        0.5 * last_1
        + 0.3 * last_3
        + 0.2 * last_12
    )
