from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample_sales.csv"
)


def main():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=48, freq="MS")
    companies = ["Company A", "Company B", "Company C"]

    products = [
        ("P001", "OTC"),
        ("P002", "OTC"),
        ("P003", "Prescription"),
        ("P004", "Prescription"),
        ("P005", "Wellness"),
        ("P006", "Wellness"),
        ("P007", "OTC"),
        ("P008", "Prescription"),
    ]

    rows = []

    for ci, company in enumerate(companies):
        for pi, (product, category) in enumerate(products):
            base = 90 + pi * 18 + ci * 12
            trend = 0.8 + 0.12 * pi
            amplitude = 12 + 2 * (pi % 3)

            for t, date in enumerate(dates):
                seasonal = amplitude * np.sin(
                    2 * np.pi * (date.month - 1) / 12
                )

                promotion = 1 if rng.random() < 0.12 else 0
                noise = rng.normal(0, 8)

                sales = max(
                    5,
                    round(
                        base
                        + trend * t
                        + seasonal
                        + promotion * 18
                        + noise,
                        1,
                    ),
                )

                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "company": company,
                        "product_id": product,
                        "category": category,
                        "sales": sales,
                        "price": round(
                            8
                            + pi * 1.8
                            + ci * 0.5
                            + rng.normal(0, 0.25),
                            2,
                        ),
                        "promotion": promotion,
                        "stockout_days": int(
                            rng.choice(
                                [0, 0, 0, 0, 1, 2, 3, 5],
                                p=[
                                    0.25,
                                    0.20,
                                    0.15,
                                    0.10,
                                    0.10,
                                    0.08,
                                    0.07,
                                    0.05,
                                ],
                            )
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Generated {len(df):,} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
