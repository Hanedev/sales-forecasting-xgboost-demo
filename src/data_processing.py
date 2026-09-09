from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "company",
    "product_id",
    "category",
    "sales",
    "price",
    "promotion",
    "stockout_days",
}


def load_sales_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(
        ["company", "product_id", "date"]
    ).reset_index(drop=True)
