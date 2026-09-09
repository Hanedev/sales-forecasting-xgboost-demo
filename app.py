from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.auth import current_role, login_form, logout_button
from src.data_processing import load_sales_data
from src.forecasting import (
    DirectMultiHorizonForecaster,
    weighted_baseline,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_sales.csv"

st.set_page_config(
    page_title="Sales Forecasting XGBoost Demo",
    page_icon="📈",
    layout="wide",
)

if not login_form():
    st.stop()

logout_button()

st.sidebar.success(
    f"Signed in as {st.session_state['username']} "
    f"({st.session_state['role']})"
)

st.title("Sales Forecasting — XGBoost Demo")
st.caption(
    "Independent portfolio demo built with synthetic data. "
    "No employer data, source code, model artifacts or "
    "confidential business rules are included."
)


@st.cache_data
def load_data():
    return load_sales_data(DATA_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame):
    return DirectMultiHorizonForecaster(
        horizons=6
    ).fit(df)


df = load_data()
model = train_model(df)

st.sidebar.header("Selection")

company = st.sidebar.selectbox(
    "Company",
    sorted(df["company"].unique()),
)

product_options = sorted(
    df.loc[
        df["company"] == company,
        "product_id",
    ].unique()
)

product = st.sidebar.selectbox(
    "Product",
    product_options,
)

history = (
    df[
        (df["company"] == company)
        & (df["product_id"] == product)
    ]
    .sort_values("date")
    .copy()
)

forecasts = model.predict_latest(
    df,
    company,
    product,
)

baseline_m1 = weighted_baseline(
    history["sales"]
)

latest_sales = float(
    history.iloc[-1]["sales"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest sales",
    f"{latest_sales:,.1f}",
)

col2.metric(
    "XGBoost M+1",
    f"{forecasts.iloc[0]['xgboost_forecast']:,.1f}",
)

col3.metric(
    "Weighted baseline M+1",
    f"{baseline_m1:,.1f}",
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history["date"],
        y=history["sales"],
        mode="lines+markers",
        name="History",
    )
)

fig.add_trace(
    go.Scatter(
        x=forecasts["forecast_date"],
        y=forecasts["xgboost_forecast"],
        mode="lines+markers",
        name="XGBoost forecast",
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

comparison = forecasts.copy()

comparison["xgboost_forecast"] = (
    comparison["xgboost_forecast"].round(1)
)

comparison["weighted_baseline"] = None

comparison.loc[
    comparison.index[0],
    "weighted_baseline",
] = round(baseline_m1, 1)

st.subheader("Forecast table")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

if current_role() == "admin":
    st.subheader(
        "Admin — Model information"
    )

    st.write(
        "This section illustrates simple role-based access."
    )

    st.json(
        {
            "forecasting_strategy": "direct multi-horizon",
            "number_of_models": 6,
            "algorithm": "XGBoost",
            "horizons": [
                "M+1",
                "M+2",
                "M+3",
                "M+4",
                "M+5",
                "M+6",
            ],
            "data_source": (
                "synthetic portfolio dataset"
            ),
        }
    )

with st.expander(
    "About this portfolio project"
):
    st.markdown(
        '''
        This repository is an **independent demonstration project**
        inspired by real-world sales forecasting problems.

        The public version intentionally uses:
        - synthetic data;
        - a simplified feature set;
        - newly written portfolio code;
        - generic business entities;
        - lightweight authentication and role-based access.

        It does **not** reproduce an employer's proprietary
        implementation.
        '''
    )
