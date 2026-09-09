from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.auth import current_role, login_form, logout_button
from src.data_processing import load_sales_data
from src.evaluation import temporal_holdout_backtest
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


@st.cache_data
def run_backtest(df: pd.DataFrame):
    return temporal_holdout_backtest(
        df,
        horizons=6,
        holdout_months=6,
    )


df = load_data()
model = train_model(df)

forecast_tab, evaluation_tab = st.tabs(
    ["Forecast", "Model evaluation"]
)

with forecast_tab:
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

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Series",
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

with evaluation_tab:
    st.subheader("Temporal holdout backtest")

    max_date = pd.Timestamp(df["date"].max())
    holdout_start = max_date - pd.DateOffset(months=5)

    st.caption(
        "Evaluation uses synthetic target months from "
        f"{holdout_start:%B %Y} to {max_date:%B %Y}. "
        "Training rows are restricted to earlier target dates "
        "to avoid target leakage."
    )

    with st.spinner("Running temporal backtest..."):
        horizon_metrics, m1_comparison = run_backtest(df)

    xgb_m1 = m1_comparison.loc[
        m1_comparison["Method"] == "XGBoost"
    ].iloc[0]

    baseline_m1_metrics = m1_comparison.loc[
        m1_comparison["Method"] == "Weighted baseline"
    ].iloc[0]

    delta_wmape = (
        baseline_m1_metrics["wMAPE (%)"]
        - xgb_m1["wMAPE (%)"]
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "XGBoost M+1 wMAPE",
        f"{xgb_m1['wMAPE (%)']:.2f}%",
    )

    metric2.metric(
        "Baseline M+1 wMAPE",
        f"{baseline_m1_metrics['wMAPE (%)']:.2f}%",
    )

    metric3.metric(
        "XGBoost M+1 MAE",
        f"{xgb_m1['MAE']:.2f}",
    )

    metric4.metric(
        "wMAPE difference",
        f"{delta_wmape:+.2f} pts",
        help=(
            "Positive means XGBoost has lower wMAPE "
            "than the weighted baseline."
        ),
    )

    st.markdown("#### M+1 — XGBoost vs weighted baseline")

    comparison_display = m1_comparison.copy()

    for column in [
        "MAE",
        "RMSE",
        "sMAPE (%)",
        "wMAPE (%)",
    ]:
        comparison_display[column] = (
            comparison_display[column].round(2)
        )

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### XGBoost performance by horizon")

    horizon_display = horizon_metrics.copy()

    for column in [
        "MAE",
        "RMSE",
        "sMAPE (%)",
        "wMAPE (%)",
    ]:
        horizon_display[column] = (
            horizon_display[column].round(2)
        )

    st.dataframe(
        horizon_display,
        use_container_width=True,
        hide_index=True,
    )

    metric_chart = go.Figure()

    metric_chart.add_trace(
        go.Bar(
            x=horizon_metrics["Horizon"],
            y=horizon_metrics["wMAPE (%)"],
            name="XGBoost wMAPE",
        )
    )

    metric_chart.update_layout(
        xaxis_title="Forecast horizon",
        yaxis_title="wMAPE (%)",
        showlegend=False,
    )

    st.plotly_chart(
        metric_chart,
        use_container_width=True,
    )

    st.info(
        "These metrics measure performance only on the "
        "synthetic portfolio dataset. They are not production "
        "results and should not be interpreted as employer "
        "performance figures."
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
            "evaluation": "6-month temporal target holdout",
            "metrics": [
                "MAE",
                "RMSE",
                "sMAPE",
                "wMAPE",
            ],
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
        """
        This repository is an **independent demonstration project**
        inspired by real-world sales forecasting problems.

        The public version intentionally uses:
        - synthetic data;
        - a simplified feature set;
        - newly written portfolio code;
        - generic business entities;
        - lightweight authentication and role-based access;
        - a transparent temporal holdout evaluation.

        It does **not** reproduce an employer's proprietary
        implementation.
        """
    )
