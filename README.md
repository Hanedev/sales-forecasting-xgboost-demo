# Sales Forecasting with XGBoost — Multi-Horizon Demo

A portfolio project demonstrating an end-to-end sales forecasting workflow with **Python, feature engineering, XGBoost, Streamlit and lightweight authentication**.

> **Portfolio disclaimer**
>
> This repository is an independent demonstration project inspired by real-world forecasting problems I worked on professionally.
> It contains **no employer source code, production data, model artifacts, confidential business rules, coefficients or proprietary information**.
> All data included in this repository is synthetic and the public implementation was written specifically for this portfolio.

## Live demo

**Streamlit:** https://sale-forcasting.streamlit.app/

Demo access:

```text
username: demo
password: demo123
```

## What the project does

The application lets a user:

- sign in through a lightweight authentication layer;
- select a company and a product;
- inspect historical monthly sales;
- generate six direct forecasts from **M+1 to M+6**;
- compare the M+1 XGBoost prediction with a simple weighted statistical baseline;
- visualize historical sales and future forecasts in a Streamlit dashboard;
- demonstrate simple role-based access with `demo` and optional `admin` roles.

## Architecture

```text
Authentication
      |
      v
Synthetic sales data
      |
      v
Data loading & validation
      |
      v
Feature engineering
(lags, rolling averages, seasonality)
      |
      v
Six direct XGBoost models
M+1 ... M+6
      |
      v
Streamlit application
      |
      +--> Historical sales
      +--> Forecast table
      +--> XGBoost vs baseline
      +--> Admin-only technical info
```

## Tech stack

- Python
- Pandas / NumPy
- XGBoost
- scikit-learn
- Streamlit
- Plotly
- PBKDF2-SHA256 password hashing
- Streamlit session state and secrets

## Project structure

```text
sales-forecasting-xgboost-demo/
├── app.py
├── data/
│   └── sample_sales.csv
├── scripts/
│   ├── generate_synthetic_data.py
│   └── generate_password_hash.py
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── auth.py
│   ├── data_processing.py
│   ├── features.py
│   └── forecasting.py
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The repository includes a deterministic **synthetic dataset** covering:

- 3 generic companies;
- 8 products;
- 48 months of monthly history;
- sales;
- price;
- promotions;
- stockout days;
- product category.

Regenerate it with:

```bash
python scripts/generate_synthetic_data.py
```

## Feature engineering

The demo uses:

- sales lags: 1, 2, 3, 6 and 12 months;
- rolling means over 3 and 6 months;
- calendar seasonality using sine/cosine encoding;
- time trend;
- price;
- promotion indicator;
- stockout days.

## Multi-horizon forecasting

The project uses a **direct multi-horizon strategy**.

Six separate XGBoost regressors are trained:

```text
Model H1 -> M+1
Model H2 -> M+2
Model H3 -> M+3
Model H4 -> M+4
Model H5 -> M+5
Model H6 -> M+6
```

## Baseline

For M+1, the application also computes a simple weighted baseline:

```text
50% × last month
30% × average of last 3 months
20% × average of last 12 months
```

## Authentication

The authentication layer is intentionally lightweight. Its purpose is to demonstrate access control in a Data/ML application, not to replace a full identity provider.

### Recruiter demo account

```text
username: demo
password: demo123
```

Only a salted **PBKDF2-SHA256 hash** of this password is stored in source control.

### Optional admin role

Generate an admin password hash:

```bash
python scripts/generate_password_hash.py "your-secure-password"
```

Copy:

```text
.streamlit/secrets.toml.example
```

to:

```text
.streamlit/secrets.toml
```

and insert the generated values.

The real `secrets.toml` is ignored by Git and must never be committed.

The optional admin role exposes a small technical model-information section to demonstrate role-based access.

## Run locally

```bash
git clone https://github.com/Hanedev/sales-forecasting-xgboost-demo.git
cd sales-forecasting-xgboost-demo

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What this project demonstrates

- translation of a forecasting problem into a reproducible ML pipeline;
- temporal feature engineering;
- direct multi-horizon forecasting;
- XGBoost model integration;
- Streamlit application development;
- lightweight authentication and session management;
- secrets handling;
- simple role-based access;
- comparison of ML predictions with a transparent baseline;
- separation of authentication, data processing, features and forecasting logic.

## Author

**Abdoul Aziz HANE**  
Data & AI Engineer

- GitHub: https://github.com/Hanedev
- LinkedIn: https://www.linkedin.com/in/hane-abdoul-aziz
