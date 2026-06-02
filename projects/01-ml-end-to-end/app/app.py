import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

PROCESSED_DATA_DIR = "data/processed/aqi_features.csv"
FORECAST_HOURS = 72

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm25_rolling_6h", "pm25_rolling_24h"
]
TARGET_COL = "pm25"

# Basic Page Configuration
st.set_page_config(
    layout="wide",
    page_title="AQI City Forecaster",
)
st.title("AQI City Forecaster")
st.markdown("_Forecasting PM2.5 pollution for Hyderabad, Karachi, and Lahore using XGBoost_")

# Sidebar
st.sidebar.header("Settings")
city = st.sidebar.selectbox(
    "Select a city",
    ["Hyderabad", "Karachi", "Lahore"]
)
forecast_days = st.sidebar.slider(
    "Forecast horizon",
    min_value=1,
    max_value=7,
    value=3
)
FORECAST_HOURS = forecast_days * 24

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DATA_DIR, parse_dates=["datetime"])
    return df

df = load_data()

# Train XGBoost with time-series cross validation
@st.cache_resource
def train_model(city_name: str):
    city_df = df[df["city"] == city_name].copy().sort_values("datetime")

    X = city_df[FEATURE_COLS]
    y = city_df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_maes = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_maes.append(mean_absolute_error(y_val, preds))

    avg_mae = np.mean(fold_maes)

    # Final model trained on all data
    final_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)
    return final_model, avg_mae, city_df

with st.spinner(f"Training XGBoost model for {city}..."):
    model, mae, city_df = train_model(city)

# Generate future forecast iteratively
def forecast_future(model, city_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    last_time = city_df["datetime"].iloc[-1]
    history = city_df["pm25"].tolist()
    future_rows = []

    for i in range(1, hours + 1):
        next_time = last_time + pd.Timedelta(hours=i)

        row = {
            "hour":             next_time.hour,
            "day_of_week":      next_time.dayofweek,
            "month":            next_time.month,
            "is_weekend":       int(next_time.dayofweek >= 5),
            "pm25_lag_1h":      history[-1],
            "pm25_lag_3h":      history[-3] if len(history) >= 3  else history[-1],
            "pm25_lag_24h":     history[-24] if len(history) >= 24 else history[-1],
            "pm25_rolling_6h":  np.mean(history[-6:]),
            "pm25_rolling_24h": np.mean(history[-24:]),
        }

        pred = model.predict(pd.DataFrame([row]))[0]
        pred = max(0, pred)
        history.append(pred)
        future_rows.append({"datetime": next_time, "yhat": pred})

    return pd.DataFrame(future_rows)

future_df = forecast_future(model, city_df, FORECAST_HOURS)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("City", city)
col2.metric("Cross-Val MAE", f"{mae:.2f} µg/m³")
col3.metric("Forecast Horizon", f"{forecast_days} days")

# Plot
st.subheader(f"{city} PM2.5 Forecast")
fig, ax = plt.subplots(figsize=(14, 5))

# Last 60 days of actual data
cutoff = city_df["datetime"].max() - pd.Timedelta(days=60)
recent = city_df[city_df["datetime"] >= cutoff]
ax.plot(recent["datetime"], recent["pm25"],
        color="steelblue", linewidth=0.8, label="Actual PM2.5")

# Future forecast
ax.plot(future_df["datetime"], future_df["yhat"],
        color="tomato", linewidth=2, label=f"XGBoost Forecast (MAE {mae:.1f})")

# WHO guideline
ax.axhline(y=15, color="green", linestyle="--", linewidth=1, label="WHO guideline (15 µg/m³)")

ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Raw Forecast Table
with st.expander("Show raw forecast data"):
    st.dataframe(
        future_df.rename(columns={"yhat": "forecast_pm25"}).reset_index(drop=True)
    )