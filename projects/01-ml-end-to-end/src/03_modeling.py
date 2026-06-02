import os
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

PROCESSED_DATA_PATH = "data/processed/aqi_features.csv"
MODELS_PATH = "models/"
PLOTS_PATH = "assets/demo-screenshots/"
FORECAST_HOURS = 72 # for 3-day forecast

def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["datetime"])
    return df

#Prophet baseline
def run_prophet(df: pd.DataFrame, city: str) -> float:
    """Train Prophet and return its MAE."""
    city_df = df[df["city"] == city][["datetime", "pm25"]].copy()
    city_df = city_df.rename(columns={"datetime": "ds", "pm25": "y"})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
    )
    model.fit(city_df)

    future = model.make_future_dataframe(periods=0, freq="h")
    forecast = model.predict(future)

    mae = mean_absolute_error(city_df["y"], forecast["yhat"])
    print(f"Prophet MAE for {city}: {mae:.2f} µg/m³")
    return mae

# XGBoost Model
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h", "pm25_rolling_6h", "pm25_rolling_24h"
]
TARGET_COL = "pm25"

def run_xgboost(df: pd.DataFrame, city: str) -> tuple:
    """
    Trains XGBoost using time-series cross validation.
    Returns trained model and MAE.
    """
    city_df = df[df["city"] == city].copy().sort_values("datetime")

    X = city_df[FEATURE_COLS]
    y = city_df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
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
        mae = mean_absolute_error(y_val, preds)
        fold_maes.append(mae)
        print(f"Fold {fold + 1} MAE: {mae:.2f} µg/m³")

    avg_mae = np.mean(fold_maes)
    print(f"XGBoost avg MAE for {city}: {avg_mae:.2f} µg/m³")

    # train final model on *all* data
    final_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)
    return final_model, avg_mae, city_df

# Generates Forecast
def forecast_future(model, city_df: pd.DataFrame) -> pd.DataFrame:
    last_row = city_df.iloc[-1].copy()
    last_time = city_df["datetime"].iloc[-1]
    history = city_df["pm25"].tolist()
    future_rows = []

    for i in range(1, FORECAST_HOURS + 1):
        next_time = last_time + pd.Timedelta(hours=i)

        # Builds the feature row
        row = {
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "pm25_lag_1h": history[-1],
            "pm25_lag_3h": history[-3] if len(history) >= 3  else history[-1],
            "pm25_lag_24h": history[-24] if len(history) >= 24 else history[-1],
            "pm25_rolling_6h": np.mean(history[-6:]),
            "pm25_rolling_24h": np.mean(history[-24:]),
        }

        pred = model.predict(pd.DataFrame([row]))[0]
        pred = max(0, pred)  # fixes negative PM2.5 issue
        history.append(pred)
        future_rows.append({"datetime": next_time, "yhat": pred})

    return pd.DataFrame(future_rows)

# Plot
def plot_forecast(city_df: pd.DataFrame, future_df: pd.DataFrame, city: str, mae: float):
    """Plots ast 60 days of actual + Plots 72-hour forecast."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # last 60 days of actual data
    cutoff = city_df["datetime"].max() - pd.Timedelta(days=60)
    recent = city_df[city_df["datetime"] >= cutoff]
    ax.plot(recent["datetime"], recent["pm25"], color="steelblue", linewidth=0.8, label="Actual PM2.5")

    # future forecast
    ax.plot(future_df["datetime"], future_df["yhat"], color="tomato", linewidth=2, label=f"XGBoost Forecast (MAE {mae:.1f})")

    # WHO guideline
    ax.axhline(y=15, color="green", linestyle="--", linewidth=1, label="WHO guideline (15 µg/m³)")

    ax.set_title(f"{city} — PM2.5 Forecast (XGBoost)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_PATH, f"05_forecast_{city.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: 05_forecast_{city.lower()}.png")

# Main
def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)

    print("Loading processed data...")
    df = load_data()
    print(f"{len(df)} rows loaded\n")

    cities = df["city"].unique()
    for city in cities:
        print(f"{'='*40}")
        print(f"Processing {city}...")
        print(f"{'='*40}")

        print("Running Prophet...")
        run_prophet(df, city)

        print("[Primary] Running XGBoost...")
        model, mae, city_df = run_xgboost(df, city)

        print("Generating future forecast...")
        future_df = forecast_future(model, city_df)

        plot_forecast(city_df, future_df, city, mae)
        print()

    print("Modeling complete.")

if __name__ == "__main__":
    main()