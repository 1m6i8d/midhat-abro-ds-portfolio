import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

PROCESSED_DATA_PATH = "data/processed/aqi_features.csv"
MODELS_PATH = "models/"
PLOTS_PATH = "assets/demo-screenshots/"
FORECAST_HOURS = 72 # for 3 day forecast

def prepare_prophet_df(df: pd.DataFrame, city: str) -> pd.DataFrame:
    # prophet requires two columns:
    city_df = df[df["city"] == city][["datetime", "pm25"]].copy()
    city_df = city_df.rename(columns={"datetime": "ds", "pm25": "y"})
    # ds for datetime
    # y for value to be forecast
    return city_df

def train_and_forecast(prophet_df: pd.DataFrame, city: str) -> tuple:
    """trains and returns a Prophet model and forecast dataframe"""
    print(f"Training model for {city}...")
    model = Prophet(
        daily_seasonality=True, # learns hour-of-day patterns
        weekly_seasonality=True, # learns day-of-week patterns
        yearly_seasonality=True, # learns month-of-year patterns
        interval_width=0.95, # 95% confidence interval
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq="h")
    forecast = model.predict(future)
    return model, forecast

def evaluate(prophet_df: pd.DataFrame, forecast: pd.DataFrame, city: str):
    """Prints evaluation metrics for training data"""
    merged = prophet_df.merge(forecast[["ds", "yhat"]], how="left", on="ds")

    mae = (merged["y"] - merged["yhat"]).abs().mean()
    print(f"MAE for {city} is {mae:.4f} µg/m³")

def plot_forecast(model, forecast: pd.DataFrame, city: str):
    """Saves a forecast plot for the last 30 days of data and future predictions."""
    fig = model.plot(forecast)
    plt.title(f"PM2.5 Forecast for {city} (using Prophet)")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    path = os.path.join(PLOTS_PATH, f"05_forecast_{city.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: 05_forecast_{city.lower()}.png")

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["datetime"])
    print(f"{len(df)} rows loaded")

    cities = df["city"].unique()

    for city in cities:
        print(f"\nProcessing {city}...")
        prophet_df = prepare_prophet_df(df, city)
        model, forecast = train_and_forecast(prophet_df, city)
        evaluate(prophet_df, forecast, city)
        plot_forecast(model, forecast, city)

    print("\nModeling complete.")


if __name__ == "__main__":
    main()