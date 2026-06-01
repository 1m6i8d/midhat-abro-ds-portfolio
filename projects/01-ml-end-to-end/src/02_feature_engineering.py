import os
import pandas as pd

RAW_DATA_PATH = "data/raw/aqi_pakistan.csv"
PROCESSED_DATA_PATH = "data/processed/aqi_features.csv"

# Load Data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)
    return df

# Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for city, group in df.groupby("city"):
        g = group.copy()

        # time features
        # week: 0=Monday - 6=Sunday
        g["hour"] = g["datetime"].dt.hour
        g["day_of_week"] = g["datetime"].dt.dayofweek
        g["month"] = g["datetime"].dt.month
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)

        # lag features, past pm2.5 values
        g["pm25_lag_1h"] = g["pm25"].shift(1)
        g["pm25_lag_3h"] = g["pm25"].shift(3)
        g["pm25_lag_24h"] = g["pm25"].shift(24)

        # average recent history
        g["pm25_rolling_6h"] = g["pm25"].shift(1).rolling(6).mean()
        g["pm25_rolling_24h"] = g["pm25"].shift(1).rolling(24).mean()

        # drop NaN
        g = g.dropna()

        results.append(g)

    return pd.concat(results, ignore_index=True)

# Main
def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"{len(df)} rows loaded")

    print("Engineering features...")
    df_engineered = engineer_features(df)
    print(f"{len(df_engineered)} rows processed")
    print(f"columns: {df_engineered.columns.tolist()}")

    df_engineered.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved to {PROCESSED_DATA_PATH}")
    print(df_engineered.head())

if __name__ == "__main__":
    main()