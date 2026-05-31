import requests
import pandas as pd
import os

CITIES = {
    "Hyderabad": {"latitude": 25.37, "longitude": 68.37},
    "Karachi":   {"latitude": 24.86, "longitude": 67.01},
    "Lahore":    {"latitude": 31.55, "longitude": 74.35},
}
RAW_DATA_PATH = "data/raw/aqi_pakistan.csv"
BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

def fetch_aqi_data(city: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch hourly PM2.5 for Pakistani cities using Open-Meteo"""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     "pm2_5",
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "timezone":   "Asia/Karachi",
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    times  = data["hourly"]["time"]
    values = data["hourly"]["pm2_5"]

    df = pd.DataFrame({"datetime": times, "pm25": values})
    df["city"] = city
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def main():
    os.makedirs("data/raw", exist_ok=True)

    all_dfs = []
    for city, coords in CITIES.items():
        print(f"Fetching data for {city}...")
        df = fetch_aqi_data(city, coords["latitude"], coords["longitude"])
        print(f"  → {len(df)} rows fetched")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(RAW_DATA_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {RAW_DATA_PATH}")
    print(combined.head(10))


if __name__ == "__main__":
    main()