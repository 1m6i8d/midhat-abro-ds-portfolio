import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    run_eda(combined)

def run_eda(df: pd.DataFrame):
    """Generate exploratory plots for the combined AQI dataframe."""
    os.makedirs("assets/demo-screenshots", exist_ok=True)
    sns.set_style("whitegrid")

    # Plot 1, PM2.5 over time per city
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, city in zip(axes, CITIES.keys()):
        city_df = df[df["city"] == city]
        ax.plot(city_df["datetime"], city_df["pm25"]) #, linewidth=0.6
        ax.set_title(f"{city}: Hourly PM2.5") #(2024)
        ax.set_ylabel("PM2.5 (µg/m³)")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("assets/demo-screenshots/01_timeseries.png", dpi=150)
    plt.close()
    print("Saved: 01_timeseries.png")

    # Plot 2, monthly average pm2.5 per city
    df["month"] = df["datetime"].dt.month
    monthly = df.groupby(["city", "month"])["pm25"].mean().reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="month", y="pm25", hue="city", marker="o")
    plt.title("Monthly Average PM2.5 by City (2024)")
    plt.xlabel("Month")
    plt.ylabel("Avg PM2.5 (µg/m³)")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.tight_layout()
    plt.savefig("assets/demo-screenshots/02_monthly_avg.png", dpi=150)
    plt.close()
    print("Saved: 02_monthly_avg.png")

    # Plot 3, pm2.5 distribution per city
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="city", y="pm25")
    plt.title("PM2.5 Distribution by City (2024)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig("assets/demo-screenshots/03_boxplot.png", dpi=150)
    plt.close()
    print("Saved: 03_boxplot.png")

    # Plot 4, avg pm2.5 by hour of day
    df["hour"] = df["datetime"].dt.hour
    hourly = df.groupby(["city", "hour"])["pm25"].mean().reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hourly, x="hour", y="pm25", hue="city", marker="o")
    plt.title("Average PM2.5 by Hour of Day (2024)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg PM2.5 (µg/m³)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig("assets/demo-screenshots/04_hourly_pattern.png", dpi=150)
    plt.close()
    print("Saved: 04_hourly_pattern.png")

    print("\nEDA complete.")

    # EDA Findings
    # 1. Lahore is significantly more polluted than Karachi and Hyderabad year-round,
    # Main reasons are it's inland geography and industries.
    # 2. Pollution spiked after October and settled around after February,
    # This period is the "smog season", where temperature inversions trap pollutants
    # 3. A secondary small spike in July is likely due to monsoon and dust-storms.
    # 4. Lahore shows a strong diurnal daily cycle where pollution is high at night and low in afternoon,
    #    Karachi and Hyderabad are nearly flat all day as sea breezes ventilate constantly

if __name__ == "__main__":
    main()