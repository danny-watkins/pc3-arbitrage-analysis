"""
data_loader.py
--------------
Author: Danny Watkins
Date: 2025
Description: 
This script fetches historical U.S. Treasury yield curve data and VIX data from the FRED API.
Additionally, it downloads significant monetary policy changes (based on Federal Funds Rate),
generates a covariance matrix plot of the yield curve data, and saves all data to the 'data' folder.
*Must register and get unique FRED API key (FREE) from https://fred.stlouisfed.org/docs/api/api_key.html*
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from environment variable
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("❌ FRED API Key is missing. Set it as an environment variable.")

fred = Fred(api_key=FRED_API_KEY)

# Define maturities to fetch from FRED
maturities = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS20", "DGS30"]

# Fetch Yield Curve Data for the last 10 years (adjust this date range as needed)
logging.info("Fetching yield curve data from FRED...")
yield_curve = pd.DataFrame({maturity: fred.get_series(maturity, start_date="2015-01-01") for maturity in maturities})

# Handle missing values by forward-filling as opposed to .dropna()
yield_curve.ffill(inplace=True)

# Ensure the date column is properly set as the index
yield_curve.index = pd.to_datetime(yield_curve.index)

# Filter data to include only the last 10 years (2013-2023)
filtered_yield_curve = yield_curve.loc['2015-01-01':'2025-01-01']

# Save the yield curve dataset for analysis
os.makedirs("data", exist_ok=True)
filtered_yield_curve.to_csv("data/yield_curve_data.csv", index_label="Date")
logging.info(f"✅ Yield curve data saved successfully. Total rows: {len(filtered_yield_curve)}")

# Fetch VIX Data from FRED (CBOE Volatility Index)
logging.info("Fetching VIX data from FRED...")
vix_data = fred.get_series("VIXCLS", start_date="2013-01-01")
vix_data = vix_data.dropna()
vix_data = vix_data.loc['2015-01-01':'2025-01-01']

# Convert to DataFrame and save
vix_df = pd.DataFrame(vix_data, columns=["VIX"])
vix_df.index = pd.to_datetime(vix_df.index)
vix_df.to_csv("data/vix_data.csv", index_label="Date")
logging.info(f"✅ VIX data saved successfully. Total rows: {len(vix_df)}")

# Fetch Federal Funds Rate (Effective Federal Funds Rate - FEDFUNDS)
logging.info("Fetching Federal Funds Rate data from FRED...")
fed_funds = fred.get_series("FEDFUNDS", start_date="2015-01-01")
fed_funds = fed_funds.dropna()
fed_funds.index = pd.to_datetime(fed_funds.index)

# Calculate Daily Changes to Detect Major Policy Shifts
fed_funds_changes = fed_funds.diff().abs()
significant_changes = fed_funds_changes > 0.25  # Threshold for significant rate change (0.25%)

# Create DataFrame for Monetary Policy Changes
monetary_policy_changes = pd.DataFrame({
    "Date": fed_funds.index,
    "Rate Change": fed_funds_changes,
    "Significant Change": significant_changes.astype(int)
}).dropna()

# Filter for significant changes only and limit to the last 10 years
significant_events = monetary_policy_changes[monetary_policy_changes["Significant Change"] == 1]
significant_events = significant_events[
    (significant_events["Date"] >= "2013-01-01") & (significant_events["Date"] <= "2025-01-01")
]

# Save to CSV
significant_events[["Date", "Significant Change"]].to_csv("data/monetary_policy.csv", index=False)
logging.info(f"✅ Monetary policy change data saved successfully. Total significant changes: {len(significant_events)}")

# Plotting the covariance matrix of the yield curve data
def plot_covariance_matrix(data):
    maturities = data.columns
    cov_matrix = np.cov(data, rowvar=False)

    # Plotting the covariance matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cov_matrix, cmap="viridis")
    plt.title("Covariance Matrix of Yield Curve Data", pad=20)
    plt.xlabel("Maturities")
    plt.ylabel("Maturities")
    plt.xticks(range(len(maturities)), maturities, rotation=45)
    plt.yticks(range(len(maturities)), maturities)
    plt.colorbar(cax, label="Covariance")
    plt.tight_layout()

    # Save the plot to the visuals folder
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/covariance_matrix.png")
    plt.close()
    print("✅ Covariance matrix plot saved to visuals/covariance_matrix.png")

# Generate and save covariance matrix plot
plot_covariance_matrix(filtered_yield_curve)


