"""
data_loader.py
--------------
Author: Danny Watkins
Date: 2025
Description: 
This script fetches historical U.S. Treasury yield curve data 
from the FRED API and stores it in a CSV file for further analysis.
Additionally, it generates a covariance matrix plot of the yield curve data 
and saves it to the 'visuals' folder.
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

# Fetch Data for the last 10 years (adjust this date range as needed)
logging.info("Fetching yield curve data from FRED...")
yield_curve = pd.DataFrame({maturity: fred.get_series(maturity, start_date="2013-01-01") for maturity in maturities})

# Handle missing values by forward-filling as opposed to .dropna()
yield_curve.ffill(inplace=True)

# Ensure the date column is properly set as the index
yield_curve.index = pd.to_datetime(yield_curve.index)

# Filter data to include only the last 10 years (2013-2023)
filtered_yield_curve = yield_curve.loc['2013-01-01':'2023-12-31']

# Save the dataset for analysis
os.makedirs("data", exist_ok=True)
filtered_yield_curve.to_csv("data/yield_curve_data.csv", index_label="Date")
logging.info(f"✅ Yield curve data saved successfully. Total rows: {len(filtered_yield_curve)}")

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


