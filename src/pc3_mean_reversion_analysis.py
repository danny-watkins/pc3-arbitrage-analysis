""" 
PC Mean Reversion Analysis
-------------------------------
Author: Danny Watkins
Date: 2025
Description:
Analyzes the mean-reverting behavior of PC1, PC2, and PC3 across different market regimes.
Market regimes include low volatility, high volatility, and monetary policy change periods.
Generates visualizations for PC1, PC2, PC3, half-life analysis, and correlation with VIX.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Ensure visuals directory exists
os.makedirs("visuals", exist_ok=True)

# Load Configuration
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
best_maturity = config['belly_maturity']

# Strip any prefix from best maturity (like 'PC3_')
best_maturity = best_maturity.replace("PC3_", "")

# Load Data
start_date = "2015-01-01"
end_date = "2025-01-01"

# Load rolling PCA results (contains PC1, PC2, PC3 values)
rolling_pca_results = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
rolling_pca_results.set_index("date", inplace=True)

# Debug: Print column names to verify structure
print("Available columns in rolling_pca_results:", rolling_pca_results.columns)

# Extract specific maturity PC columns (PC1, PC2, PC3 for best maturity)
pc_columns = [
    f'PC1_{best_maturity}',
    f'PC2_{best_maturity}',
    f'PC3_{best_maturity}'
]

# Validate that the columns exist
missing_columns = [col for col in pc_columns if col not in rolling_pca_results.columns]
if missing_columns:
    print(f"❌ Missing columns in rolling PCA results: {missing_columns}")
    exit(1)

# Load VIX data
vix_data = pd.read_csv("data/vix_data.csv", index_col=0, parse_dates=True)

# Load monetary policy data
monetary_policy_data = pd.read_csv("data/monetary_policy.csv", parse_dates=["Date"])

# Validate data
if rolling_pca_results.empty or vix_data.empty or monetary_policy_data.empty:
    print("❌ One or more datasets are empty. Please check file paths and contents.")
    exit(1)

# Align data to the same timeline
rolling_pca_results = rolling_pca_results.loc[start_date:end_date, pc_columns]
vix_data = vix_data.loc[start_date:end_date]
monetary_policy_data = monetary_policy_data[(monetary_policy_data["Date"] >= start_date) & (monetary_policy_data["Date"] <= end_date)]

# Merge data
merged_data = pd.merge(rolling_pca_results, vix_data, left_index=True, right_index=True, how='inner')

# Merge with monetary policy data
monetary_policy_data['Monetary Policy Change'] = 1
monetary_policy_data.set_index("Date", inplace=True)
merged_data = pd.merge(merged_data, monetary_policy_data, left_index=True, right_index=True, how='left')
merged_data['Monetary Policy Change'].fillna(0, inplace=True)

# Regime Classification
volatility_threshold = merged_data['VIX'].median()
merged_data['Market Regime'] = np.where(merged_data['VIX'] < volatility_threshold, 'Low Volatility', 'High Volatility')
merged_data['Market Regime'] = np.where(merged_data['Monetary Policy Change'] != 0, 'Monetary Policy Change', merged_data['Market Regime'])

# Mean Reversion Calculation (Half-Life)
def calculate_half_life(series):
    series = series - series.mean()
    lagged = series.shift(1).fillna(0)
    delta = series - lagged
    beta = np.polyfit(lagged, delta, 1)[0]
    half_life = -np.log(2) / beta
    return half_life

# Initialize results dictionary
results = {}

# Half-Life and Correlation Analysis
for pc in pc_columns:
    half_lives = {}
    correlations = {}
    for regime in merged_data['Market Regime'].unique():
        regime_data = merged_data[merged_data['Market Regime'] == regime][pc]
        half_life = calculate_half_life(regime_data)
        half_lives[regime] = half_life
        correlation = regime_data.corr(merged_data.loc[regime_data.index, 'VIX'])
        correlations[regime] = correlation
    results[pc] = {'half_lives': half_lives, 'correlations': correlations}

# Print all merged data for analysis
print("\nMerged Data Sample:\n", merged_data.head())

# Print half-lives and correlations
print("\nMean Reversion Analysis Results:")
for pc in pc_columns:
    print(f"\nResults for {pc}:")
    print("Half-Lives by Market Regime:")
    for regime, half_life in results[pc]['half_lives'].items():
        print(f"  {regime}: {half_life:.2f} days")
    print("Correlations with VIX:")
    for regime, correlation in results[pc]['correlations'].items():
        print(f"  {regime}: {correlation:.2f}")

# Plotting Half-Life Comparison
plt.figure(figsize=(12, 6))
for pc in pc_columns:
    plt.bar(results[pc]['half_lives'].keys(), results[pc]['half_lives'].values(), label=pc)
plt.title('Half-Life Comparison of PC1, PC2, and PC3 (Best Maturity) by Market Regime')
plt.xlabel('Market Regime')
plt.ylabel('Half-Life (Days)')
plt.legend()
plt.savefig("visuals/half_life_comparison.png")
plt.close()

print("✅ Half-life comparison visuals saved.")
