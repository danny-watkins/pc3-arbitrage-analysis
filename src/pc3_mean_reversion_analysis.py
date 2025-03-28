""" 
pc3_mean_reversion_analysis.py
-------------------------------
Author: Danny Watkins
Date: 2025
Description:
Analyzes the mean-reverting behavior of PC3 across different market regimes.
Market regimes include low volatility, high volatility, and monetary policy change periods.
Generates visualizations for PC3, half-life analysis, and correlation with VIX.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load Configuration
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
best_maturity = config['belly_maturity']

# Load Data
start_date = "2015-01-01"
end_date = "2025-01-01"

# Load rolling PCA results (contains PC3 values)
rolling_pca_results = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
rolling_pca_results.set_index("date", inplace=True)

# Load VIX data
vix_data = pd.read_csv("data/vix_data.csv", index_col=0, parse_dates=True)

# Load monetary policy data
monetary_policy_data = pd.read_csv("data/monetary_policy.csv", parse_dates=["Date"])

# Validate data
if rolling_pca_results.empty or vix_data.empty or monetary_policy_data.empty:
    print("❌ One or more datasets are empty. Please check file paths and contents.")
    exit(1)

# Ensure all data is aligned to the same timeline
rolling_pca_results = rolling_pca_results.loc[start_date:end_date]
vix_data = vix_data.loc[start_date:end_date]
monetary_policy_data = monetary_policy_data[(monetary_policy_data["Date"] >= start_date) & (monetary_policy_data["Date"] <= end_date)]

# Determine the correct PC3 column based on belly maturity
pc3_column = best_maturity

# Merge data
merged_data = pd.merge(rolling_pca_results[[pc3_column]], vix_data, left_index=True, right_index=True, how='inner')
merged_data.columns = ['PC3', 'VIX']

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

# Half-Life and Correlation Analysis
regimes = merged_data['Market Regime'].unique()
half_lives = {}
correlations = {}
for regime in regimes:
    regime_data = merged_data[merged_data['Market Regime'] == regime]['PC3']
    half_life = calculate_half_life(regime_data)
    half_lives[regime] = half_life
    correlation = regime_data.corr(merged_data.loc[regime_data.index, 'VIX'])
    correlations[regime] = correlation

print("\nPC3 Mean-Reversion Half-Lives by Regime:")
for regime, half_life in half_lives.items():
    print(f"{regime}: {half_life:.2f} days")

print("\nCorrelation between PC3 and VIX by Regime:")
for regime, correlation in correlations.items():
    print(f"{regime}: {correlation:.2f}")

# Plotting
plt.figure(figsize=(14, 10))

# Plot 1: PC3 Time Series
plt.subplot(2, 2, 1)
sns.lineplot(data=merged_data, x=merged_data.index, y='PC3', label="PC3", color='blue')
plt.title("PC3 Time Series")
plt.xlabel("Date")
plt.ylabel("PC3 Value")
plt.grid(True)

# Plot 2: Half-Life by Regime
plt.subplot(2, 2, 2)
bars = plt.bar(half_lives.keys(), half_lives.values(), color=['green', 'red', 'orange'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f} days', ha='center', va='bottom')
plt.title('Half-Life of PC3 by Market Regime')
plt.xlabel('Market Regime')
plt.ylabel('Half-Life (Days)')

# Plot 3: Correlation Heatmap
plt.subplot(2, 2, 3)
sns.heatmap(pd.DataFrame(correlations, index=[0]), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation between PC3 and VIX by Market Regime')

plt.tight_layout()
plt.show()
