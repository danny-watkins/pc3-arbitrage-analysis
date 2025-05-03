# pc3_mean_reversion_analysis.py
# -------------------------------
# Author: Danny Watkins
# Updated: 2025
# Description:
# Analyzes mean-reversion properties of PC1, PC2, PC3 scores
# using Half-Life calculations and Augmented Dickey-Fuller tests.
# Visualizes PC score behaviors and saves clean plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller

# Ensure directories exist
os.makedirs("visuals", exist_ok=True)

# Load Data
rolling_pca = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
vix_data = pd.read_csv("data/vix_data.csv", parse_dates=["Date"])
monetary_policy = pd.read_csv("data/monetary_policy.csv", parse_dates=["Date"])

# Merge Datasets
rolling_pca.set_index("date", inplace=True)
vix_data.set_index("Date", inplace=True)
monetary_policy.set_index("Date", inplace=True)

# Focus only on PC1_Score, PC2_Score, PC3_Score
pc_scores = rolling_pca[["PC1_Score", "PC2_Score", "PC3_Score"]]

# Merge VIX into PCA scores
merged = pc_scores.join(vix_data, how="inner")
merged = merged.rename(columns={"VIX": "VIX_Index"})

# Add Monetary Policy Changes
merged["Monetary_Policy_Change"] = 0
merged.loc[merged.index.intersection(monetary_policy.index), "Monetary_Policy_Change"] = 1


# Classify Market Regimes
vol_threshold = merged["VIX_Index"].median()
merged["Market_Regime"] = np.where(
    merged["Monetary_Policy_Change"] == 1, "Policy Change",
    np.where(merged["VIX_Index"] < vol_threshold, "Low Volatility", "High Volatility")
)

# Mean Reversion Metrics
def calculate_half_life(series):
    series = series.dropna()
    lagged_series = series.shift(1).dropna()
    delta = series.diff().dropna()
    beta = np.polyfit(lagged_series, delta, 1)[0]
    half_life = -np.log(2) / beta
    return half_life

# Analyze each PC score
analysis_results = {}
for pc in ["PC1_Score", "PC2_Score", "PC3_Score"]:
    series = merged[pc]
    half_life = calculate_half_life(series)
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(series.dropna())
    
    analysis_results[pc] = {
        "Half_Life": half_life,
        "ADF_Statistic": adf_stat,
        "ADF_pvalue": adf_pvalue
    }

# Print Results
print("\n📈 Mean Reversion Analysis (Full Period):\n")
for pc, metrics in analysis_results.items():
    print(f"{pc}:")
    print(f"  Half-Life: {metrics['Half_Life']:.2f} days")
    print(f"  ADF Statistic: {metrics['ADF_Statistic']:.3f}")
    print(f"  ADF p-value: {metrics['ADF_pvalue']:.5f}")
    if metrics['ADF_pvalue'] < 0.05:
        print("  ✅ Stationary (mean-reverting)")
    else:
        print("  ❌ Not stationary (non-mean-reverting)")
    print("")

# 📈 Visualization: PC1 vs PC2 vs PC3 over time
plt.figure(figsize=(14, 7))
plt.plot(merged.index, merged["PC1_Score"], label="PC1 Score (Level)", color="green", alpha=0.7)
plt.plot(merged.index, merged["PC2_Score"], label="PC2 Score (Slope)", color="orange", alpha=0.7)
plt.plot(merged.index, merged["PC3_Score"], label="PC3 Score (Curvature)", color="blue", linewidth=2)
plt.title("PC1 vs PC2 vs PC3 Scores Over Time")
plt.xlabel("Date")
plt.ylabel("Score Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("visuals/mean_reversion_scores.png")
plt.close()

print("✅ Visual of PC1, PC2, PC3 scores saved as 'visuals/mean_reversion_scores.png'")
