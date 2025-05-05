"""
pc3_mean_reversion_analysis.py
-------------------------------
Author: Danny Watkins
Updated: 2025
Description:
Analyzes mean-reversion properties of Smoothed PC1, PC2, PC3 scores
using Half-Life calculations, Augmented Dickey-Fuller tests,
Mean Absolute Deviation, and Zero Crossing counts.
Adds VIX context through overlay and correlation heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from statsmodels.tsa.stattools import adfuller

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config.get("data_dir", "data")
visuals_dir = os.path.join(config.get("visuals_dir", "visuals"), "macro_sensitivity")
os.makedirs(visuals_dir, exist_ok=True)

# -------------------- Load Data --------------------
smoothed_pcs = pd.read_csv(os.path.join(data_dir, "fourier_smoothed_pc_scores.csv"), parse_dates=["Date"])
smoothed_pcs.set_index("Date", inplace=True)
vix_data = pd.read_csv(os.path.join(data_dir, "vix_data.csv"), parse_dates=["Date"])
vix_data.set_index("Date", inplace=True)

pc_scores = smoothed_pcs[["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]]

# -------------------- Mean Reversion Metrics --------------------
def calculate_half_life(series):
    series = series.dropna()
    lagged = series.shift(1).dropna()
    delta = series.diff().dropna()
    beta = np.polyfit(lagged, delta, 1)[0]
    return -np.log(2) / beta

analysis_results = {}
for pc in pc_scores.columns:
    series = pc_scores[pc]
    half_life = calculate_half_life(series)
    adf_stat, adf_pvalue, *_ = adfuller(series.dropna())
    analysis_results[pc] = {
        "Half_Life": half_life,
        "ADF_Statistic": adf_stat,
        "ADF_pvalue": adf_pvalue
    }

print("\n[INFO] Mean Reversion Analysis (Smoothed PC Scores):\n")
for pc, metrics in analysis_results.items():
    print(f"{pc}:")
    print(f"  Half-Life: {metrics['Half_Life']:.2f} days")
    print(f"  ADF Statistic: {metrics['ADF_Statistic']:.3f}")
    print(f"  ADF p-value: {metrics['ADF_pvalue']:.5f}")
    print("  [INFO] Stationary" if metrics['ADF_pvalue'] < 0.05 else "  [INFO] Not stationary")
    print("")

# -------------------- Descriptive Metrics --------------------
mad_results = {}
zero_crossing_results = {}
for pc in pc_scores.columns:
    series = pc_scores[pc].dropna()
    mad_results[pc] = np.mean(np.abs(series))
    zero_crossing_results[pc] = ((series.shift(1) * series) < 0).sum()

print("\n[INFO] Smoothed Behavior Summary:")
print("-" * 40)
for pc in pc_scores.columns:
    print(f"{pc}:")
    print(f"  Mean Absolute Deviation: {mad_results[pc]:.4f}")
    print(f"  Zero Crossings: {zero_crossing_results[pc]:.2f}")
    print("-" * 40)

# -------------------- Plot with VIX Overlay --------------------
vix_norm = (vix_data["VIX"] - vix_data["VIX"].mean()) / vix_data["VIX"].std()
vix_norm = vix_norm.loc[pc_scores.index.min():pc_scores.index.max()]

plt.figure(figsize=(14, 7))
plt.plot(pc_scores.index, pc_scores["Smoothed_PC1_Score"], label="PC1 (Level)", color="green", alpha=0.7)
plt.plot(pc_scores.index, pc_scores["Smoothed_PC2_Score"], label="PC2 (Slope)", color="orange", alpha=0.7)
plt.plot(pc_scores.index, pc_scores["Smoothed_PC3_Score"], label="PC3 (Curvature)", color="blue", linewidth=2)
plt.plot(vix_norm.index, vix_norm, label="VIX (Z-Score)", color="gray", linestyle="--", alpha=0.5)
plt.title("Smoothed PC Scores Over Time with VIX")
plt.xlabel("Date")
plt.ylabel("Score Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "smoothed_pc_scores_with_vix.png"))
plt.close()
print(f"[INFO] Saved plot of smoothed PC scores with VIX overlay to '{visuals_dir}/smoothed_pc_scores_with_vix.png'")

# -------------------- Heatmap: PCs vs VIX --------------------
merged = pc_scores.join(vix_data["VIX"], how="inner")
corr_matrix = merged.corr().loc[
    ["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"], ["VIX"]
]

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of Smoothed PCs with VIX")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "pc_vix_correlation_heatmap.png"))
plt.close()
print(f"[INFO] Saved PC–VIX correlation heatmap to '{visuals_dir}/pc_vix_correlation_heatmap.png'")
