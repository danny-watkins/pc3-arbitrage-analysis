"""
pc_macro_sensitivity_analysis.py
---------------------------------
Author: Danny Watkins
Date: 2025
Description:
Computes and visualizes correlation of PC scores (PC1–PC3) with VIX, segmented by market volatility regime.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config.get("data_dir", "data")
visuals_dir = os.path.join(config.get("visuals_dir", "visuals"), "macro_sensitivity")

# -------------------- Setup --------------------
sns.set(style="whitegrid", context="talk")
os.makedirs(visuals_dir, exist_ok=True)

# -------------------- Load Data --------------------
rolling_pca = pd.read_csv(os.path.join(data_dir, "rolling_pca_results.csv"), parse_dates=["date"])
vix_data = pd.read_csv(os.path.join(data_dir, "vix_data.csv"), parse_dates=["Date"])

rolling_pca.set_index("date", inplace=True)
vix_data.set_index("Date", inplace=True)

# -------------------- Merge and Segment --------------------
merged = rolling_pca.join(vix_data, how="inner")
vix_median = merged["VIX"].median()
merged["VIX_Regime"] = np.where(merged["VIX"] > vix_median, "High VIX", "Low VIX")

# -------------------- Compute Correlations --------------------
results = []
for pc in ["PC1_Score", "PC2_Score", "PC3_Score"]:
    for regime in ["High VIX", "Low VIX"]:
        subset = merged[merged["VIX_Regime"] == regime]
        corr = subset[pc].corr(subset["VIX"])
        results.append({"PC": pc, "Regime": regime, "Correlation": corr})

corr_df = pd.DataFrame(results)

# -------------------- Plot --------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=corr_df, x="PC", y="Correlation", hue="Regime", palette="muted")
plt.title("Correlation of PC Scores with VIX by Market Regime")
plt.ylabel("Pearson Correlation")
plt.ylim(-1, 1)
plt.axhline(0, linestyle="--", color="gray")
plt.tight_layout()
plot_path = os.path.join(visuals_dir, "pc_vix_correlation_by_regime.png")
plt.savefig(plot_path)
plt.close()

print(f"[INFO] Saved correlation by regime plot to '{plot_path}'")
