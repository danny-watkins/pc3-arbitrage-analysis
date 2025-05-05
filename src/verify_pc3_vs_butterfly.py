"""
verify_pc3_vs_butterfly.py
---------------------------
Author: Danny Watkins
Date: 2025
Description:
Compares Fourier-smoothed PC3 signal with:
(1) 2s5s10s butterfly curvature,
(2) A PCA-weighted synthetic curvature portfolio using PC3 eigenvector weights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# -------------------- Load Config --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config["data_dir"]
visuals_dir = os.path.join(config["visuals_dir"], "comparisons")
os.makedirs(visuals_dir, exist_ok=True)

# -------------------- Load Data --------------------
pc3 = pd.read_csv(f"{data_dir}/fourier_filtered_signals.csv", parse_dates=["Date"])
yield_curve = pd.read_csv(f"{data_dir}/yield_curve_data.csv", parse_dates=["Date"])
rolling_pca = pd.read_csv(f"{data_dir}/rolling_pca_results.csv", parse_dates=["date"])

pc3.set_index("Date", inplace=True)
yield_curve.set_index("Date", inplace=True)
rolling_pca.set_index("date", inplace=True)

# -------------------- PART 1: 2s5s10s Butterfly --------------------
yield_curve["2s5s10s_Curvature"] = 2 * yield_curve["DGS5"] - yield_curve["DGS2"] - yield_curve["DGS10"]

merged = pd.merge(
    pc3[["Smoothed_PC3_Score"]],
    yield_curve[["2s5s10s_Curvature"]],
    left_index=True,
    right_index=True,
    how="inner"
).dropna()

correlation = merged["Smoothed_PC3_Score"].corr(merged["2s5s10s_Curvature"])
print(f"[INFO] Correlation with 2s5s10s Butterfly: {correlation:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(merged.index, merged["Smoothed_PC3_Score"], label="Fourier-Smoothed PC3", color="blue")
plt.plot(merged.index, merged["2s5s10s_Curvature"], label="2s5s10s Butterfly Curvature", color="orange")
plt.title(f"Fourier-Smoothed PC3 vs. 2s5s10s Butterfly\nCorrelation = {correlation:.2f}")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{visuals_dir}/pc3_vs_2s5s10s_correlation.png")
plt.close()
print(f"[INFO] Saved: {visuals_dir}/pc3_vs_2s5s10s_correlation.png")

# -------------------- PART 2: PCA-Weighted Synthetic Curvature --------------------
pc3_weights = rolling_pca[["PC3_DGS2", "PC3_DGS5", "PC3_DGS10"]]
pc3_weights.columns = ["w2", "w5", "w10"]

yields_subset = yield_curve[["DGS2", "DGS5", "DGS10"]]
weights_and_yields = pc3_weights.join(yields_subset, how="inner").dropna()

weights_and_yields["Synthetic_PC3_Score"] = (
    weights_and_yields["w2"] * weights_and_yields["DGS2"] +
    weights_and_yields["w5"] * weights_and_yields["DGS5"] +
    weights_and_yields["w10"] * weights_and_yields["DGS10"]
)

synthetic_comparison = pd.merge(
    pc3[["Smoothed_PC3_Score"]],
    weights_and_yields[["Synthetic_PC3_Score"]],
    left_index=True,
    right_index=True,
    how="inner"
).dropna()

synthetic_corr = synthetic_comparison["Smoothed_PC3_Score"].corr(synthetic_comparison["Synthetic_PC3_Score"])
print(f"[INFO] Correlation with PCA-Weighted Synthetic Curvature: {synthetic_corr:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(synthetic_comparison.index, synthetic_comparison["Smoothed_PC3_Score"], label="Fourier-Smoothed PC3", color="blue")
plt.plot(synthetic_comparison.index, synthetic_comparison["Synthetic_PC3_Score"], label="PCA-Weighted Synthetic Trade", color="green")
plt.title(f"Fourier-Smoothed PC3 vs. PCA-Weighted Curvature\nCorrelation = {synthetic_corr:.2f}")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{visuals_dir}/pc3_vs_pca_weighted_synthetic.png")
plt.close()
print(f"[INFO] Saved: {visuals_dir}/pc3_vs_pca_weighted_synthetic.png")
