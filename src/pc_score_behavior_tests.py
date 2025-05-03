# pc_score_behavior_tests.py
# ---------------------------
# Author: Danny Watkins
# Updated: 2025
# Description:
# Analyzes deeper mean-reversion behavior of Smoothed PC1, PC2, PC3.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory
os.makedirs("visuals", exist_ok=True)

# Load Fourier Smoothed PC Scores
smoothed_pcs = pd.read_csv("data/fourier_smoothed_pc_scores.csv", parse_dates=["Date"])
smoothed_pcs.set_index("Date", inplace=True)

# Focus on Smoothed PC scores
pc_scores = smoothed_pcs[["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]]

# 1️⃣ Plot Histograms
plt.figure(figsize=(14, 6))
for i, pc in enumerate(["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]):
    plt.subplot(1, 3, i+1)
    plt.hist(pc_scores[pc].dropna(), bins=50, alpha=0.7, label=pc)
    plt.title(f"Histogram of {pc}")
    plt.xlabel("Score Value")
    plt.ylabel("Frequency")
    plt.grid(True)
plt.tight_layout()
plt.savefig("visuals/smoothed_pc_score_histograms.png")
print("✅ Saved histogram of Smoothed PCs to 'visuals/smoothed_pc_score_histograms.png'")

# 2️⃣ Mean Absolute Deviation from 0
mad_results = {}
for pc in ["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]:
    mad = np.mean(np.abs(pc_scores[pc].dropna()))
    mad_results[pc] = mad
    print(f"🔎 Mean Absolute Deviation from 0 for {pc}: {mad:.4f}")

# 3️⃣ Total Zero Crossings
zero_crossing_results = {}
for pc in ["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]:
    series = pc_scores[pc].dropna()
    zero_crossings = ((series.shift(1) * series) < 0).sum()
    zero_crossing_results[pc] = zero_crossings
    print(f"🔎 Total Zero Crossings for {pc}: {zero_crossings}")


# 📋 Summary Printout
print("\n🧠 Smoothed Behavior Summary:")
print("-" * 40)
for pc in ["Smoothed_PC1_Score", "Smoothed_PC2_Score", "Smoothed_PC3_Score"]:
    print(f"{pc}:")
    print(f"  Mean Absolute Deviation: {mad_results[pc]:.4f}")
    print(f"  Zero Crossings/Year: {zero_crossing_results[pc]:.2f}")
    print("-" * 40)
