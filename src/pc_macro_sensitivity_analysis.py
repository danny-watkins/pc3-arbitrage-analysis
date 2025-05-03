# pc_macro_sensitivity_analysis.py
# ---------------------------------
# Author: Danny Watkins
# Date: 2025
# Description:
# Tests sensitivity of PC1, PC2, and PC3 to macro factors (VIX, monetary policy).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output folders exist
os.makedirs("visuals", exist_ok=True)

# Load Data
rolling_pca = pd.read_csv("data/rolling_pca_results.csv", parse_dates=["date"])
vix_data = pd.read_csv("data/vix_data.csv", parse_dates=["Date"])
monetary_policy = pd.read_csv("data/monetary_policy.csv", parse_dates=["Date"])

rolling_pca.set_index("date", inplace=True)
vix_data.set_index("Date", inplace=True)
monetary_policy.set_index("Date", inplace=True)

# Merge PC scores with VIX
merged = rolling_pca.join(vix_data, how="inner")

# 1️⃣ Correlation between PC scores and VIX
print("\n📈 Correlation with VIX:")
for pc in ["PC1_Score", "PC2_Score", "PC3_Score"]:
    corr = merged[pc].corr(merged["VIX"])
    print(f"{pc} vs VIX Correlation: {corr:.4f}")

# 2️⃣ Behavior around monetary policy changes
print("\n📈 Behavior around Monetary Policy Changes:")

# Create 'Policy Event' flag
merged["Policy_Event"] = 0
merged.loc[merged.index.intersection(monetary_policy.index), "Policy_Event"] = 1

# Window: +/- 10 trading days around event
window = 10
event_windows = []

for event_date in monetary_policy.index:
    window_dates = pd.date_range(event_date - pd.Timedelta(days=window*1.5), event_date + pd.Timedelta(days=window*1.5))
    window_data = merged.loc[merged.index.intersection(window_dates)]
    event_windows.append(window_data)

event_data = pd.concat(event_windows)

# Average PC behavior around events
avg_pc1 = event_data.groupby(event_data.index - event_data.index[0])["PC1_Score"].mean()
avg_pc2 = event_data.groupby(event_data.index - event_data.index[0])["PC2_Score"].mean()
avg_pc3 = event_data.groupby(event_data.index - event_data.index[0])["PC3_Score"].mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(avg_pc1.index.days, avg_pc1.values, label="PC1_Score", color="green")
plt.plot(avg_pc2.index.days, avg_pc2.values, label="PC2_Score", color="orange")
plt.plot(avg_pc3.index.days, avg_pc3.values, label="PC3_Score", color="blue")
plt.axvline(x=0, linestyle="--", color="black", label="Policy Event")
plt.title("Average PC Score Behavior Around Monetary Policy Changes")
plt.xlabel("Days from Event")
plt.ylabel("Average Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visuals/pc_behavior_around_policy_events.png")
plt.close()

print("✅ Saved PC behavior plot around policy events to 'visuals/pc_behavior_around_policy_events.png'")

# 3️⃣ Variance of PCs in High VIX vs Low VIX periods
vix_median = merged["VIX"].median()
merged["VIX_Regime"] = np.where(merged["VIX"] > vix_median, "High VIX", "Low VIX")

print("\n📈 Variance of PCs during High vs Low VIX Regimes:")
for pc in ["PC1_Score", "PC2_Score", "PC3_Score"]:
    var_high = merged[merged["VIX_Regime"] == "High VIX"][pc].var()
    var_low = merged[merged["VIX_Regime"] == "Low VIX"][pc].var()
    print(f"{pc} Variance - High VIX: {var_high:.6f}, Low VIX: {var_low:.6f}")
