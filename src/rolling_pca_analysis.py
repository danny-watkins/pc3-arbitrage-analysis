"""
rolling_pca_analysis.py
-------------------------------
Author: Danny Watkins
Date: 2025
Description:
Applies Rolling Principal Component Analysis (PCA) to yield curve data,
saves formatted results for Fourier analysis and trading,
and generates PCA explained variance visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import os

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
mid_maturities = config["mid_maturities"]
start_date = config["start_date"]
end_date = config["end_date"]
window_size = config["window_size"]
pca_components = config["pca_components"]
data_dir = config["data_dir"]
visuals_dir = os.path.join(config["visuals_dir"], "pca_analysis")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

# -------------------- Load and Standardize Data --------------------
yield_curve = pd.read_csv(f"{data_dir}/yield_curve_data.csv", parse_dates=["Date"], index_col="Date")
scaler = StandardScaler()
yield_curve_scaled = scaler.fit_transform(yield_curve)

# -------------------- Rolling PCA --------------------
rolling_pca_results = []

for i in range(len(yield_curve) - window_size):
    window_data = yield_curve_scaled[i : i + window_size]
    pca = PCA(n_components=pca_components)
    pca.fit(window_data)

    pca_output = {
        "date": yield_curve.index[i + window_size],
        "Explained_Variance_PC1": pca.explained_variance_ratio_[0],
        "Explained_Variance_PC2": pca.explained_variance_ratio_[1],
        "Explained_Variance_PC3": pca.explained_variance_ratio_[2],
    }

    for j, maturity in enumerate(yield_curve.columns):
        pca_output[f"PC1_{maturity}"] = pca.components_[0][j]
        pca_output[f"PC2_{maturity}"] = pca.components_[1][j]
        pca_output[f"PC3_{maturity}"] = pca.components_[2][j]

    pc1_series = np.dot(window_data, pca.components_[0])
    pc2_series = np.dot(window_data, pca.components_[1])
    pc3_series = np.dot(window_data, pca.components_[2])
    pca_output["PC1_Score"] = pc1_series[-1]
    pca_output["PC2_Score"] = pc2_series[-1]
    pca_output["PC3_Score"] = pc3_series[-1]

    rolling_pca_results.append(pca_output)

# -------------------- Save Output --------------------
rolling_pca_df = pd.DataFrame(rolling_pca_results)
rolling_pca_df.to_csv(f"{data_dir}/rolling_pca_results.csv", index=False)

print(f"[INFO] Total rows generated: {len(rolling_pca_df)}")
print(f"[INFO] Rolling PCA results saved to '{data_dir}/rolling_pca_results.csv'")

# -------------------- Plot: Explained Variance Over Time --------------------
rolling_pca_df["date"] = pd.to_datetime(rolling_pca_df["date"])

plt.figure(figsize=(12, 6))
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC1"], label="PC1", color="green")
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC2"], label="PC2", color="orange")
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC3"], label="PC3", color="blue")
plt.title("Explained Variance of Principal Components Over Time")
plt.xlabel("Date")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{visuals_dir}/pca_explained_variance.png")
plt.close()
print(f"[INFO] Saved time series of explained variance to '{visuals_dir}/pca_explained_variance.png'")

# -------------------- Plot: Bar Chart of Latest PCA Fit --------------------
explained_variance = pca.explained_variance_ratio_
components = ["PC1", "PC2", "PC3"]

plt.figure(figsize=(12, 6))
bars = plt.bar(components, explained_variance, color=["green", "orange", "blue"])
for bar, var in zip(bars, explained_variance):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{var * 100:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Explained Variance Contribution of PC1, PC2, and PC3")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.ylim(0, 1)
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{visuals_dir}/explained_variance_contribution.png")
plt.close()
print(f"[INFO] Saved bar chart of explained variance to '{visuals_dir}/explained_variance_contribution.png'")

