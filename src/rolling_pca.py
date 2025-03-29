"""
Rolling PCA Analysis
-------------------------------
Author: Danny Watkins
Date: 2025
Description:
Applies Rolling Principal Component Analysis (PCA) to yield curve data,
saves properly formatted numerical results for Fourier analysis and trading,
and generates PCA visualizations.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import os

# Load Configuration
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
mid_maturities = config['mid_maturities']
start_date = config['start_date']
end_date = config['end_date']

# Load Yield Curve Data
yield_curve = pd.read_csv("data/yield_curve_data.csv", parse_dates=["Date"], index_col="Date")

# Standardize the Data
scaler = StandardScaler()
yield_curve_scaled = scaler.fit_transform(yield_curve)

# Define rolling window size (250 trading days = 1 year of daily data)
window_size = 250
rolling_pca_results = []

# Apply Rolling PCA
for i in range(len(yield_curve) - window_size):
    # Select rolling window
    window_data = yield_curve_scaled[i : i + window_size]

    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(window_data)

    # Store explained variance ratios
    pca_output = {
        "date": yield_curve.index[i + window_size],
        "Explained_Variance_PC1": pca.explained_variance_ratio_[0],
        "Explained_Variance_PC2": pca.explained_variance_ratio_[1],
        "Explained_Variance_PC3": pca.explained_variance_ratio_[2],
    }

    # Store PCA component weights for each maturity
    for j, maturity in enumerate(yield_curve.columns):
        pca_output[f"PC1_{maturity}"] = pca.components_[0][j]
        pca_output[f"PC2_{maturity}"] = pca.components_[1][j]
        pca_output[f"PC3_{maturity}"] = pca.components_[2][j]

    # Compute the actual PC scores (PC1, PC2, PC3 movements)
    pc1_series = np.dot(window_data, pca.components_[0])
    pc2_series = np.dot(window_data, pca.components_[1])
    pc3_series = np.dot(window_data, pca.components_[2])

    # Save the most recent PC scores
    pca_output["PC1_Score"] = pc1_series[-1]
    pca_output["PC2_Score"] = pc2_series[-1]
    pca_output["PC3_Score"] = pc3_series[-1]

    rolling_pca_results.append(pca_output)

# Convert results to DataFrame
rolling_pca_df = pd.DataFrame(rolling_pca_results)

# Save to CSV
os.makedirs("data", exist_ok=True)
rolling_pca_df.to_csv("data/rolling_pca_results.csv", index=False)
print(f"✅ Total rows generated: {len(rolling_pca_df)}")
print("✅ Rolling PCA results saved successfully.")

# ------------------ VISUALIZATIONS ------------------

# Convert date column to datetime for plotting
rolling_pca_df["date"] = pd.to_datetime(rolling_pca_df["date"])

# Plot Explained Variance of PC1, PC2, PC3 Over Time
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC1"], label="PC1", color="green")
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC2"], label="PC2", color="orange")
plt.plot(rolling_pca_df["date"], rolling_pca_df["Explained_Variance_PC3"], label="PC3", color="blue")
plt.title("Explained Variance of Principal Components Over Time")
plt.xlabel("Date")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.grid()
plt.savefig("visuals/pca_explained_variance.png")
plt.close()

print("✅ PCA Explained Variance visualization saved.")

# ------------------ VISUALIZATION: Explained Variance ------------------
# Explained Variance Bar Plot with Percentages and No Grid Lines
explained_variance = [pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1], pca.explained_variance_ratio_[2]]
components = ["PC1", "PC2", "PC3"]

plt.figure(figsize=(12, 6))
bars = plt.bar(components, explained_variance, color=["green", "orange", "blue"])

# Add percentage labels above the bars
for bar, var in zip(bars, explained_variance):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{var*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Explained Variance Contribution of PC1, PC2, and PC3")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.ylim(0, 1)
plt.grid(False)  # Remove grid lines
plt.savefig("visuals/explained_variance_contribution.png")
plt.close()

print("✅ Explained Variance Contribution visualization saved.")
