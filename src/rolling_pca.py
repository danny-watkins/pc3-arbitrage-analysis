"""
rolling_pca.py
--------------
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

# Load Yield Curve Data
yield_curve = pd.read_csv("data/yield_curve_data.csv", parse_dates=["Date"], index_col="Date")

# Ensure maturities are sorted the same way as `identify_butterfly_maturities.py`
def extract_maturity_number(maturity_label):
    maturity = maturity_label.replace("DGS", "").replace("MO", "").replace("Y", "")
    is_month = "MO" in maturity_label  # True for MO maturities
    try:
        return (is_month, int(maturity))  # Return (is_month flag, maturity value)
    except ValueError:
        return (False, None)  # Return invalid flag

def custom_maturity_sort(maturity_label):
    is_month, num = extract_maturity_number(maturity_label)
    return (not is_month, num)  # Sort by (is_month flag, numeric value)

# Sort maturities correctly before PCA
sorted_maturities = sorted(yield_curve.columns, key=custom_maturity_sort)
yield_curve = yield_curve[sorted_maturities]

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
    for j, maturity in enumerate(sorted_maturities):  
        pca_output[f"PC1_{maturity}"] = pca.components_[0][j]
        pca_output[f"PC2_{maturity}"] = pca.components_[1][j]
        pca_output[f"PC3_{maturity}"] = pca.components_[2][j]

    # Compute the actual PC3 time series (PC3 dislocations)
    pc3_series = np.dot(window_data, pca.components_[2])

    # Save the most recent PC3 dislocation
    pca_output["PC3_Dislocation"] = pc3_series[-1]

    rolling_pca_results.append(pca_output)

# Convert results to DataFrame
rolling_pca_df = pd.DataFrame(rolling_pca_results)

# Save to CSV
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
plt.show()

# Plot PC3 Dislocation Time Series
plt.figure(figsize=(12, 6))
plt.plot(rolling_pca_df["date"], rolling_pca_df["PC3_Dislocation"], color="blue")
plt.title("PC3 Dislocation Time Series")
plt.xlabel("Date")
plt.ylabel("PC3 Dislocation")
plt.grid()
plt.savefig("visuals/pc3_dislocation_timeseries.png")
plt.show()

# Compute average explained variance for PC1, PC2, and PC3
explained_variance_pc1 = rolling_pca_df["Explained_Variance_PC1"].mean()
explained_variance_pc2 = rolling_pca_df["Explained_Variance_PC2"].mean()
explained_variance_pc3 = rolling_pca_df["Explained_Variance_PC3"].mean()

# Store variance contributions
variance_contributions = [explained_variance_pc1, explained_variance_pc2, explained_variance_pc3]
labels = ["PC1", "PC2", "PC3"]

# Print variance percentages to console
print("\n📊 Explained Variance Contributions:")
print(f"PC1: {explained_variance_pc1:.2%}")
print(f"PC2: {explained_variance_pc2:.2%}")
print(f"PC3: {explained_variance_pc3:.2%}")

# Plot explained variance contributions
plt.figure(figsize=(8, 6))
plt.bar(labels, variance_contributions, color=["green", "orange", "blue"])
plt.xlabel("Principal Component")
plt.ylabel("Average Explained Variance Ratio")
plt.title("Explained Variance Contribution of PC1, PC2, and PC3")
plt.ylim(0, 1)

# Annotate the bars with percentage values
for i, v in enumerate(variance_contributions):
    plt.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=12, fontweight="bold")

# Save the visual
plt.savefig("visuals/pca_explained_variance.png")
plt.show()

print("✅ Explained variance visualization saved to visuals folder.")
