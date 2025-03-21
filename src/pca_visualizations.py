"""
pca_visualizations.py
----------------------
Author: Danny Watkins
Date: 2025
Description:
Generates PCA visualizations for yield curve data, including both 2D and 3D plots.
Visualizations include geometric interpretation and projections onto principal components,
with both original and centered data. Arrows indicate principal component directions.
Additionally, generates visualizations for PC3 dislocations and mean-reversion analysis,
including overlays with VIX and monetary policy changes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def load_yield_curve_data(filepath):
    """ Load the yield curve data from CSV. """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

# Geometric Interpretation Plot (2D)
def geometric_interpretation_2d(data):
    centered_data = data - np.mean(data, axis=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], label='Original Data', alpha=0.6, color='blue')
    plt.scatter(centered_data.iloc[:, 0], centered_data.iloc[:, 1], label='Centered Data', alpha=0.6, color='red')
    plt.title("2D Geometric Interpretation: Original vs. Centered Data")
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.savefig("visuals/2d_geometric_interpretation.png")
    plt.show()

# Geometric Interpretation Plot (3D)
def geometric_interpretation_3d(data):
    centered_data = data - np.mean(data, axis=0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], label='Original Data', alpha=0.6, color='blue')
    ax.scatter(centered_data.iloc[:, 0], centered_data.iloc[:, 1], centered_data.iloc[:, 2], label='Centered Data', alpha=0.6, color='red')
    ax.set_title("3D Geometric Interpretation: Original vs. Centered Data")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_zlabel(data.columns[2])
    ax.legend()
    plt.savefig("visuals/3d_geometric_interpretation.png")
    plt.show()

# Projection Visualization (2D)
def projection_visualization_2d(data):
    centered_data = data - np.mean(data, axis=0)
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(centered_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(centered_data.iloc[:, 0], centered_data.iloc[:, 1], label='Centered Data', alpha=0.6, color='blue')
    plt.scatter(projected_data[:, 0], projected_data[:, 1], label='Projected Data', alpha=0.6, color='green', marker='x')
    plt.title("2D Projection of Data onto Principal Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig("visuals/2d_projection_visualization.png")
    plt.show()

# Projection Visualization (3D) with arrows
def projection_visualization_3d(data):
    centered_data = data - np.mean(data, axis=0)
    pca = PCA(n_components=3)
    projected_data = pca.fit_transform(centered_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centered_data.iloc[:, 0], centered_data.iloc[:, 1], centered_data.iloc[:, 2], 
               label='Centered Data', alpha=0.2, color='blue')
    ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], 
               label='Projected Data', alpha=0.2, color='green', marker='x')

    # Adding principal component arrows
    arrow_length = 3
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', linewidth=3, label='PC1', alpha=0.8)
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color='purple', linewidth=3, label='PC2', alpha=0.8)
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color='orange', linewidth=3, label='PC3', alpha=0.8)

    ax.set_title("3D Projection of Data onto Principal Components with Arrows")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.savefig("visuals/3d_projection_with_arrows.png")
    plt.show()

# PC3 Mean-Reversion Visualization with Enhanced Overlay and Correlation Analysis
def pc3_mean_reversion_visualization(data, vix, policy_changes):
    # Align all data to the same timeline (2013-2023)
    start_date = "2013-01-01"
    end_date = "2023-12-31"
    data = data.loc[start_date:end_date]
    vix = vix.loc[start_date:end_date]
    policy_changes = policy_changes[(policy_changes['Date'] >= start_date) & (policy_changes['Date'] <= end_date)]

    # Check if data is loaded correctly
    if data.empty or vix.empty or policy_changes.empty:
        print("❌ One or more data frames are empty. Please check data loading.")
        return
    else:
        print("✅ All data frames are loaded correctly.")
        print(f"PC3 data sample:\n{data.head()}")
        print(f"VIX data sample:\n{vix.head()}")
        print(f"Policy change data sample:\n{policy_changes.head()}")

    plt.figure(figsize=(14, 12))

    # Main PC3 and VIX Plot (Overlay)
    ax1 = plt.subplot(3, 1, 1)
    sns.lineplot(data=data, x=data.index, y='PC3', label="PC3 Dislocations", color="blue", linewidth=1.5)
    sns.lineplot(data=data, x=data.index, y='VIX', label="VIX Index", color="purple", linestyle="--", linewidth=1.2)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Mean Reversion Line")
    plt.title("PC3 Dislocations and VIX Overlay with Policy Change Markers")
    plt.xlabel("Date")
    plt.ylabel("PC3 Dislocation / VIX Level")
    plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    # Highlight Policy Changes
    for date in policy_changes['Date']:
        plt.axvline(x=date, color='red', linestyle=':', linewidth=1, alpha=0.7, label="Policy Change" if date == policy_changes['Date'].iloc[0] else "")

    plt.legend(loc='upper left')

    # Rolling Correlation Plot
    ax2 = plt.subplot(3, 1, 2)
    rolling_corr = data['PC3'].rolling(window=30).corr(data['VIX'])
    sns.lineplot(data=rolling_corr, label="30-Day Rolling Correlation (PC3 vs VIX)", color="green")
    plt.title("Rolling Correlation between PC3 and VIX")
    plt.xlabel("Date")
    plt.ylabel("Correlation Coefficient")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Correlation Coefficient Bar Plot by Market Regime
    ax3 = plt.subplot(3, 1, 3)
    regimes = data['Market Regime'].unique()
    correlations = {}
    for regime in regimes:
        regime_data = data[data['Market Regime'] == regime]
        correlation = regime_data['PC3'].corr(regime_data['VIX'])
        correlations[regime] = correlation

    sns.barplot(x=list(correlations.keys()), y=list(correlations.values()), palette="viridis")
    plt.title("Correlation between PC3 and VIX by Market Regime")
    plt.xlabel("Market Regime")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)

    plt.tight_layout()

    # Save the plot to the visuals folder
    output_path = "visuals/pc3_mean_reversion_enhanced_viz.png"
    try:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Visualization saved successfully to {output_path}")
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
    plt.close()
