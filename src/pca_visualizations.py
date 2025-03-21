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

# Example usage
if __name__ == "__main__":
    # Load data
    yield_curve_data = load_yield_curve_data("data/yield_curve_data.csv")

    # Generate visualizations using the correct maturities (2Y and 10Y)
    geometric_interpretation_2d(yield_curve_data[['DGS2', 'DGS10']])
    geometric_interpretation_3d(yield_curve_data[['DGS2', 'DGS5', 'DGS10']])
    projection_visualization_2d(yield_curve_data[['DGS2', 'DGS10']])
    projection_visualization_3d(yield_curve_data[['DGS2', 'DGS5', 'DGS10']])
