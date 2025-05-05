"""
pca_geometry_visuals.py
------------------------
Author: Danny Watkins
Date: 2025
Description:
Generates PCA visualizations for yield curve data, including both 2D and 3D plots.
Visualizations include geometric interpretation and projections onto principal components,
with both original and centered data. Arrows indicate principal component directions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import json

# -------------------- Load Configuration --------------------
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
data_dir = config.get("data_dir", "data")
visual_dir = os.path.join(config.get("visuals_dir", "visuals"), "pca_geometry")
os.makedirs(visual_dir, exist_ok=True)

# -------------------- Load Data --------------------
def load_yield_curve_data():
    return pd.read_csv(os.path.join(data_dir, "yield_curve_data.csv"), index_col=0, parse_dates=True)

# -------------------- 2D Geometric Interpretation --------------------
def geometric_interpretation_2d(data):
    centered = data - data.mean()
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], label="Original Data", alpha=0.6, color="blue")
    plt.scatter(centered.iloc[:, 0], centered.iloc[:, 1], label="Centered Data", alpha=0.6, color="red")
    plt.title("2D Geometric Interpretation: Original vs. Centered Data")
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, "2d_geometric_interpretation.png"))
    plt.close()

# -------------------- 3D Geometric Interpretation --------------------
def geometric_interpretation_3d(data):
    centered = data - data.mean()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], label="Original Data", alpha=0.6, color="blue")
    ax.scatter(centered.iloc[:, 0], centered.iloc[:, 1], centered.iloc[:, 2], label="Centered Data", alpha=0.6, color="red")
    ax.set_title("3D Geometric Interpretation: Original vs. Centered Data")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_zlabel(data.columns[2])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, "3d_geometric_interpretation.png"))
    plt.close()

# -------------------- 2D Projection --------------------
def projection_visualization_2d(data):
    centered = data - data.mean()
    pca = PCA(n_components=2)
    projected = pca.fit_transform(centered)
    plt.figure(figsize=(8, 6))
    plt.scatter(centered.iloc[:, 0], centered.iloc[:, 1], label="Centered Data", alpha=0.6, color="blue")
    plt.scatter(projected[:, 0], projected[:, 1], label="Projected Data", alpha=0.6, color="green", marker="x")
    plt.title("2D Projection onto Principal Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, "2d_projection_visualization.png"))
    plt.close()

# -------------------- 3D Projection with Arrows --------------------
def projection_visualization_3d(data):
    centered = data - data.mean()
    pca = PCA(n_components=3)
    projected = pca.fit_transform(centered)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centered.iloc[:, 0], centered.iloc[:, 1], centered.iloc[:, 2], label="Centered Data", alpha=0.2, color="blue")
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], label="Projected Data", alpha=0.2, color="green", marker="x")

    arrow_len = 3
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color="red", linewidth=3, label="PC1", alpha=0.8)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color="purple", linewidth=3, label="PC2", alpha=0.8)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color="orange", linewidth=3, label="PC3", alpha=0.8)

    ax.set_title("3D Projection with Principal Component Arrows")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, "3d_projection_with_arrows.png"))
    plt.close()

# -------------------- Run Visualizations --------------------
if __name__ == "__main__":
    data = load_yield_curve_data()
    geometric_interpretation_2d(data[["DGS2", "DGS10"]])
    geometric_interpretation_3d(data[["DGS2", "DGS5", "DGS10"]])
    projection_visualization_2d(data[["DGS2", "DGS10"]])
    projection_visualization_3d(data[["DGS2", "DGS5", "DGS10"]])
    
    
print("[INFO] PCA geometry visualizations completed and saved.")

